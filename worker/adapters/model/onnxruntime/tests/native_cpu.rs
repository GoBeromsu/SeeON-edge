//! Opt-in real CPU inference against independently recorded Python ORT CPU outputs.
//! No GPU golden is read, rewritten or replaced by these comparisons.
use std::ffi::CString;
use std::path::{Path, PathBuf};

use seeon_onnxruntime_native::{ErrorKind, Input, Model, Output, Threads};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must name the explicit CPU verification input"))
}

fn sha(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest.iter() {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four bytes")))
        .collect()
}

fn model_path(root: &Path, role: &str) -> PathBuf {
    root.join(match role {
        "fall" => "fall/pose-bbox56-gru/model.onnx",
        "stored_pose" => "pose/yolo26n-pose.onnx",
        "bed" => "bed/yolo26l-seg.onnx",
        _ => panic!("unrecognized test model role"),
    })
}

fn case_input(root: &Path, role: &str, case: &Value) -> (Vec<f32>, Vec<i64>) {
    if role == "fall" {
        let all = std::fs::read(root.join("fall/windows.f32")).expect("fall inputs");
        let window = usize::try_from(case["window"].as_u64().expect("window index"))
            .expect("bounded window index");
        let start = window.checked_mul(30 * 56 * 4).expect("window offset");
        let end = start.checked_add(30 * 56 * 4).expect("window end");
        return (
            floats(all.get(start..end).expect("existing window")),
            vec![1, 30, 56],
        );
    }
    let name = case["input"].as_str().expect("frame identifier");
    assert!(
        name.bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    );
    let directory = if role == "bed" { "bed" } else { "pose" };
    let data = std::fs::read(root.join(format!("{directory}/{name}.images.f32")))
        .expect("recorded model input");
    let size = if role == "bed" { 1280 } else { 640 };
    (floats(&data), vec![1, 3, size, size])
}

#[test]
#[ignore = "requires pinned ORT CPU runtime, pretrained models, inputs and Python CPU receipt"]
fn real_models_match_existing_python_cpu_every_output() {
    let runtime = required_path("SEEON_TEST_ORT_RUNTIME");
    let models = required_path("SEEON_TEST_ORT_MODELS");
    let fixtures = required_path("SEEON_TEST_ORT_FIXTURES");
    let receipt_path = required_path("SEEON_TEST_ORT_CPU_RECEIPT");
    let reference_root = receipt_path.parent().expect("CPU reference directory");
    let receipt = std::fs::read(&receipt_path).expect("independent Python CPU receipt");
    assert!(receipt.len() <= 1024 * 1024);
    let receipt: Value = serde_json::from_slice(&receipt).expect("CPU receipt JSON");
    assert_eq!(receipt["runtime_version"], "1.29.0");
    assert_eq!(
        sha(&std::fs::read(&runtime).expect("actual ORT runtime")),
        receipt["runtime_sha256"]
    );
    let cases = receipt["cases"].as_array().expect("CPU cases");
    assert_eq!(
        cases.len(),
        132,
        "124 fall windows and four frames per image model"
    );
    for role in ["fall", "stored_pose", "bed"] {
        let bytes = std::fs::read(model_path(&models, role)).expect("existing ONNX asset");
        let digest = sha(&bytes);
        let threads = if role == "fall" {
            Threads::Default
        } else {
            Threads::Single
        };
        let mut model = Model::open(&runtime, &bytes, threads).expect("real Rust CPU owner");
        let info = model.info().clone();
        assert_eq!(info.runtime_version, "1.29.0");
        assert_eq!(info.threads, threads);
        assert_eq!(info.input_count, 1);
        let input_name = CString::new(if role == "fall" { "window" } else { "images" }).unwrap();
        let selected: Vec<_> = cases.iter().filter(|case| case["role"] == role).collect();
        assert_eq!(selected.len(), if role == "fall" { 124 } else { 4 });
        for case in selected {
            assert_eq!(digest, case["model_sha256"]);
            let (values, shape) = case_input(&fixtures, role, case);
            let input_bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(sha(&input_bytes), case["input_sha256"]);
            let expected = case["outputs"].as_array().expect("Python CPU outputs");
            assert_eq!(info.output_count as usize, expected.len());
            let names: Vec<_> = (0..expected.len())
                .map(|index| {
                    CString::new(if role == "fall" {
                        "84".to_owned()
                    } else {
                        format!("output{index}")
                    })
                    .unwrap()
                })
                .collect();
            let shapes: Vec<Vec<i64>> = expected
                .iter()
                .map(|output| {
                    output["shape"]
                        .as_array()
                        .expect("shape")
                        .iter()
                        .map(|value| value.as_i64().expect("dimension"))
                        .collect()
                })
                .collect();
            let mut buffers: Vec<Vec<f32>> = shapes
                .iter()
                .map(|shape| {
                    let count = shape
                        .iter()
                        .try_fold(1usize, |count, dimension| {
                            assert!(*dimension > 0);
                            count.checked_mul(usize::try_from(*dimension).unwrap())
                        })
                        .expect("bounded output dimensions");
                    assert!(count <= 16 * 1024 * 1024);
                    vec![f32::NAN; count]
                })
                .collect();
            let mut outputs: Vec<_> = names
                .iter()
                .zip(buffers.iter_mut())
                .map(|(name, data)| Output { name, data })
                .collect();
            let result = model
                .run(
                    Input {
                        name: &input_name,
                        data: &values,
                        shape: &shape,
                    },
                    &mut outputs,
                )
                .expect("actual CPU inference through Rust");
            assert_eq!(result.shapes().len(), expected.len());
            for (index, actual_shape) in result.shapes().iter().enumerate() {
                assert_eq!(actual_shape.dimensions(), shapes[index]);
                assert_eq!(actual_shape.elements(), buffers[index].len());
                assert!(buffers[index].iter().all(|value| value.is_finite()));
                let name = expected[index]["reference_path"]
                    .as_str()
                    .expect("reference file");
                assert!(name.starts_with("reference-") && name.ends_with(".f32"));
                assert!(
                    name.bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.'))
                );
                let raw = std::fs::read(reference_root.join(name)).expect("Python CPU values");
                assert_eq!(
                    sha(&raw),
                    expected[index]["python_cpu_sha256"],
                    "{role} {} output{index}",
                    case["input"]
                );
                let reference = floats(&raw);
                assert_eq!(reference.len(), buffers[index].len());
                for (element, (actual, oracle)) in buffers[index].iter().zip(reference).enumerate()
                {
                    assert!(oracle.is_finite());
                    assert_eq!(
                        actual.to_bits(),
                        oracle.to_bits(),
                        "{role} {} output{index} element{element}",
                        case["input"]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires pinned ORT CPU runtime and the existing fall ONNX model"]
fn refused_input_is_reusable_but_failed_output_remains_poisoned() {
    let runtime = required_path("SEEON_TEST_ORT_RUNTIME");
    let bytes = std::fs::read(model_path(&required_path("SEEON_TEST_ORT_MODELS"), "fall"))
        .expect("existing fall ONNX");
    let mut model = Model::open(&runtime, &bytes, Threads::Single).expect("CPU owner");
    let name = CString::new("window").unwrap();
    let output_name = CString::new("84").unwrap();
    let values = vec![0.25; 30 * 56];
    let mut output = [987.25];
    let invalid = model
        .run(
            Input {
                name: &name,
                data: &values,
                shape: &[0, 30, 56],
            },
            &mut [Output {
                name: &output_name,
                data: &mut output,
            }],
        )
        .expect_err("invalid input");
    assert_eq!(invalid.kind(), ErrorKind::InvalidArgument);
    assert_eq!(output, [987.25]);
    model
        .run(
            Input {
                name: &name,
                data: &values,
                shape: &[1, 30, 56],
            },
            &mut [Output {
                name: &output_name,
                data: &mut output,
            }],
        )
        .expect("input refusal did not poison");
    assert!(output[0].is_finite());
    let mut empty = [];
    let failed = model
        .run(
            Input {
                name: &name,
                data: &values,
                shape: &[1, 30, 56],
            },
            &mut [Output {
                name: &output_name,
                data: &mut empty,
            }],
        )
        .expect_err("output capacity failure");
    assert_eq!(failed.kind(), ErrorKind::Output);
    output[0] = 987.25;
    let poisoned = model
        .run(
            Input {
                name: &name,
                data: &values,
                shape: &[1, 30, 56],
            },
            &mut [Output {
                name: &output_name,
                data: &mut output,
            }],
        )
        .expect_err("sticky poison");
    assert_eq!(poisoned.kind(), ErrorKind::Poisoned);
    assert_eq!(output, [987.25]);
    assert_eq!(model.info().runtime_version, "1.29.0");
}
