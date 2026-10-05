//! Opt-in CPU role checks against independent Python ORT CPU references.
//! The GPU manifest pins inputs and models only: no CUDA output is compared,
//! rewritten, or replaced, and no accelerator acceptance is asserted here.

use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use seeon_onnxruntime_native::{ErrorKind, Model, Threads};
use seeon_worker::bed_input::{BedInputError, Letterbox};
use seeon_worker::pose_bbox56::PoseBbox56Row;
use seeon_worker::stored_pose::{OUTPUT_SHAPE, PersonBox, StoredPoseError, person_boxes};
use seeon_worker_runtime::cpu::bed::{BedCpu, BedCpuError};
use seeon_worker_runtime::cpu::fall::{FallCpu, FallCpuError};
use seeon_worker_runtime::cpu::stored_pose::{StoredPoseCpu, StoredPoseCpuError};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MANIFEST_SHA256: &str = "1184583122b90f5ecbe83a5f5b9c59e740a4c4701e4600b1a038c5db728cb82b";
const WINDOW_BYTES: usize = 30 * 56 * 4;
const MAX_TENSOR_BYTES: usize = 32 * 1024 * 1024;

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"));
    let text = value
        .to_str()
        .unwrap_or_else(|| panic!("{name} must be UTF-8"));
    assert!(
        !text.trim().is_empty() && !text.contains('\0'),
        "{name} must be a nonblank path"
    );
    PathBuf::from(value)
}

fn read(path: &Path, limit: usize) -> Vec<u8> {
    let metadata = fs::metadata(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert!(
        metadata.is_file() && metadata.len() > 0,
        "{} must be a nonempty file",
        path.display()
    );
    let length = usize::try_from(metadata.len()).expect("file length fits usize");
    assert!(
        length <= limit,
        "{} exceeds the file size bound",
        path.display()
    );
    let file = fs::File::open(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut bytes = Vec::with_capacity(length);
    file.take(u64::try_from(limit).expect("bounded read limit") + 1)
        .read_to_end(&mut bytes)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        length,
        "{} changed while reading",
        path.display()
    );
    bytes
}

fn sha(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest.iter() {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} must be a string"))
}

fn array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"))
}

fn pin(bytes: &[u8], digest: &str, label: &str) {
    assert!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f')),
        "{label} must have a SHA-256 digest"
    );
    assert_eq!(sha(bytes), digest, "{label} SHA-256");
}

fn relative(root: &Path, name: &str) -> PathBuf {
    let path = Path::new(name);
    assert!(
        !name.is_empty()
            && !name.contains('\0')
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "invalid relative fixture path {name:?}"
    );
    root.join(path)
}

fn shape(spec: &Value, expected: &[usize]) -> usize {
    let dimensions = array(spec, "shape");
    assert!(
        !dimensions.is_empty() && dimensions.len() <= 8,
        "bounded tensor rank"
    );
    assert_eq!(dimensions.len(), expected.len(), "tensor rank");
    let count = dimensions
        .iter()
        .zip(expected)
        .try_fold(1usize, |count, (dimension, expected)| {
            let dimension =
                usize::try_from(dimension.as_u64().expect("positive integer dimension"))
                    .expect("dimension fits usize");
            assert!(dimension > 0, "positive tensor dimension");
            assert_eq!(dimension, *expected, "literal tensor shape");
            count.checked_mul(dimension)
        })
        .expect("bounded shape product");
    assert!(
        count
            .checked_mul(4)
            .is_some_and(|bytes| bytes <= MAX_TENSOR_BYTES),
        "bounded float32 tensor"
    );
    count
}

fn floats(bytes: &[u8], count: usize) -> Vec<f32> {
    let length = count.checked_mul(4).expect("bounded float32 byte count");
    assert!(
        length > 0 && length <= MAX_TENSOR_BYTES,
        "bounded float32 input"
    );
    assert_eq!(bytes.len(), length, "float32 shape and byte length");
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let value = f32::from_le_bytes(chunk.try_into().expect("four bytes"));
            assert!(value.is_finite(), "finite fixture/reference float32");
            value
        })
        .collect()
}

fn bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} value count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} element {index} is finite"
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} element {index} float32 bits"
        );
    }
}

fn rows(bytes: &[u8]) -> Vec<PoseBbox56Row> {
    floats(bytes, 30 * 56)
        .chunks_exact(56)
        .map(|row| row.try_into().expect("56-value oldest-first row"))
        .collect()
}

struct Assets {
    runtime: PathBuf,
    models: PathBuf,
    fixtures: PathBuf,
    references: PathBuf,
    manifest: Value,
    receipt: Value,
}

impl Assets {
    fn open() -> Self {
        let runtime = required_path("SEEON_TEST_ORT_RUNTIME");
        let models = required_path("SEEON_TEST_ORT_MODELS");
        let fixtures = required_path("SEEON_TEST_ORT_FIXTURES");
        let receipt_path = required_path("SEEON_TEST_ORT_CPU_RECEIPT");
        assert!(
            models.is_dir(),
            "SEEON_TEST_ORT_MODELS must be an existing directory"
        );
        assert!(
            fixtures.is_dir(),
            "SEEON_TEST_ORT_FIXTURES must be an existing directory"
        );
        let references = receipt_path
            .parent()
            .expect("CPU reference directory")
            .to_path_buf();
        let manifest_bytes = read(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gpu/manifest.json"),
            1024 * 1024,
        );
        pin(
            &manifest_bytes,
            MANIFEST_SHA256,
            "restored GPU manifest (input identity only)",
        );
        let manifest: Value =
            serde_json::from_slice(&manifest_bytes).expect("fixture manifest JSON");
        assert_eq!(manifest["schema"], "seeon-gpu-oracle-fixtures/v2");
        let receipt: Value = serde_json::from_slice(&read(&receipt_path, 1024 * 1024))
            .expect("independent Python CPU receipt JSON");
        assert_eq!(
            receipt["scope"],
            "Python ORT CPU reference; not a CUDA oracle or GPU acceptance"
        );
        assert_eq!(receipt["manifest_sha256"], MANIFEST_SHA256);
        assert_eq!(receipt["runtime_version"], "1.29.0");
        pin(
            &read(&runtime, 512 * 1024 * 1024),
            text(&receipt, "runtime_sha256"),
            "SEEON_TEST_ORT_RUNTIME",
        );
        let cases = array(&receipt, "cases");
        assert_eq!(
            cases.len(),
            132,
            "124 fall windows, four stored-pose and four bed frames"
        );
        for (role, expected) in [("fall", 124), ("stored_pose", 4), ("bed", 4)] {
            assert_eq!(
                cases.iter().filter(|case| case["role"] == role).count(),
                expected,
                "{role} case count"
            );
        }
        for case in cases {
            let role = text(case, "role");
            assert_eq!(
                case["providers"],
                json!(["CPUExecutionProvider"]),
                "CPU-only reference provider"
            );
            assert_eq!(
                case["threads"].as_u64(),
                Some(if role == "fall" { 0 } else { 1 }),
                "reference thread policy"
            );
            assert_eq!(
                array(case, "outputs").len(),
                if role == "bed" { 2 } else { 1 },
                "{role} reference output count"
            );
        }
        Self {
            runtime,
            models,
            fixtures,
            references,
            manifest,
            receipt,
        }
    }

    fn model(&self, role: &str, threads: Threads) -> Model {
        let path = match role {
            "fall" => "fall/pose-bbox56-gru/model.onnx",
            "stored_pose" => "pose/yolo26n-pose.onnx",
            "bed" => "bed/yolo26l-seg.onnx",
            _ => panic!("unknown CPU role"),
        };
        let bytes = read(&self.models.join(path), 512 * 1024 * 1024);
        pin(
            &bytes,
            text(&self.manifest["models"][role], "onnx_sha256"),
            role,
        );
        let model = Model::open(&self.runtime, &bytes, threads).expect("open real CPU model");
        assert_eq!(model.info().runtime_version, "1.29.0");
        assert_eq!(model.info().threads, threads);
        assert_eq!(model.info().input_count, 1);
        assert_eq!(model.info().output_count, if role == "bed" { 2 } else { 1 });
        model
    }

    fn fixture(&self, name: &str) -> Vec<u8> {
        let spec = &self.manifest["fixtures"][name];
        assert!(
            spec.is_object(),
            "{name} must be pinned by manifest.fixtures"
        );
        let count = usize::try_from(spec["bytes"].as_u64().expect("fixture byte length"))
            .expect("fixture length fits usize");
        assert!(
            count > 0 && count <= MAX_TENSOR_BYTES,
            "bounded fixture length"
        );
        let bytes = read(&relative(&self.fixtures, name), count);
        assert_eq!(bytes.len(), count, "{name} fixture length");
        pin(&bytes, text(spec, "sha256"), name);
        bytes
    }

    fn tensor(&self, spec: &Value, expected: &[usize]) -> Vec<u8> {
        let count = shape(spec, expected);
        let bytes = self.fixture(text(spec, "path"));
        assert_eq!(
            bytes.len(),
            count.checked_mul(4).expect("tensor byte count"),
            "pinned input shape"
        );
        bytes
    }

    fn case(&self, role: &str, input: &str, window: Option<usize>, bytes: &[u8]) -> &Value {
        let window = window.map_or(Value::Null, |index| json!(index));
        let selected: Vec<_> = array(&self.receipt, "cases")
            .iter()
            .filter(|case| {
                case["role"] == role && case["input"] == input && case["window"] == window
            })
            .collect();
        assert_eq!(
            selected.len(),
            1,
            "exactly one {role}/{input}/{window} CPU reference"
        );
        let case = selected[0];
        assert_eq!(
            text(case, "model_sha256"),
            text(&self.manifest["models"][role], "onnx_sha256"),
            "{role} reference model identity"
        );
        pin(bytes, text(case, "input_sha256"), "reference input");
        case
    }

    fn reference(&self, case: &Value, index: usize, expected: &[usize]) -> Vec<f32> {
        let output = array(case, "outputs")
            .get(index)
            .expect("CPU reference output index");
        let count = shape(output, expected);
        let name = text(output, "reference_path");
        assert!(
            name.starts_with("reference-")
                && name.ends_with(".f32")
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.'))
                && Path::new(name).components().count() == 1,
            "CPU references must be sibling files"
        );
        let bytes = read(
            &relative(&self.references, name),
            count.checked_mul(4).expect("reference byte count"),
        );
        pin(&bytes, text(output, "python_cpu_sha256"), name);
        floats(&bytes, count)
    }

    fn image_input(&self, role: &str, entry: &Value) -> Vec<u8> {
        let (directory, side) = if role == "bed" {
            ("bed", 1280)
        } else {
            ("pose", 640)
        };
        assert_eq!(
            text(&entry["images"], "path"),
            format!("{directory}/{}.images.f32", text(entry, "id"))
        );
        self.tensor(&entry["images"], &[1, 3, side, side])
    }

    fn rgb(&self, entry: &Value) -> (Vec<u8>, i64, i64) {
        let name = text(entry, "frame");
        assert_eq!(name, format!("frames/{}.rgb", text(entry, "id")));
        let bytes = self.fixture(name);
        let header = bytes.get(..16).expect("SPRGB001 header, width and height");
        assert_eq!(&header[..8], b"SPRGB001", "RGB fixture magic");
        let width = u32::from_le_bytes(header[8..12].try_into().expect("u32 width"));
        let height = u32::from_le_bytes(header[12..16].try_into().expect("u32 height"));
        assert!(width > 0 && height > 0, "positive RGB dimensions");
        let pixels = usize::try_from(width)
            .expect("width fits usize")
            .checked_mul(usize::try_from(height).expect("height fits usize"))
            .and_then(|count| count.checked_mul(3))
            .expect("bounded packed RGB length");
        assert!(pixels <= MAX_TENSOR_BYTES, "bounded RGB fixture");
        assert_eq!(
            bytes.len(),
            pixels.checked_add(16).expect("RGB file length"),
            "exact packed RGB length"
        );
        assert_eq!((width, height), (640, 360), "pinned frame dimensions");
        (bytes[16..].to_vec(), i64::from(width), i64::from(height))
    }
}

#[test]
#[ignore = "requires pinned CPU models and references"]
fn fall_cpu_scores_all_pinned_windows_oldest_first_bit_exact() {
    let assets = Assets::open();
    let spec = &assets.manifest["fall"]["windows"];
    assert_eq!(text(spec, "path"), "fall/windows.f32");
    let windows = assets.tensor(spec, &[124, 30, 56]);
    let first = rows(
        windows
            .get(..WINDOW_BYTES)
            .expect("first complete fall window"),
    );
    let mut owner = FallCpu::new(assets.model("fall", Threads::Default));
    assert_eq!(owner.score(&[]), Err(FallCpuError::Window));
    assert_eq!(owner.score(&first[..29]), Err(FallCpuError::Window));
    let mut long = first.clone();
    long.push(first[29]);
    assert_eq!(owner.score(&long), Err(FallCpuError::Window));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut malformed = first.clone();
        malformed[29][55] = value;
        assert_eq!(owner.score(&malformed), Err(FallCpuError::Window));
    }
    let mut executed = 0;
    for (index, window) in windows.chunks_exact(WINDOW_BYTES).enumerate() {
        let case = assets.case("fall", "windows", Some(index), window);
        let reference = assets.reference(case, 0, &[1, 1]);
        // Do not reverse, resample, calibrate, or apply a sigmoid to these rows.
        let logit = owner
            .score(&rows(window))
            .expect("valid fall window after refusals");
        bits(
            &[logit],
            &reference,
            &format!("fall window {index} raw logit"),
        );
        executed += 1;
    }
    assert_eq!(
        executed, 124,
        "all oldest-first windows executed on one reusable owner"
    );
}

#[test]
#[ignore = "requires pinned CPU models and references"]
fn bed_cpu_infers_real_rgb_and_preserves_pinned_input_bits() {
    let assets = Assets::open();
    let entries = array(&assets.manifest, "bed");
    assert_eq!(entries.len(), 4, "four pinned bed frames");
    let mut owner = BedCpu::new(assets.model("bed", Threads::Single));
    for entry in entries {
        let (rgb, width, height) = assets.rgb(entry);
        let input = assets.image_input("bed", entry);
        let case = assets.case("bed", text(entry, "id"), None, &input);
        let detections = assets.reference(case, 0, &[1, 300, 38]);
        let protos = assets.reference(case, 1, &[1, 32, 320, 320]);
        for (bytes, width, height, error) in [
            (&[][..], 0, height, BedInputError::Dimensions),
            (&[][..], width, -1, BedInputError::Dimensions),
            (&[][..], i64::MAX, 2, BedInputError::DimensionsOverflow),
            (
                &rgb[..rgb.len() - 1],
                width,
                height,
                BedInputError::ImageLength,
            ),
        ] {
            assert_eq!(
                owner
                    .infer(bytes, width, height)
                    .expect_err("malformed RGB"),
                BedCpuError::Input(error)
            );
        }
        {
            let raw = owner
                .infer(&rgb, width, height)
                .expect("actual RGB bed inference after refusals");
            assert_eq!(raw.detections.len(), 300 * 38);
            assert_eq!(raw.protos.len(), 32 * 320 * 320);
            assert_eq!(
                (raw.letterbox.source_width, raw.letterbox.source_height),
                (width, height)
            );
            assert_eq!(
                raw.letterbox,
                Letterbox {
                    source_height: 360,
                    source_width: 640,
                    scale: 2.0,
                    resized_height: 720,
                    resized_width: 1280,
                    pad_top: 280,
                    pad_left: 0,
                }
            );
            bits(
                raw.detections,
                &detections,
                &format!("bed {} ordered raw detections", entry["id"]),
            );
            bits(
                raw.protos,
                &protos,
                &format!("bed {} ordered raw prototypes", entry["id"]),
            );
        }
        // The borrowed raw result is released before observing the owner's tensor.
        bits(
            owner.input(),
            &floats(&input, 3 * 1280 * 1280),
            "bed pinned RGB preprocessing",
        );
    }
}

fn boxes(actual: &[PersonBox], expected: &[PersonBox], id: &str) {
    assert!(
        !expected.is_empty(),
        "{id} CPU reference must contain people"
    );
    assert_eq!(actual.len(), expected.len(), "{id} ordered person count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for (component, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{id} ordered person {index} component {component} float64 bits"
            );
        }
    }
}

#[test]
#[ignore = "requires pinned CPU models and references"]
fn stored_pose_cpu_infers_real_rgb_and_returns_ordered_person_boxes() {
    let assets = Assets::open();
    let entries = array(&assets.manifest, "stored_pose");
    assert_eq!(entries.len(), 4, "four pinned stored-pose frames");
    assert_eq!(OUTPUT_SHAPE, [1, 300, 57]);
    let mut owner = StoredPoseCpu::new(assets.model("stored_pose", Threads::Single), 0.25)
        .expect("explicit valid threshold");
    for entry in entries {
        let id = text(entry, "id");
        let (rgb, width, height) = assets.rgb(entry);
        let input = assets.image_input("stored_pose", entry);
        let case = assets.case("stored_pose", id, None, &input);
        let reference = assets.reference(case, 0, &[1, 300, 57]);
        // Reusing the core decoder is a wiring check, not a second decoder oracle.
        let expected = person_boxes(&reference, &OUTPUT_SHAPE, width, height, 0.25)
            .expect("decode independent CPU raw reference");
        for (bytes, width, height, error) in [
            (&[][..], 0, height, StoredPoseError::Dimensions),
            (&[][..], width, -1, StoredPoseError::Dimensions),
            (&[][..], i64::MAX, 2, StoredPoseError::DimensionsOverflow),
            (
                &rgb[..rgb.len() - 1],
                width,
                height,
                StoredPoseError::ImageLength,
            ),
        ] {
            assert_eq!(
                owner.infer(bytes, width, height),
                Err(StoredPoseCpuError::Input(error))
            );
        }
        let actual = owner
            .infer(&rgb, width, height)
            .expect("actual RGB stored-pose inference after refusals");
        boxes(&actual, &expected, id);
    }
}

#[test]
#[ignore = "requires pinned CPU models and references"]
fn wrong_model_roles_poison_before_later_malformed_input() {
    let assets = Assets::open();
    let windows = assets.tensor(&assets.manifest["fall"]["windows"], &[124, 30, 56]);
    let valid = rows(
        windows
            .get(..WINDOW_BYTES)
            .expect("first complete fall window"),
    );
    let entry = array(&assets.manifest, "bed")
        .first()
        .expect("pinned RGB frame");
    let (rgb, width, height) = assets.rgb(entry);

    let mut fall = FallCpu::new(assets.model("stored_pose", Threads::Default));
    assert_eq!(
        fall.score(&valid),
        Err(FallCpuError::Native(ErrorKind::InvalidArgument))
    );
    assert_eq!(fall.score(&[]), Err(FallCpuError::Poisoned));
    let mut nonfinite = valid.clone();
    nonfinite[29][55] = f32::NAN;
    assert_eq!(fall.score(&nonfinite), Err(FallCpuError::Poisoned));
    assert_eq!(fall.score(&valid), Err(FallCpuError::Poisoned));

    let mut bed = BedCpu::new(assets.model("fall", Threads::Single));
    assert_eq!(
        bed.infer(&rgb, width, height).expect_err("wrong bed model"),
        BedCpuError::Native(ErrorKind::InvalidArgument)
    );
    assert_eq!(
        bed.infer(&[], 0, -1)
            .expect_err("poison precedes bad dimensions"),
        BedCpuError::Poisoned
    );
    assert_eq!(
        bed.infer(&rgb[..rgb.len() - 1], width, height)
            .expect_err("poison precedes bad length"),
        BedCpuError::Poisoned
    );
    assert_eq!(
        bed.infer(&rgb, width, height)
            .expect_err("bed poison is permanent"),
        BedCpuError::Poisoned
    );

    let mut pose = StoredPoseCpu::new(assets.model("bed", Threads::Single), 0.25)
        .expect("valid stored-pose threshold");
    assert_eq!(
        pose.infer(&rgb, width, height),
        Err(StoredPoseCpuError::Native(ErrorKind::InvalidArgument))
    );
    assert_eq!(pose.infer(&[], 0, -1), Err(StoredPoseCpuError::Poisoned));
    assert_eq!(
        pose.infer(&rgb[..rgb.len() - 1], width, height),
        Err(StoredPoseCpuError::Poisoned)
    );
    assert_eq!(
        pose.infer(&rgb, width, height),
        Err(StoredPoseCpuError::Poisoned)
    );
}
