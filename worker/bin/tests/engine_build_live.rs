//! Ignored actual-GPU live pose build. Missing selected prerequisites fail.

use std::fs;
use std::path::{Path, PathBuf};

use seeon_ml_worker::engine_build::{LiveBuildRequest, build_live_pose};
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::seam::{IdSource, RandomIds};

const TEMPLATE: &str = include_str!("../../adapters/deepstream/configs/nvinfer-yolo26-pose.txt");

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_STORED_POSE_ONNX, ML_WORKER_IMAGE, and image-owned /opt/seeon/nvdsinfer-observer/build-result.json plus /opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so"]
fn live_pose_receipt_records_one_fresh_build() {
    let onnx_path = std::env::var("SEEON_TEST_STORED_POSE_ONNX")
        .expect("SEEON_TEST_STORED_POSE_ONNX is required");
    let image = std::env::var("ML_WORKER_IMAGE").expect("ML_WORKER_IMAGE is required");
    for path in [
        "/opt/seeon/nvdsinfer-observer/build-result.json",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so",
    ] {
        assert!(Path::new(path).is_file(), "{path} is required");
    }
    let onnx = fs::read(&onnx_path).expect("stored pose ONNX");
    let expected = sha256_hex(&onnx);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "live-pose-gpu-{}-{}",
        std::process::id(),
        RandomIds.uuid4().expect("id")
    ));
    fs::create_dir(&dir).expect("scratch");
    let engine = dir.join("pose.engine");
    let request = LiveBuildRequest {
        onnx: &onnx,
        expected_onnx_sha256: &expected,
        engine: &engine,
        image_digest: &image,
        infer_config: TEMPLATE,
        batch_size: 2,
    };
    let receipt = build_live_pose(request).expect("fresh live pose build");
    let value = receipt.document();
    println!("live_build_artifacts={}", dir.display());
    println!("live_build_receipt={value}");
    assert_eq!(value["precision"], "fp16");
    assert_eq!(value["onnx_sha256"], expected);
    assert_eq!(
        value["engine_sha256"],
        sha256_hex(&fs::read(&engine).unwrap())
    );
    assert_eq!(value["input"], "images");
    assert!(value["trt_version"].as_i64().unwrap() > 0);
    let library = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so";
    assert_eq!(
        value["observer_library_sha256"],
        sha256_hex(&fs::read(library).unwrap())
    );
    assert_eq!(value["min_dimensions"], serde_json::json!([1, 3, 640, 640]));
    assert_eq!(value["opt_dimensions"], serde_json::json!([2, 3, 640, 640]));
    assert_eq!(value["max_dimensions"], serde_json::json!([2, 3, 640, 640]));
    let scratch: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".live-pose-")
        })
        .collect();
    assert_eq!(scratch.len(), 1);
    let log = fs::read_to_string(scratch[0].join("build.log")).unwrap();
    let observations: Vec<_> = log
        .lines()
        .filter(|line| line.starts_with("INFO: SEEON_BUILD_OBSERVATION_V3 "))
        .collect();
    assert_eq!(observations.len(), 1);
    let observed = observations[0];
    for field in [
        format!(
            "tf32={}",
            u8::from(value["tf32_enabled"].as_bool().unwrap())
        ),
        format!("trt={}", value["trt_version"].as_i64().unwrap()),
        format!("device={}", value["device"].as_i64().unwrap()),
    ] {
        assert!(observed.split_whitespace().any(|part| part == field));
    }
    let (major, minor) = value["compute_capability"]
        .as_str()
        .unwrap()
        .split_once('.')
        .unwrap();
    assert!(observed.contains(&format!("sm_major={major} sm_minor={minor} ")));
    assert!(observed.ends_with(&format!(
        "device_name={}",
        value["device_name"].as_str().unwrap()
    )));
    assert_eq!(
        value["image_digest"],
        image
            .rsplit_once('@')
            .map_or(image.as_str(), |(_, digest)| digest)
    );
    let before = fs::read(&engine).unwrap();
    assert!(build_live_pose(request).is_err());
    assert_eq!(fs::read(&engine).unwrap(), before);
}
