//! FP32 build receipt refusals without CUDA, plus one ignored actual-GPU loop.
//! Expected ONNX digests and GPU identity come from the fixed runtime fixture
//! manifest, never from generated Rust truth. Missing GPU prerequisites fail.

use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use seeon_deepstream_native::StateError;
use seeon_ml_worker::engine_build::{BuildError, BuildRequest, build_fp32};
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::run::ModelRole;
use seeon_ml_worker::seam::{IdSource, RandomIds};
use serde_json::Value;

const MANIFEST: &str = include_str!("../../runtime/inference/tests/fixtures/gpu/manifest.json");
const IMAGE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

struct Scratch(PathBuf);

impl Deref for Scratch {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let result = fs::remove_dir_all(&self.0);
        if !std::thread::panicking() {
            result.expect("remove owned engine fixture");
        } else if let Err(error) = result {
            eprintln!("owned engine fixture cleanup failed: {error}");
        }
    }
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("fixed gpu manifest")
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("field {key} must be a string"))
}

fn request<'a>(
    role: ModelRole,
    onnx: &'a [u8],
    expected: &'a str,
    engine: &'a Path,
    image: &'a str,
    device: i32,
) -> BuildRequest<'a> {
    BuildRequest {
        role,
        onnx,
        expected_onnx_sha256: expected,
        engine,
        image_digest: image,
        device,
    }
}

fn scratch(name: &str) -> Scratch {
    let unique = RandomIds.uuid4().expect("fixture identity");
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "engine-build-{name}-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("fresh owned scratch");
    Scratch(path)
}

#[test]
fn malformed_or_mismatched_source_sha_is_refused_before_cuda() {
    let dir = scratch("sha");
    let engine = dir.join("bed.engine");
    let bytes = b"not-an-onnx-model";
    let malformed = "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    let error = build_fp32(request(ModelRole::Bed, bytes, malformed, &engine, IMAGE, 0))
        .expect_err("uppercase digest");
    assert!(matches!(error, BuildError::SourceDigest), "{error}");
    let short = "0123456789abcdef";
    let error = build_fp32(request(ModelRole::Bed, bytes, short, &engine, IMAGE, 0))
        .expect_err("short digest");
    assert!(matches!(error, BuildError::SourceDigest), "{error}");
    let other = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let error = build_fp32(request(ModelRole::Bed, bytes, other, &engine, IMAGE, 0))
        .expect_err("mismatched digest");
    assert!(matches!(error, BuildError::SourceDigest), "{error}");
    assert!(!engine.exists(), "refused source must not create an engine");
}

#[test]
fn malformed_image_identity_and_arguments_are_refused_without_cuda() {
    let dir = scratch("args");
    let engine = dir.join("pose.engine");
    let bytes = b"captured-onnx";
    let expected = sha256_hex(bytes);
    let images = [
        "latest",
        "sha256:NOT-LOWERCASE-HEX-0123456789abcdef0123456789abcdef0123456789abcdef",
        "sha256:0123",
        "ghcr.io/example/worker:latest",
        "name@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
        "",
    ];
    for image in images {
        let error = build_fp32(request(
            ModelRole::StoredPose,
            bytes,
            &expected,
            &engine,
            image,
            0,
        ))
        .expect_err(image);
        assert!(matches!(error, BuildError::ImageDigest), "{image}: {error}");
    }
    let cases = [
        (ModelRole::Fall, Path::new(""), 0, StateError::InvalidPath),
        (
            ModelRole::Fall,
            engine.as_path(),
            -1,
            StateError::InvalidDevice,
        ),
        (ModelRole::Bed, engine.as_path(), 0, StateError::InvalidOnnx),
    ];
    for (role, path, device, native) in cases {
        let source: &[u8] = if native == StateError::InvalidOnnx {
            b""
        } else {
            bytes
        };
        let digest = sha256_hex(source);
        let error =
            build_fp32(request(role, source, &digest, path, IMAGE, device)).expect_err("argument");
        assert!(
            matches!(error, BuildError::Native(found) if found == native),
            "{error}"
        );
    }
    assert!(
        !engine.exists(),
        "argument refusal must not create an engine"
    );
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_{BED,STORED_POSE,FALL}_ONNX and ML_WORKER_IMAGE"]
fn fp32_receipts_record_actual_builds_and_refuse_rebuild() {
    let manifest = manifest();
    let image = std::env::var("ML_WORKER_IMAGE").expect("ML_WORKER_IMAGE is required");
    assert!(
        seeon_ml_worker::config::model_bundle::identity::deployment_image_digest(&image).is_some(),
        "ML_WORKER_IMAGE must be a real reference@sha256 builder image id"
    );
    let roles = [
        (
            ModelRole::Bed,
            "SEEON_TEST_BED_ONNX",
            "bed.engine",
            text(&manifest["models"]["bed"], "onnx_sha256"),
        ),
        (
            ModelRole::StoredPose,
            "SEEON_TEST_STORED_POSE_ONNX",
            "stored-pose.engine",
            text(&manifest["models"]["stored_pose"], "onnx_sha256"),
        ),
        (
            ModelRole::Fall,
            "SEEON_TEST_FALL_ONNX",
            "fall.engine",
            text(&manifest["models"]["fall"], "onnx_sha256"),
        ),
    ];
    let dir = scratch("gpu");
    for (role, var, file, expected) in roles {
        let onnx_path = std::env::var(var).unwrap_or_else(|_| panic!("{var} is required"));
        let onnx = fs::read(&onnx_path).unwrap_or_else(|error| panic!("{var} unreadable: {error}"));
        assert_eq!(
            sha256_hex(&onnx),
            *expected,
            "{var} is not the fixture model"
        );
        let engine = dir.join(file);
        let receipt = build_fp32(request(role, &onnx, expected, &engine, &image, 0))
            .unwrap_or_else(|error| panic!("{var} build: {error}"));
        let value = receipt.document();
        assert_eq!(value["engine"], file);
        assert_eq!(value["onnx_sha256"], expected);
        let bytes = fs::read(&engine).expect("engine");
        assert_eq!(value["engine_sha256"], sha256_hex(&bytes));
        assert_eq!(value["precision"], "fp32");
        assert_eq!(value["tf32_enabled"], false);
        let trt = value["trt_version"].as_i64().expect("trt version");
        assert!(
            (100_000..110_000).contains(&trt),
            "TensorRT 10 required, got {trt}"
        );
        assert_eq!(value["device_name"], text(&manifest["gpu"], "name"));
        assert_eq!(
            value["compute_capability"],
            text(&manifest["gpu"], "compute_capability")
        );
        assert_eq!(value["device"], 0);
        let (input, dimensions) = match role {
            ModelRole::Bed => ("images", vec![1, 3, 1280, 1280]),
            ModelRole::StoredPose => ("images", vec![1, 3, 640, 640]),
            ModelRole::Fall => ("window", vec![1, 30, 56]),
        };
        assert_eq!(value["input"], input);
        assert_eq!(value["dimensions"], serde_json::json!(dimensions));
        assert_eq!(
            value["image_digest"],
            image
                .rsplit_once('@')
                .map_or(image.as_str(), |(_, digest)| digest)
        );
        assert_ne!(
            value["image_digest"], IMAGE,
            "successful receipt must use the real image id"
        );
        let before = bytes;
        let again = build_fp32(request(role, &onnx, expected, &engine, &image, 0));
        assert!(again.is_err(), "{file} rebuild returned a second receipt");
        assert_eq!(fs::read(&engine).expect("engine after rebuild"), before);
    }
}
