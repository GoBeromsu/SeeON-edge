//! Resource-gated integration: aggregate only fresh, real four-engine receipts.
//! Retain owned build artifacts for inspection; no expected data is regenerated.

use std::fs;
use std::path::{Path, PathBuf};

use seeon_ml_worker::engine_build::{
    BuildRequest, BuiltEngine, EngineReceipt, EngineSet, FlowArtifacts, IdentityRequest,
    LiveBuildRequest, build_fp32, build_live_pose, publish_identity,
};
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::run::ModelRole;
use seeon_ml_worker::seam::{IdSource, RandomIds};
use serde_json::Value;

const MANIFEST: &str = include_str!("../../runtime/rust/tests/fixtures/gpu/manifest.json");
const INFER: &str = include_str!("../../adapters/deepstream/configs/nvinfer-yolo26-pose.txt");

fn source(variable: &str, expected: &str) -> Vec<u8> {
    let path = std::env::var(variable).unwrap_or_else(|_| panic!("{variable} is required"));
    let bytes = fs::read(path).expect("read provisioned ONNX");
    assert_eq!(sha256_hex(&bytes), expected, "frozen ONNX identity");
    bytes
}

fn fp32(role: ModelRole, bytes: &[u8], expected: &str, path: &Path, image: &str) -> EngineReceipt {
    build_fp32(BuildRequest {
        role,
        onnx: bytes,
        expected_onnx_sha256: expected,
        engine: path,
        image_digest: image,
        device: 0,
    })
    .expect("fresh native FP32 engine")
}

#[test]
#[ignore = "requires GPU, pinned observer SDK, parser/tracker, three SEEON_TEST_*_ONNX inputs and ML_WORKER_IMAGE"]
fn aggregate_contains_four_fresh_measured_engine_receipts() {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    let expected = |role: &str| manifest["models"][role]["onnx_sha256"].as_str().unwrap();
    let image = std::env::var("ML_WORKER_IMAGE").expect("actual builder image required");
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "aggregate-gpu-{}-{}",
        std::process::id(),
        RandomIds.uuid4().expect("owned fixture identity")
    ));
    fs::create_dir(&root).expect("exclusive owned build directory");
    println!("aggregate_build_artifacts={}", root.display());
    let pose = source("SEEON_TEST_STORED_POSE_ONNX", expected("stored_pose"));
    let bed = source("SEEON_TEST_BED_ONNX", expected("bed"));
    let fall = source("SEEON_TEST_FALL_ONNX", expected("fall"));
    let live_path = root.join("live-pose.engine");
    let stored_path = root.join("stored-pose.engine");
    let bed_path = root.join("bed.engine");
    let fall_path = root.join("fall.engine");
    let served = root.join("nvinfer-served.txt");
    let mut engine_keys = 0;
    let mut batch_keys = 0;
    let mut config = String::new();
    for line in INFER.lines() {
        if line.starts_with("model-engine-file=") {
            engine_keys += 1;
            config.push_str(&format!("model-engine-file={}\n", live_path.display()));
        } else if line.starts_with("batch-size=") {
            batch_keys += 1;
            config.push_str("batch-size=2\n");
        } else {
            config.push_str(line);
            config.push('\n');
        }
    }
    assert_eq!((engine_keys, batch_keys), (1, 1));
    fs::write(&served, &config).expect("owned served config");
    let live_receipt = build_live_pose(LiveBuildRequest {
        onnx: &pose,
        expected_onnx_sha256: expected("stored_pose"),
        engine: &live_path,
        image_digest: &image,
        infer_config: &config,
        batch_size: 2,
    })
    .expect("fresh nvinfer live engine");
    let stored_receipt = fp32(
        ModelRole::StoredPose,
        &pose,
        expected("stored_pose"),
        &stored_path,
        &image,
    );
    let bed_receipt = fp32(ModelRole::Bed, &bed, expected("bed"), &bed_path, &image);
    let fall_receipt = fp32(ModelRole::Fall, &fall, expected("fall"), &fall_path, &image);
    let parser =
        Path::new("/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo26_pose.so");
    let tracker =
        Path::new("/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so");
    let tracker_config = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("adapters/deepstream/configs/config_tracker_NvDCF_perf.yml");
    let destination = root.join("engine-identity.json");
    let request = || IdentityRequest {
        engines: EngineSet {
            live_pose: BuiltEngine {
                receipt: &live_receipt,
                path: &live_path,
            },
            stored_pose: BuiltEngine {
                receipt: &stored_receipt,
                path: &stored_path,
            },
            bed: BuiltEngine {
                receipt: &bed_receipt,
                path: &bed_path,
            },
            fall: BuiltEngine {
                receipt: &fall_receipt,
                path: &fall_path,
            },
        },
        flow: FlowArtifacts {
            parser_lib: parser,
            infer_config: &served,
            tracker_config: &tracker_config,
            tracker_library: tracker,
        },
        image_digest: &image,
        batch_size: 2,
        destination: &destination,
    };
    publish_identity(request()).expect("publish only fresh measured receipts");
    let bytes = fs::read(&destination).unwrap();
    let identity: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(identity["schema_version"], 1);
    assert_eq!(identity["batch_size"], 2);
    assert_eq!(identity["engines"].as_object().unwrap().len(), 4);
    for (role, receipt, engine) in [
        ("live_pose", &live_receipt, &live_path),
        ("stored_pose", &stored_receipt, &stored_path),
        ("bed", &bed_receipt, &bed_path),
        ("fall", &fall_receipt, &fall_path),
    ] {
        assert_eq!(&identity["engines"][role], receipt.document());
        assert_eq!(
            identity["engines"][role]["engine_sha256"],
            sha256_hex(&fs::read(engine).unwrap())
        );
        assert_eq!(
            identity["engines"][role]["device_name"],
            manifest["gpu"]["name"]
        );
        assert_eq!(
            identity["engines"][role]["compute_capability"],
            manifest["gpu"]["compute_capability"]
        );
    }
    for (key, path) in [
        ("parser_lib_sha256", parser),
        ("infer_config_sha256", served.as_path()),
        ("tracker_config_sha256", tracker_config.as_path()),
        ("tracker_library_sha256", tracker),
    ] {
        assert_eq!(identity["flow"][key], sha256_hex(&fs::read(path).unwrap()));
    }
    assert!(
        publish_identity(request()).is_err(),
        "identity is publish-once"
    );
    assert_eq!(fs::read(&destination).unwrap(), bytes);
    println!("aggregate_identity_sha256={}", sha256_hex(&bytes));
}
