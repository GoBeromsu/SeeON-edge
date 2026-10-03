//! Hermetic provider-specific reader coverage and resource-gated integration
//! of fresh four-engine receipts. GPU artifacts and frozen goldens are retained.

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
        engines: EngineSet::TensorRt {
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

mod hybrid_reader {
    // Synthetic file-contract facts only: no model, native, or GPU execution.
    use std::collections::BTreeMap;

    use super::{IdSource, PathBuf, RandomIds, Value, fs, sha256_hex};
    use seeon_ml_worker::config::model_bundle::identity::{
        EnginePaths, IdentityError, IdentityInputs, IdentityKind, verify_aggregate,
    };
    use serde_json::json;

    const IMAGE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const POSE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const BED: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const FALL: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SERVING: &[u8] = b"[property]\nmodel-engine-file=live_pose.engine\nbatch-size=2\n";

    struct Fixture {
        root: PathBuf,
        engines: [PathBuf; 4],
        observer: PathBuf,
        flow: Vec<(&'static str, PathBuf)>,
        identity: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "hybrid-identity-{}-{}",
                std::process::id(),
                RandomIds.uuid4().expect("owned fixture identity")
            ));
            fs::create_dir(&root).expect("exclusive fixture directory");
            let engines = ["live_pose", "stored_pose", "bed", "fall"]
                .map(|role| root.join(format!("{role}.engine")));
            fs::write(&engines[0], b"synthetic-live-engine").unwrap();
            let observer = root.join("observer.so");
            fs::write(&observer, b"synthetic-observer-sdk").unwrap();
            let mut flow = Vec::new();
            for (key, name, bytes) in [
                ("infer_config_sha256", "infer.txt", SERVING),
                (
                    "tracker_config_sha256",
                    "tracker.txt",
                    b"synthetic-tracker-config",
                ),
                (
                    "tracker_library_sha256",
                    "tracker.so",
                    b"synthetic-tracker-library",
                ),
                (
                    "parser_lib_sha256",
                    "parser.so",
                    b"synthetic-parser-library",
                ),
            ] {
                let path = root.join(name);
                fs::write(&path, bytes).unwrap();
                flow.push((key, path));
            }
            let fixture = Self {
                identity: root.join("identity.json"),
                root,
                engines,
                observer,
                flow,
            };
            fixture.write(&fixture.document());
            fixture
        }

        fn receipt(&self, index: usize, source: &str, extra: Value) -> Value {
            let mut receipt = json!({
                "engine": self.engines[index].file_name().unwrap().to_str().unwrap(),
                "engine_sha256": sha256_hex(&fs::read(&self.engines[index]).unwrap()),
                "onnx_sha256": source,
                "image_digest": IMAGE,
                "device": 0,
                "device_name": "synthetic-contract-device",
                "compute_capability": "1.0",
                "trt_version": 1,
            });
            receipt
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            receipt
        }

        fn document(&self) -> Value {
            let flow = self
                .flow
                .iter()
                .map(|(key, path)| (*key, sha256_hex(&fs::read(path).unwrap())))
                .collect::<BTreeMap<_, _>>();
            json!({
                "schema_version": 2,
                "batch_size": 2,
                "engines": {"live_pose": self.receipt(0, POSE, json!({
                    "precision": "fp16",
                    "tf32_enabled": true,
                    "input": "images",
                    "min_dimensions": [1, 3, 640, 640],
                    "opt_dimensions": [2, 3, 640, 640],
                    "max_dimensions": [2, 3, 640, 640],
                    "observer_library_sha256": sha256_hex(&fs::read(&self.observer).unwrap()),
                }))},
                "flow": flow,
                "auxiliary": {
                    "runtime": "onnxruntime",
                    "provider": "cpu",
                    "models": {
                        "stored_pose": {"onnx_sha256": POSE},
                        "bed": {"onnx_sha256": BED},
                        "fall": {"onnx_sha256": FALL},
                    },
                },
            })
        }

        fn tensor_rt_document(&self) -> Value {
            let mut value = self.document();
            value["schema_version"] = json!(1);
            value.as_object_mut().unwrap().remove("auxiliary");
            for (index, (role, source, input, dimensions)) in [
                ("stored_pose", POSE, "images", json!([1, 3, 640, 640])),
                ("bed", BED, "images", json!([1, 3, 1280, 1280])),
                ("fall", FALL, "window", json!([1, 30, 56])),
            ]
            .into_iter()
            .enumerate()
            {
                fs::write(&self.engines[index + 1], format!("synthetic-{role}-engine")).unwrap();
                value["engines"][role] = self.receipt(
                    index + 1,
                    source,
                    json!({
                        "precision": "fp32", "tf32_enabled": false,
                        "input": input, "dimensions": dimensions,
                    }),
                );
            }
            value
        }

        fn inputs(&self) -> IdentityInputs<'_> {
            IdentityInputs {
                engines: EnginePaths::OnnxRuntimeCpu {
                    live_pose: &self.engines[0],
                },
                pose_onnx_sha256: POSE,
                bed_onnx_sha256: BED,
                fall_onnx_sha256: FALL,
                flow: &self.flow,
                observer_library: &self.observer,
                image_digest: IMAGE,
                configured_batch: Some(2),
                deployed_batch: Some(2),
            }
        }

        fn tensor_rt_inputs(&self) -> IdentityInputs<'_> {
            let mut inputs = self.inputs();
            inputs.engines = EnginePaths::TensorRt {
                live_pose: &self.engines[0],
                stored_pose: &self.engines[1],
                bed: &self.engines[2],
                fall: &self.engines[3],
            };
            inputs
        }

        fn write(&self, value: &Value) {
            fs::write(&self.identity, serde_json::to_vec(value).unwrap()).unwrap();
        }

        fn admit(&self) -> Result<BTreeMap<String, String>, IdentityError> {
            verify_aggregate(&self.identity, self.inputs())
        }

        fn rejects(&self, value: &Value, kind: IdentityKind, subject: &str) {
            self.write(value);
            assert_refusal(self.admit(), kind, subject);
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.root) {
                eprintln!("owned hybrid fixture cleanup failed: {error}");
                assert!(
                    std::thread::panicking(),
                    "owned fixture cleanup must succeed"
                );
            }
        }
    }

    fn assert_refusal(
        result: Result<BTreeMap<String, String>, IdentityError>,
        kind: IdentityKind,
        subject: &str,
    ) {
        let error = result.expect_err("tampered identity must refuse");
        assert_eq!((error.kind, error.subject), (kind, subject.to_owned()));
    }

    #[test]
    fn valid_hybrid_projects_only_live_facts_without_auxiliary_engines() {
        let fixture = Fixture::new();
        let value = fixture.document();
        let before = fs::read(&fixture.identity).unwrap();
        let live = value["engines"]["live_pose"].as_object().unwrap();
        let admitted = fixture
            .admit()
            .expect("schema2 with only a live engine admits");
        assert_eq!(admitted.len(), live.len() + 1);
        for (key, expected) in live {
            assert_eq!(
                admitted[key],
                expected
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| expected.to_string())
            );
        }
        assert_eq!(admitted["batch_size"], "2");
        assert!(!admitted.contains_key("auxiliary"));
        assert!(!admitted.contains_key("schema_version"));
        assert_eq!(
            fs::read(&fixture.identity).unwrap(),
            before,
            "reader preserves bytes"
        );
        assert!(fixture.engines[1..].iter().all(|path| !path.exists()));
        let mut request = fixture.inputs();
        request.image_digest = "registry.invalid/ml-worker@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            verify_aggregate(&fixture.identity, request).unwrap(),
            admitted
        );
        let mut value = value;
        value["engines"]["live_pose"]["tf32_enabled"] = json!(false);
        fixture.write(&value);
        assert_eq!(fixture.admit().unwrap()["tf32_enabled"], "false");
    }

    #[test]
    fn schema_and_provider_are_explicit_and_never_fall_back() {
        let fixture = Fixture::new();
        assert_refusal(
            verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs()),
            IdentityKind::Schema,
            "identity",
        );
        for version in [json!(0), json!(1), json!(3), json!("2"), json!(2.0)] {
            let mut value = fixture.document();
            value["schema_version"] = version;
            fixture.rejects(&value, IdentityKind::Schema, "identity");
        }
        for (key, invalid) in [
            ("runtime", json!("tensorrt")),
            ("runtime", json!("ONNXRuntime")),
            ("runtime", json!(null)),
            ("provider", json!("cuda")),
            ("provider", json!("CPU")),
            ("provider", json!("CPUExecutionProvider")),
            ("provider", json!(0)),
        ] {
            let mut value = fixture.document();
            value["auxiliary"][key] = invalid;
            fixture.rejects(&value, IdentityKind::Schema, "auxiliary");
        }
        let gpu = fixture.tensor_rt_document();
        fixture.write(&gpu);
        verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs())
            .expect("schema1 remains valid");
        assert_refusal(fixture.admit(), IdentityKind::Schema, "identity");
    }

    #[test]
    fn hybrid_envelopes_reject_unknown_missing_and_non_object_members() {
        let fixture = Fixture::new();
        for (pointer, subject) in [
            ("", "identity"),
            ("/engines", "engines"),
            ("/flow", "flow"),
            ("/auxiliary", "auxiliary"),
            ("/auxiliary/models", "auxiliary"),
            ("/auxiliary/models/stored_pose", "auxiliary"),
            ("/auxiliary/models/bed", "auxiliary"),
            ("/auxiliary/models/fall", "auxiliary"),
        ] {
            let mut value = fixture.document();
            value
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unexpected".to_owned(), json!(true));
            fixture.rejects(&value, IdentityKind::Schema, subject);
        }
        for (pointer, keys, subject) in [
            (
                "",
                &[
                    "schema_version",
                    "batch_size",
                    "engines",
                    "flow",
                    "auxiliary",
                ][..],
                "identity",
            ),
            ("/engines", &["live_pose"][..], "engines"),
            (
                "/auxiliary",
                &["runtime", "provider", "models"][..],
                "auxiliary",
            ),
            (
                "/auxiliary/models",
                &["stored_pose", "bed", "fall"][..],
                "auxiliary",
            ),
            (
                "/flow",
                &[
                    "infer_config_sha256",
                    "tracker_config_sha256",
                    "tracker_library_sha256",
                    "parser_lib_sha256",
                ][..],
                "flow",
            ),
        ] {
            for key in keys {
                let mut value = fixture.document();
                value
                    .pointer_mut(pointer)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(*key);
                fixture.rejects(&value, IdentityKind::Schema, subject);
            }
        }
        for (pointer, kind, subject) in [
            ("", IdentityKind::NotObject, "identity"),
            ("/engines", IdentityKind::Schema, "engines"),
            ("/engines/live_pose", IdentityKind::Schema, "engines"),
            ("/flow", IdentityKind::Schema, "flow"),
            ("/auxiliary", IdentityKind::Schema, "auxiliary"),
            ("/auxiliary/models", IdentityKind::Schema, "auxiliary"),
            ("/auxiliary/models/bed", IdentityKind::Schema, "auxiliary"),
        ] {
            for invalid in [json!(null), json!([]), json!("not-an-object")] {
                let mut value = fixture.document();
                *value.pointer_mut(pointer).unwrap() = invalid;
                fixture.rejects(&value, kind, subject);
            }
        }
    }

    #[test]
    fn cpu_sources_require_exact_lowercase_hashes_and_admitted_source_identity() {
        let fixture = Fixture::new();
        for role in ["stored_pose", "bed", "fall"] {
            for invalid in [
                json!(""),
                json!("a".repeat(63)),
                json!("a".repeat(65)),
                json!("A".repeat(64)),
                json!("g".repeat(64)),
                json!(0),
                json!(null),
            ] {
                let mut value = fixture.document();
                value["auxiliary"]["models"][role]["onnx_sha256"] = invalid;
                fixture.rejects(&value, IdentityKind::Schema, "auxiliary");
            }
            let mut value = fixture.document();
            value["auxiliary"]["models"][role]
                .as_object_mut()
                .unwrap()
                .remove("onnx_sha256");
            fixture.rejects(&value, IdentityKind::Schema, "auxiliary");
            let mut value = fixture.document();
            value["auxiliary"]["models"][role]["engine"] = json!("model.onnx");
            fixture.rejects(&value, IdentityKind::Schema, "auxiliary");
            let mut value = fixture.document();
            value["auxiliary"]["models"][role]["onnx_sha256"] = json!("b".repeat(64));
            fixture.rejects(
                &value,
                if role == "stored_pose" {
                    IdentityKind::Schema
                } else {
                    IdentityKind::DigestMismatch
                },
                if role == "stored_pose" {
                    "auxiliary"
                } else {
                    role
                },
            );
        }
        let mut value = fixture.document();
        value["engines"]["live_pose"]["onnx_sha256"] = json!("b".repeat(64));
        value["auxiliary"]["models"]["stored_pose"]["onnx_sha256"] = json!("b".repeat(64));
        fixture.rejects(&value, IdentityKind::DigestMismatch, "live_pose");
        fixture.write(&fixture.document());
        let wrong = "0".repeat(64);
        for role in ["live_pose", "bed", "fall"] {
            let mut request = fixture.inputs();
            match role {
                "live_pose" => request.pose_onnx_sha256 = &wrong,
                "bed" => request.bed_onnx_sha256 = &wrong,
                "fall" => request.fall_onnx_sha256 = &wrong,
                _ => unreachable!(),
            }
            assert_refusal(
                verify_aggregate(&fixture.identity, request),
                IdentityKind::DigestMismatch,
                role,
            );
        }
    }

    #[test]
    fn hybrid_live_receipt_keeps_engine_image_hardware_and_precision_checks() {
        let fixture = Fixture::new();
        for (key, invalid, kind, subject) in [
            (
                "engine",
                json!("other.engine"),
                IdentityKind::DigestMismatch,
                "live_pose",
            ),
            (
                "engine",
                json!("/tmp/live_pose.engine"),
                IdentityKind::Schema,
                "engines",
            ),
            ("engine", json!(".."), IdentityKind::Schema, "engines"),
            (
                "engine",
                json!("live\0.engine"),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "engine_sha256",
                json!("0".repeat(64)),
                IdentityKind::DigestMismatch,
                "live_pose",
            ),
            (
                "engine_sha256",
                json!("A".repeat(64)),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "onnx_sha256",
                json!("g".repeat(64)),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "image_digest",
                json!(format!("sha256:{}", "f".repeat(64))),
                IdentityKind::ImageDigest,
                "image_digest",
            ),
            ("device", json!(1), IdentityKind::Schema, "engines"),
            ("device_name", json!(" "), IdentityKind::Schema, "engines"),
            (
                "device_name",
                json!("invalid\0device"),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "compute_capability",
                json!("01.0"),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "compute_capability",
                json!("1"),
                IdentityKind::Schema,
                "engines",
            ),
            ("trt_version", json!(0), IdentityKind::Schema, "engines"),
            ("trt_version", json!("1"), IdentityKind::Schema, "engines"),
            ("precision", json!("fp32"), IdentityKind::Schema, "engines"),
            (
                "tf32_enabled",
                json!("true"),
                IdentityKind::Schema,
                "engines",
            ),
            ("input", json!("window"), IdentityKind::Schema, "engines"),
            (
                "min_dimensions",
                json!([2, 3, 640, 640]),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "opt_dimensions",
                json!([1, 3, 640, 640]),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "max_dimensions",
                json!([3, 3, 640, 640]),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "max_dimensions",
                json!([2, 3, 640.5, 640]),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "observer_library_sha256",
                json!("A".repeat(64)),
                IdentityKind::Schema,
                "engines",
            ),
            (
                "observer_library_sha256",
                json!("0".repeat(64)),
                IdentityKind::DigestMismatch,
                "observer_library_sha256",
            ),
        ] {
            let mut value = fixture.document();
            value["engines"]["live_pose"][key] = invalid;
            fixture.rejects(&value, kind, subject);
        }
        let value = fixture.document();
        for key in value["engines"]["live_pose"].as_object().unwrap().keys() {
            let mut missing = value.clone();
            missing["engines"]["live_pose"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            fixture.rejects(&missing, IdentityKind::Schema, "engines");
        }
        fixture.write(&value);
        let mut request = fixture.inputs();
        request.image_digest = "registry.invalid/ml-worker:latest";
        assert_refusal(
            verify_aggregate(&fixture.identity, request),
            IdentityKind::ImageDigest,
            "image_digest",
        );
    }

    #[test]
    fn hybrid_live_observer_and_all_flow_files_are_still_bound_to_content() {
        let fixture = Fixture::new();
        let artifacts = [
            ("live_pose", fixture.engines[0].clone()),
            ("observer_library_sha256", fixture.observer.clone()),
        ]
        .into_iter()
        .chain(fixture.flow.iter().cloned());
        for (subject, path) in artifacts {
            let original = fs::read(&path).unwrap();
            fs::write(&path, b"tampered-artifact").unwrap();
            assert_refusal(fixture.admit(), IdentityKind::DigestMismatch, subject);
            fs::remove_file(&path).unwrap();
            assert_refusal(fixture.admit(), IdentityKind::ArtifactUnreadable, subject);
            fs::write(&path, &original).unwrap();
        }
        fixture.admit().expect("restored artifacts admit");
        for (key, _) in &fixture.flow {
            let mut value = fixture.document();
            value["flow"][*key] = json!("A".repeat(64));
            fixture.rejects(&value, IdentityKind::DigestInvalid, key);
        }
        fixture.write(&fixture.document());
        for alternate in [
            fixture.flow[..3].to_vec(),
            vec![fixture.flow[0].clone(); 4],
            vec![
                ("unknown", fixture.flow[0].1.clone()),
                fixture.flow[1].clone(),
                fixture.flow[2].clone(),
                fixture.flow[3].clone(),
            ],
        ] {
            let mut request = fixture.inputs();
            request.flow = &alternate;
            assert_refusal(
                verify_aggregate(&fixture.identity, request),
                IdentityKind::Schema,
                "flow",
            );
        }
    }

    #[test]
    fn hybrid_rejects_fingerprinted_build_capable_or_unparseable_infer_config() {
        let fixture = Fixture::new();
        for key in [
            "onnx-file",
            "model-file",
            "proto-file",
            "uff-file",
            "tlt-encoded-model",
            "custom-network-config",
            "engine-create-func-name",
        ] {
            fs::write(
                &fixture.flow[0].1,
                format!(
                    " [ property ] \n model-engine-file=live_pose.engine\n {key} =/models/source\n"
                ),
            )
            .unwrap();
            fixture.rejects(
                &fixture.document(),
                IdentityKind::Schema,
                "infer_config_sha256",
            );
        }
        for bytes in [b"[property]\n\0".as_slice(), b"[property]\n\xff".as_slice()] {
            fs::write(&fixture.flow[0].1, bytes).unwrap();
            fixture.rejects(
                &fixture.document(),
                IdentityKind::Schema,
                "infer_config_sha256",
            );
        }
        fs::write(&fixture.flow[0].1, SERVING).unwrap();
        fixture.write(&fixture.document());
        fixture.admit().expect("engine-only restored config admits");
    }

    #[test]
    fn hybrid_batch_contract_and_deployed_coverage_remain_enforced() {
        let fixture = Fixture::new();
        for invalid in [
            json!(0),
            json!(17),
            json!(-1),
            json!(2.0),
            json!("2"),
            json!(u64::MAX),
        ] {
            let mut value = fixture.document();
            value["batch_size"] = invalid;
            fixture.rejects(&value, IdentityKind::BatchSize, "batch_size");
        }
        fixture.write(&fixture.document());
        let mut request = fixture.inputs();
        request.configured_batch = Some(1);
        assert_refusal(
            verify_aggregate(&fixture.identity, request),
            IdentityKind::BatchSize,
            "batch_size",
        );
        for (deployed, kind) in [
            (-1, IdentityKind::NegativeDeployedBatch),
            (3, IdentityKind::BatchNotCovering),
        ] {
            let mut request = fixture.inputs();
            request.deployed_batch = Some(deployed);
            assert_refusal(
                verify_aggregate(&fixture.identity, request),
                kind,
                "batch_size",
            );
        }
        for deployed in [0, 1, 2] {
            let mut request = fixture.inputs();
            request.deployed_batch = Some(deployed);
            verify_aggregate(&fixture.identity, request)
                .expect("deployed batch covered by live profile");
        }
        let mut value = fixture.document();
        value["batch_size"] = json!(1);
        fixture.write(&value);
        let mut request = fixture.inputs();
        request.configured_batch = Some(1);
        request.deployed_batch = Some(1);
        assert_refusal(
            verify_aggregate(&fixture.identity, request),
            IdentityKind::Schema,
            "engines",
        );
    }

    #[test]
    fn schema1_never_admits_hybrid_receipts_or_missing_gpu_artifacts() {
        let fixture = Fixture::new();
        let mut hybrid = fixture.document();
        hybrid["schema_version"] = json!(1);
        fixture.write(&hybrid);
        assert_refusal(
            verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs()),
            IdentityKind::Schema,
            "identity",
        );
        hybrid.as_object_mut().unwrap().remove("auxiliary");
        fixture.write(&hybrid);
        assert_refusal(
            verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs()),
            IdentityKind::Schema,
            "engines",
        );
        let gpu = fixture.tensor_rt_document();
        fixture.write(&gpu);
        verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs())
            .expect("valid schema1 baseline");
        for (index, role) in ["live_pose", "stored_pose", "bed", "fall"]
            .into_iter()
            .enumerate()
        {
            let mut missing = gpu.clone();
            missing["engines"].as_object_mut().unwrap().remove(role);
            fixture.write(&missing);
            assert_refusal(
                verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs()),
                IdentityKind::Schema,
                "engines",
            );
            fixture.write(&gpu);
            let bytes = fs::read(&fixture.engines[index]).unwrap();
            fs::remove_file(&fixture.engines[index]).unwrap();
            assert_refusal(
                verify_aggregate(&fixture.identity, fixture.tensor_rt_inputs()),
                IdentityKind::ArtifactUnreadable,
                role,
            );
            fs::write(&fixture.engines[index], bytes).unwrap();
        }
    }

    #[test]
    #[ignore = "requires the image-owned SDK observer file; file-only CLI check, no GPU/model execution"]
    fn check_config_cli_admits_cpu_sources_and_refuses_wrong_provider_or_changed_bytes() {
        let fixture = Fixture::new();
        let models = fixture.root.join("models");
        let fall_root = models.join("fall/pose-bbox56-gru");
        fs::create_dir_all(fall_root.join("conformance")).unwrap();
        fs::create_dir_all(models.join("bed")).unwrap();
        let pose = models.join("pose.onnx");
        let bed = models.join("bed/yolo26l-seg.onnx");
        fs::write(&pose, b"synthetic-pose-source").unwrap();
        fs::write(&bed, b"synthetic-bed-source").unwrap();
        // These documents prove only packaged file integrity, not fall metadata
        // validity, ONNX execution, GPU hardware, or usable runtime startup.
        let mut members = Vec::new();
        for (name, bytes) in [
            ("model.onnx", b"synthetic-fall-source".as_slice()),
            ("model.pt", b"synthetic-weights"),
            ("calibration.json", b"{}"),
            ("conformance/fixture.json", b"{}"),
        ] {
            fs::write(fall_root.join(name), bytes).unwrap();
            members.push(json!({
                "relative_path": name, "sha256": sha256_hex(bytes), "size": bytes.len(),
            }));
        }
        fs::write(
            fall_root.join("bundle-manifest.json"),
            serde_json::to_vec(&json!({"files": members})).unwrap(),
        )
        .unwrap();
        let mut document = fixture.document();
        let pose_sha = sha256_hex(&fs::read(&pose).unwrap());
        document["engines"]["live_pose"]["onnx_sha256"] = json!(pose_sha);
        document["engines"]["live_pose"]["observer_library_sha256"] = json!(sha256_hex(
            &fs::read("/opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so")
                .expect("selected file-only integration requires the SDK observer"),
        ));
        document["auxiliary"]["models"]["stored_pose"]["onnx_sha256"] = json!(pose_sha);
        document["auxiliary"]["models"]["bed"]["onnx_sha256"] =
            json!(sha256_hex(&fs::read(&bed).unwrap()));
        document["auxiliary"]["models"]["fall"]["onnx_sha256"] =
            json!(sha256_hex(&fs::read(fall_root.join("model.onnx")).unwrap()));
        fixture.write(&document);
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_ml-worker"));
        command.current_dir(&fixture.root).env_clear();
        if let Some(loader) = std::env::var_os("LD_LIBRARY_PATH") {
            command.env("LD_LIBRARY_PATH", loader);
        }
        command
            .env("RELAY_TOKEN", "file-only-provider-test")
            .env("ML_WORKER_IMAGE", IMAGE)
            .env("ML_WORKER_FLOW_BATCH_SIZE", "2")
            .env("ML_WORKER_FLOW_ENGINE_PATH", &fixture.engines[0])
            .env("ML_WORKER_FLOW_ENGINE_IDENTITY_PATH", &fixture.identity)
            .env("ML_WORKER_FLOW_ONNX_PATH", &pose)
            .env(
                "ML_WORKER_MODEL_SELECTION_PATH",
                fixture.root.join("absent-selection.json"),
            )
            .args(["check-config", "--state-dir"])
            .arg(fixture.root.join("state"));
        for (key, path) in &fixture.flow {
            let environment = match *key {
                "infer_config_sha256" => "ML_WORKER_FLOW_INFER_CONFIG",
                "tracker_config_sha256" => "ML_WORKER_FLOW_TRACKER_CONFIG",
                "tracker_library_sha256" => "ML_WORKER_FLOW_TRACKER_LIBRARY",
                "parser_lib_sha256" => "ML_WORKER_FLOW_PARSER_LIBRARY",
                _ => unreachable!("fixture Flow artifact"),
            };
            command.env(environment, path);
        }
        // Duplicate flags intentionally exercise the same last-value contract
        // as the real dispatcher, without rebuilding the fixture between requests.
        command.arg("--auxiliary-runtime=onnxruntime-cpu");
        let accepted = config_output(&mut command);
        assert!(
            accepted.status.success(),
            "{}",
            String::from_utf8_lossy(&accepted.stderr)
        );
        assert!(fixture.engines[1..].iter().all(|path| !path.exists()));
        command.arg("--auxiliary-runtime=tensorrt");
        assert_eq!(config_output(&mut command).status.code(), Some(3));
        command.arg("--auxiliary-runtime=onnxruntime-cpu");
        assert!(config_output(&mut command).status.success());
        fs::write(&bed, b"changed-after-identity").unwrap();
        assert_eq!(config_output(&mut command).status.code(), Some(3));
    }

    fn config_output(command: &mut std::process::Command) -> std::process::Output {
        use std::process::Stdio;
        use std::time::{Duration, Instant};
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return child.wait_with_output().unwrap(),
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                result => {
                    let _ = child.kill();
                    let output = child.wait_with_output().expect("reap file-only CLI child");
                    panic!("file-only check-config did not finish: {result:?}; {output:?}");
                }
            }
        }
    }
}
