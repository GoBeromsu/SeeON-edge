use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use rustix::fs::{CWD, FileType, Mode};
use serde_json::{Value, json};

use super::super::EngineReceipt;
use super::{
    BuiltEngine, EngineSet, FlowArtifacts, IdentityError, IdentityRequest, publish_identity,
};
use crate::records::id::sha256_hex;
use crate::seam::{IdSource, RandomIds};

const IMAGE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OBSERVER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!(".identity-test-{}", RandomIds.uuid4().expect("id")));
        fs::create_dir(&path).expect("exclusive scratch");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove owned identity fixture");
    }
}

fn engine(dir: &Path, name: &str, bytes: &[u8]) -> (EngineReceipt, PathBuf) {
    let path = dir.join(name);
    fs::write(&path, bytes).expect("engine");
    let digest = sha256_hex(bytes);
    let role = name.trim_end_matches(".engine");
    let document = match role {
        "live_pose" => json!({
            "engine": name,
            "onnx_sha256": SOURCE,
            "engine_sha256": digest,
            "precision": "fp16",
            "tf32_enabled": true,
            "trt_version": 101600,
            "device_name": "NVIDIA GeForce RTX 5070 Ti",
            "compute_capability": "12.0",
            "device": 0,
            "input": "images",
            "image_digest": IMAGE,
            "min_dimensions": [1, 3, 640, 640],
            "opt_dimensions": [2, 3, 640, 640],
            "max_dimensions": [2, 3, 640, 640],
            "observer_library_sha256": OBSERVER,
        }),
        "stored_pose" => static_document(name, &digest, "images", json!([1, 3, 640, 640])),
        "bed" => static_document(name, &digest, "images", json!([1, 3, 1280, 1280])),
        "fall" => static_document(name, &digest, "window", json!([1, 30, 56])),
        _ => panic!("unknown role"),
    };
    (EngineReceipt { document }, path)
}

fn static_document(name: &str, digest: &str, input: &str, dimensions: Value) -> Value {
    json!({
        "engine": name,
        "onnx_sha256": SOURCE,
        "engine_sha256": digest,
        "precision": "fp32",
        "tf32_enabled": false,
        "trt_version": 101600,
        "device_name": "NVIDIA GeForce RTX 5070 Ti",
        "compute_capability": "12.0",
        "device": 0,
        "input": input,
        "dimensions": dimensions,
        "image_digest": IMAGE,
    })
}

struct FlowFiles {
    parser_lib: PathBuf,
    infer_config: PathBuf,
    tracker_config: PathBuf,
    tracker_library: PathBuf,
}

impl FlowFiles {
    fn new(dir: &Path) -> Self {
        let files = Self {
            parser_lib: dir.join("parser.so"),
            infer_config: dir.join("infer.txt"),
            tracker_config: dir.join("tracker.txt"),
            tracker_library: dir.join("tracker.so"),
        };
        for path in [
            &files.parser_lib,
            &files.infer_config,
            &files.tracker_config,
            &files.tracker_library,
        ] {
            fs::write(
                path,
                format!("flow-{}", path.file_name().unwrap().to_string_lossy()),
            )
            .expect("flow");
        }
        files
    }

    fn artifacts(&self) -> FlowArtifacts<'_> {
        FlowArtifacts {
            parser_lib: &self.parser_lib,
            infer_config: &self.infer_config,
            tracker_config: &self.tracker_config,
            tracker_library: &self.tracker_library,
        }
    }
}

fn set<'a>(built: &'a [(EngineReceipt, PathBuf); 4]) -> EngineSet<'a> {
    EngineSet {
        live_pose: BuiltEngine {
            receipt: &built[0].0,
            path: &built[0].1,
        },
        stored_pose: BuiltEngine {
            receipt: &built[1].0,
            path: &built[1].1,
        },
        bed: BuiltEngine {
            receipt: &built[2].0,
            path: &built[2].1,
        },
        fall: BuiltEngine {
            receipt: &built[3].0,
            path: &built[3].1,
        },
    }
}

fn four(dir: &Path) -> [(EngineReceipt, PathBuf); 4] {
    [
        engine(dir, "live_pose.engine", b"live"),
        engine(dir, "stored_pose.engine", b"stored"),
        engine(dir, "bed.engine", b"bed"),
        engine(dir, "fall.engine", b"fall"),
    ]
}

fn request<'a>(
    engines: EngineSet<'a>,
    flow: FlowArtifacts<'a>,
    destination: &'a Path,
) -> IdentityRequest<'a> {
    IdentityRequest {
        engines,
        flow,
        image_digest: IMAGE,
        batch_size: 2,
        destination,
    }
}

#[test]
fn publishes_exact_four_roles_and_preserves_live_tf32() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let destination = scratch.0.join("engine-identity.json");
    publish_identity(request(set(&built), artifacts, &destination)).expect("publish");
    let raw = fs::read(&destination).expect("identity");
    assert!(raw.ends_with(b"\n"));
    let document: Value = serde_json::from_slice(&raw).expect("json");
    assert_eq!(document["schema_version"], 1);
    assert_eq!(document["batch_size"], 2);
    assert_eq!(&document["engines"]["live_pose"], built[0].0.document());
    assert_eq!(&document["engines"]["stored_pose"], built[1].0.document());
    assert_eq!(&document["engines"]["bed"], built[2].0.document());
    assert_eq!(&document["engines"]["fall"], built[3].0.document());
    assert_eq!(document["engines"]["live_pose"]["tf32_enabled"], true);
    assert_eq!(
        &document["flow"]["parser_lib_sha256"],
        &json!(sha256_hex(b"flow-parser.so"))
    );
    assert_eq!(
        &document["flow"]["infer_config_sha256"],
        &json!(sha256_hex(b"flow-infer.txt"))
    );
    assert_eq!(
        &document["flow"]["tracker_config_sha256"],
        &json!(sha256_hex(b"flow-tracker.txt"))
    );
    assert_eq!(
        &document["flow"]["tracker_library_sha256"],
        &json!(sha256_hex(b"flow-tracker.so"))
    );
    let before = fs::read(&destination).expect("first");
    assert!(matches!(
        publish_identity(request(set(&built), artifacts, &destination)),
        Err(IdentityError::Io)
    ));
    assert_eq!(fs::read(&destination).expect("unchanged"), before);
    assert!(!scratch.0.read_dir().unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".identity-")
    }));
}

#[test]
fn swapped_roles_and_static_profile_mismatch_refuse() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let mut swapped = set(&built);
    swapped.bed = BuiltEngine {
        receipt: &built[3].0,
        path: &built[3].1,
    };
    swapped.fall = BuiltEngine {
        receipt: &built[2].0,
        path: &built[2].1,
    };
    let destination = scratch.0.join("swapped.json");
    assert!(matches!(
        publish_identity(request(swapped, artifacts, &destination)),
        Err(IdentityError::Receipt)
    ));
    assert!(!destination.exists());
    let mut receipt = built[1].0.document().clone();
    receipt["dimensions"] = json!([1, 3, 641, 640]);
    let wrong = EngineReceipt { document: receipt };
    let mut mismatched = set(&built);
    mismatched.stored_pose.receipt = &wrong;
    assert!(matches!(
        publish_identity(request(mismatched, artifacts, &destination)),
        Err(IdentityError::Receipt)
    ));
    assert!(!destination.exists());
}

#[test]
fn differing_source_image_and_hardware_refuse() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let destination = scratch.0.join("refused.json");
    let mut changed = built[1].0.document().clone();
    changed["onnx_sha256"] =
        json!("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    let source = EngineReceipt { document: changed };
    let mut engines = set(&built);
    engines.stored_pose.receipt = &source;
    assert!(matches!(
        publish_identity(request(engines, artifacts, &destination)),
        Err(IdentityError::Receipt)
    ));
    let mut image = built[2].0.document().clone();
    image["image_digest"] =
        json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
    let image = EngineReceipt { document: image };
    let mut engines = set(&built);
    engines.bed.receipt = &image;
    assert!(matches!(
        publish_identity(request(engines, artifacts, &destination)),
        Err(IdentityError::Image)
    ));
    let mut device = built[3].0.document().clone();
    device["device"] = json!(1);
    let device = EngineReceipt { document: device };
    let mut engines = set(&built);
    engines.fall.receipt = &device;
    assert!(matches!(
        publish_identity(request(engines, artifacts, &destination)),
        Err(IdentityError::Receipt)
    ));
    assert!(!destination.exists());
    assert!(!scratch.0.read_dir().unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".identity-")
    }));
}
#[test]
fn differing_valid_hardware_fields_refuse_separately() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let destination = scratch.0.join("hardware.json");
    let cases = [
        (2_usize, "trt_version", json!(101601)),
        (3, "device_name", json!("NVIDIA GeForce RTX 4090")),
        (1, "compute_capability", json!("8.9")),
    ];
    for (index, key, value) in cases {
        let mut document = built[index].0.document().clone();
        document[key] = value;
        let receipt = EngineReceipt { document };
        let mut engines = set(&built);
        match index {
            1 => engines.stored_pose.receipt = &receipt,
            2 => engines.bed.receipt = &receipt,
            _ => engines.fall.receipt = &receipt,
        }
        assert!(matches!(
            publish_identity(request(engines, artifacts, &destination)),
            Err(IdentityError::Receipt)
        ));
    }
    assert!(!destination.exists());
}

#[test]
fn malformed_sha_fields_refuse_even_when_pose_hashes_match() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let destination = scratch.0.join("sha.json");
    let bad = json!("not-a-sha");
    let mut live = built[0].0.document().clone();
    let mut stored = built[1].0.document().clone();
    live["onnx_sha256"] = bad.clone();
    stored["onnx_sha256"] = bad.clone();
    let live = EngineReceipt { document: live };
    let stored = EngineReceipt { document: stored };
    let mut engines = set(&built);
    engines.live_pose.receipt = &live;
    engines.stored_pose.receipt = &stored;
    assert!(matches!(
        publish_identity(request(engines, artifacts, &destination)),
        Err(IdentityError::Receipt)
    ));
    for index in [2_usize, 3] {
        let mut document = built[index].0.document().clone();
        document["onnx_sha256"] = bad.clone();
        let receipt = EngineReceipt { document };
        let mut engines = set(&built);
        if index == 2 {
            engines.bed.receipt = &receipt;
        } else {
            engines.fall.receipt = &receipt;
        }
        assert!(matches!(
            publish_identity(request(engines, artifacts, &destination)),
            Err(IdentityError::Receipt)
        ));
    }
    assert!(!destination.exists());
}

#[test]
fn modified_engine_and_invalid_flow_refuse() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    fs::write(&built[0].1, b"changed-live").expect("mutate");
    let destination = scratch.0.join("modified.json");
    assert!(matches!(
        publish_identity(request(set(&built), artifacts, &destination)),
        Err(IdentityError::Output)
    ));
    assert!(!destination.exists());
    fs::write(&built[0].1, b"live").expect("restore known engine bytes");
    let missing = scratch.0.join("missing.so");
    let mut bad = artifacts;
    bad.parser_lib = &missing;
    assert!(matches!(
        publish_identity(request(set(&built), bad, &destination)),
        Err(IdentityError::Flow)
    ));
}
#[test]
fn flow_symlink_is_followed_for_current_artifact_bytes() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let mut artifacts = files.artifacts();
    let linked = scratch.0.join("parser.link");
    symlink(artifacts.parser_lib, &linked).expect("symlink");
    artifacts.parser_lib = &linked;
    let destination = scratch.0.join("linked.json");
    publish_identity(request(set(&built), artifacts, &destination)).expect("follows flow symlink");
    let document: Value = serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
    assert_eq!(
        &document["flow"]["parser_lib_sha256"],
        &json!(sha256_hex(b"flow-parser.so"))
    );
}

#[test]
fn fifo_flow_refuses_without_blocking() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let fifo = scratch.0.join("parser.fifo");
    rustix::fs::mknodat(CWD, &fifo, FileType::Fifo, Mode::RUSR | Mode::WUSR, 0).expect("fifo");
    let mut piped = artifacts;
    piped.parser_lib = &fifo;
    let destination = scratch.0.join("fifo.json");
    let started = std::time::Instant::now();
    assert!(matches!(
        publish_identity(request(set(&built), piped, &destination)),
        Err(IdentityError::Flow)
    ));
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
    assert!(!destination.exists());
}

#[test]
fn existing_destination_bytes_stay_and_preflight_creates_nothing() {
    let scratch = Scratch::new();
    let built = four(&scratch.0);
    let files = FlowFiles::new(&scratch.0);
    let artifacts = files.artifacts();
    let destination = scratch.0.join("kept.json");
    fs::write(&destination, b"kept-bytes").expect("existing");
    assert!(matches!(
        publish_identity(request(set(&built), artifacts, &destination)),
        Err(IdentityError::Io)
    ));
    assert_eq!(fs::read(&destination).unwrap(), b"kept-bytes");
    let absent = scratch.0.join("absent-parent").join("identity.json");
    let mut bad = request(set(&built), artifacts, &absent);
    bad.batch_size = 0;
    assert!(matches!(publish_identity(bad), Err(IdentityError::Batch)));
    assert!(!scratch.0.join("absent-parent").exists());
}
