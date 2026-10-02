//! CPU-only admission of the schema_version 1 aggregate identity.
//!
//! Every byte in this file is a synthetic contract fixture. Device names,
//! TensorRT versions, compute capabilities, and SHA-256 values are published
//! profile facts, not measurements from this process. Nothing here calls
//! CUDA, TensorRT, or a native observer, and no result is native provenance.
//!
//! Paths are owned under `CARGO_TARGET_TMPDIR` or `CARGO_MANIFEST_DIR`. The
//! process working directory and environment are not consulted. Frozen golden
//! files are read and never rewritten.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::iter::repeat_n;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use seeon_ml_worker::config::model_bundle::identity::{
    EnginePaths, IdentityInputs, IdentityKind, fingerprint, verify_aggregate,
};
use seeon_ml_worker::seam::{IdSource, RandomIds};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const IMAGE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const POSE_ONNX: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BED_ONNX: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const FALL_ONNX: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
const DEVICE: &str = "synthetic-contract-device";
const COMPUTE: &str = "1.0";
const TRT_VERSION: i64 = 1;

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let serial = RandomIds.uuid4().expect("owned directory identity");
        let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "aggregate-admission-{}-{label}-{serial}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("exclusive scratch");
        Self(directory)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("owned admission fixture cleanup failed: {error}");
            assert!(
                std::thread::panicking(),
                "owned fixture cleanup must succeed"
            );
        }
    }
}

struct World {
    root: Scratch,
    engines: [PathBuf; 4],
    flow: [PathBuf; 4],
    flow_inputs: Vec<(&'static str, PathBuf)>,
    observer: PathBuf,
    identity: PathBuf,
    live_tf32: bool,
    batch: u32,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write_file(path: &Path, body: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent");
    }
    fs::write(path, body).expect("fixture");
}

fn read_bytes(path: &Path) -> Vec<u8> {
    fs::read(path).expect("preserved bytes")
}

fn common(engine: &str, digest: &str, onnx: &str) -> Value {
    json!({
        "engine": engine,
        "engine_sha256": digest,
        "onnx_sha256": onnx,
        "trt_version": TRT_VERSION,
        "device_name": DEVICE,
        "compute_capability": COMPUTE,
        "device": 0,
        "image_digest": IMAGE,
    })
}

fn with_common(engine: &str, digest: &str, onnx: &str, extra: Value) -> Value {
    let mut document = common(engine, digest, onnx);
    let fields = document.as_object_mut().expect("object");
    for (key, value) in extra.as_object().expect("extra").clone() {
        fields.insert(key, value);
    }
    document
}

fn live_entry(digest: &str, batch: u32, tf32: bool, observer: &str) -> Value {
    let batch = i64::from(batch);
    with_common(
        "live_pose.engine",
        digest,
        POSE_ONNX,
        json!({
            "precision": "fp16",
            "tf32_enabled": tf32,
            "input": "images",
            "min_dimensions": [1, 3, 640, 640],
            "opt_dimensions": [batch, 3, 640, 640],
            "max_dimensions": [batch, 3, 640, 640],
            "observer_library_sha256": observer,
        }),
    )
}

fn static_entry(name: &str, digest: &str, onnx: &str, input: &str, dimensions: Value) -> Value {
    with_common(
        name,
        digest,
        onnx,
        json!({
            "precision": "fp32",
            "tf32_enabled": false,
            "input": input,
            "dimensions": dimensions,
        }),
    )
}

fn flow_value(world: &World) -> Value {
    json!({
        "infer_config_sha256": sha256_hex(&read_bytes(&world.flow[0])),
        "tracker_config_sha256": sha256_hex(&read_bytes(&world.flow[1])),
        "tracker_library_sha256": sha256_hex(&read_bytes(&world.flow[2])),
        "parser_lib_sha256": sha256_hex(&read_bytes(&world.flow[3])),
    })
}

fn document(world: &World) -> Value {
    let digests = world
        .engines
        .iter()
        .map(|path| sha256_hex(&read_bytes(path)))
        .collect::<Vec<_>>();
    let observer = sha256_hex(&read_bytes(&world.observer));
    json!({
        "schema_version": 1,
        "batch_size": world.batch,
        "engines": {
            "live_pose": live_entry(&digests[0], world.batch, world.live_tf32, &observer),
            "stored_pose": static_entry(
                "stored_pose.engine", &digests[1], POSE_ONNX, "images", json!([1, 3, 640, 640])
            ),
            "bed": static_entry(
                "bed.engine", &digests[2], BED_ONNX, "images", json!([1, 3, 1280, 1280])
            ),
            "fall": static_entry(
                "fall.engine", &digests[3], FALL_ONNX, "window", json!([1, 30, 56])
            ),
        },
        "flow": flow_value(world),
    })
}

fn world(label: &str) -> World {
    let root = Scratch::new(label);
    let engines = [
        root.0.join("live_pose.engine"),
        root.0.join("stored_pose.engine"),
        root.0.join("bed.engine"),
        root.0.join("fall.engine"),
    ];
    for (path, body) in engines
        .iter()
        .zip([b"live" as &[u8], b"stored", b"bed", b"fall"])
    {
        write_file(path, body);
    }
    let flow = [
        root.0.join("infer.txt"),
        root.0.join("tracker.txt"),
        root.0.join("tracker.so"),
        root.0.join("parser.so"),
    ];
    for (path, body) in flow.iter().zip([
        b"synthetic-infer" as &[u8],
        b"synthetic-tracker-config",
        b"synthetic-tracker-lib",
        b"synthetic-parser",
    ]) {
        write_file(path, body);
    }
    let observer = root.0.join("observer.so");
    write_file(&observer, b"synthetic-observer-sdk");
    let identity = root.0.join("identity.json");
    let built = World {
        root,
        engines,
        flow_inputs: [
            "infer_config_sha256",
            "tracker_config_sha256",
            "tracker_library_sha256",
            "parser_lib_sha256",
        ]
        .into_iter()
        .zip(flow.iter().cloned())
        .collect(),
        flow,
        observer,
        identity,
        live_tf32: true,
        batch: 2,
    };
    write_file(
        &built.identity,
        &serde_json::to_vec(&document(&built)).expect("document"),
    );
    built
}

fn inputs(world: &World) -> IdentityInputs<'_> {
    IdentityInputs {
        engines: EnginePaths {
            live_pose: &world.engines[0],
            stored_pose: &world.engines[1],
            bed: &world.engines[2],
            fall: &world.engines[3],
        },
        pose_onnx_sha256: POSE_ONNX,
        bed_onnx_sha256: BED_ONNX,
        fall_onnx_sha256: FALL_ONNX,
        flow: &world.flow_inputs,
        observer_library: &world.observer,
        image_digest: IMAGE,
        configured_batch: Some(world.batch),
        deployed_batch: Some(i128::from(world.batch)),
    }
}

fn admit(world: &World) -> Result<BTreeMap<String, String>, (IdentityKind, String)> {
    let request = inputs(world);
    verify_aggregate(&world.identity, request).map_err(|error| (error.kind, error.subject))
}

fn projected(world: &World) -> BTreeMap<String, String> {
    let live = document(world)["engines"]["live_pose"].clone();
    let mut media = live
        .as_object()
        .expect("live object")
        .iter()
        .map(|(key, value)| {
            (
                key.clone(),
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    media.insert("batch_size".to_owned(), world.batch.to_string());
    media
}

fn rewrite(world: &World, value: &Value) {
    write_file(
        &world.identity,
        &serde_json::to_vec(value).expect("rewritten document"),
    );
}

fn mutate_entry(
    world: &mut World,
    role: &str,
    edit: impl FnOnce(&mut serde_json::Map<String, Value>),
) {
    let mut value = document(world);
    let entry = value["engines"][role]
        .as_object_mut()
        .expect("engine entry");
    edit(entry);
    rewrite(world, &value);
}

#[test]
fn valid_four_entry_aggregate_projects_live_media_facts() {
    let fixture = world("valid");
    let before = read_bytes(&fixture.identity);
    let admitted = admit(&fixture).expect("synthetic schema1 admits");
    assert_eq!(admitted, projected(&fixture));
    assert_eq!(admitted["tf32_enabled"], "true");
    assert_eq!(admitted["batch_size"], "2");
    assert_eq!(admitted["precision"], "fp16");
    assert_eq!(admitted["engine"], "live_pose.engine");
    assert!(!admitted.contains_key("infer_config_sha256"));
    assert!(!admitted.contains_key("schema_version"));
    assert_eq!(
        read_bytes(&fixture.identity),
        before,
        "reader preserves bytes"
    );
}

#[test]
fn legacy_flat_golden_is_explicitly_refused() {
    let fixture = world("legacy");
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/d/engine-identity.json");
    let original = read_bytes(&golden);
    fs::copy(&golden, &fixture.identity).expect("copy frozen golden; never rewrite it");
    assert_eq!(read_bytes(&golden), original);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "identity".to_owned()))
    );
    assert_eq!(read_bytes(&golden), original, "golden bytes stay frozen");
}

#[test]
fn malformed_and_missing_role_documents_refuse() {
    let fixture = world("malformed");
    let mut value = document(&fixture);
    value["engines"]
        .as_object_mut()
        .expect("engines")
        .remove("fall");
    rewrite(&fixture, &value);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "engines".to_owned()))
    );

    value = document(&fixture);
    value["engines"]["bed"] = json!("not-an-entry");
    rewrite(&fixture, &value);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "engines".to_owned()))
    );

    value = document(&fixture);
    value.as_object_mut().expect("root").remove("engines");
    rewrite(&fixture, &value);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "identity".to_owned()))
    );

    write_file(&fixture.identity, b"[]");
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::NotObject, "identity".to_owned()))
    );
    write_file(&fixture.identity, b"{");
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::IdentityUnreadable, "identity".to_owned()))
    );
}

#[test]
fn incorrect_source_engine_and_image_hashes_refuse() {
    let mut fixture = world("hashes");
    mutate_entry(&mut fixture, "bed", |entry| {
        entry.insert("onnx_sha256".to_owned(), json!("0".repeat(64)));
    });
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::DigestMismatch, "bed".to_owned()))
    );

    mutate_entry(&mut fixture, "fall", |entry| {
        entry.insert("engine_sha256".to_owned(), json!("0".repeat(64)));
    });
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::DigestMismatch, "fall".to_owned()))
    );

    mutate_entry(&mut fixture, "stored_pose", |entry| {
        entry.insert(
            "image_digest".to_owned(),
            json!(format!("sha256:{}", "f".repeat(64))),
        );
    });
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::ImageDigest, "image_digest".to_owned()))
    );

    let mut value = document(&fixture);
    value["engines"]["live_pose"]["onnx_sha256"] = json!("b".repeat(64));
    value["engines"]["stored_pose"]["onnx_sha256"] = json!("b".repeat(64));
    rewrite(&fixture, &value);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::DigestMismatch, "live_pose".to_owned()))
    );
}

#[test]
fn mismatched_hardware_and_native_tf32_refuse_without_claiming_observation() {
    let mut fixture = world("hardware");
    mutate_entry(&mut fixture, "fall", |entry| {
        entry.insert("compute_capability".to_owned(), json!("9.0"));
    });
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "engines".to_owned())),
        "published contract mismatch, not a GPU observation"
    );

    mutate_entry(&mut fixture, "bed", |entry| {
        entry.insert("tf32_enabled".to_owned(), json!(true));
    });
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "engines".to_owned()))
    );

    fixture.live_tf32 = false;
    rewrite(&fixture, &document(&fixture));
    let admitted = admit(&fixture).expect("live tf32 false is a valid published bool");
    assert_eq!(admitted["tf32_enabled"], "false");
}

#[test]
fn observer_sdk_and_flow_artifact_mutation_refuse() {
    let fixture = world("artifacts");
    let before = read_bytes(&fixture.observer);
    write_file(&fixture.observer, b"different-synthetic-sdk");
    assert_ne!(read_bytes(&fixture.observer), before);
    assert_eq!(
        admit(&fixture),
        Err((
            IdentityKind::DigestMismatch,
            "observer_library_sha256".to_owned()
        ))
    );
    write_file(&fixture.observer, &before);

    let flow_before = read_bytes(&fixture.flow[3]);
    write_file(&fixture.flow[3], b"mutated-parser");
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::DigestMismatch, "parser_lib_sha256".to_owned()))
    );
    assert_ne!(read_bytes(&fixture.flow[3]), flow_before);
    write_file(&fixture.flow[3], &flow_before);
    assert_eq!(
        read_bytes(&fixture.identity),
        serde_json::to_vec(&document(&fixture)).expect("document"),
    );
    admit(&fixture).expect("restored artifact admits");
}

#[test]
fn configured_and_deployed_batch_bounds_refuse() {
    let fixture = world("batch");
    let mut request = inputs(&fixture);
    request.configured_batch = Some(1);
    assert_eq!(
        verify_aggregate(&fixture.identity, request)
            .expect_err("configured batch")
            .kind,
        IdentityKind::BatchSize,
    );

    let mut request = inputs(&fixture);
    request.deployed_batch = Some(3);
    let error = verify_aggregate(&fixture.identity, request).expect_err("deployed above built");
    assert_eq!(
        (error.kind, error.subject),
        (IdentityKind::BatchNotCovering, "batch_size".to_owned())
    );

    let mut request = inputs(&fixture);
    request.deployed_batch = Some(-1);
    let error = verify_aggregate(&fixture.identity, request).expect_err("negative deployed");
    assert_eq!(
        (error.kind, error.subject),
        (IdentityKind::NegativeDeployedBatch, "batch_size".to_owned())
    );

    let mut value = document(&fixture);
    value["batch_size"] = json!(0);
    rewrite(&fixture, &value);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::BatchSize, "batch_size".to_owned()))
    );

    value = document(&fixture);
    value["batch_size"] = json!(17);
    rewrite(&fixture, &value);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::BatchSize, "batch_size".to_owned()))
    );
}

#[test]
fn identity_oversize_and_fifo_refuse_without_rewriting_or_blocking() {
    let fixture = world("bounds");
    let original = read_bytes(&fixture.identity);
    let mut expanded = original.clone();
    expanded.extend(repeat_n(b' ', 64 * 1024));
    write_file(&fixture.identity, &expanded);
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::IdentityUnreadable, "identity".to_owned()))
    );

    fs::remove_file(&fixture.identity).expect("replace owned identity");
    rustix::fs::mknodat(
        rustix::fs::CWD,
        &fixture.identity,
        rustix::fs::FileType::Fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        0,
    )
    .expect("owned fifo");
    let identity = fixture.identity.clone();
    let engines = fixture.engines.clone();
    let flow_paths = fixture.flow.clone();
    let observer = fixture.observer.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader = thread::spawn(move || {
        let flow = vec![
            ("infer_config_sha256", flow_paths[0].clone()),
            ("tracker_config_sha256", flow_paths[1].clone()),
            ("tracker_library_sha256", flow_paths[2].clone()),
            ("parser_lib_sha256", flow_paths[3].clone()),
        ];
        let request = IdentityInputs {
            engines: EnginePaths {
                live_pose: &engines[0],
                stored_pose: &engines[1],
                bed: &engines[2],
                fall: &engines[3],
            },
            pose_onnx_sha256: POSE_ONNX,
            bed_onnx_sha256: BED_ONNX,
            fall_onnx_sha256: FALL_ONNX,
            flow: &flow,
            observer_library: &observer,
            image_digest: IMAGE,
            configured_batch: Some(2),
            deployed_batch: Some(2),
        };
        sender
            .send(verify_aggregate(&identity, request).map_err(|error| error.kind))
            .expect("result");
    });
    let refused = receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("identity fifo cannot block admission");
    reader.join().expect("reader joined");
    assert_eq!(refused, Err(IdentityKind::IdentityUnreadable));

    let artifact = fixture.root.0.join("engine.fifo");
    rustix::fs::mknodat(
        rustix::fs::CWD,
        &artifact,
        rustix::fs::FileType::Fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        0,
    )
    .expect("artifact fifo");
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader = thread::spawn(move || {
        sender
            .send(fingerprint(&artifact, "live_pose").map_err(|error| error.kind))
            .expect("fingerprint result");
    });
    let refused = receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("artifact fifo cannot block fingerprinting");
    reader.join().expect("fingerprint reader joined");
    assert_eq!(refused, Err(IdentityKind::ArtifactUnreadable));
}

#[test]
fn unreadable_identity_and_missing_engine_keep_distinct_subjects() {
    let fixture = world("access");
    let original = read_bytes(&fixture.identity);
    fs::remove_file(&fixture.identity).expect("replace owned identity");
    fs::create_dir(&fixture.identity).expect("unreadable identity directory");
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::IdentityUnreadable, "identity".to_owned()))
    );
    fs::remove_dir(&fixture.identity).expect("remove owned directory");
    write_file(&fixture.identity, &original);

    fs::remove_file(&fixture.engines[0]).expect("remove owned engine");
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::ArtifactUnreadable, "live_pose".to_owned()))
    );
}

#[test]
fn reader_does_not_truncate_or_replace_an_existing_identity() {
    let fixture = world("preserve");
    let before = read_bytes(&fixture.identity);
    let _ = admit(&fixture).expect("admits");
    let mut file = File::options()
        .read(true)
        .open(&fixture.identity)
        .expect("open");
    let mut after = Vec::new();
    file.read_to_end(&mut after).expect("read");
    assert_eq!(after, before);
}

#[test]
fn fingerprinted_build_capable_infer_refuses_and_engine_only_admits() {
    let fixture = world("engine-only");
    let engine_only = "\
[property]
model-engine-file=/cache/live.engine
batch-size=2
custom-lib-path=/opt/parser.so
# onnx-file=comment is not a key
";
    write_file(&fixture.flow[0], engine_only.as_bytes());
    rewrite(&fixture, &document(&fixture));
    admit(&fixture).expect("engine-only served config admits");

    let build_capable = "\
[property]
onnx-file=/models/pose.onnx
model-engine-file=/cache/live.engine
batch-size=2
";
    write_file(&fixture.flow[0], build_capable.as_bytes());
    rewrite(&fixture, &document(&fixture));
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "infer_config_sha256".to_owned())),
        "correct fingerprint does not admit a construction key"
    );

    let spaced = " \t [ property ] \t \n  onnx-file =/models/pose.onnx\n";
    write_file(&fixture.flow[0], spaced.as_bytes());
    rewrite(&fixture, &document(&fixture));
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "infer_config_sha256".to_owned()))
    );

    let unrelated = "\
[class-attrs-all]
onnx-file=ignored-outside-property
[property]
model-engine-file=/cache/live.engine
[other]
onnx-file=still-outside
";
    write_file(&fixture.flow[0], unrelated.as_bytes());
    rewrite(&fixture, &document(&fixture));
    admit(&fixture).expect("construction keys outside property stay serving text");

    let repeated = "\
[property]
model-engine-file=/cache/live.engine
[other]
model-engine-file=/elsewhere.engine
[ property ]
  model-file=/models/pose.caffemodel
";
    write_file(&fixture.flow[0], repeated.as_bytes());
    rewrite(&fixture, &document(&fixture));
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "infer_config_sha256".to_owned()))
    );

    let mut nul = engine_only.as_bytes().to_vec();
    nul.push(0);
    write_file(&fixture.flow[0], &nul);
    rewrite(&fixture, &document(&fixture));
    assert_eq!(
        admit(&fixture),
        Err((IdentityKind::Schema, "infer_config_sha256".to_owned()))
    );
}
