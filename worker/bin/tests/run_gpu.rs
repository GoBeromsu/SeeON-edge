//! Idle GPU lifecycle of the actual `ml-worker` binary. Not live-camera,
//! inference-parity, or final-image qualification. An empty admitted roster is legal.
//! Passing requires the production marker emitted only after the first successful
//! policy turn and delivery, then a clean SIGTERM. Process-alive or prepared output alone is not success.
//! Prerequisites are the actual four-engine aggregate, its sibling engines, and
//! the packaged fall and bed ONNX sources. Idle success is still the policy-loop
//! marker plus clean SIGTERM, not process survival.
//! One resource-gated case intentionally rewrites the same `compute_capability`
//! in all four recorded receipts. That is test tampering, not a new native
//! receipt or hardware observation. File admission can still pass because the
//! four entries agree; the actual GPU binary must then refuse the mismatch.
use std::collections::BTreeSet;
use std::fs;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use seeon_ml_worker::gpu::lease;
use serde_json::{Value, json};

#[path = "support/active_shutdown.rs"]
mod active_shutdown;
#[path = "support/boot_fixture.rs"]
mod fixture;
#[path = "support/gpu_process.rs"]
mod gpu_process;
#[path = "support/replay_gpu.rs"]
mod replay_gpu;
use active_shutdown::{OwnedProductStore, shutdown_during_active_recording};
use fixture::{
    ALERTS_PATH, CLIPS_PATH_PREFIX, CONFIG_PATH, EXECUTION_RECORDS_PATH, Fixture, IDENTITY_PATH,
    STATUS_PATH, Server, shutdown_config_write_cancelled,
};
use gpu_process::{shutdown_after_policy_loop, spawn_gpu, wait_for_refusal};
use replay_gpu::{assert_python_decodes, shutdown_after_accepted_trace, trace_path};

static RELAY: Mutex<()> = Mutex::new(());

fn relay_guard() -> std::sync::MutexGuard<'static, ()> {
    RELAY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
#[ignore = "requires GPU lane ordinal 0, restored engines/model bundle, isolated ml-api loopback and actual ml-worker; idle lifecycle only, not live-camera or final-image qualification"]
fn empty_roster_warms_cuda_owners_then_sigterm_exits_clean_without_facility_post_or_rtsp() {
    let shutdown_started = Arc::new(AtomicBool::new(false));
    let _guard = relay_guard();
    let fixture = Fixture::new("idle-gpu");
    let server = Server::start(
        Some(fixture.config(false)),
        Some(Arc::clone(&shutdown_started)),
    );
    let mut command = fixture.command("run");
    command.env("CUDA_VISIBLE_DEVICES", "0");
    let mut child = spawn_gpu(command);
    let shutdown = shutdown_after_policy_loop(&mut child, &shutdown_started);
    let stderr = String::from_utf8_lossy(&shutdown.stderr);
    assert!(
        shutdown.saw_policy_loop,
        "missing first successful policy-turn marker after CUDA and model warm gates: {stderr}"
    );
    assert!(
        !stderr.contains("reason=cuda_unavailable")
            && !stderr.contains("reason=engine_open")
            && !stderr.contains("reason=gpu_lease")
            && !stderr.contains("reason=engine_identity")
            && !stderr.contains("boot refused")
            && !stderr.contains("shutdown incomplete"),
        "idle GPU lifecycle refused or failed shutdown: {stderr}"
    );
    assert_eq!(shutdown.status.code(), Some(0), "{stderr}");
    assert!(
        shutdown.status.success(),
        "clean shutdown required, not process survival: {stderr}"
    );
    let requests = server.finish();
    assert_eq!(
        requests
            .iter()
            .take(2)
            .map(|request| (request.method.as_str(), request.path.as_str()))
            .collect::<Vec<_>>(),
        [("GET", IDENTITY_PATH), ("GET", CONFIG_PATH)],
        "fresh configuration must precede real boot gates",
    );
    assert!(
        requests
            .iter()
            .skip(2)
            .all(|request| { request.method == "GET" && request.path == CONFIG_PATH }),
        "after boot, only the real restart/config poll is allowed: {requests:?}"
    );
    assert!(
        requests.iter().all(|request| request.method != "POST"),
        "idle roster must not POST facility status: {requests:?}"
    );
    assert!(
        requests.iter().all(|request| request.body.is_null()),
        "identity and config GETs carry no body: {requests:?}"
    );
    fixture.assert_no_source_activation();
    drop(child);
    lease::acquire(&fixture.state)
        .expect("fixture lease reacquired after clean exit; this is not native-close proof");
    println!("IDLE_GPU_STDERR={stderr}");
}

#[test]
#[ignore = "requires GPU lane 0, actual four-engine aggregate, isolated ml-api and ml-worker"]
fn runtime_only_revision_cannot_enable_execution_records() {
    let _guard = relay_guard();
    let fixture = Fixture::new("runtime-only-revision");
    let server = Server::start(Some(fixture.config(false)), None);
    let mut command = fixture.command("run");
    command.env("CUDA_VISIBLE_DEVICES", "0");
    command.env("ML_WORKER_EXECUTION_RECORDS_ENABLED", "true");
    command.env("ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY", "4");
    command.env("ML_WORKER_EXECUTION_RECORDS_BATCH_MAX", "1");
    command.env("ML_WORKER_EXECUTION_RECORDS_FLUSH_MS", "50");
    // Deliberately invalid runtime input, never a build declaration or receipt.
    // The records wire accepts opaque identities; attribution must not rely on
    // that syntax check or on this value being present in the environment.
    command.env(
        "ML_WORKER_BUILD_REVISION",
        "runtime-only-not-a-build-declaration",
    );
    let mut child = spawn_gpu(command);
    let refusal = wait_for_refusal(&mut child);
    let stderr = String::from_utf8_lossy(&refusal.stderr);
    assert_eq!(refusal.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains(
            "output startup refused: execution records refused: missing worker_build_revision"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains("policy loop ready"), "{stderr}");
    let requests = server.finish();
    assert!(
        requests.iter().all(|request| request.method == "GET"),
        "unattributed worker must not export records or facility status: {requests:?}"
    );
    fixture.assert_no_source_activation();
    drop(child);
    lease::acquire(&fixture.state).expect("lease released after refused attribution");
    println!("RUNTIME_REVISION_REFUSAL={stderr}");
}

#[test]
/// Intentional shared capability tamper, not a native receipt or hardware observation.
#[ignore = "requires GPU lane 0, actual four-engine aggregate, isolated ml-api and ml-worker"]
fn tampered_shared_compute_capability_is_engine_identity_before_readiness() {
    let _guard = relay_guard();
    let fixture = Fixture::new("hardware-mismatch");
    let identity_path = std::path::PathBuf::from(
        fixture
            .env
            .get("ML_WORKER_FLOW_ENGINE_IDENTITY_PATH")
            .expect("owned aggregate"),
    );
    let mut identity: Value =
        serde_json::from_slice(&fs::read(&identity_path).expect("owned aggregate")).expect("JSON");
    let engines = identity["engines"]
        .as_object_mut()
        .expect("four recorded receipts");
    assert_eq!(
        engines.len(),
        4,
        "tamper the same field in every recorded receipt"
    );
    for (role, entry) in engines {
        let receipt = entry.as_object_mut().expect(role);
        let recorded = receipt["compute_capability"]
            .as_str()
            .expect("recorded compute capability");
        assert_ne!(
            recorded, "99.0",
            "{role} must not already claim the tamper value"
        );
        receipt.insert("compute_capability".to_owned(), json!("99.0"));
    }
    fs::write(&identity_path, serde_json::to_vec(&identity).expect("JSON")).expect("owned tamper");
    let server = Server::start(Some(fixture.config(false)), None);
    let mut command = fixture.command("run");
    command.env("CUDA_VISIBLE_DEVICES", "0");
    let mut child = spawn_gpu(command);
    let refusal = wait_for_refusal(&mut child);
    let stderr = String::from_utf8_lossy(&refusal.stderr);
    assert_eq!(refusal.status.code(), Some(3), "{stderr}");
    assert!(
        stderr.contains("reason=engine_identity") && stderr.contains("report=no facility"),
        "identity refusal required, not CUDA or model start: {stderr}"
    );
    assert!(
        !stderr.contains("reason=cuda_unavailable")
            && !stderr.contains("reason=engine_open")
            && !stderr.contains("reason=gpu_lease")
            && !stderr.contains(gpu_process::POLICY_LOOP_READY),
        "refusal must precede readiness and model owners: {stderr}"
    );
    let requests = server.finish();
    assert_eq!(
        requests
            .iter()
            .map(|request| (request.method.as_str(), request.path.as_str()))
            .collect::<Vec<_>>(),
        [("GET", IDENTITY_PATH), ("GET", CONFIG_PATH)],
        "empty roster reaches the post-CUDA identity gate without a facility POST"
    );
    fixture.assert_no_source_activation();
    drop(child);
    lease::acquire(&fixture.state)
        .expect("identity refusal must release the real lease before return");
}
#[test]
#[ignore = "T28 requires GPU lane 0, a genuine schema-1 four-engine aggregate with batch_size exactly 1 via SEEON_TEST_ENGINE_IDENTITY and sibling engines, packaged ONNX bundle, SEEON_TEST_MEDIA_INFER, SEEON_TEST_MEDIA_TRACKER, SEEON_TEST_RTSP_URI, isolated ml-api loopback, parent-owned /var/lib/clip-store with SEEON_TEST_CLIP_STORE_OWNER matching /var/lib/clip-store/.seeon-test-owner, and SEEON_TEST_PYTHON; batch-2, an invented profile, or missing ownership is a failure"]
fn active_camera_sigterm_during_native_recording_finalizes_clip_and_retains_queue() {
    let shutdown_started = Arc::new(AtomicBool::new(false));
    let _guard = relay_guard();
    let store = OwnedProductStore::claim();
    let preexisting = store.paths();
    let fixture = Fixture::new("active-shutdown");
    require_single_camera_batch(&fixture);
    let uri = approved_rtsp_uri();
    let server = Server::start(
        Some(active_camera_config(&uri)),
        Some(Arc::clone(&shutdown_started)),
    );
    let mut command = fixture.command("run");
    command.env("CUDA_VISIBLE_DEVICES", "0");
    command.env("ML_WORKER_EXECUTION_RECORDS_ENABLED", "true");
    command.env("ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY", "4");
    command.env("ML_WORKER_EXECUTION_RECORDS_BATCH_MAX", "1");
    command.env("ML_WORKER_EXECUTION_RECORDS_FLUSH_MS", "50");
    let mut child = spawn_gpu(command);
    let record_dir = PathBuf::from(&fixture.env["ML_WORKER_FLOW_RECORD_DIR"]);
    let shutdown = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        shutdown_during_active_recording(&mut child, &record_dir, &shutdown_started)
    })) {
        Ok(shutdown) => shutdown,
        Err(failure) => {
            let requests = server.finish();
            let alerts = requests
                .iter()
                .filter(|request| {
                    request.method == "POST" && request_route(&request.path) == ALERTS_PATH
                })
                .count();
            let status = requests.iter().rev().find(|request| {
                request.method == "POST" && request_route(&request.path) == STATUS_PATH
            });
            eprintln!(
                "active-camera refusal evidence: alerts={alerts} last_status={:?} {}",
                status.map(|request| &request.body),
                model_score_summary(&requests, ACTIVE_CAMERA_ID)
            );
            std::panic::resume_unwind(failure);
        }
    };
    let stderr = String::from_utf8_lossy(&shutdown.stderr);
    assert_eq!(shutdown.status.code(), Some(0), "{stderr}");
    assert!(
        shutdown.status.success(),
        "active recording SIGTERM must exit 0 inside the shared 25s budget: {stderr}"
    );
    assert!(
        !stderr.contains("reason=cuda_unavailable")
            && !stderr.contains("reason=engine_open")
            && !stderr.contains("reason=gpu_lease")
            && !stderr.contains("reason=engine_identity")
            && !stderr.contains("boot refused")
            && !stderr.contains("shutdown incomplete")
            && !stderr.contains("recording start refused"),
        "active camera must reach native recording, not a boot or start refusal: {stderr}"
    );
    let requests = server.finish();
    assert!(
        requests.iter().any(|request| {
            request.method == "POST"
                && request_route(&request.path) == ALERTS_PATH
                && request.body["camera_id"] == ACTIVE_CAMERA_ID
        }),
        "the admitted event must carry the configured camera UUID: {requests:?}"
    );
    assert!(requests.iter().all(|request| {
        !(request.method == "PUT" && request_route(&request.path).starts_with(CLIPS_PATH_PREFIX))
    }),);
    let published = finalized_ready_clip(store.root(), &preexisting, ACTIVE_CAMERA_ID);
    let queue = retained_clip_queue(&fixture.state, ACTIVE_CAMERA_ID);
    assert!(
        queue.iter().any(|entry| {
            entry["kind"] == "CLIP"
                && entry["clip_id"] == published["clip_id"]
                && entry["camera_id"] == ACTIVE_CAMERA_ID
                && entry["local_state"] == "READY"
        }),
        "disabled clip export must retain the durable CLIP entry: {queue:?}"
    );
    let python = std::env::var_os("SEEON_TEST_PYTHON")
        .expect("SEEON_TEST_PYTHON required when T28 is included");
    assert_backend_parses_manifest(&python, &published["manifest_path"]);
    drop(child);
    lease::acquire(&fixture.state).expect("clean active shutdown must release the real lease");
    fixture.assert_no_source_activation();
}
#[test]
#[ignore = "requires GPU lane 0, genuine schema-1 batch-1 aggregate via SEEON_TEST_ENGINE_IDENTITY and sibling engines, packaged ONNX, SEEON_TEST_MEDIA_INFER, SEEON_TEST_MEDIA_TRACKER, approved SEEON_TEST_RTSP_URI, isolated canonical ml-api, actual ml-worker, SEEON_TEST_PYTHON, and parent-owned clip-store matching SEEON_TEST_CLIP_STORE_OWNER"]
fn accepted_native_metadata_is_written_to_replay_trace_before_owned_sigterm() {
    use seeon_ml_worker::seam::{IdSource, RandomIds};
    let shutdown_started = Arc::new(AtomicBool::new(false));
    let _guard = relay_guard();
    let _store = OwnedProductStore::claim();
    let fixture = Fixture::new("replay-trace");
    require_single_camera_batch(&fixture);
    let uri = approved_rtsp_uri();
    let server = Server::start(
        Some(active_camera_config(&uri)),
        Some(Arc::clone(&shutdown_started)),
    );
    // Cargo's mounted temporary root survives both fixture Drop and container removal.
    let trace_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("native-replay-{}", RandomIds.uuid4().unwrap()));
    fs::create_dir(&trace_dir).expect("exclusive retained replay directory");
    let trace = trace_path(&trace_dir, ACTIVE_CAMERA_ID);
    eprintln!("REPLAY_RETAINED_PATH={}", trace.display());
    let mut command = fixture.command("run");
    command.env("CUDA_VISIBLE_DEVICES", "0");
    command.env("WORKER_REPLAY_TRACE_DIR", &trace_dir);
    command.env("ML_WORKER_EXECUTION_RECORDS_ENABLED", "false");
    let mut child = spawn_gpu(command);
    let shutdown =
        shutdown_after_accepted_trace(&mut child, &trace, ACTIVE_CAMERA_ID, &shutdown_started);
    let stderr = String::from_utf8_lossy(&shutdown.stderr);
    assert_eq!(shutdown.status.code(), Some(0), "{stderr}");
    assert!(
        shutdown.status.success(),
        "accepted replay trace SIGTERM must exit 0 inside the pre-signal 25s budget: {stderr}"
    );
    println!("REPLAY_GPU_STDERR={stderr}");
    let evidence =
        replay_gpu::inspect_trace(&trace, ACTIVE_CAMERA_ID).expect("saved native replay trace");
    eprintln!(
        "replay trace preserved path={} complete_rows={} frame_rows={} \
         nonempty_pose_rows={} distinct_pts={:?}",
        evidence.path.display(),
        evidence.complete_rows,
        evidence.frame_rows,
        evidence.nonempty_pose_rows,
        evidence.distinct_pts
    );
    let python = std::env::var_os("SEEON_TEST_PYTHON")
        .expect("SEEON_TEST_PYTHON required when the replay GPU case is included");
    assert_python_decodes(&python, &trace);
    drop(child);
    lease::acquire(&fixture.state).expect("clean replay shutdown must release the real lease");
    fixture.assert_no_source_activation();
    let _requests = server.finish();
}
const ACTIVE_CAMERA_ID: &str = "11111111-1111-4111-8111-111111111111";
fn request_route(target: &str) -> &str {
    let without_query = target.split_once('?').map_or(target, |(path, _)| path);
    without_query
        .strip_prefix("http://ml-api:8000")
        .unwrap_or(without_query)
}
const MODEL_SCORE: &str = "model.score";

fn model_score_summary(requests: &[fixture::Request], camera_id: &str) -> String {
    let mut ids = BTreeSet::new();
    let mut fall = Vec::new();
    let mut logit = Vec::new();
    let mut first = None;
    let mut last = None;
    for request in requests {
        if request.method != "POST" || request_route(&request.path) != EXECUTION_RECORDS_PATH {
            continue;
        }
        let Some(records) = request.body["records"].as_array() else {
            continue;
        };
        for record in records {
            if record["camera_id"] != camera_id || record["record_kind"] != MODEL_SCORE {
                continue;
            }
            let Some(record_id) = record["record_id"].as_str() else {
                continue;
            };
            if seeon_ml_worker::records::id::sha256_hex_field(record_id, "record_id").is_err() {
                continue;
            }
            if !ids.insert(record_id.to_owned()) {
                continue;
            }
            let payload = &record["payload"];
            if let Some(value) = finite_number(&payload["fall_transition"]) {
                fall.push(value);
            }
            if let Some(value) = finite_number(&payload["raw_logit"]) {
                logit.push(value);
            }
            let point = format!(
                "fall_transition={} raw_logit={}",
                number_text(&payload["fall_transition"]),
                number_text(&payload["raw_logit"])
            );
            if first.is_none() {
                first = Some(point.clone());
            }
            last = Some(point);
        }
    }
    format!(
        "model.score camera={camera_id} unique_ids={} count={} \
         fall_transition[{}] raw_logit[{}] first={:?} last={:?}",
        ids.len(),
        ids.len(),
        span(&fall),
        span(&logit),
        first.as_deref().unwrap_or("absent"),
        last.as_deref().unwrap_or("absent"),
    )
}

fn finite_number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|number| number.is_finite())
}

fn number_text(value: &Value) -> String {
    match finite_number(value) {
        Some(number) => number.to_string(),
        None => "absent".to_owned(),
    }
}

fn span(values: &[f64]) -> String {
    match (
        values.iter().copied().reduce(f64::min),
        values.iter().copied().reduce(f64::max),
    ) {
        (Some(min), Some(max)) => format!("min={min} max={max} n={}", values.len()),
        _ => "absent".to_owned(),
    }
}

#[test]
fn model_score_summary_reports_unique_finite_bounds_without_inventing_absent_fields() {
    let camera = ACTIVE_CAMERA_ID;
    let other = "22222222-2222-4222-8222-222222222222";
    let requests = vec![
        fixture::Request {
            method: "POST".into(),
            path: EXECUTION_RECORDS_PATH.into(),
            body: json!({
                "records": [
                    {
                        "record_id": "ab".repeat(32),
                        "camera_id": camera,
                        "record_kind": MODEL_SCORE,
                        "payload": {"fall_transition": 0.12, "raw_logit": 0.3}
                    },
                    {
                        "record_id": "ab".repeat(32),
                        "camera_id": camera,
                        "record_kind": MODEL_SCORE,
                        "payload": {"fall_transition": 9.0, "raw_logit": 9.0}
                    },
                    {
                        "record_id": "cd".repeat(32),
                        "camera_id": camera,
                        "record_kind": MODEL_SCORE,
                        "payload": {"fall_transition": 0.5}
                    },
                    {
                        "record_id": "ef".repeat(32),
                        "camera_id": other,
                        "record_kind": MODEL_SCORE,
                        "payload": {"fall_transition": 0.9, "raw_logit": 4.0}
                    },
                    {
                        "record_id": "11".repeat(32),
                        "camera_id": camera,
                        "record_kind": "policy.decision",
                        "payload": {"fall_transition": 0.01}
                    }
                ]
            }),
        },
        fixture::Request {
            method: "POST".into(),
            path: ALERTS_PATH.into(),
            body: json!({"camera_id": camera}),
        },
    ];
    let summary = model_score_summary(&requests, camera);
    assert!(summary.contains("unique_ids=2 count=2"), "{summary}");
    assert!(
        summary.contains("fall_transition[min=0.12 max=0.5 n=2]"),
        "{summary}"
    );
    assert!(
        summary.contains("raw_logit[min=0.3 max=0.3 n=1]"),
        "{summary}"
    );
    assert!(
        summary.contains("first=\"fall_transition=0.12 raw_logit=0.3\""),
        "{summary}"
    );
    assert!(
        summary.contains("last=\"fall_transition=0.5 raw_logit=absent\""),
        "{summary}"
    );
    assert!(
        !summary.contains("raw_logit=4") && !summary.contains("fall_transition=0.9"),
        "{summary}"
    );
    let empty = model_score_summary(&[], camera);
    assert!(empty.contains("unique_ids=0 count=0"), "{empty}");
    assert!(
        empty.contains("fall_transition[absent]") && empty.contains("raw_logit[absent]"),
        "{empty}"
    );
}

fn require_single_camera_batch(fixture: &Fixture) {
    let path = &fixture.env["ML_WORKER_FLOW_ENGINE_IDENTITY_PATH"];
    let document: Value = serde_json::from_slice(&fs::read(path).expect("owned aggregate"))
        .expect("owned aggregate JSON");
    assert_eq!(
        document["batch_size"], 1,
        "genuine batch-one aggregate required"
    );
    assert_eq!(fixture.env["ML_WORKER_FLOW_BATCH_SIZE"], "1");
}

fn approved_rtsp_uri() -> String {
    let uri = std::env::var("SEEON_TEST_RTSP_URI")
        .expect("SEEON_TEST_RTSP_URI required; missing approved synthetic stream is a failure");
    assert!(
        uri.starts_with("rtsp://") && uri.len() > "rtsp://".len() && !uri.contains('\0'),
        "SEEON_TEST_RTSP_URI must be one approved rtsp URI"
    );
    uri
}

fn active_camera_config(uri: &str) -> Value {
    json!({
        "cameras": [{
            "camera_id": ACTIVE_CAMERA_ID,
            "facility_id": "facility-1",
            "rtsp_url": uri,
            "domains": ["fall", "bed_exit"]
        }],
        "registry_version": 7,
        "restart_epoch": 9,
        "clip_export_enabled": false,
        "clip_export_version": 0
    })
}

fn finalized_ready_clip(store: &Path, preexisting: &BTreeSet<PathBuf>, camera_id: &str) -> Value {
    let clips = store.join("clips");
    let mut manifests = Vec::new();
    for entry in
        fs::read_dir(&clips).unwrap_or_else(|error| panic!("published clips required: {error}"))
    {
        let entry = entry.expect("clip entry");
        let path = entry.path();
        if preexisting.contains(&path)
            || !path.is_dir()
            || path.file_name().and_then(|name| name.to_str()) == Some(".staging")
        {
            continue;
        }
        let manifest_path = path.join("manifest.json");
        let media_path = path.join("clip.mp4");
        let manifest: Value = serde_json::from_slice(
            &fs::read(&manifest_path).unwrap_or_else(|error| panic!("manifest required: {error}")),
        )
        .expect("manifest JSON");
        let media =
            fs::metadata(&media_path).unwrap_or_else(|error| panic!("media required: {error}"));
        assert!(
            media.is_file() && media.len() > 0,
            "finalized media must be a nonempty file"
        );
        let duration = manifest["duration_ms"].as_i64().expect("duration_ms");
        let start = manifest["clip_start_at"].as_str().expect("clip_start_at");
        let end = manifest["clip_end_at"].as_str().expect("clip_end_at");
        assert!(duration > 0, "nonzero duration required: {manifest}");
        assert!(end > start, "valid clip bounds required: {start} .. {end}");
        assert_eq!(manifest["finalized"], true);
        assert_eq!(manifest["camera_id"], camera_id, "{manifest}");
        assert_eq!(manifest["state"], "READY");
        assert_eq!(manifest["video_available"], true);
        let mut published = manifest;
        published["manifest_path"] = Value::String(manifest_path.display().to_string());
        manifests.push(published);
    }
    assert_eq!(
        manifests.len(),
        1,
        "one finalized clip for {camera_id} required, found {manifests:?}"
    );
    manifests.pop().expect("one clip")
}

fn retained_clip_queue(state: &Path, camera_id: &str) -> Vec<Value> {
    let queue = state.join("delivery-queue");
    let mut entries = Vec::new();
    for entry in
        fs::read_dir(&queue).unwrap_or_else(|error| panic!("durable queue required: {error}"))
    {
        let entry = entry.expect("queue entry");
        let name = entry.file_name();
        let name = name.to_str().unwrap_or("");
        if !name.ends_with(".json") || name.starts_with('.') {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(entry.path()).expect("queue JSON"))
            .expect("durable queue entry");
        entries.push(value);
    }
    assert!(
        entries
            .iter()
            .any(|entry| entry["kind"] == "CLIP" && entry["camera_id"] == camera_id),
        "disabled clip export must retain the configured camera CLIP: {entries:?}"
    );
    entries
}

fn assert_backend_parses_manifest(python: &std::ffi::OsStr, manifest: &Value) {
    let path = manifest.as_str().expect("manifest path");
    let output = std::process::Command::new(python)
        .arg("-c")
        .arg(BACKEND_PARSE)
        .arg(path)
        .output()
        .expect("SEEON_TEST_PYTHON runs");
    assert!(
        output.status.success(),
        "backend parse_manifest_bytes failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: Value = serde_json::from_slice(&output.stdout).expect("parser JSON");
    assert_eq!(parsed[0]["verdict"], "parsed", "{parsed}");
}

const BACKEND_PARSE: &str = "
import json, sys
from backend.app.features.clips.manifest import parse_manifest_bytes
with open(sys.argv[1], 'rb') as handle:
    manifest = parse_manifest_bytes(handle.read())
if manifest is None:
    print(json.dumps([{'verdict': 'rejected'}]))
else:
    print(json.dumps([{'verdict': 'parsed', 'event_refs': list(manifest.event_refs)}]))
";
#[test]
fn shutdown_config_write_tolerance_stays_inside_owned_sigterm() {
    let flag = AtomicBool::new(false);
    let broken = Error::new(ErrorKind::BrokenPipe, "peer closed");
    let reset = Error::new(ErrorKind::ConnectionReset, "peer reset");
    let aborted = Error::new(ErrorKind::ConnectionAborted, "peer aborted");
    assert!(!shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        Some(&flag),
        &broken
    ));
    flag.store(true, Ordering::SeqCst);
    assert!(shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        Some(&flag),
        &broken
    ));
    assert!(shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        Some(&flag),
        &reset
    ));
    assert!(shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        Some(&flag),
        &aborted
    ));
    assert!(!shutdown_config_write_cancelled(
        "POST",
        CONFIG_PATH,
        Some(&flag),
        &broken
    ));
    assert!(!shutdown_config_write_cancelled(
        "GET",
        IDENTITY_PATH,
        Some(&flag),
        &broken
    ));
    assert!(!shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        None,
        &broken
    ));
    let refused = Error::new(ErrorKind::ConnectionRefused, "not a cancelled write");
    assert!(!shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        Some(&flag),
        &refused
    ));
    let timed_out = Error::new(ErrorKind::TimedOut, "write deadline");
    assert!(!shutdown_config_write_cancelled(
        "GET",
        CONFIG_PATH,
        Some(&flag),
        &timed_out
    ));
}
#[test]
fn replay_trace_inspection_rejects_partial_malformed_and_non_native_rows() {
    use seeon_ml_worker::seam::{IdSource, RandomIds};
    let dir = std::env::temp_dir().join(format!(
        "seeon-replay-inspect-{}",
        RandomIds.uuid4().unwrap()
    ));
    fs::create_dir(&dir).expect("exclusive inspect fixture");
    let path = dir.join("trace.jsonl");
    let camera = ACTIVE_CAMERA_ID;
    fs::write(&path, "{\"version\":\"replay-trace-v2\"}\n{\"camera_id\"").expect("partial");
    assert_eq!(
        replay_gpu::inspect_trace(&path, camera),
        Err(replay_gpu::TraceInspectError::Insufficient)
    );
    fs::write(&path, "{\"version\":\"replay-trace-v2\"}\n{not-json}\n").expect("malformed");
    assert_eq!(
        replay_gpu::inspect_trace(&path, camera),
        Err(replay_gpu::TraceInspectError::Malformed)
    );
    fs::write(&path, format!("{}{}\n", header(), row(camera, 1, "[]"))).expect("short");
    assert_eq!(
        replay_gpu::inspect_trace(&path, camera),
        Err(replay_gpu::TraceInspectError::Insufficient)
    );
    let wrong = row("other-camera", 1, &pose());
    fs::write(&path, format!("{}{wrong}\n{wrong}\n{wrong}\n", header())).expect("camera");
    assert_eq!(
        replay_gpu::inspect_trace(&path, camera),
        Err(replay_gpu::TraceInspectError::Camera)
    );
    let wide = row(camera, 1, &pose()).replace("\"frame_width\":640", "\"frame_width\":1280");
    fs::write(&path, format!("{}{wide}\n{wide}\n{wide}\n", header())).expect("dims");
    assert_eq!(
        replay_gpu::inspect_trace(&path, camera),
        Err(replay_gpu::TraceInspectError::Dimensions)
    );
    let bad = row(camera, 1, &pose()).replace("0.1", "NaN");
    fs::write(&path, format!("{}{bad}\n{bad}\n{bad}\n", header())).expect("nonfinite");
    assert_eq!(
        replay_gpu::inspect_trace(&path, camera),
        Err(replay_gpu::TraceInspectError::Malformed)
    );
    let valid = format!(
        "{}{}\n{}\n{}\n{}\ntrailing",
        header(),
        row(camera, 0, "[]").replace("\"source_event\":\"frame\"", "\"source_event\":\"open\""),
        row(camera, 1, &pose()),
        row(camera, 2, &pose()),
        row(camera, 3, "[]")
    );
    fs::write(&path, &valid).expect("valid");
    let evidence = replay_gpu::inspect_trace(&path, camera).expect("complete lines");
    assert_eq!(evidence.frame_rows, 3);
    assert_eq!(evidence.nonempty_pose_rows, 2);
    assert_eq!(evidence.distinct_pts, vec![1, 2, 3]);
    let huge = dir.join("huge.jsonl");
    fs::write(&huge, header()).expect("oversized seed");
    assert_eq!(
        replay_gpu::inspect_trace_bounded(&huge, camera, 1),
        Err(replay_gpu::TraceInspectError::Oversized)
    );
    fs::remove_dir_all(&dir).expect("remove owned fixture");
}

fn header() -> &'static str {
    seeon_ml_worker::trace_out::HEADER_LINE
}

fn pose() -> String {
    let point = "[0.1,0.2,0.3]";
    let points = std::iter::repeat_n(point, 17).collect::<Vec<_>>().join(",");
    format!(
        "[{{\"track_id\":7,\"lifecycle\":\"tracked\",\"bbox\":[0.1,0.2,0.3,0.4,0.5],\
         \"keypoints\":[{points}]}}]"
    )
}

fn row(camera: &str, pts: u64, tracks: &str) -> String {
    format!(
        "{{\"camera_id\":\"{camera}\",\"seq\":{pts},\"pts_ns\":{pts},\"epoch\":1,\
         \"source_event\":\"frame\",\"source\":\"nvdcf\",\"tracks\":{tracks},\
         \"bed_polygon_id\":null,\"bed_polygon\":null,\"bed_polygon_image_size\":null,\
         \"night_window_active\":false,\"frame_width\":640,\"frame_height\":360}}"
    )
}
