//! T17: the telemetry bodies the Rust worker builds are the bodies the
//! Python worker sends, and the product backend accepts them.
//!
//! Oracles: `r/heartbeat.json`, `r/runtime-status-steady.json` and
//! `r/runtime-status-boot-failure.json`, compared under the manifest
//! `json-value` rule (drop each golden's `normalised` fields on both sides);
//! and the backend `RelayRuntimeStatusRequest` (`extra="forbid"`) behind
//! `POST /api/v1/relay/runtime-status` on an isolated PostgreSQL sandbox.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::SystemTime;

use serde_json::Value;

use seeon_deepstream_native::{GpuDeviceReport, NvmlStatus};
use seeon_ml_worker::telemetry::gpu::{GpuStatus, unix_seconds};
use seeon_ml_worker::telemetry::heartbeat::Heartbeat;
use seeon_ml_worker::telemetry::status::{
    CameraStatus, ClipExportStatus, ClipRecorderStatus, DecodeStatus, DetectionStatus,
    FacilityStatus, WorkerStatus, parse_acceptance,
};

fn worker_wire(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(relative)
}

fn golden(relative: &str) -> Value {
    serde_json::from_slice(&fs::read(worker_wire(relative)).expect("golden is readable"))
        .expect("golden is JSON")
}

fn normalised_fields(relative: &str) -> Vec<String> {
    let manifest = golden("manifest.json");
    let entry = manifest["goldens"]
        .as_array()
        .expect("goldens list")
        .iter()
        .find(|entry| entry["path"] == relative)
        .unwrap_or_else(|| panic!("{relative} is in the manifest"))
        .clone();
    assert_eq!(entry["compare"], "json-value");
    entry["normalised"]
        .as_array()
        .expect("normalised list")
        .iter()
        .map(|field| field.as_str().expect("field path").to_owned())
        .collect()
}

/// Drops one dotted field path; `name[]` walks every member of an array.
fn drop_field(value: &mut Value, path: &[&str]) {
    let Some((head, rest)) = path.split_first() else {
        return;
    };
    if let Some(array_key) = head.strip_suffix("[]") {
        if let Some(Value::Array(items)) = value.get_mut(array_key) {
            for item in items {
                drop_field(item, rest);
            }
        }
    } else if rest.is_empty() {
        if let Value::Object(members) = value {
            members.remove(*head);
        }
    } else if let Some(member) = value.get_mut(*head) {
        drop_field(member, rest);
    }
}

fn key_paths(value: &Value, prefix: &str, paths: &mut BTreeSet<String>) {
    match value {
        Value::Object(members) => {
            for (key, member) in members {
                let path = format!("{prefix}.{key}");
                paths.insert(path.clone());
                key_paths(member, &path, paths);
            }
        }
        Value::Array(items) => {
            for item in items {
                key_paths(item, &format!("{prefix}[]"), paths);
            }
        }
        _ => {}
    }
}

fn key_set(value: &Value) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    key_paths(value, "", &mut paths);
    paths
}

/// The manifest `json-value` rule; the dropped fields must still be present
/// on the Rust side, so the key sets are compared before dropping.
fn assert_json_value_equal(rust_body: &[u8], relative: &str) {
    let mut rust: Value = serde_json::from_slice(rust_body).expect("Rust body is JSON");
    let mut expected = golden(relative);
    assert_eq!(key_set(&rust), key_set(&expected), "{relative} key set");
    for field in normalised_fields(relative) {
        let path: Vec<&str> = field.split('.').collect();
        drop_field(&mut rust, &path);
        drop_field(&mut expected, &path);
    }
    assert_eq!(rust, expected, "{relative} values");
}

fn now_sec() -> f64 {
    unix_seconds(SystemTime::now())
}

/// A healthy facility; every normalised field carries a live value.
fn steady_status() -> FacilityStatus {
    FacilityStatus {
        facility_id: "facility-1".into(),
        cameras: vec![CameraStatus {
            camera_id: "cmsnw6rjc01vhlh01oswn99yq".into(),
            decode: DecodeStatus {
                requested: "auto".into(),
                selected: Some("nvdec".into()),
                fallback_count: 0,
                last_reason: None,
                updated_at_sec: now_sec(),
            },
            measured_fps: None,
            detection: Some(DetectionStatus {
                expected: false,
                inference_admitted: 0,
                inference_succeeded: 0,
                inference_overwritten: 0,
                decision_completed: 0,
            }),
        }],
        clip_recorder: ClipRecorderStatus {
            available: true,
            dropped_frames: Some(0),
            dropped_events: Some(0),
            failed_writes: Some(0),
            finalized_clips: Some(4),
            video_unavailable_clips: Some(1),
            active_clips: Some(0),
            encoder: Some("h264_nvenc".into()),
        },
        clip_export: ClipExportStatus {
            enabled: true,
            version: 2,
        },
        gpu: Some(GpuStatus::from_report(
            Ok(GpuDeviceReport {
                nvml: NvmlStatus::Ok,
                cuda_context_ok: true,
                driver_version: Some("host driver".into()),
                device_name: Some("host device".into()),
            }),
            now_sec(),
        )),
        worker: Some(WorkerStatus::current(true, now_sec(), None)),
        delivery_queue: None,
    }
}

/// The body a worker sends when the CUDA profile failed to boot on a host
/// without the NVML library.
fn boot_failure_status() -> FacilityStatus {
    FacilityStatus {
        facility_id: "facility-1".into(),
        cameras: Vec::new(),
        clip_recorder: ClipRecorderStatus::unavailable(),
        clip_export: ClipExportStatus {
            enabled: false,
            version: 0,
        },
        gpu: Some(GpuStatus::from_report(
            Ok(GpuDeviceReport {
                nvml: NvmlStatus::LibraryMissing,
                cuda_context_ok: false,
                driver_version: None,
                device_name: None,
            }),
            now_sec(),
        )),
        worker: Some(WorkerStatus::current(
            false,
            now_sec(),
            Some("cuda_unavailable".into()),
        )),
        delivery_queue: None,
    }
}

/// The first attempt of a fresh sender: `seq` 1, no accepted generation.
fn first_body(status: &FacilityStatus) -> Vec<u8> {
    status.body(1, None).expect("status body builds")
}

#[test]
fn heartbeat_body_equals_the_python_golden() {
    let heartbeat = Heartbeat {
        camera_id: "cmsnw6rjc01vhlh01oswn99yq".into(),
        facility_id: "facility-1".into(),
        config_version: 7,
    };
    assert_json_value_equal(
        &heartbeat.body().expect("heartbeat body builds"),
        "r/heartbeat.json",
    );
}

#[test]
fn steady_runtime_status_body_equals_the_python_golden() {
    assert_json_value_equal(
        &first_body(&steady_status()),
        "r/runtime-status-steady.json",
    );
}

#[test]
fn boot_failure_runtime_status_body_equals_the_python_golden() {
    assert_json_value_equal(
        &first_body(&boot_failure_status()),
        "r/runtime-status-boot-failure.json",
    );
}

const BACKEND_SNIPPET: &str = r#"
import json, os, sys
import psycopg
from fastapi.testclient import TestClient
from backend.app.features.status.runtime_status_store import get_runtime_status_store
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import open_product_sandbox, product_audit_runtime
bodies = json.loads(sys.stdin.read())
dsn = os.environ["SEEON_TEST_POSTGRES_DSN"]
out = []
with psycopg.connect(dsn, autocommit=True) as admin:
    with open_product_sandbox(admin, dsn) as sandbox:
        app = postgres_api_app(sandbox, product_audit_runtime(sandbox))
        app.state.edge_relay_token = "relay-token"
        app.state.camera_inventory = {
            "camera-1": {"camera_id": "camera-1", "facility_id": "facility-1"}}
        client = TestClient(app)
        for body in bodies:
            response = client.post(
                "/api/v1/relay/runtime-status",
                content=body.encode("utf-8"),
                headers={"Authorization": "Bearer relay-token",
                         "Content-Type": "application/json"})
            facilities = get_runtime_status_store(app).snapshot()["facilities"]
            out.append({"status": response.status_code, "response": response.text,
                        "stored": facilities.get("facility-1")})
print(json.dumps(out))
"#;

/// The reviewed backend-answer table: the HTTP status the relay route
/// returns for each way the backend can treat a runtime-status body.
#[derive(Debug, PartialEq, Eq)]
enum Backend {
    /// 200: validated by `RelayRuntimeStatusRequest` and recorded.
    Recorded,
    /// 409: `RuntimeStatusStore.record` refused an old generation or seq.
    Stale,
    /// 422: the forbid model rejected an unknown or mistyped field.
    Rejected,
    Other(u64),
}

fn backend(status: u64) -> Backend {
    match status {
        200 => Backend::Recorded,
        409 => Backend::Stale,
        422 => Backend::Rejected,
        other => Backend::Other(other),
    }
}

fn post_to_backend(bodies: &[Vec<u8>]) -> Vec<Value> {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is set");
    assert!(std::env::var_os("SEEON_TEST_POSTGRES_DSN").is_some());
    let texts: Vec<String> = bodies
        .iter()
        .map(|body| String::from_utf8(body.clone()).expect("body is UTF-8"))
        .collect();
    let input = serde_json::to_vec(&texts).expect("bodies encode");
    let mut child = Command::new(python)
        .arg("-c")
        .arg(BACKEND_SNIPPET)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("python spawns");
    child
        .stdin
        .take()
        .expect("python stdin")
        .write_all(&input)
        .expect("bodies written");
    let output = child.wait_with_output().expect("python exits");
    assert!(output.status.success(), "backend snippet exited non-zero");
    let Value::Array(results) =
        serde_json::from_slice(&output.stdout).expect("snippet prints JSON")
    else {
        panic!("snippet prints a list");
    };
    results
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON and SEEON_TEST_POSTGRES_DSN"]
fn backend_records_the_rust_runtime_status_bodies() {
    let statuses = [steady_status(), boot_failure_status()];
    let bodies: Vec<Vec<u8>> = statuses.iter().map(first_body).collect();
    let results = post_to_backend(&bodies);
    assert_eq!(results.len(), bodies.len());
    for (body, result) in bodies.iter().zip(&results) {
        let sent: Value = serde_json::from_slice(body).expect("body is JSON");
        let status = result["status"].as_u64().expect("HTTP status");
        assert_eq!(backend(status), Backend::Recorded);

        let response = result["response"].as_str().expect("response text");
        let generation = parse_acceptance(response.as_bytes())
            .expect("the Rust sender accepts the backend answer");

        let stored = &result["stored"];
        assert_eq!(stored["generation"], Value::from(generation));
        assert_eq!(stored["seq"], sent["seq"]);
        assert_eq!(stored["gpu"], sent["gpu"]);
        assert_eq!(stored["clip_recorder"], sent["clip_recorder"]);
        assert_eq!(stored["clip_export"], sent["clip_export"]);
        for field in ["alive", "pid", "started_at_sec"] {
            assert_eq!(
                stored["worker"][field], sent["worker"][field],
                "worker.{field}"
            );
        }
        let camera_ids = |cameras: &Value| -> Vec<Value> {
            cameras
                .as_array()
                .expect("camera list")
                .iter()
                .map(|camera| camera["camera_id"].clone())
                .collect()
        };
        assert_eq!(camera_ids(&stored["cameras"]), camera_ids(&sent["cameras"]));
    }
}
