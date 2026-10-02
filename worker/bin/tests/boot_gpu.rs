//! Actual executable gates, not helper substitutes. GPU fields are never normalized.
//! Identity admission uses the one schema-1 aggregate. Tampering changes the
//! owned live-pose engine bytes or removes `engines.live_pose.onnx_sha256`.
//! It does not remove a legacy flat field or skip a missing prerequisite.
//! The golden refusal requires genuinely unavailable NVML and hidden CUDA devices.
//! Passing on a CPU builder does not qualify GPU inference or a final image.
use std::fs;
use std::io::Read;
use std::process::{Child, Command, ExitStatus};
use std::sync::Mutex;
use std::time::Duration;

use seeon_ml_worker::gpu::lease;
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::seam::{Clock, SystemClock};
use serde_json::Value;

#[path = "support/boot_fixture.rs"]
mod fixture;
use fixture::{CONFIG_PATH, Fixture, IDENTITY_PATH, Request, STATUS_PATH, Server};

// The production authority has one fixed port. Each test creates fresh independent state.
static RELAY: Mutex<()> = Mutex::new(());

fn relay_guard() -> std::sync::MutexGuard<'static, ()> {
    // The lock protects only the port, not shared mutable fixture state.
    // A failed case must not poison independent cases after RAII cleanup.
    RELAY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn failure_post(requests: &[Request]) -> &Value {
    let observed: Vec<_> = requests
        .iter()
        .map(|r| (r.method.as_str(), r.path.as_str()))
        .collect();
    assert_eq!(
        observed,
        [
            ("GET", IDENTITY_PATH),
            ("GET", CONFIG_PATH),
            ("POST", STATUS_PATH)
        ],
        "fresh config precedes boot; one bounded failure report, no retry/event/model-score request"
    );
    &requests[2].body
}
fn no_post(requests: &[Request]) {
    assert_eq!(
        requests
            .iter()
            .map(|r| (r.method.as_str(), r.path.as_str()))
            .collect::<Vec<_>>(),
        [("GET", IDENTITY_PATH), ("GET", CONFIG_PATH)]
    );
}
fn assert_golden(actual: &Value) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/r/runtime-status-boot-failure.json");
    let golden = fs::read(path).expect("restore pinned parity inputs before running this gate");
    assert_eq!(
        sha256_hex(&golden),
        "2f91f86b6ca49256ac2959ccca21111a2fc689e9eaacc9d87f02151ac56b1925"
    );
    let expected: Value = serde_json::from_slice(&golden).expect("unchanged golden");
    let mut actual = actual.clone();
    for pointer in [
        "/gpu/captured_at_sec",
        "/worker/started_at_sec",
        "/worker/pid",
    ] {
        let value = actual
            .pointer_mut(pointer)
            .expect("volatile field is present");
        assert!(value.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0));
        *value = expected
            .pointer(pointer)
            .expect("golden volatile field")
            .clone();
    }
    assert_eq!(actual, expected);
}
fn assert_identity_failure(body: &Value) {
    assert_eq!(body["facility_id"], "facility-1");
    assert_eq!(body["worker"]["alive"], false);
    assert_eq!(body["worker"]["profile_boot_error"], "engine_identity");
    assert_eq!(body["clip_recorder"]["available"], false);
    assert_eq!(body["cameras"], serde_json::json!([]));
    assert_eq!(body["seq"], 1);
}

#[test]
#[ignore = "requires GPU lane, real model/engine inputs, no NVML utility capability and isolated ml-api loopback alias"]
fn configured_cuda_refusal_matches_fixed_golden_through_binary_even_when_report_is_503() {
    let _guard = relay_guard();
    let fixture = Fixture::new("configured");
    let server = Server::start(Some(fixture.config(true)), None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    let requests = server.finish();
    assert_golden(failure_post(&requests));
    fixture.assert_no_source_activation();
}

#[test]
#[ignore = "requires GPU lane, real model/engine inputs and isolated ml-api loopback alias"]
fn empty_roster_reaches_cuda_gate_without_fabricating_a_facility_post() {
    let _guard = relay_guard();
    let fixture = Fixture::new("empty");
    let server = Server::start(Some(fixture.config(false)), None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    no_post(&server.finish());
    fixture.assert_no_source_activation();
    let diagnostic = String::from_utf8_lossy(&stderr);
    assert!(diagnostic.contains("reason=cuda_unavailable"));
    assert!(diagnostic.contains("report=no facility"));
    println!("BOOT_EMPTY_DIAGNOSTIC={}", String::from_utf8_lossy(&stderr));
}

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn changed_flow_engine_is_refused_before_cuda_through_binary() {
    let _guard = relay_guard();
    let fixture = Fixture::new("changed-engine");
    let engine = owned_path(&fixture, "ML_WORKER_FLOW_ENGINE_PATH");
    let mut bytes = fs::read(&engine).expect("owned engine");
    bytes[0] ^= 1;
    fs::write(&engine, bytes).expect("tamper only owned copy");
    let server = Server::start(Some(fixture.config(true)), None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_identity_failure(failure_post(&server.finish()));
    fixture.assert_no_source_activation();
}

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn missing_flow_identity_entry_is_refused_before_cuda_through_binary() {
    let _guard = relay_guard();
    let fixture = Fixture::new("missing-identity");
    let identity_path = owned_path(&fixture, "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH");
    let mut identity: Value =
        serde_json::from_slice(&fs::read(&identity_path).expect("identity")).expect("JSON");
    let removed = identity["engines"]["live_pose"]
        .as_object_mut()
        .expect("live_pose receipt")
        .remove("onnx_sha256");
    assert!(
        removed.is_some(),
        "tamper removes engines.live_pose.onnx_sha256"
    );
    fs::write(&identity_path, serde_json::to_vec(&identity).expect("JSON"))
        .expect("owned identity");
    let server = Server::start(Some(fixture.config(true)), None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_identity_failure(failure_post(&server.finish()));
    fixture.assert_no_source_activation();
}

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn recorded_pose_source_must_match_the_deployed_onnx() {
    let _guard = relay_guard();
    let fixture = Fixture::new("changed-source");
    let identity_path = owned_path(&fixture, "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH");
    let mut identity: Value =
        serde_json::from_slice(&fs::read(&identity_path).expect("identity")).expect("JSON");
    // Keep the two pose entries mutually consistent, so the actual deployed
    // ONNX binding—not merely the shared-source rule—must reject this tamper.
    for role in ["live_pose", "stored_pose"] {
        identity["engines"][role]["onnx_sha256"] = Value::String("0".repeat(64));
    }
    fs::write(&identity_path, serde_json::to_vec(&identity).expect("JSON"))
        .expect("tamper only owned identity");
    let server = Server::start(Some(fixture.config(true)), None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_identity_failure(failure_post(&server.finish()));
    fixture.assert_no_source_activation();
}

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn missing_fresh_config_and_lkg_preempts_a_contended_lease_without_post() {
    let _guard = relay_guard();
    let fixture = Fixture::new("no-config");
    let _lease = lease::acquire(&fixture.state).expect("held lease");
    let server = Server::start(None, None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    no_post(&server.finish());
    fixture.assert_no_source_activation();
}

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn admitted_empty_roster_preserves_real_lease_contention_exit() {
    let _guard = relay_guard();
    let fixture = Fixture::new("empty-contended");
    let _lease = lease::acquire(&fixture.state).expect("held lease");
    let server = Server::start(Some(fixture.config(false)), None);
    let (status, stderr) = run(fixture.command("run"));
    assert_eq!(
        status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    no_post(&server.finish());
    fixture.assert_no_source_activation();
    let diagnostic = String::from_utf8_lossy(&stderr);
    assert!(diagnostic.contains("reason=gpu_lease"));
    assert!(diagnostic.contains("report=no facility"));
    println!(
        "BOOT_EMPTY_LEASE_DIAGNOSTIC={}",
        String::from_utf8_lossy(&stderr)
    );
}

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn static_check_config_remains_network_free_after_run_dispatch_is_added() {
    let _guard = relay_guard();
    let fixture = Fixture::new("check-config");
    let server = Server::start(Some(fixture.config(true)), None);
    let (status, stderr) = run(fixture.command("check-config"));
    assert_eq!(
        status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert!(server.finish().is_empty());
    fixture.assert_no_source_activation();
}

/// CONFIG-endpoint fallback only: identity remains served. Not full backend-down acceptance.
#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn config_endpoint_503_reuses_exact_lkg_bytes_and_reaches_lease_gate() {
    let _guard = relay_guard();
    for (label, configured) in [("lkg-configured", true), ("lkg-empty", false)] {
        let fixture = Fixture::new(label);
        let seeded = Server::start(Some(fixture.config(configured)), None);
        let (status, stderr) = run(fixture.command("run"));
        assert_eq!(
            status.code(),
            Some(4),
            "{label} seed: {}",
            String::from_utf8_lossy(&stderr)
        );
        let seed_requests = seeded.finish();
        if configured {
            assert_golden(failure_post(&seed_requests));
        } else {
            no_post(&seed_requests);
        }
        fixture.assert_no_source_activation();
        let cached = lkg_bytes(&fixture);
        assert!(
            cached.contains_key(std::path::Path::new("current.json")),
            "{label} real binary must seed LKG"
        );
        let _lease = lease::acquire(&fixture.state).expect("held lease");
        let failed = Server::start(None, None);
        let (status, stderr) = run(fixture.command("run"));
        let diagnostic = String::from_utf8_lossy(&stderr);
        assert_eq!(status.code(), Some(3), "{label} LKG: {diagnostic}");
        assert_eq!(lkg_bytes(&fixture), cached, "{label} cached bytes changed");
        let requests = failed.finish();
        fixture.assert_no_source_activation();
        assert!(diagnostic.contains("reason=gpu_lease"), "{diagnostic}");
        assert!(
            !diagnostic.contains("worker config pull refused"),
            "{diagnostic}"
        );
        if configured {
            let report = failure_post(&requests);
            assert_eq!(report["facility_id"], "facility-1");
            assert_eq!(report["worker"]["profile_boot_error"], "gpu_lease");
        } else {
            no_post(&requests);
            assert!(diagnostic.contains("report=no facility"), "{diagnostic}");
        }
    }
}

fn lkg_bytes(fixture: &Fixture) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let root = fixture.state.join("config-lkg");
    let mut files = std::collections::BTreeMap::new();
    let mut pending = vec![root.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("LKG dir") {
            let path = entry.expect("LKG entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().is_some_and(|name| name != ".lock") {
                files.insert(
                    path.strip_prefix(&root).expect("below LKG").to_path_buf(),
                    fs::read(&path).expect("LKG bytes"),
                );
            }
        }
    }
    files
}
struct OwnedChild(Child);

#[test]
#[ignore = "requires real model/engine inputs and isolated ml-api loopback alias"]
fn actual_startup_window_admission_precedes_lease_and_cache_publication() {
    let _guard = relay_guard();
    for (start, zone, exit, drops) in [
        ("1:00", "UTC", 2, 0),
        ("0１:00", "UTC", 3, 0),
        ("1:00", "Seeon/Not_A_Zone", 3, 1),
    ] {
        let fixture = Fixture::new("window-admission");
        let _lease = lease::acquire(&fixture.state).expect("held real lease");
        let mut config = fixture.config(false);
        config["detection_windows"] = serde_json::json!({
            "fall":{"start":start,"end":"02:00","tz":zone}
        });
        let server = Server::start(Some(config), None);
        let (status, stderr) = run(fixture.command("run"));
        let diagnostic = String::from_utf8_lossy(&stderr);
        assert_eq!(status.code(), Some(exit), "{start:?}/{zone}: {diagnostic}");
        no_post(&server.finish());
        fixture.assert_no_source_activation();
        assert_eq!(diagnostic.matches("ALWAYS/24-7").count(), drops);
        if exit == 2 {
            assert!(diagnostic.contains("worker config pull skipped: malformed payload"));
            assert!(!fixture.state.join("config-lkg").exists());
        } else {
            assert!(diagnostic.contains("reason=gpu_lease"));
            assert!(diagnostic.contains("report=no facility"));
            assert!(fixture.state.join("config-lkg/current.json").is_file());
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn run(mut command: Command) -> (ExitStatus, Vec<u8>) {
    let mut child = OwnedChild(command.spawn().expect("actual ml-worker binary"));
    let clock = SystemClock::new();
    let mut status = None;
    poll_until(
        &clock,
        clock.monotonic() + Duration::from_secs(20),
        "binary boot refusal",
        || {
            status = child.0.try_wait().expect("process status");
            status.is_some()
        },
    )
    .expect("binary exits before bounded boot deadline");
    let mut stderr = Vec::new();
    child
        .0
        .stderr
        .take()
        .expect("stderr pipe")
        .read_to_end(&mut stderr)
        .expect("stderr");
    (status.expect("exited"), stderr)
}
fn owned_path(fixture: &Fixture, key: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(fixture.env.get(key).expect(key))
}
