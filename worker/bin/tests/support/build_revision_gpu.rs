//! Connected qualification of one actual packaged `ml-worker`. Included only
//! by `run_gpu`. The selected binary is the test override or the Cargo binary.
//! Parent revision and image are checked, never invented. The image marker
//! must already be mounted; this module never writes it.
//!
//! Three unique finite `model.score` record ids are positive score transport.
//! Identical finite scores are valid. This is not native recording and not
//! PostgreSQL durability. The fixture ACK remains in memory. T28 stays separate.
use std::collections::BTreeSet;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::records::id::sha256_hex_field;
use seeon_ml_worker::seam::{Clock, SystemClock};
use serde_json::Value;

use super::fixture::{EXECUTION_RECORDS_PATH, Fixture, Request, Server, request_route};
use super::gpu_process::{GpuChild, capture_pipe, signal_owned};

const MODEL_SCORE: &str = "model.score";
pub const REQUIRED_SCORES: usize = 3;
/// Same bound as idle CUDA and model-owner readiness. Not a sleep.
const SCORE_WAIT: Duration = Duration::from_secs(45);
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(25);
const SHUTDOWN_REAP_ALLOWANCE: Duration = Duration::from_secs(2);
const IMAGE_MARKER: &str = "/opt/seeon/ml-worker-image-revision";
const PARENT_REVISION: &str = "ML_WORKER_BUILD_REVISION";
const PARENT_IMAGE: &str = "ML_WORKER_IMAGE";

pub struct ScoreShutdown {
    pub status: ExitStatus,
    pub stderr: Vec<u8>,
}

/// Parent `ML_WORKER_BUILD_REVISION`: 40 lowercase hex digits, not all zero.
pub fn parent_revision() -> String {
    let revision = std::env::var(PARENT_REVISION).unwrap_or_else(|_| {
        panic!("{PARENT_REVISION} required; a missing declaration is a failure")
    });
    assert!(
        is_revision(&revision),
        "{PARENT_REVISION} must be 40 lowercase hex digits and not all zero"
    );
    revision
}

/// Normalized deployment digest of parent `ML_WORKER_IMAGE`, equal to the
/// fixture aggregate image authority. A `name@sha256:` reference is not kept.
pub fn parent_image(fixture: &Fixture) -> String {
    let image = std::env::var(PARENT_IMAGE)
        .unwrap_or_else(|_| panic!("{PARENT_IMAGE} required; a missing declaration is a failure"));
    let digest = seeon_ml_worker::config::model_bundle::identity::deployment_image_digest(&image)
        .unwrap_or_else(|| panic!("{PARENT_IMAGE} must carry a sha256 deployment digest"))
        .to_owned();
    assert_eq!(
        digest, fixture.env["ML_WORKER_IMAGE"],
        "parent image digest must equal fixture aggregate image authority"
    );
    digest
}

/// The marker must already be mounted. Tests and production do not create or
/// rewrite `/opt/seeon/ml-worker-image-revision`.
pub fn require_image_marker(revision: &str, contradictory: bool) {
    let metadata = std::fs::symlink_metadata(IMAGE_MARKER)
        .unwrap_or_else(|error| panic!("{IMAGE_MARKER} must already be a regular marker: {error}"));
    assert!(
        metadata.file_type().is_file() && metadata.len() <= 41,
        "{IMAGE_MARKER} must already be a bounded regular marker"
    );
    let mut text = std::fs::read_to_string(IMAGE_MARKER)
        .unwrap_or_else(|error| panic!("read existing {IMAGE_MARKER}: {error}"));
    if text.ends_with('\n') {
        text.pop();
    }
    let matches = is_revision(&text) && text == revision;
    assert!(
        is_revision(&text) && matches != contradictory,
        "{IMAGE_MARKER} must already {} the runtime declaration",
        if contradictory { "contradict" } else { "match" }
    );
}

/// Waits for three unique finite actual score ids, then SIGTERM only this child.
/// The readiness clock is rechecked after the condition poll. Success is exit 0
/// inside the pre-signal 25s budget. The reap allowance is cleanup only.
pub fn shutdown_after_actual_scores(
    child: &mut GpuChild,
    server: &Server,
    camera_id: &str,
    revision: &str,
    image: &str,
    shutdown_started: &AtomicBool,
) -> ScoreShutdown {
    let stdout = child.take_stdout();
    let stderr = child.take_stderr();
    let stdout_done = std::thread::spawn(move || capture_pipe(stdout));
    let stderr_done = std::thread::spawn(move || capture_pipe(stderr));
    let clock = SystemClock::new();
    let ready_deadline = clock.monotonic() + SCORE_WAIT;
    let mut early = None;
    let ready = poll_until(&clock, ready_deadline, "actual model.score records", || {
        early = child.try_wait();
        early.is_some()
            || attributed_score_ids(&server.observed(), camera_id, revision, image).len()
                >= REQUIRED_SCORES
    });
    let ready_late = clock.monotonic() > ready_deadline;
    let scores = attributed_score_ids(&server.observed(), camera_id, revision, image).len();
    if ready.is_err() || ready_late || early.is_some() || scores < REQUIRED_SCORES {
        if early.is_none() {
            child.kill_owned();
        }
        let captured = join_drains(stdout_done, stderr_done);
        panic!(
            "three unique finite actual model.score ids required before SIGTERM; \
             status={early:?} wait={ready:?} late={ready_late} observed={scores}: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    let shutdown_deadline = clock.monotonic() + SHUTDOWN_BUDGET;
    shutdown_started.store(true, Ordering::SeqCst);
    signal_owned(child.pid());
    let mut status = None;
    let mut observed_exit = None;
    let stopped = poll_until(&clock, shutdown_deadline, "score SIGTERM shutdown", || {
        status = child.try_wait();
        if status.is_some() {
            observed_exit = Some(clock.monotonic());
        }
        status.is_some()
    });
    if stopped.is_err() || observed_exit.is_none_or(|time| time > shutdown_deadline) {
        let cleanup = clock.monotonic() + SHUTDOWN_REAP_ALLOWANCE;
        let _ = poll_until(&clock, cleanup, "failed score shutdown cleanup", || {
            child.try_wait().is_some()
        });
        if child.try_wait().is_none() {
            child.kill_owned();
        }
        let captured = join_drains(stdout_done, stderr_done);
        panic!(
            "SIGTERM did not produce an observed exit inside the pre-signal 25s budget: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    ScoreShutdown {
        status: status.expect("exited"),
        stderr: join_drains(stdout_done, stderr_done),
    }
}

/// Unique record ids with a finite score and the expected provenance.
/// Identical score values count separately. Duplicate ids count once.
pub fn attributed_score_ids(
    requests: &[Request],
    camera_id: &str,
    revision: &str,
    image: &str,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    for request in requests {
        if request.method != "POST" || request_route(&request.path) != EXECUTION_RECORDS_PATH {
            continue;
        }
        let provenance = &request.body["provenance"];
        if provenance["worker_build_revision"] != revision
            || provenance["worker_image_digest"] != image
        {
            continue;
        }
        let Some(records) = request.body["records"].as_array() else {
            continue;
        };
        for record in records {
            if let Some(record_id) = score_id(record, camera_id) {
                seen.insert(record_id);
            }
        }
    }
    seen
}

fn score_id(record: &Value, camera_id: &str) -> Option<String> {
    if record["camera_id"] != camera_id || record["record_kind"] != MODEL_SCORE {
        return None;
    }
    let record_id = record["record_id"].as_str()?;
    sha256_hex_field(record_id, "record_id").ok()?;
    record["payload"]["fall_transition"]
        .as_f64()
        .filter(|value| value.is_finite())?;
    Some(record_id.to_owned())
}

fn is_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn join_drains(
    stdout_done: std::thread::JoinHandle<Vec<u8>>,
    stderr_done: std::thread::JoinHandle<Vec<u8>>,
) -> Vec<u8> {
    let stderr = stderr_done.join().expect("stderr drain");
    let _stdout = stdout_done.join().expect("stdout drain");
    stderr
}
