//! T34: `policy/coverage.rs` (G16) against `d/observation-coverage.json`,
//! recorded from the Python `ObservationCoverage` at 030aaf1. Every golden
//! call passes its host time, so the fake clock is only read by the
//! defaulting case below. Durations are compared as values, `None` for
//! unknown; a refusal is compared by class only.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use seeon_ml_worker::policy::coverage::{
    ActualObservation, ForeignIdentity, MetadataObservation, ObservationCoverage, ObservationGap,
    ObservationIdentity, ObservationRecovery,
};
use seeon_ml_worker::seam::Clock;
use serde_json::Value;
use sha2::{Digest, Sha256};

const GOLDEN_SHA256_PREFIX: &str = "a8295753967a";
const FAKE_MONOTONIC_SECONDS: u64 = 5000;

/// The recipe's monotonic fake: always 5000 s, and time never moves.
struct FakeClock;

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        Duration::from_secs(FAKE_MONOTONIC_SECONDS)
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    fn pause(&self, _limit: Duration) {}
}

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/d/observation-coverage.json");
    let bytes = std::fs::read(&path).expect("observation-coverage golden is readable");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert!(
        digest.starts_with(GOLDEN_SHA256_PREFIX),
        "golden sha256 {digest}"
    );
    serde_json::from_slice(&bytes).expect("observation-coverage golden is JSON")
}

fn text(value: &Value, key: &str) -> String {
    value[key].as_str().expect(key).to_owned()
}

fn unsigned(value: &Value, key: &str) -> u64 {
    value[key].as_u64().expect(key)
}

fn identity(value: &Value) -> ObservationIdentity {
    ObservationIdentity {
        worker_boot_id: text(value, "worker_boot_id"),
        camera_id: text(value, "camera_id"),
        source_generation: unsigned(value, "source_generation"),
        stream_epoch: unsigned(value, "stream_epoch"),
    }
}

fn actual(value: &Value) -> Option<ActualObservation> {
    (!value.is_null()).then(|| ActualObservation {
        identity: identity(&value["identity"]),
        seq: unsigned(value, "seq"),
        source_pts_ns: value["source_pts_ns"].as_u64(),
        native_publish_sequence: unsigned(value, "native_publish_sequence"),
        host_time: value["host_time"].as_f64().expect("host_time"),
    })
}

fn gap(value: &Value) -> Option<ObservationGap> {
    (!value.is_null()).then(|| ObservationGap {
        identity: identity(&value["identity"]),
        last_actual: actual(&value["last_actual"]),
        loss_detected_host_time: value["loss_detected_host_time"]
            .as_f64()
            .expect("loss_detected_host_time"),
    })
}

fn recovery(value: &Value) -> Option<ObservationRecovery> {
    (!value.is_null()).then(|| ObservationRecovery {
        gap: gap(&value["gap"]).expect("a recovery closes a gap"),
        next_actual: actual(&value["next_actual"]).expect("a recovery has its input"),
        host_observation_duration: value["host_observation_duration"].as_f64(),
        source_duration_ns: value["source_duration_ns"].as_u64(),
        native_publish_sequence_gap: value["native_publish_sequence_gap"].as_u64(),
    })
}

/// The golden's `observe` arguments; `source_pts` is in nanoseconds.
fn observation(arguments: &Value) -> MetadataObservation {
    MetadataObservation {
        identity: identity(arguments),
        seq: unsigned(arguments, "seq"),
        source_pts_ns: arguments["source_pts"].as_u64(),
        native_publish_sequence: unsigned(arguments, "native_publish_sequence"),
    }
}

/// Python raises `ValueError` for a foreign identity; no other class occurs.
fn refusal(step: &Value) -> Option<ForeignIdentity> {
    step.get("raised").map(|raised| {
        assert_eq!(raised["class"], "ValueError", "unmapped raised class");
        ForeignIdentity
    })
}

fn binding() -> ObservationIdentity {
    ObservationIdentity {
        worker_boot_id: "boot-a".to_owned(),
        camera_id: "cmsnw6rjc01vhlh01oswn99yq".to_owned(),
        source_generation: 1,
        stream_epoch: 0,
    }
}

fn coverage() -> ObservationCoverage {
    ObservationCoverage::new(binding(), Arc::new(FakeClock))
}

fn frame(identity: ObservationIdentity, seq: u64, pts_ns: Option<u64>) -> MetadataObservation {
    MetadataObservation {
        identity,
        seq,
        source_pts_ns: pts_ns,
        native_publish_sequence: seq + 1,
    }
}

#[test]
fn every_golden_step_returns_and_leaves_the_recorded_state() {
    let golden = golden();
    let steps = golden["steps"].as_array().expect("steps");
    assert_eq!(steps.len(), 22);
    let mut coverage = coverage();
    for step in steps {
        let label = text(step, "label");
        let arguments = &step["arguments"];
        match step["call"].as_str().expect("call") {
            "detect_gap" => {
                let host_time = arguments["host_time"].as_f64();
                let returned = coverage.detect_gap(host_time);
                assert_eq!(returned, gap(&step["returned"]), "{label}");
            }
            "observe" => {
                let host_time = arguments["host_time"].as_f64();
                let returned = coverage.observe(observation(arguments), host_time);
                let expected = match refusal(step) {
                    Some(refused) => Err(refused),
                    None => Ok(recovery(&step["returned"])),
                };
                assert_eq!(returned, expected, "{label}");
            }
            "rebind" => coverage.rebind(identity(&arguments["binding"])),
            call => panic!("unknown golden call {call}"),
        }
        let state = &step["state_after"];
        assert_eq!(
            coverage.last_actual(),
            actual(&state["last_actual"]).as_ref(),
            "{label}"
        );
        assert_eq!(
            coverage.open_gap(),
            gap(&state["open_gap"]).as_ref(),
            "{label}"
        );
    }
}

#[test]
fn a_foreign_identity_is_refused_and_the_gap_stays_open() {
    let foreign: [fn(&mut ObservationIdentity); 4] = [
        |identity| identity.worker_boot_id = "boot-foreign".to_owned(),
        |identity| identity.camera_id = "camera-foreign".to_owned(),
        |identity| identity.source_generation = 2,
        |identity| identity.stream_epoch = 1,
    ];
    for change in foreign {
        let mut coverage = coverage();
        coverage
            .observe(frame(binding(), 0, Some(0)), Some(1.0))
            .expect("the expected binding is accepted");
        let open = coverage.detect_gap(Some(2.0));
        let mut identity = binding();
        change(&mut identity);
        let refused = coverage.observe(frame(identity, 1, Some(33_333_333)), Some(3.0));
        assert_eq!(refused, Err(ForeignIdentity));
        assert_eq!(coverage.open_gap(), open.as_ref());
        assert_eq!(coverage.last_actual().map(|last| last.seq), Some(0));
    }
}

#[test]
fn backwards_host_time_or_pts_is_unknown_never_negative() {
    let mut coverage = coverage();
    coverage
        .observe(frame(binding(), 0, Some(2_000_000_000)), Some(50.0))
        .expect("accepted");
    coverage.detect_gap(Some(51.0));
    let backwards = coverage
        .observe(frame(binding(), 1, Some(1_000_000_000)), Some(49.0))
        .expect("accepted")
        .expect("the open gap closes");
    assert_eq!(backwards.host_observation_duration, None);
    assert_eq!(backwards.source_duration_ns, None);
    assert_eq!(backwards.native_publish_sequence_gap, Some(0));

    coverage.detect_gap(Some(52.0));
    let equal = coverage
        .observe(frame(binding(), 2, Some(1_000_000_000)), Some(49.0))
        .expect("accepted")
        .expect("the open gap closes");
    assert_eq!(equal.host_observation_duration, Some(0.0));
    assert_eq!(equal.source_duration_ns, Some(0));
}

#[test]
fn an_omitted_host_time_reads_the_injected_monotonic_clock() {
    let seconds = Duration::from_secs(FAKE_MONOTONIC_SECONDS).as_secs_f64();
    let mut coverage = coverage();
    let gap = coverage.detect_gap(None).expect("no gap was open");
    assert_eq!(gap.loss_detected_host_time, seconds);
    let recovery = coverage
        .observe(frame(binding(), 0, None), None)
        .expect("accepted")
        .expect("the open gap closes");
    assert_eq!(recovery.next_actual.host_time, seconds);
}
