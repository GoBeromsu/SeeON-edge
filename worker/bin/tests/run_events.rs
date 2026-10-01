//! Strategy: parent's "Admitted event reaches durable queue" row.
//! Oracles: FlowEvidenceBinding, real Python stager/queue and delivery builder.
//! Clock and filesystem failure are controlled; no fake inference or stager.

use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_ml_worker::delivery::sender::event_body;
use seeon_ml_worker::delivery::{AdmissionFault, DeliveryQueue, QueueError};
use seeon_ml_worker::json::{Json, JsonError};
use seeon_ml_worker::policy::emit::{EmitError, Stager};

use seeon_ml_worker::records::Record;
use seeon_ml_worker::records::builder::{Frame, Stream};
use seeon_ml_worker::records::id::canonical;
use seeon_ml_worker::run::events::{EventDelivery, EventDeliveryError};
use seeon_ml_worker::seam::Clock;
use seeon_worker::episode::BusinessEvent;
use serde_json::{Value, json};

type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
const ID: &str = "a5e15ff2-90fd-4764-be74-a7da4f573cc9";
const WALL_NS: u64 = 1_600_000_000_123_456_789;
const DETECTED_AT: &str = "2020-09-13T12:26:40.123456Z";

struct FixedClock {
    time: SystemTime,
    reads: AtomicUsize,
    nonwall_calls: AtomicUsize,
}

impl FixedClock {
    fn at(time: SystemTime) -> Self {
        Self {
            time,
            reads: AtomicUsize::new(0),
            nonwall_calls: AtomicUsize::new(0),
        }
    }

    fn wall_fixture() -> Self {
        Self::at(UNIX_EPOCH + Duration::from_nanos(WALL_NS))
    }
}

impl Clock for FixedClock {
    fn wall(&self) -> SystemTime {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.time
    }

    fn monotonic(&self) -> Duration {
        self.nonwall_calls.fetch_add(1, Ordering::SeqCst);
        Duration::from_secs(99)
    }

    fn pause(&self, _limit: Duration) {
        self.nonwall_calls.fetch_add(1, Ordering::SeqCst);
    }
}

fn fresh_dir(name: &str) -> io::Result<PathBuf> {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("run_events")
        .join(name);
    match fs::remove_dir_all(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn event() -> BusinessEvent {
    BusinessEvent {
        domain: "fall".into(),
        event_type: "fall".into(),
        identity: ID.into(),
        camera_id: "camera-7".into(),
        facility_id: "facility-9".into(),
        time_sec: 4.75,
        probability: Some(0.87),
        person_id: Some(12),
        bed_id: None,
    }
}

fn stream() -> Stream {
    Stream {
        camera_id: "camera-7".into(),
        worker_boot_id: "boot-origin".into(),
        source_generation: 17,
        stream_epoch: 23,
    }
}

fn frame() -> Frame {
    Frame {
        frame_seq: 129,
        source_pts_ns: Some(-4_750_000_000),
    }
}

fn stager() -> Result<Stager> {
    Ok(Stager::new(
        "camera-7",
        "facility-9",
        7,
        Some(&"a".repeat(64)),
    )?)
}

fn wire(record: &Record) -> Result<Value> {
    Ok(serde_json::from_str(&canonical(&record.to_json())?)?)
}

fn assert_receipt(record: &Record, outcome: &str, reason: Option<&str>) -> Result {
    let actual = wire(record)?;
    assert_eq!(actual["record_kind"], "event.delivery");
    assert_eq!(actual["camera_id"], "camera-7");
    assert_eq!(actual["worker_boot_id"], "boot-origin");
    assert_eq!(actual["source_generation"], 17);
    assert_eq!(actual["stream_epoch"], 23);
    assert_eq!(actual["frame_seq"], 129);
    assert_eq!(actual["source_pts_ns"], -4_750_000_000_i64);
    assert_eq!(actual["observed_at_ns"], WALL_NS);
    assert_eq!(actual["causal_unit_id"], ID);
    assert_eq!(actual["outcome"], outcome);
    assert_eq!(
        actual["payload"],
        json!({
            "edge_event_id": ID, "event_type": "fall", "domain": "fall",
            "queue": "delivery", "reason": reason,
        })
    );
    Ok(())
}

const PYTHON: &str = r#"
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from shared.events.delivery_queue import DeliveryQueue, EntryKind
import worker.pipeline.output.event_sink as event_sink
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.diagnostics.emit_delivery import event_delivery_record
from worker.types import BusinessEvent
case = json.load(sys.stdin)
stager = DurableEvidenceStager(Path(sys.argv[2]), camera_id='camera-7',
    facility_id='facility-9', config_version=7, runtime_manifest_sha256='a' * 64)
event = BusinessEvent(**case['event'])
class Trigger:
    camera_id = event.camera_id
class NoClipRecorder:
    def on_event(self, trigger, received, *, detected_at):
        assert received is event
        assert trigger.camera_id == event.camera_id
        assert detected_at.tzinfo is not None
        return None
sink = event_sink.EvidenceEventSink(
    stager, recorder=NoClipRecorder(),
    now=lambda: datetime(2020, 9, 13, 12, 26, 40, 123456, tzinfo=timezone.utc),
)
# Run the real sink, not a second implementation of its payload construction.
# Optional recording is outside this event-admission boundary.
sink.emit_for_frame(event, Trigger())
for label, value in (("true", True), ("false", False), ("object", {}), ("array", [])):
    try:
        event_sink._event_audit({label: value})
    except ValueError:
        assert label in {"object", "array"}
    else:
        assert label in {"true", "false"}

python_events = [entry for entry in stager.queue.entries() if entry["kind"] == EntryKind.EVENT]
assert len(python_events) == 1
assert list(DeliveryQueue(Path(sys.argv[1])).entries()) == python_events
expected = event_delivery_record(camera_id='camera-7', worker_boot_id='boot-origin',
    source_generation=17, stream_epoch=23, frame_seq=129, source_pts_ns=-4750000000,
    edge_event_id='a5e15ff2-90fd-4764-be74-a7da4f573cc9', event_type='fall', domain='fall',
    admitted=True, observed_at_ns=1600000000123456789)
assert expected is not None
assert case['record'] == expected.to_json()
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with the canonical Python queue and event-record consumers"]
fn durable_admission_precedes_one_receipt_and_matches_python_consumers() -> Result {
    let root = fresh_dir("python-admission")?;
    let queue = DeliveryQueue::open(&root.join("rust"), true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let event = event();
    let delivery = EventDelivery::new(&clock, &stager, &queue);
    let audit = vec![
        ("decision_trace_id".into(), Json::Str("d".repeat(64))),
        ("model_version".into(), Json::Str("model-α".into())),
        ("clock_source".into(), Json::Bool(true)),
        ("detector_version".into(), Json::Bool(false)),
        ("config_version".into(), Json::Int(999)),
        ("runtime_manifest_sha256".into(), Json::Str("b".repeat(64))),
    ];
    let mut receipts = Vec::new();
    let mut at_receipt = Vec::new();
    let staged = delivery.stage(&event, &stream(), frame(), Some(&audit), &mut |record| {
        // Public queue reads in the callback prove durability precedes observation.
        at_receipt.push(queue.entries());
        receipts.push(record);
    })?;
    assert_eq!(staged.event_ref, ID);
    assert_eq!(staged.detected_at, DETECTED_AT);
    assert!(staged.admission.accepted);
    assert!(!staged.admission.already_admitted);
    assert_eq!(clock.nonwall_calls.load(Ordering::SeqCst), 0);
    assert_eq!(receipts.len(), 1);
    assert_receipt(&receipts[0], "admitted", None)?;
    let observed = at_receipt.pop().ok_or("missing queue observation")??;
    assert_eq!(observed.len(), 1);
    assert_eq!(observed[0]["edge_event_id"], ID);
    assert_eq!(observed[0]["detected_at"], DETECTED_AT);
    // Reopen, rather than trusting an in-memory entry or envelope constructor.
    assert_eq!(
        DeliveryQueue::open(queue.directory(), true)?.entries()?,
        observed
    );
    let case = json!({
        "event": {
            "domain": "fall", "event_type": "fall", "identity": ID,
            "camera_id": "camera-7", "facility_id": "facility-9",
            "time_sec": 4.75, "probability": 0.87, "person_id": 12, "bed_id": None::<i64>,
            "audit": {
                "decision_trace_id": "d".repeat(64), "model_version": "model-α",
                "clock_source": true, "detector_version": false,
                "config_version": 999, "runtime_manifest_sha256": "b".repeat(64),
            },
        },
        "record": wire(&receipts[0])?,
    });
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = Command::new(
        std::env::var_os("SEEON_TEST_PYTHON")
            .ok_or("SEEON_TEST_PYTHON is required for the canonical consumer")?,
    )
    .arg("-c")
    .arg(PYTHON)
    .arg(queue.directory())
    .arg(root.join("python"))
    .current_dir(&repo)
    .env("PYTHONPATH", &repo)
    .stdin(Stdio::piped())
    .stderr(Stdio::inherit())
    .spawn()?;
    child
        .stdin
        .take()
        .ok_or("missing Python stdin")?
        .write_all(&serde_json::to_vec(&case)?)?;
    assert!(
        child.wait()?.success(),
        "canonical Python queue and receipt disagree"
    );
    Ok(())
}

#[test]
fn queue_conflict_preserves_original_bytes_and_actual_duplicate_result() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("conflict")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let event = event();
    let delivery = EventDelivery::new(&clock, &stager, &queue);
    let mut receipts = Vec::new();
    delivery.stage(&event, &stream(), frame(), None, &mut |r| receipts.push(r))?;

    let original = queue.entries()?;
    let duplicate = delivery.stage(&event, &stream(), frame(), None, &mut |r| receipts.push(r))?;

    assert!(duplicate.admission.accepted && duplicate.admission.already_admitted);
    // No retry inside the component: a later caller attempt with changed wall
    // time is a real immutable-content conflict, not permission to replace it.
    let later = FixedClock::at(clock.time + Duration::from_secs(1));
    let failed = EventDelivery::new(&later, &stager, &queue).stage(
        &event,
        &stream(),
        frame(),
        None,
        &mut |r| receipts.push(r),
    );

    assert!(matches!(failed, Err(EventDeliveryError::Refused(result))
        if !result.accepted && result.fault == Some(AdmissionFault::Conflict)
            && !result.already_admitted));
    assert_eq!(queue.entries()?, original);
    assert_eq!(receipts.len(), 3);
    assert_receipt(&receipts[0], "admitted", None)?;
    assert_receipt(&receipts[1], "admitted", None)?;
    let refusal = wire(&receipts[2])?;
    assert_eq!(refusal["outcome"], "refused");
    assert_eq!(refusal["payload"]["reason"], "conflict");
    assert_eq!(refusal["observed_at_ns"], WALL_NS + 1_000_000_000);
    Ok(())
}

#[test]
fn camera_and_stager_binding_mismatches_never_reach_the_queue() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("binding")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let mut event = event();
    event.camera_id = "different-camera".into();
    let mut receipts = Vec::new();
    let failure = EventDelivery::new(&clock, &stager, &queue).stage(
        &event,
        &stream(),
        frame(),
        None,
        &mut |r| receipts.push(r),
    );

    assert!(matches!(failure, Err(EventDeliveryError::CameraMismatch)));
    assert_eq!(queue.accepted_count()?, 0);
    assert!(receipts.is_empty());
    assert_eq!(clock.reads.load(Ordering::SeqCst), 0);
    event.camera_id = "camera-7".into();
    for (camera, facility) in [
        ("other-camera", "facility-9"),
        ("camera-7", "other-facility"),
    ] {
        let wrong = Stager::new(camera, facility, 7, None)?;
        let mut receipts = Vec::new();
        let failure = EventDelivery::new(&clock, &wrong, &queue).stage(
            &event,
            &stream(),
            frame(),
            None,
            &mut |r| receipts.push(r),
        );
        assert!(matches!(failure, Err(EventDeliveryError::StagerIdentity)));
        assert_eq!(receipts.len(), 1);
        assert_receipt(
            &receipts[0],
            "refused",
            Some("ValueError: stager identity does not match admitted event"),
        )?;
    }
    assert!(queue.entries()?.is_empty());
    Ok(())
}

#[test]
fn real_stager_serialization_refusals_emit_once_and_never_admit() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("serialization")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let delivery = EventDelivery::new(&clock, &stager, &queue);
    for (time, probability) in [(f64::NAN, Some(0.87)), (4.75, Some(f64::INFINITY))] {
        let mut event = event();
        event.time_sec = time;
        event.probability = probability;
        let mut receipts = Vec::new();
        let failure = delivery.stage(&event, &stream(), frame(), None, &mut |r| receipts.push(r));

        assert!(matches!(
            failure,
            Err(EventDeliveryError::Stage(EmitError::Json(
                JsonError::NonFinite
            )))
        ));
        assert_eq!(receipts.len(), 1);
        assert_receipt(
            &receipts[0],
            "refused",
            Some("ValueError: envelope json: out of range float values are not JSON compliant"),
        )?;
        assert_eq!(queue.accepted_count()?, 0);
    }
    Ok(())
}

#[test]
fn real_queue_io_failure_is_not_success_or_an_internal_retry() -> Result {
    let root = fresh_dir("queue-io")?;
    let queue = DeliveryQueue::open(&root.join("live"), true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let event = event();
    fs::rename(queue.directory(), root.join("retained"))?;
    fs::write(queue.directory(), b"filesystem failure evidence")?;
    let mut receipts = Vec::new();
    let failure = EventDelivery::new(&clock, &stager, &queue).stage(
        &event,
        &stream(),
        frame(),
        None,
        &mut |r| receipts.push(r),
    );

    assert!(matches!(
        failure,
        Err(EventDeliveryError::Queue(QueueError::Io(_)))
    ));
    assert_eq!(receipts.len(), 1);
    assert_receipt(&receipts[0], "refused", Some("NotADirectoryError"))?;
    assert_eq!(fs::read(queue.directory())?, b"filesystem failure evidence");
    assert!(
        DeliveryQueue::open(&root.join("retained"), true)?
            .entries()?
            .is_empty()
    );
    assert_eq!(clock.nonwall_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn unrepresentable_identity_provenance_and_wall_time_fail_closed() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("invalid-context")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let delivery = EventDelivery::new(&clock, &stager, &queue);
    let mut invalid_event = event();
    invalid_event.identity = "camera-7:fall:episode-1".into();
    let mut receipts = Vec::new();
    assert!(matches!(
        delivery.stage(&invalid_event, &stream(), frame(), None, &mut |r| receipts
            .push(r)),
        Err(EventDeliveryError::Identity)
    ));
    let event = event();
    let mut invalid_stream = stream();
    invalid_stream.worker_boot_id.clear();
    assert!(matches!(
        delivery.stage(&event, &invalid_stream, frame(), None, &mut |r| receipts
            .push(r)),
        Err(EventDeliveryError::Record(_))
    ));
    for time in [
        UNIX_EPOCH - Duration::from_nanos(1),
        UNIX_EPOCH + Duration::from_secs(u64::MAX / 1_000_000_000 + 1),
    ] {
        let bad_clock = FixedClock::at(time);
        assert!(matches!(
            EventDelivery::new(&bad_clock, &stager, &queue).stage(
                &event,
                &stream(),
                frame(),
                None,
                &mut |r| receipts.push(r)
            ),
            Err(EventDeliveryError::WallTime)
        ));
    }
    assert!(
        receipts.is_empty(),
        "do not invent canonical provenance for invalid inputs"
    );
    assert!(queue.entries()?.is_empty());
    Ok(())
}

#[test]
fn whole_second_wall_clock_and_absent_probability_preserve_domain_time() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("whole-second")?, true)?;
    let clock = FixedClock::at(UNIX_EPOCH + Duration::from_secs(1_600_000_000));
    let stager = stager()?;
    let mut event = event();
    event.time_sec = -1.25;
    event.probability = None;
    event.domain = "bed_exit".into();
    event.event_type = "bed-exit".into();
    event.person_id = None;
    event.bed_id = None;
    let mut receipts = Vec::new();
    let staged = EventDelivery::new(&clock, &stager, &queue).stage(
        &event,
        &stream(),
        Frame {
            frame_seq: 0,
            source_pts_ns: None,
        },
        None,
        &mut |r| receipts.push(r),
    )?;
    assert_eq!(staged.detected_at, "2020-09-13T12:26:40Z");
    assert_eq!(receipts.len(), 1);
    let record = wire(&receipts[0])?;
    assert_eq!(record["frame_seq"], 0);
    assert!(record["source_pts_ns"].is_null());
    assert_eq!(record["payload"]["event_type"], "bed-exit");
    assert_eq!(record["payload"]["domain"], "bed_exit");
    let entries = queue.entries()?;
    assert_eq!(entries[0]["detected_at"], staged.detected_at);
    let bytes = event_body(&entries[0]).map_err(|_| "queued event is not consumable by sender")?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert!(body["probability"].is_null());
    assert_eq!(body["event_type"], "bed-exit");
    assert_eq!(body["evidence"]["domain"], "bed_exit");
    assert_eq!(body["evidence"]["time_sec"], -1.25);
    assert_eq!(body["evidence"]["identity"], ID);
    assert_eq!(
        body["audit"],
        serde_json::json!({
            "config_version": 7,
            "runtime_manifest_sha256": "a".repeat(64),
        })
    );
    assert!(body["evidence"].get("person_id").is_none());
    assert!(body["evidence"].get("bed_id").is_none());
    Ok(())
}

#[test]
fn receipt_observes_post_admission_time_not_the_detection_time() -> Result {
    struct AdmissionClock<'a>(&'a DeliveryQueue);
    impl Clock for AdmissionClock<'_> {
        fn wall(&self) -> SystemTime {
            let accepted = self.0.accepted_count().expect("real queue is readable") > 0;
            UNIX_EPOCH + Duration::from_nanos(WALL_NS + if accepted { 10_000_000_000 } else { 0 })
        }
        fn monotonic(&self) -> Duration {
            Duration::ZERO
        }
        fn pause(&self, _limit: Duration) {
            panic!("event admission must not wait on its clock");
        }
    }
    let queue = DeliveryQueue::open(&fresh_dir("receipt-time")?, true)?;
    let clock = AdmissionClock(&queue);
    let stager = stager()?;
    let event = event();
    let mut receipts = Vec::new();
    let staged = EventDelivery::new(&clock, &stager, &queue).stage(
        &event,
        &stream(),
        frame(),
        None,
        &mut |record| receipts.push(record),
    )?;
    assert_eq!(staged.detected_at, DETECTED_AT);
    assert_eq!(receipts.len(), 1);
    assert_eq!(
        wire(&receipts[0])?["observed_at_ns"],
        WALL_NS + 10_000_000_000
    );
    assert_eq!(queue.entries()?[0]["detected_at"], DETECTED_AT);
    Ok(())
}
#[test]
fn zero_and_nonzero_entity_ids_and_scalar_audit_are_preserved() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("entity-audit")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let mut event = event();
    event.person_id = Some(0);
    event.bed_id = Some(9);
    let audit = vec![
        ("decision_trace_id".into(), Json::Str("d".repeat(64))),
        ("model_version".into(), Json::Str("lstm-v3".into())),
        ("operating_threshold".into(), Json::Float(0.62)),
        ("clock_source".into(), Json::Bool(false)),
        ("detector_version".into(), Json::Bool(true)),
        ("config_version".into(), Json::Int(1)),
    ];
    let mut receipts = Vec::new();
    EventDelivery::new(&clock, &stager, &queue).stage(
        &event,
        &stream(),
        frame(),
        Some(&audit),
        &mut |record| receipts.push(record),
    )?;
    let bytes = event_body(&queue.entries()?[0]).map_err(|_| "queued event is not consumable")?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert_eq!(body["evidence"]["person_id"], 0);
    assert_eq!(body["evidence"]["bed_id"], 9);
    assert_eq!(body["audit"]["decision_trace_id"], "d".repeat(64));
    assert_eq!(body["audit"]["model_version"], "lstm-v3");
    assert_eq!(body["audit"]["operating_threshold"], 0.62);
    assert_eq!(body["audit"]["clock_source"], false);
    assert_eq!(body["audit"]["detector_version"], true);

    // Stager config authority overwrites a caller-supplied version and fills
    // the configured manifest. The supplied trace id is not replaced.
    assert_eq!(body["audit"]["config_version"], 7);
    assert_eq!(body["audit"]["runtime_manifest_sha256"], "a".repeat(64));
    assert_eq!(receipts.len(), 1);
    Ok(())
}

#[test]
fn nested_audit_is_refused_before_clock_or_queue() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("bad-audit")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = stager()?;
    let event = event();
    for audit in [
        vec![("decision_trace_id".into(), Json::Object(Vec::new()))],
        vec![("decision_trace_id".into(), Json::Array(Vec::new()))],
    ] {
        let mut receipts = Vec::new();
        let failure = EventDelivery::new(&clock, &stager, &queue).stage(
            &event,
            &stream(),
            frame(),
            Some(&audit),
            &mut |record| receipts.push(record),
        );
        assert!(matches!(failure, Err(EventDeliveryError::Audit)));
        assert!(receipts.is_empty());
    }
    assert_eq!(clock.reads.load(Ordering::SeqCst), 0);
    assert!(queue.entries()?.is_empty());
    Ok(())
}

#[test]
fn absent_audit_is_omitted_rather_than_defaulted() -> Result {
    let queue = DeliveryQueue::open(&fresh_dir("absent-audit")?, true)?;
    let clock = FixedClock::wall_fixture();
    let stager = Stager::new("camera-7", "facility-9", 7, None)?;
    let mut receipts = Vec::new();
    EventDelivery::new(&clock, &stager, &queue).stage(
        &event(),
        &stream(),
        frame(),
        None,
        &mut |record| receipts.push(record),
    )?;
    let bytes = event_body(&queue.entries()?[0]).map_err(|_| "queued event is not consumable")?;
    let body: Value = serde_json::from_slice(&bytes)?;
    assert!(body.get("audit").is_none());
    assert_eq!(body["evidence"]["person_id"], 12);
    assert!(body["evidence"].get("bed_id").is_none());
    assert_eq!(receipts.len(), 1);
    Ok(())
}
