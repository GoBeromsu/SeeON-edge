//! Strategy: notes/test-strategy-rust-s4.md (adapter/provenance/lifecycle rows).
//! Oracles: Python emit_policy.py, reviewed worker-wire fixtures, and the
//! Stage4 ownership contract. Network and clock alone are controlled seams.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_deepstream_native::GpuMetrics;
use seeon_ml_worker::policy::emit::{ModelEvidence, ModelScore};
use seeon_ml_worker::records::builder::{AuthorityRole, DecisionSource, Frame, Numeric, Stream};
use seeon_ml_worker::records::id::{ContractError, canonical};
use seeon_ml_worker::records::{Lanes, Provenance, RELAY_TIMEOUT, Record};
use seeon_ml_worker::relay::RelayClient;
use seeon_ml_worker::relay::wire::DeliveryDisposition;
use seeon_ml_worker::run::decision::{DecisionError, adapt_decision};
use seeon_ml_worker::run::exporter::{self, JoinError, SpawnError};
use seeon_ml_worker::run::records::{RecordError, decision_record, model_score_record};
use seeon_ml_worker::seam::Clock;
use seeon_worker::trace::{
    DecisionTraceMissingReason as Missing, DecisionTraceReason as Reason, DecisionTraceSnapshot,
    DecisionTraceState as State, DecisionTraceValueName as Name, NumericTraceValue, TraceFloat,
};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const WALL_NS: u64 = 1_787_000_000_000_000_000;
// Transport/deadlock safety only, never an elapsed-time assertion or test oracle.
const SAFETY: Duration = Duration::from_secs(30);

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/r")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn stream() -> Stream {
    Stream {
        camera_id: "cmsnw6rjc01vhlh01oswn99yq".into(),
        worker_boot_id: "boot-0001".into(),
        source_generation: 1,
        stream_epoch: 1,
    }
}

fn frame() -> Frame {
    Frame {
        frame_seq: 129,
        source_pts_ns: Some(4_966_666_657),
    }
}

fn value(record: &Record) -> Value {
    serde_json::from_str(&canonical(&record.to_json()).unwrap()).unwrap()
}

fn snapshot(track: Option<u64>, bed: Option<u64>, triggered: bool) -> DecisionTraceSnapshot {
    DecisionTraceSnapshot::new(
        Reason::TransitionConfirmed,
        (State::TransitionCandidate, State::TransitionConfirmed),
        triggered,
        track,
        bed,
        BTreeMap::from([
            (Name::TransitionVotes, NumericTraceValue::Integer(2)),
            (
                Name::TransitionThreshold,
                NumericTraceValue::Float(TraceFloat::new(1.0).unwrap()),
            ),
        ]),
        BTreeMap::from([(Name::FallenProbability, Missing::ClassifierStrideNotDue)]),
    )
    .unwrap()
}

#[test]
fn decision_bridge_preserves_domain_kinds_tokens_subjects_and_absence() {
    let source = DecisionSource {
        generation: Some(3),
        module_qualified_id: Some("fall.v2".into()),
        authority_role: AuthorityRole::Authoritative,
        decision_trace_id: Some("trace-7".into()),
    };
    for (track, bed, triggered) in [(Some(7), Some(9), true), (None, None, false)] {
        let snapshot = snapshot(track, bed, triggered);
        let adapted = adapt_decision(&snapshot).unwrap();
        assert!(
            adapted
                .values
                .contains(&("transition_votes".into(), Numeric::Int(2)))
        );
        assert!(
            adapted
                .values
                .contains(&("transition_threshold".into(), Numeric::Float(1.0)))
        );
        let record = decision_record(&stream(), frame(), WALL_NS, &snapshot, &source).unwrap();
        let wire = value(&record);
        assert_eq!(
            wire["payload"],
            json!({
                "reason": "transition-confirmed", "previous_state": "transition-candidate",
                "current_state": "transition-confirmed", "triggered": triggered,
                "track_id": track, "bed_id": bed,
                "values": {"transition_votes": 2, "transition_threshold": 1.0},
                "missing_values": {"fallen_probability": "classifier-stride-not-due"},
                "module_qualified_id": "fall.v2", "authority_role": "authoritative",
                "decision_trace_id": "trace-7"
            })
        );
        assert!(wire["payload"]["values"]["transition_votes"].is_u64());
        assert!(wire["payload"]["values"]["transition_threshold"].is_f64());
        assert_eq!(
            wire["outcome"],
            if triggered {
                "triggered"
            } else {
                "transition-confirmed"
            }
        );
        assert_eq!(record.body().observed_at_ns, WALL_NS);
        assert_eq!(record.body().frame_seq, Some(129));
        assert_eq!(record.body().source_pts_ns, Some(4_966_666_657));
    }
}

#[test]
fn decision_bridge_refuses_integer_truncation() {
    let overflow = usize::try_from(i128::from(i64::MAX) + 1).unwrap();
    let snapshot = DecisionTraceSnapshot::new(
        Reason::BelowThreshold,
        (State::Clear, State::Clear),
        false,
        None,
        None,
        BTreeMap::from([(Name::TransitionVotes, NumericTraceValue::Integer(overflow))]),
        BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(
        adapt_decision(&snapshot),
        Err(DecisionError::IntegerOutOfRange(Name::TransitionVotes))
    );
}

fn score() -> ModelScore {
    let golden = fixture("execution-records.model-evidence.json");
    let runner = &golden["runner_output"];
    ModelScore {
        track_id: 1,
        generation: Some(0),
        fall_transition: runner["fall_transition"].as_f64(),
        background: runner["background"].as_f64(),
        fallen: runner["fallen"].as_f64(),
        evidence: Some(ModelEvidence {
            raw_logit: runner["raw_logit"].as_f64().unwrap(),
            applied_temperature: runner["applied_temperature"].as_f64().unwrap(),
        }),
    }
}

// Same native counter fixture as wire_records.rs; no inference substitute.
fn accelerator(elapsed: u64) -> AcceleratorEvidence {
    let before = GpuMetrics {
        attempted: 4,
        succeeded: 4,
        failed: 0,
        host_to_device_bytes: 1_000,
        device_to_host_bytes: 200,
        elapsed_ns: 4_000,
        device: 0,
    };
    let after = GpuMetrics {
        attempted: 5,
        succeeded: 5,
        failed: 0,
        host_to_device_bytes: 1_600,
        device_to_host_bytes: 260,
        elapsed_ns: 4_000 + elapsed,
        device: 0,
    };
    AcceleratorEvidence::from_delta(
        &before,
        &after,
        0,
        EngineDigest::new([0xab; 32]),
        Precision::Fp32,
    )
    .unwrap()
}

fn record() -> Record {
    model_score_record(&stream(), frame(), WALL_NS, &score(), &accelerator(789)).unwrap()
}

#[test]
fn model_bridge_binds_calibration_and_runtime_receipt_before_hashing() {
    let mut expected =
        fixture("execution-records.model-evidence.json")["batch"]["records"][0].clone();
    expected["payload"]["accelerator"] = json!({
        "provider": "tensorrt", "precision": "fp32", "device_ordinal": 0,
        "engine_sha256": "ab".repeat(32), "call_seq": 5, "attempted": 1,
        "succeeded": 1, "failed": 0, "h2d_bytes": 600, "d2h_bytes": 60, "elapsed_ns": 789
    });
    expected.as_object_mut().unwrap().remove("record_id");
    // Expected hash input is the reviewed Python body plus the accelerator
    // contract, not Rust builder output; these fixture floats have identical reprs.
    let expected_id: String = Sha256::digest(serde_json::to_vec(&expected).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    expected["record_id"] = json!(expected_id);
    let first = record();
    assert_eq!(value(&first), expected);
    let changed =
        model_score_record(&stream(), frame(), WALL_NS, &score(), &accelerator(790)).unwrap();
    assert_ne!(first.record_id(), changed.record_id());
    let mut next_stream = stream();
    next_stream.source_generation = 7;
    next_stream.stream_epoch = 9;
    let changed =
        model_score_record(&next_stream, frame(), WALL_NS, &score(), &accelerator(789)).unwrap();
    assert_eq!(changed.body().source_generation, 7);
    assert_eq!(changed.body().stream_epoch, 9);
    assert_ne!(first.record_id(), changed.record_id());
    let mut no_calibration = score();
    no_calibration.evidence = None;
    no_calibration.generation = None;
    let wire = value(
        &model_score_record(
            &stream(),
            frame(),
            WALL_NS,
            &no_calibration,
            &accelerator(789),
        )
        .unwrap(),
    );
    assert!(wire["payload"]["generation"].is_null());
    assert!(wire["payload"].get("raw_logit").is_none());
    assert!(wire["payload"].get("class_origins").is_none());
    assert!(wire["payload"].get("accelerator").is_some());
}

#[test]
fn model_bridge_refuses_signed_id_loss_and_missing_probabilities() {
    let base = score();
    let cases = [
        (
            ModelScore {
                track_id: -1,
                ..base
            },
            RecordError::NegativeIdentity("track_id"),
        ),
        (
            ModelScore {
                generation: Some(-1),
                ..base
            },
            RecordError::NegativeIdentity("generation"),
        ),
        (
            ModelScore {
                fall_transition: None,
                ..base
            },
            RecordError::MissingScore("fall_transition"),
        ),
        (
            ModelScore {
                background: None,
                ..base
            },
            RecordError::MissingScore("background"),
        ),
        (
            ModelScore {
                fallen: None,
                ..base
            },
            RecordError::MissingScore("fallen"),
        ),
    ];
    for (score, error) in cases {
        assert_eq!(
            model_score_record(&stream(), frame(), WALL_NS, &score, &accelerator(789)),
            Err(error)
        );
    }
}

fn provenance() -> Provenance {
    Provenance {
        worker_build_revision: "abc123".into(),
        worker_image_digest: "sha256:deadbeef".into(),
        model_digest: "model-1".into(),
        calibration_digest: "cal-1".into(),
        preprocessing_identity: "pose-bbox56/v1".into(),
        policy_identity: "fall.policy:2".into(),
        // Explicit fixture identity: published SHA256 of the bytes {}. The
        // parent's effective-config recipe is deliberately not implemented here.
        config_digest: "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a".into(),
    }
}

/// Controlled pause boundary. A pause never advances time without a permit.
struct GateClock {
    now: AtomicU64,
    paused: SyncSender<Duration>,
    permits: Mutex<Receiver<()>>,
}

impl Clock for GateClock {
    fn monotonic(&self) -> Duration {
        Duration::from_nanos(self.now.load(Ordering::Acquire))
    }
    fn wall(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_nanos(WALL_NS)
    }
    fn pause(&self, limit: Duration) {
        self.paused.send(limit).unwrap();
        self.permits.lock().unwrap().recv_timeout(SAFETY).unwrap();
        self.now
            .fetch_add(u64::try_from(limit.as_nanos()).unwrap(), Ordering::AcqRel);
    }
}

fn gate_clock() -> (Arc<GateClock>, Receiver<Duration>, SyncSender<()>) {
    let (paused_tx, paused_rx) = mpsc::sync_channel(1);
    let (permit_tx, permit_rx) = mpsc::sync_channel(1);
    (
        Arc::new(GateClock {
            now: AtomicU64::new(0),
            paused: paused_tx,
            permits: Mutex::new(permit_rx),
        }),
        paused_rx,
        permit_tx,
    )
}

/// Deadline clock for native join polls. No wall clock or fixed sleep.
#[derive(Default)]
struct JoinClock(AtomicU64);
impl Clock for JoinClock {
    fn monotonic(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Acquire))
    }
    fn wall(&self) -> SystemTime {
        UNIX_EPOCH
    }
    fn pause(&self, limit: Duration) {
        thread::yield_now();
        self.0
            .fetch_add(u64::try_from(limit.as_nanos()).unwrap(), Ordering::AcqRel);
    }
}

struct Relay {
    base: String,
    posted: Receiver<Value>,
    reply: SyncSender<u16>,
    thread: JoinHandle<()>,
}

// Existing records_exporter.rs loopback pattern, with owned thread and a
// response gate so timeout/late-policy-record ordering is deterministic.
fn relay(requests: usize) -> Relay {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (posted_tx, posted) = mpsc::sync_channel(1);
    let (reply, replies) = mpsc::sync_channel(1);
    let thread = thread::spawn(move || {
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(SAFETY)).unwrap();
            stream.set_write_timeout(Some(SAFETY)).unwrap();
            let mut bytes = Vec::new();
            let (head, size) = loop {
                let mut chunk = [0; 4096];
                let read = stream.read(&mut chunk).unwrap();
                assert_ne!(read, 0);
                bytes.extend_from_slice(&chunk[..read]);
                let mut headers = [httparse::EMPTY_HEADER; 32];
                let mut request = httparse::Request::new(&mut headers);
                if let httparse::Status::Complete(head) = request.parse(&bytes).unwrap() {
                    assert_eq!(request.method, Some("POST"));
                    let header = request
                        .headers
                        .iter()
                        .find(|h| h.name.eq_ignore_ascii_case("content-length"))
                        .unwrap();
                    let size = std::str::from_utf8(header.value)
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    break (head, size);
                }
            };
            while bytes.len() < head + size {
                let mut chunk = [0; 4096];
                let read = stream.read(&mut chunk).unwrap();
                assert_ne!(read, 0);
                bytes.extend_from_slice(&chunk[..read]);
            }
            let body: Value = serde_json::from_slice(&bytes[head..head + size]).unwrap();
            posted_tx.send(body.clone()).unwrap();
            let status = replies.recv_timeout(SAFETY).unwrap();
            let response = if status == 200 {
                json!({"batch_id": body["batch_id"], "accepted": body["records"].as_array().unwrap().len(),
                    "duplicates": 0, "rejected": [], "storage_state": "committed", "committed_at_ns": WALL_NS})
            } else {
                json!({})
            };
            let response = serde_json::to_vec(&response).unwrap();
            write!(stream, "HTTP/1.1 {status} Reply\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len()).unwrap();
            stream.write_all(&response).unwrap();
            stream.flush().unwrap();
        }
    });
    Relay {
        base,
        posted,
        reply,
        thread,
    }
}

fn start(lanes: Arc<Lanes>, base: &str, clock: Arc<dyn Clock>) -> exporter::Handle {
    let client = RelayClient::new(base, "<relay-token>", RELAY_TIMEOUT).unwrap();
    exporter::spawn(lanes, client, provenance(), 1, 50, clock).unwrap()
}

#[test]
fn composed_exporter_preserves_pending_lane_records_and_provenance() {
    let relay = relay(1);
    let lanes = Arc::new(Lanes::new(4).unwrap());
    let client = RelayClient::new(&relay.base, "<relay-token>", RELAY_TIMEOUT).unwrap();
    let pipeline = seeon_ml_worker::records::compose::Composed {
        settings: seeon_ml_worker::config::env::ExecutionRecordsSettings {
            lane_capacity: 4,
            batch_max: 1,
            flush_ms: 50,
        },
        exporter: seeon_ml_worker::records::Exporter::new(
            Arc::clone(&lanes),
            client,
            provenance(),
            1,
            50,
        )
        .unwrap(),
        lanes: Arc::clone(&lanes),
    };
    assert!(lanes.try_emit(record()));
    let (clock, _paused, _permit) = gate_clock();
    let mut owner = exporter::spawn_composed(pipeline, clock).unwrap();
    let body = relay.posted.recv_timeout(SAFETY).unwrap();
    assert_eq!(body["records"][0], value(&record()));
    assert_eq!(
        body["provenance"]["config_digest"],
        provenance().config_digest
    );
    owner.request_stop();
    relay.reply.send(200).unwrap();
    relay.thread.join().unwrap();
    let report = owner
        .join(&JoinClock::default(), Duration::from_secs(3600))
        .unwrap();
    assert!(report.finished);
    assert!(!report.pending);
    assert!(!lanes.has_work());
    assert!(report.failures.is_empty());
    assert_eq!(
        report
            .receipts
            .iter()
            .map(|receipt| receipt.accepted)
            .sum::<u64>(),
        1
    );
}

#[test]
fn exporter_drains_late_policy_records_and_retains_timed_out_owner() {
    let relay = relay(2);
    let lanes = Arc::new(Lanes::new(4).unwrap());
    let (clock, _paused, _permit) = gate_clock();
    assert!(lanes.try_emit(record()));
    let mut owner = start(Arc::clone(&lanes), &relay.base, clock);
    let first = relay.posted.recv_timeout(SAFETY).unwrap();
    let mut expected_provenance = fixture("execution-records.json")["provenance"].clone();
    expected_provenance["config_digest"] = json!(provenance().config_digest);
    assert_eq!(first["provenance"], expected_provenance);
    assert_eq!(first["records"][0], value(&record()));
    assert_eq!(
        owner.join(&JoinClock::default(), Duration::ZERO),
        Err(JoinError::Timeout)
    );
    assert!(!owner.is_finished());
    assert!(!owner.report().finished);
    let policy = decision_record(
        &stream(),
        frame(),
        WALL_NS,
        &snapshot(Some(7), None, true),
        &DecisionSource {
            generation: Some(3),
            module_qualified_id: Some("fall.v2".into()),
            authority_role: AuthorityRole::Authoritative,
            decision_trace_id: None,
        },
    )
    .unwrap();
    assert!(lanes.try_emit(policy.clone()));
    assert!(owner.request_stop().newly_requested);
    assert!(!owner.request_stop().newly_requested);
    relay.reply.send(200).unwrap();
    let second = relay.posted.recv_timeout(SAFETY).unwrap();
    assert_eq!(second["records"][0], value(&policy));
    relay.reply.send(200).unwrap();
    relay.thread.join().unwrap();
    let report = owner
        .join(&JoinClock::default(), Duration::from_secs(3600))
        .unwrap();
    assert!(report.finished);
    assert!(!report.pending);
    assert!(!lanes.has_work());
    assert!(report.failures.is_empty());
    assert_eq!(report.receipts.iter().map(|r| r.accepted).sum::<u64>(), 2);
    assert_eq!(
        report.receipts[0].batch_id,
        first["batch_id"].as_str().unwrap()
    );
    assert_eq!(
        report.receipts[1].batch_id,
        second["batch_id"].as_str().unwrap()
    );
    assert_eq!(
        owner.join(&JoinClock::default(), Duration::ZERO),
        Ok(report)
    );
}

#[test]
fn exporter_stop_reports_uninterruptible_idle_wait_then_joins() {
    let lanes = Arc::new(Lanes::new(4).unwrap());
    let (clock, paused, permit) = gate_clock();
    // No request is made for empty lanes; an actual loopback listener still
    // owns the endpoint throughout the test.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let mut owner = start(lanes, &base, clock.clone());
    let first_pause = paused.recv_timeout(SAFETY).unwrap();
    let stop = owner.request_stop();
    assert_eq!(stop.idle_wait_bound, Duration::from_millis(50));
    let join_clock = JoinClock::default();
    let deadline = Duration::from_millis(25);
    assert_eq!(owner.join(&join_clock, deadline), Err(JoinError::Timeout));
    assert_eq!(join_clock.monotonic(), deadline);
    assert!(!first_pause.is_zero() && first_pause <= stop.idle_wait_bound);
    permit.send(()).unwrap();
    let mut advanced = first_pause;
    while advanced < stop.idle_wait_bound {
        let pause = paused.recv_timeout(SAFETY).unwrap();
        assert!(!pause.is_zero());
        advanced += pause;
        assert!(advanced <= stop.idle_wait_bound);
        permit.send(()).unwrap();
    }
    let report = owner
        .join(&JoinClock::default(), Duration::from_secs(3600))
        .unwrap();
    assert!(report.finished);
    assert!(!report.pending);
    assert!(report.receipts.is_empty());
    assert!(report.failures.is_empty());
    assert_eq!(clock.monotonic(), Duration::from_millis(50));
}

#[test]
fn exporter_failure_is_observable_and_stop_interrupts_existing_backoff() {
    let relay = relay(1);
    let lanes = Arc::new(Lanes::new(4).unwrap());
    let (clock, paused, permit) = gate_clock();
    assert!(lanes.try_emit(record()));
    let mut owner = start(Arc::clone(&lanes), &relay.base, clock.clone());
    relay.posted.recv_timeout(SAFETY).unwrap();
    relay.reply.send(503).unwrap();
    relay.thread.join().unwrap();
    let pending_pause = paused.recv_timeout(SAFETY).unwrap();
    let running = owner.report();
    assert!(!running.finished);
    assert!(running.pending);
    assert_eq!(running.failures.len(), 1);
    assert_eq!(running.failures[0].status_code, Some(503));
    assert_eq!(running.failures[0].disposition, DeliveryDisposition::Retry);
    owner.request_stop();
    permit.send(()).unwrap();
    let report = owner
        .join(&JoinClock::default(), Duration::from_secs(3600))
        .unwrap();
    assert!(report.finished);
    assert!(report.pending);
    assert!(lanes.has_work());
    assert!(report.receipts.is_empty());
    assert_eq!(report.failures, running.failures);
    assert_eq!(clock.monotonic(), pending_pause);
}

#[test]
fn exporter_refuses_missing_provenance_and_unbounded_transport() {
    let lanes = Arc::new(Lanes::new(1).unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let client = RelayClient::new(&base, "<relay-token>", RELAY_TIMEOUT).unwrap();
    let mut invalid = provenance();
    invalid.config_digest.clear();
    let error = exporter::spawn(
        lanes.clone(),
        client,
        invalid,
        1,
        50,
        Arc::new(JoinClock::default()),
    );
    assert!(matches!(
        error,
        Err(SpawnError::Provenance(ContractError::Identity(
            "config_digest"
        )))
    ));
    for timeout in [Duration::ZERO, RELAY_TIMEOUT + Duration::from_nanos(1)] {
        let client = RelayClient::new(&base, "<relay-token>", timeout).unwrap();
        let error = exporter::spawn(
            lanes.clone(),
            client,
            provenance(),
            1,
            50,
            Arc::new(JoinClock::default()),
        );
        assert!(matches!(error, Err(SpawnError::TransportTimeout)));
    }
}

#[test]
fn exporter_honors_failure_backoff_before_sending_only_the_loss_gap() {
    let relay = relay(2);
    let lanes = Arc::new(Lanes::new(4).unwrap());
    let (clock, paused, permit) = gate_clock();
    assert!(lanes.try_emit(record()));
    let mut owner = start(Arc::clone(&lanes), &relay.base, clock.clone());
    relay.posted.recv_timeout(SAFETY).unwrap();
    relay.reply.send(503).unwrap();
    let mut advanced = Duration::ZERO;
    let backoff = Duration::from_millis(50);
    while advanced < backoff {
        let pause = paused
            .recv_timeout(SAFETY)
            .expect("required failure backoff pause");
        assert_eq!(clock.monotonic(), advanced);
        assert!(!pause.is_zero());
        advanced += pause;
        assert!(advanced <= backoff);
        assert_eq!(owner.report().failures.len(), 1);
        permit.send(()).unwrap();
    }
    let second = relay.posted.recv_timeout(SAFETY).unwrap();
    assert_eq!(clock.monotonic(), Duration::from_millis(50));
    assert_eq!(second["records"], json!([]));
    assert_eq!(second["gaps"].as_array().unwrap().len(), 1);
    // shared.events.execution_records.WireGap declares `cause`, not record `reason`.
    assert_eq!(second["gaps"][0]["cause"], "export-failed");
    owner.request_stop();
    relay.reply.send(200).unwrap();
    relay.thread.join().unwrap();
    let report = owner
        .join(&JoinClock::default(), Duration::from_secs(3600))
        .unwrap();
    assert!(report.finished);
    assert!(!report.pending);
    assert!(!lanes.has_work());
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.receipts.len(), 1);
    assert_eq!(report.receipts[0].accepted, 0);
}

#[test]
fn exporter_panicked_thread_returns_a_typed_terminal_failure() {
    struct BrokenClock;
    impl Clock for BrokenClock {
        fn monotonic(&self) -> Duration {
            panic!("injected external clock failure")
        }
        fn wall(&self) -> SystemTime {
            UNIX_EPOCH
        }
        fn pause(&self, _: Duration) {
            panic!("injected external clock failure")
        }
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let mut owner = start(
        Arc::new(Lanes::new(1).unwrap()),
        &base,
        Arc::new(BrokenClock),
    );
    assert_eq!(
        owner.join(&JoinClock::default(), Duration::from_secs(3600)),
        Err(JoinError::Panicked)
    );
    assert!(!owner.report().finished);
    assert_eq!(
        owner.join(&JoinClock::default(), Duration::ZERO),
        Err(JoinError::Panicked)
    );
}
