//! Policy-thread parity against output recorded from the Python oracles at
//! 145637c (`tests/fixtures/worker-wire/d/policy-parity.json`, sha256 prefix
//! pinned below). The golden's `row_oracles` names the module behind each
//! section; Rust output is never a golden.
//!
//! - T29a: the association pass of `worker/adapters/deepstream/metadata.py`:
//!   IoU ties and their tie-break order, the IoU gate, the score floor and a
//!   degenerate box whose overlap is NaN (Rust maps NaN to 0.0, a disclosed
//!   deviation whose expected row comes from a Python control run).
//! - T29b: `FallDomainDecider.update` behind the `policy_pump.py` glue over a
//!   116-frame sequence with a same-bucket frame, a pts jump that yields gap
//!   rows, and a pts rollback. The windows each request carries, the replay
//!   lines, the events and the counters must match per frame.
//! - T29c: the float32 transition probability of `ort_pose_bbox56.py`, within
//!   one float32 ulp: `f32::exp` and NumPy's float32 `exp` may differ by one
//!   ulp before the division.
//! - T29d: the bed recognizer outcomes, mapped through a reviewed table to the
//!   outcomes of `submit` and `receive`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, SystemTime};

use seeon_deepstream_native::{FrameIdentity, GpuMetrics, TrackedObject};
use seeon_ml_worker::msg::{
    BedOutput, BedRequest, FALL_REQUEST_CAPACITY, FallRequest, FallResponse, GPU_REQUEST_CAPACITY,
    PosePacket,
};
use seeon_ml_worker::policy::bed::{BedRefusal, BedReply, receive, submit};
use seeon_ml_worker::policy::fall::{FallStage, score::transition_probability};
use seeon_ml_worker::policy::ingest::ingest;
use seeon_ml_worker::seam::Clock;
use seeon_worker::bed_input::Letterbox;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
use seeon_worker::trace::{DecisionTraceSnapshot, NumericTraceValue};
use seeon_worker_runtime::bed_gpu::BedGpuError;
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};
use seeon_worker_runtime::fall_gpu::FallScore;
use serde_json::Value;
use sha2::{Digest, Sha256};

const GOLDEN_SHA256_PREFIX: &str = "28cea98e4e78";
/// TEST-POLICY's approved absolute tolerance for float trace values and event
/// probabilities.
const MAX_ABS: f64 = 1e-4;
/// Stated tolerance for the transition probability (review row 12).
const TRANSITION_MAX_F32_ULP: i64 = 1;

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/d/policy-parity.json");
    let bytes = std::fs::read(&path).expect("golden");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert!(
        digest.starts_with(GOLDEN_SHA256_PREFIX),
        "golden sha256 {digest}"
    );
    serde_json::from_slice(&bytes).expect("golden json")
}

fn text(value: &Value) -> &str {
    value.as_str().expect("golden string")
}

fn f32_bits(value: &Value) -> u32 {
    u32::from_str_radix(text(value), 16).expect("float32 hex")
}

fn f64_bits(value: &Value) -> u64 {
    u64::from_str_radix(text(value), 16).expect("float64 hex")
}

fn unsigned(value: &Value) -> u64 {
    value.as_u64().expect("golden unsigned")
}

fn items(value: &Value) -> &Vec<Value> {
    value.as_array().expect("golden array")
}

/// The pose packet a golden `input` describes, on the recipe's 1280x720
/// source with valid pts.
fn packet(input: &Value) -> PosePacket {
    let sequence = unsigned(&input["sequence"]);
    let frame = FrameIdentity {
        sequence,
        pts_ns: unsigned(&input["pts_ns"]),
        pts_valid: 1,
        frame_number: i64::try_from(sequence).expect("sequence"),
        source_width: 1280,
        source_height: 720,
        ..FrameIdentity::default()
    };
    let rows = items(&input["rows"])
        .iter()
        .map(|row| {
            let values = items(row);
            std::array::from_fn(|index| f32::from_bits(f32_bits(&values[index])))
        })
        .collect();
    let objects = items(&input["objects"])
        .iter()
        .map(|object| TrackedObject {
            track_id: unsigned(&object["track_id"]),
            left: f32::from_bits(f32_bits(&object["left"])),
            top: f32::from_bits(f32_bits(&object["top"])),
            width: f32::from_bits(f32_bits(&object["width"])),
            height: f32::from_bits(f32_bits(&object["height"])),
            confidence: f32::from_bits(f32_bits(&object["confidence"])),
        })
        .collect();
    PosePacket {
        frame,
        tensor_present: true,
        rows,
        objects,
    }
}

#[test]
fn association_matches_the_python_rows() {
    let golden = golden();
    let cases = items(&golden["association"]);
    assert!(!cases.is_empty(), "no association cases");
    for case in cases {
        let name = text(&case["name"]);
        let expected = &case["expected"];
        let frame = ingest(&packet(&case["input"]))
            .unwrap_or_else(|refusal| panic!("{name}: ingest refused {refusal:?}"));
        let live: Vec<u64> = items(&expected["live_track_ids"])
            .iter()
            .map(unsigned)
            .collect();
        assert_eq!(frame.live_track_ids, live, "{name}: live tracks");
        let rows: BTreeMap<u64, Vec<u32>> = frame
            .rows
            .iter()
            .map(|(track, row)| (*track, row.iter().map(|value| value.to_bits()).collect()))
            .collect();
        let expected_rows: BTreeMap<u64, Vec<u32>> = expected["rows"]
            .as_object()
            .expect("golden rows")
            .iter()
            .map(|(track, row)| {
                let track = track.parse().expect("track id");
                (track, items(row).iter().map(f32_bits).collect())
            })
            .collect();
        assert_eq!(rows, expected_rows, "{name}: rows (float32 bits)");
        assert_eq!(
            frame.time_sec.map(f64::to_bits),
            Some(f64_bits(&expected["time_sec"])),
            "{name}: time_sec"
        );
        assert_eq!(
            frame.frame_index,
            expected["frame_index"].as_i64().expect("frame index"),
            "{name}: frame index"
        );
    }
}

#[test]
fn transition_probability_is_within_one_f32_ulp() {
    let golden = golden();
    let transition = &golden["transition"];
    let temperature = f64::from_bits(f64_bits(&transition["temperature"])) as f32;
    let cases = items(&transition["cases"]);
    assert!(!cases.is_empty(), "no transition cases");
    for case in cases {
        let logit = f32::from_bits(f32_bits(&case["logit"]));
        let expected = f64::from_bits(f64_bits(&case["transition"])) as f32;
        let actual = transition_probability(logit, temperature).expect("finite logit") as f32;
        let ulps = (i64::from(actual.to_bits()) - i64::from(expected.to_bits())).abs();
        assert!(
            ulps <= TRANSITION_MAX_F32_ULP,
            "logit {logit}: {actual:e} vs Python {expected:e} is {ulps} float32 ulp"
        );
    }
}

fn evidence() -> AcceleratorEvidence {
    let before = GpuMetrics {
        attempted: 6,
        succeeded: 5,
        failed: 1,
        host_to_device_bytes: 1000,
        device_to_host_bytes: 300,
        elapsed_ns: 4000,
        device: 2,
    };
    let after = GpuMetrics {
        attempted: 7,
        succeeded: 6,
        failed: 1,
        host_to_device_bytes: 1123,
        device_to_host_bytes: 345,
        elapsed_ns: 4789,
        device: 2,
    };
    AcceleratorEvidence::from_delta(
        &before,
        &after,
        2,
        EngineDigest::new(std::array::from_fn(|index| index as u8)),
        Precision::Fp32,
    )
    .expect("evidence")
}

/// The recorder's `window_sha256`: sha256 of the 30x56 float32 values, row
/// major and little endian, that the runner received.
fn window_sha256(request: &FallRequest) -> String {
    let mut hasher = Sha256::new();
    for row in request.window.iter() {
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The canonical replay lines of `tests_support/native_yolo_parity.py`, as
/// `worker/runtime/rust/tests/gpu_parity.rs` renders them.
fn render(
    lines: &mut Vec<String>,
    frame: i64,
    evaluated: bool,
    events: &[BusinessEvent],
    snapshots: &[DecisionTraceSnapshot],
) {
    let optional = |value: Option<u64>| value.map_or_else(|| "-".to_owned(), |v| v.to_string());
    lines.push(format!("F\t{frame}\t{}", u8::from(evaluated)));
    for event in events {
        lines.push(format!(
            "E\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{}\t{}\t{}",
            event.domain,
            event.event_type,
            event.identity,
            event.camera_id,
            event.facility_id,
            event.time_sec.to_bits(),
            event
                .probability
                .map_or_else(|| "-".to_owned(), |p| format!("{:016x}", p.to_bits())),
            optional(event.person_id),
            optional(event.bed_id),
        ));
    }
    for snapshot in snapshots {
        let mut values: Vec<_> = snapshot.values().iter().collect();
        values.sort_by_key(|(name, _)| name.as_str());
        let mut missing: Vec<_> = snapshot.missing_values().iter().collect();
        missing.sort_by_key(|(name, _)| name.as_str());
        lines.push(format!(
            "T\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            snapshot.reason,
            snapshot.previous_state,
            snapshot.current_state,
            u8::from(snapshot.triggered),
            optional(snapshot.track_id),
            optional(snapshot.bed_id),
            values.len(),
            missing.len(),
        ));
        for (name, value) in values {
            lines.push(match value {
                NumericTraceValue::Integer(value) => format!("V\t{name}\tI\t{value}"),
                NumericTraceValue::Float(value) => {
                    format!("V\t{name}\tF\t{:016x}", value.get().to_bits())
                }
            });
        }
        for (name, reason) in missing {
            lines.push(format!("M\t{name}\t{reason}"));
        }
    }
}

/// Exact, except float trace values and event probabilities within `MAX_ABS`.
fn lines_match(actual: &str, expected: &str) -> bool {
    if actual == expected {
        return true;
    }
    let actual: Vec<&str> = actual.split('\t').collect();
    let expected: Vec<&str> = expected.split('\t').collect();
    let float_field = match (expected[0], expected.len()) {
        ("V", 4) if expected[2] == "F" => 3,
        ("E", 10) => 7,
        _ => return false,
    };
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(&expected)
            .enumerate()
            .all(|(index, (a, e))| {
                if index != float_field || a == e {
                    return a == e;
                }
                match (u64::from_str_radix(a, 16), u64::from_str_radix(e, 16)) {
                    (Ok(a), Ok(e)) => (f64::from_bits(a) - f64::from_bits(e)).abs() <= MAX_ABS,
                    _ => false,
                }
            })
}

#[test]
fn fall_sequence_replays_the_python_lines() {
    let golden = golden();
    let sequence = &golden["sequence"];
    let decider = FallPolicyDecider::new(
        "cmsnw6rjc01vhlh01oswn99yq",
        "facility-1",
        "boot",
        "epoch",
        0,
        FallPolicy::default(),
        FallCapacities {
            retained_tracks: 64,
            generation_identities: 64,
            episodes: 64,
            vote_window: 5,
        },
    )
    .expect("decider");
    let temperature = f64::from_bits(f64_bits(&sequence["temperature"]));
    let mut stage = FallStage::new(decider, temperature).expect("stage");
    let (requests, runner) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let frames = items(&sequence["frames"]);
    assert!(!frames.is_empty(), "no sequence frames");
    let mut events_total = 0;
    for frame in frames {
        let ordinal = unsigned(&frame["ordinal"]);
        let ingested = ingest(&packet(&frame["input"]))
            .unwrap_or_else(|refusal| panic!("ordinal {ordinal}: ingest refused {refusal:?}"));
        let mut events = stage
            .observe(&ingested, &requests)
            .unwrap_or_else(|error| panic!("ordinal {ordinal}: observe {error:?}"));
        let sent: Vec<FallRequest> = runner.try_iter().collect();
        let predictions = items(&frame["valid_predictions"]);
        let mut windows: Vec<(u64, String)> = sent
            .iter()
            .map(|request| (request.track_id, window_sha256(request)))
            .collect();
        windows.sort();
        let mut expected_windows: Vec<(u64, String)> = predictions
            .iter()
            .map(|p| (unsigned(&p["track"]), text(&p["window_sha256"]).to_owned()))
            .collect();
        expected_windows.sort();
        assert_eq!(windows, expected_windows, "ordinal {ordinal}: fall windows");
        for request in sent {
            let logit = predictions
                .iter()
                .find(|p| unsigned(&p["track"]) == request.track_id)
                .map(|p| f32::from_bits(f32_bits(&p["logit"])))
                .expect("logit for the requested track");
            let response = FallResponse {
                frame: request.frame,
                track_id: request.track_id,
                score: Ok(FallScore {
                    logit,
                    evidence: evidence(),
                }),
            };
            events.extend(
                stage
                    .consume(response)
                    .unwrap_or_else(|error| panic!("ordinal {ordinal}: consume {error:?}")),
            );
        }
        let mut lines = Vec::new();
        render(
            &mut lines,
            ingested.frame_index,
            stage.decider().last_update_evaluated(),
            &events,
            stage.decider().last_trace_snapshots(),
        );
        let expected_lines: Vec<&str> = items(&frame["replay_lines"]).iter().map(text).collect();
        assert_eq!(
            lines.len(),
            expected_lines.len(),
            "ordinal {ordinal}: {lines:#?} vs Python {expected_lines:#?}"
        );
        for (actual, expected) in lines.iter().zip(&expected_lines) {
            assert!(
                lines_match(actual, expected),
                "ordinal {ordinal}: {actual:?} vs Python {expected:?}"
            );
        }
        assert_eq!(
            events.len() as u64,
            unsigned(&frame["events"]),
            "ordinal {ordinal}: events"
        );
        assert_eq!(
            stage.counters().resample_gap_rows,
            unsigned(&frame["resample_gap_rows_total"]),
            "ordinal {ordinal}: resample gap rows"
        );
        assert_eq!(
            stage.decider().track_id_switch_absorbed_total(),
            unsigned(&frame["track_id_switch_absorbed_total"]),
            "ordinal {ordinal}: track id switches absorbed"
        );
        events_total += events.len() as u64;
    }
    assert_eq!(
        events_total,
        unsigned(&sequence["events_total"]),
        "events total"
    );
}

/// A monotonic clock that only moves when the code under test pauses.
struct FakeClock {
    now: Mutex<Duration>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            now: Mutex::new(Duration::from_secs(100)),
        }
    }
}

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        *self.now.lock().expect("clock")
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + self.monotonic()
    }

    fn pause(&self, limit: Duration) {
        *self.now.lock().expect("clock") += limit;
    }
}

fn bed_output() -> BedOutput {
    BedOutput {
        detections: vec![0.25, 0.5],
        protos: vec![0.75],
        letterbox: Letterbox {
            source_height: 720,
            source_width: 1280,
            scale: 0.5,
            resized_height: 360,
            resized_width: 640,
            pad_top: 140,
            pad_left: 0,
        },
        evidence: evidence(),
    }
}

fn classify(outcome: Result<BedReply, BedRefusal>) -> String {
    match outcome {
        Ok(Ok(_)) => "Ok(Ok(BedOutput))".to_owned(),
        Ok(Err(BedGpuError::Output)) => "Ok(Err(BedGpuError::Output))".to_owned(),
        Ok(Err(other)) => format!("Ok(Err({other:?}))"),
        Err(refusal) => format!("Err(BedRefusal::{refusal:?})"),
    }
}

/// Submits one frame, lets the runner answer with `answer`, then receives.
fn answered(answer: Option<BedReply>) -> Result<BedReply, BedRefusal> {
    let clock = FakeClock::new();
    let (requests, runner) = mpsc::sync_channel::<BedRequest>(GPU_REQUEST_CAPACITY);
    let replies = submit(&requests, vec![0; 12], 2, 2)?;
    let request = runner.try_recv().expect("the request reached the runner");
    let deadline = clock.monotonic() + Duration::from_secs(1);
    match answer {
        Some(reply) => {
            request.reply.send(reply).expect("reply");
            receive(&clock, deadline, &replies)
        }
        None => {
            // The runner holds the request without answering.
            let outcome = receive(&clock, deadline, &replies);
            drop(request);
            outcome
        }
    }
}

fn unavailable() -> Vec<Result<BedReply, BedRefusal>> {
    let clock = FakeClock::new();
    let (requests, runner) = mpsc::sync_channel::<BedRequest>(GPU_REQUEST_CAPACITY);
    // The runner took the request and went away without answering.
    let replies = submit(&requests, vec![0; 12], 2, 2).expect("submit");
    drop(runner.try_recv().expect("the request reached the runner"));
    let dropped = receive(&clock, clock.monotonic() + Duration::from_secs(1), &replies);
    // No runner at all.
    drop(runner);
    let absent = submit(&requests, vec![0; 12], 2, 2).map(|_| Ok(bed_output()));
    vec![dropped, absent]
}

fn full() -> Result<BedReply, BedRefusal> {
    let (requests, _runner) = mpsc::sync_channel::<BedRequest>(GPU_REQUEST_CAPACITY);
    let _held: Vec<Receiver<BedReply>> = (0..GPU_REQUEST_CAPACITY)
        .map(|_| submit(&requests, vec![0; 12], 2, 2).expect("a free slot"))
        .collect();
    submit(&requests, vec![0; 12], 2, 2).map(|_| Ok(bed_output()))
}

#[test]
fn bed_outcomes_map_to_the_reviewed_classes() {
    let golden = golden();
    let cases = items(&golden["bed"]);
    assert!(!cases.is_empty(), "no bed cases");
    for case in cases {
        let name = text(&case["case"]);
        let outcomes = match name {
            "ok" | "not_found" => vec![answered(Some(Ok(bed_output())))],
            "runner_raises" => vec![answered(Some(Err(BedGpuError::Output)))],
            "timeout" => vec![answered(None)],
            "unavailable" => unavailable(),
            "full" => vec![full()],
            other => panic!("unreviewed bed case {other}"),
        };
        for outcome in outcomes {
            if let Ok(Ok(output)) = &outcome {
                assert_eq!(
                    output.detections,
                    bed_output().detections,
                    "{name}: the runner's tensors pass through"
                );
            }
            assert_eq!(
                classify(outcome),
                text(&case["rust_outcome"]),
                "{name}: Python {}",
                text(&case["python_outcome"])
            );
        }
    }
}
