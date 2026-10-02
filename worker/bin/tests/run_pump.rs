//! CPU message composition against the public ingest/FallStage boundaries.
//! Oracles: Stage4 routing/exit/receipt contracts, not copied policy math.

use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_deepstream_native::{FrameIdentity, GpuMetrics, StateError, TrackedObject};
use seeon_ml_worker::exit::Exit;
use seeon_ml_worker::msg::{FALL_REQUEST_CAPACITY, FallRequest, FallResponse, PosePacket};
use seeon_ml_worker::policy::fall::{DecisionUpdate, FallStage, FallStageError};
use seeon_ml_worker::policy::ingest::IngestRefusal;
use seeon_ml_worker::run::policy::FallResponseError;
use seeon_ml_worker::run::pump::{CameraPolicy, PolicyPump, PolicySink, PumpError};
use seeon_ml_worker::seam::Clock;
use seeon_worker::detection_window::{DetectionWindow, DetectionWindowError};
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
use seeon_worker::trace::{
    DecisionTraceMissingReason, DecisionTraceReason, DecisionTraceSnapshot, DecisionTraceValueName,
};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, EvidenceError, Precision};
use seeon_worker_runtime::fall_gpu::{FallGpuError, FallScore};

#[derive(Debug)]
struct Receipt {
    frame: FrameIdentity,
    index: i64,
    time: f64,
    snapshots: Vec<DecisionTraceSnapshot>,
    generations: Vec<Option<u64>>,
    events: Vec<BusinessEvent>,
}

#[derive(Debug, PartialEq)]
enum Kind {
    Decision,
    Score,
}

#[derive(Default)]
struct Sink {
    decisions: Vec<Receipt>,
    scores: Vec<(FrameIdentity, u64, FallScore)>,
    order: Vec<Kind>,
    failures: Vec<(FrameIdentity, PumpError)>,
}

impl PolicySink for Sink {
    fn decision(&mut self, update: DecisionUpdate<'_>) {
        self.order.push(Kind::Decision);
        self.decisions.push(Receipt {
            frame: update.frame,
            index: update.frame_index,
            time: update.time_sec,
            snapshots: update.snapshots.to_vec(),
            generations: update
                .snapshots
                .iter()
                .map(|snapshot| snapshot.track_id.and_then(|id| update.generation_for(id)))
                .collect(),
            events: update.events.to_vec(),
        });
    }

    fn score(
        &mut self,
        frame: FrameIdentity,
        track_id: u64,
        score: &FallScore,
    ) -> Result<(), seeon_ml_worker::run::pump::PumpError> {
        self.order.push(Kind::Score);
        self.scores.push((frame, track_id, *score));
        Ok(())
    }
    fn frame_failure(&mut self, frame: FrameIdentity, error: &PumpError) -> Result<(), PumpError> {
        self.failures.push((frame, error.clone()));
        Ok(())
    }
}

struct FailClock;

impl Clock for FailClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }

    fn wall(&self) -> SystemTime {
        panic!("clock must not be read")
    }

    fn pause(&self, _limit: Duration) {}
}

struct WallClock {
    wall: Mutex<SystemTime>,
    reads: std::sync::atomic::AtomicUsize,
}

impl Clock for WallClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }

    fn wall(&self) -> SystemTime {
        self.reads.fetch_add(1, Ordering::SeqCst);
        *self.wall.lock().unwrap()
    }

    fn pause(&self, _limit: Duration) {}
}

fn wall_at(seconds: i64) -> SystemTime {
    if seconds >= 0 {
        UNIX_EPOCH + Duration::from_secs(seconds as u64)
    } else {
        UNIX_EPOCH - Duration::from_secs(seconds.unsigned_abs())
    }
}

impl WallClock {
    fn set(&self, seconds: i64) {
        *self.wall.lock().unwrap() = wall_at(seconds);
    }
}

fn clock_at(seconds: i64) -> Arc<WallClock> {
    Arc::new(WallClock {
        wall: Mutex::new(wall_at(seconds)),
        reads: std::sync::atomic::AtomicUsize::new(0),
    })
}

fn idle_clock() -> Arc<dyn Clock> {
    clock_at(1_700_000_000)
}

fn open(cameras: Vec<CameraPolicy>) -> PolicyPump {
    PolicyPump::new(cameras, None, idle_clock()).expect("ungated roster")
}

fn utc_window(start: &str, end: &str) -> DetectionWindow {
    DetectionWindow::from_zoneinfo_dir(start, end, "UTC", Path::new("/usr/share/zoneinfo"))
        .expect("real UTC TZif")
}

fn inactive(source_id: u32) -> CameraPolicy {
    CameraPolicy {
        source_id,
        stage: None,
    }
}

fn camera(source_id: u32, capacity: usize) -> CameraPolicy {
    let decider = FallPolicyDecider::new(
        format!("camera-{source_id}"),
        "facility-1",
        "boot-1",
        "epoch-1",
        0,
        FallPolicy::default(),
        FallCapacities {
            retained_tracks: capacity,
            generation_identities: capacity,
            episodes: capacity,
            vote_window: 5,
        },
    )
    .expect("valid test policy");
    CameraPolicy {
        source_id,
        stage: Some(FallStage::new(decider, 1.0).expect("valid calibration")),
    }
}

fn packet(source_id: u32, sequence: u64, tracks: &[u64]) -> PosePacket {
    let mut rows = Vec::new();
    let mut objects = Vec::new();
    for (slot, &track_id) in tracks.iter().enumerate() {
        let left = 10.0 + slot as f32 * 100.0;
        let mut row = [0.0; 57];
        row[..5].copy_from_slice(&[left, 20.0, left + 80.0, 200.0, 0.9]);
        for point in 0..17 {
            row[6 + point * 3] = left + 40.0;
            row[7 + point * 3] = 30.0 + point as f32 * 8.0;
            row[8 + point * 3] = 0.9;
        }
        rows.push(row);
        objects.push(TrackedObject {
            track_id,
            left,
            top: 20.0,
            width: 80.0,
            height: 180.0,
            confidence: 0.9,
        });
    }
    PosePacket {
        frame: FrameIdentity {
            source_id,
            sequence,
            pts_ns: sequence * 100_000_000,
            pts_valid: 1,
            frame_number: sequence as i64,
            source_width: 640,
            source_height: 360,
            ..FrameIdentity::default()
        },
        tensor_present: true,
        rows,
        objects,
    }
}

fn awaiting(
    pump: &mut PolicyPump,
    source_id: u32,
    tracks: &[u64],
    sender: &SyncSender<FallRequest>,
    receiver: &Receiver<FallRequest>,
) -> Vec<FallRequest> {
    let mut sink = Sink::default();
    for sequence in 0..120 {
        pump.observe(&packet(source_id, sequence, tracks), sender, &mut sink)
            .expect("valid synthetic packet");
        let requests: Vec<_> = receiver.try_iter().collect();
        if !requests.is_empty() {
            assert_eq!(requests.len(), tracks.len());
            assert!(
                requests
                    .iter()
                    .all(|request| request.frame.source_id == source_id)
            );
            return requests;
        }
    }
    panic!("fixed synthetic sequence did not produce a due window");
}

fn refused(request: &FallRequest) -> FallResponse {
    FallResponse {
        frame: request.frame,
        track_id: request.track_id,
        score: Err(FallGpuError::Window),
    }
}

fn score() -> FallScore {
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
    FallScore {
        logit: 1.25,
        evidence: AcceleratorEvidence::from_delta(
            &before,
            &after,
            2,
            EngineDigest::new(std::array::from_fn(|index| index as u8)),
            Precision::Fp32,
        )
        .expect("public CPU test evidence"),
    }
}

fn assert_missing(receipt: &Receipt, tracks: &[u64]) {
    assert_eq!(
        receipt
            .snapshots
            .iter()
            .map(|snapshot| snapshot.track_id)
            .collect::<Vec<_>>(),
        tracks.iter().copied().map(Some).collect::<Vec<_>>()
    );
    for snapshot in &receipt.snapshots {
        assert_eq!(
            snapshot
                .missing_values()
                .get(&DecisionTraceValueName::FallTransitionProbability),
            Some(&DecisionTraceMissingReason::AdapterReturnedNoData)
        );
    }
    assert!(receipt.events.is_empty());
}

#[test]
fn duplicate_sources_refuse_configuration_and_empty_roster_is_idle() {
    let error = PolicyPump::new(vec![camera(19, 1), camera(19, 1)], None, idle_clock())
        .err()
        .expect("duplicate source must refuse construction");
    assert_eq!(error, PumpError::DuplicateSource(19));
    assert_eq!(error.exit(), Exit::Config);
    assert!(matches!(
        PolicyPump::new(vec![inactive(19), camera(19, 1)], None, idle_clock()),
        Err(PumpError::DuplicateSource(19))
    ));
    let mut pump = PolicyPump::new(Vec::new(), None, idle_clock()).expect("empty roster");
    let mut sink = Sink::default();
    pump.flush(&mut sink).expect("idle flush");
    pump.flush(&mut sink).expect("repeat idle flush");
    assert!(sink.order.is_empty());
    let (sender, _receiver) = mpsc::sync_channel(1);
    assert_eq!(
        pump.observe(&packet(19, 0, &[1]), &sender, &mut sink),
        Err(PumpError::UnknownSource(19))
    );
    assert_eq!(pump.pending_scores(), 0);
    assert!(sink.order.is_empty());
}

#[test]
fn interleaved_sources_keep_same_track_ids_and_pending_decisions_separate() {
    let mut pump = open(vec![camera(19, 1), camera(3, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let first = awaiting(&mut pump, 3, &[7], &sender, &receiver);
    let second = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    assert_eq!(pump.pending_scores(), 2);
    let stop = AtomicBool::new(false);
    let mut sink = Sink::default();
    pump.consume(refused(&first[0]), &stop, &mut sink).unwrap();
    assert_eq!(pump.pending_scores(), 1);
    assert_eq!(sink.decisions.len(), 1);
    assert_eq!(sink.decisions[0].frame, first[0].frame);
    pump.consume(refused(&second[0]), &stop, &mut sink).unwrap();
    assert_eq!(pump.pending_scores(), 0);
    assert_eq!(sink.decisions.len(), 2);
    assert_eq!(sink.decisions[1].frame, second[0].frame);
    assert!(sink.scores.is_empty());
    assert!(!stop.load(Ordering::SeqCst));
}

#[test]
fn routing_and_ingest_refusals_leave_pending_policy_and_outputs_untouched() {
    let mut pump = open(vec![camera(19, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    let mut unknown = packet(3, 100, &[8]);
    unknown.frame.source_width = 0;
    let mut bad_width = packet(19, 100, &[8]);
    bad_width.frame.source_width = 0;
    let mut bad_height = packet(19, 100, &[8]);
    bad_height.frame.source_height = 0;
    let mut bad_sequence = packet(19, 100, &[8]);
    bad_sequence.frame.sequence = u64::MAX;
    let mut sink = Sink::default();
    for (input, expected) in [
        (unknown, PumpError::UnknownSource(3)),
        (
            bad_width,
            PumpError::Ingest {
                source_id: 19,
                cause: IngestRefusal::SourceSize,
            },
        ),
        (
            bad_height,
            PumpError::Ingest {
                source_id: 19,
                cause: IngestRefusal::SourceSize,
            },
        ),
        (
            bad_sequence,
            PumpError::Ingest {
                source_id: 19,
                cause: IngestRefusal::Sequence,
            },
        ),
    ] {
        assert_eq!(expected.exit(), Exit::Runtime);
        assert_eq!(pump.observe(&input, &sender, &mut sink), Err(expected));
        assert_eq!(pump.pending_scores(), 1);
        assert!(sink.order.is_empty());
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }
    pump.consume(refused(&requests[0]), &AtomicBool::new(false), &mut sink)
        .unwrap();
    assert_eq!(sink.decisions.len(), 1);
    assert_eq!(sink.decisions[0].frame, requests[0].frame);
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn all_fatal_variants_stop_before_unknown_source_routing_or_output() {
    let mut pump = open(vec![camera(19, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    for cause in [
        FallGpuError::Native(StateError::ExecutionFailed),
        FallGpuError::Output,
        FallGpuError::Evidence(EvidenceError::NoDeviceToHost),
        FallGpuError::Poisoned,
    ] {
        let mut response = refused(&requests[0]);
        response.frame.source_id = 3;
        response.score = Err(cause);
        let stop = AtomicBool::new(false);
        let mut sink = Sink::default();
        let error = pump.consume(response, &stop, &mut sink).unwrap_err();
        assert_eq!(
            error,
            PumpError::Response(FallResponseError::Accelerator(cause))
        );
        assert_eq!(error.exit(), Exit::FatalAccelerator);
        assert!(stop.load(Ordering::SeqCst));
        assert!(sink.order.is_empty());
        assert_eq!(pump.pending_scores(), 1);
    }
    let mut sink = Sink::default();
    pump.flush(&mut sink).unwrap();
    assert_eq!(sink.decisions.len(), 1);
    assert_eq!(sink.decisions[0].frame, requests[0].frame);
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn window_partial_stale_and_duplicate_responses_complete_only_once() {
    let mut pump = open(vec![camera(19, 2)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7, 8], &sender, &receiver);
    let stop = AtomicBool::new(false);
    let mut sink = Sink::default();
    let mut stale = refused(&requests[0]);
    stale.frame.sequence += 1000;
    pump.consume(stale, &stop, &mut sink).unwrap();
    let mut wrong_track = refused(&requests[0]);
    wrong_track.track_id = 99;
    pump.consume(wrong_track, &stop, &mut sink).unwrap();
    assert_eq!(pump.pending_scores(), 2);
    assert!(sink.order.is_empty());
    pump.consume(refused(&requests[0]), &stop, &mut sink)
        .unwrap();
    assert_eq!(pump.pending_scores(), 1);
    pump.consume(refused(&requests[0]), &stop, &mut sink)
        .unwrap();
    assert_eq!(pump.pending_scores(), 1);
    assert!(sink.order.is_empty());
    pump.consume(refused(&requests[1]), &stop, &mut sink)
        .unwrap();
    assert_eq!(pump.pending_scores(), 0);
    pump.consume(refused(&requests[1]), &stop, &mut sink)
        .unwrap();
    pump.flush(&mut sink).unwrap();
    assert_eq!(sink.order, vec![Kind::Decision]);
    let receipt = &sink.decisions[0];
    assert_eq!(receipt.frame, requests[0].frame);
    assert_eq!(receipt.index, requests[0].frame.sequence as i64);
    assert_eq!(receipt.time, requests[0].frame.pts_ns as f64 / 1e9);
    assert_missing(receipt, &[7, 8]);
    assert!(!stop.load(Ordering::SeqCst));
}

#[test]
fn flush_uses_declared_roster_order_and_does_not_repeat_completed_updates() {
    let mut pump = open(vec![camera(19, 1), camera(3, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let second = awaiting(&mut pump, 3, &[8], &sender, &receiver);
    let first = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    assert_eq!(pump.pending_scores(), 2);
    let mut sink = Sink::default();
    pump.flush(&mut sink).unwrap();
    assert_eq!(pump.pending_scores(), 0);
    assert_eq!(sink.decisions.len(), 2);
    assert_eq!(sink.decisions[0].frame, first[0].frame);
    assert_eq!(sink.decisions[1].frame, second[0].frame);
    assert_missing(&sink.decisions[0], &[7]);
    assert_missing(&sink.decisions[1], &[8]);
    pump.flush(&mut sink).unwrap();
    pump.consume(refused(&first[0]), &AtomicBool::new(false), &mut sink)
        .unwrap();
    assert_eq!(sink.order, vec![Kind::Decision, Kind::Decision]);
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn real_score_frame_track_and_evidence_arrive_before_completion() {
    let mut pump = open(vec![camera(19, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    let expected = score();
    let request = &requests[0];
    let stop = AtomicBool::new(false);
    let mut sink = Sink::default();
    pump.consume(
        FallResponse {
            frame: request.frame,
            track_id: request.track_id,
            score: Ok(expected),
        },
        &stop,
        &mut sink,
    )
    .unwrap();
    assert_eq!(
        sink.scores,
        vec![(request.frame, request.track_id, expected)]
    );
    assert_eq!(sink.order, vec![Kind::Score, Kind::Decision]);
    assert_eq!(sink.decisions[0].frame, request.frame);
    assert_eq!(sink.decisions[0].snapshots[0].track_id, Some(7));
    assert!(
        sink.decisions[0].snapshots[0]
            .values()
            .contains_key(&DecisionTraceValueName::FallTransitionProbability)
    );
    assert_eq!(pump.pending_scores(), 0);
    pump.consume(
        FallResponse {
            frame: request.frame,
            track_id: request.track_id,
            score: Ok(expected),
        },
        &stop,
        &mut sink,
    )
    .unwrap();
    assert_eq!(
        sink.scores,
        vec![(request.frame, request.track_id, expected)]
    );
    assert_eq!(sink.order, vec![Kind::Score, Kind::Decision]);
    assert!(!stop.load(Ordering::SeqCst));
}

#[test]
fn nonfatal_unknown_response_is_a_routing_error_without_score_receipt() {
    let mut pump = PolicyPump::new(Vec::new(), None, idle_clock()).unwrap();
    let stop = AtomicBool::new(false);
    let mut sink = Sink::default();
    for result in [Ok(score()), Err(FallGpuError::Window)] {
        let error = pump
            .consume(
                FallResponse {
                    frame: packet(3, 0, &[7]).frame,
                    track_id: 7,
                    score: result,
                },
                &stop,
                &mut sink,
            )
            .unwrap_err();
        assert_eq!(error, PumpError::UnknownSource(3));
        assert_eq!(error.exit(), Exit::Runtime);
        assert!(!stop.load(Ordering::SeqCst));
        assert!(sink.order.is_empty());
    }
}

#[test]
fn observe_keeps_earlier_receipt_when_later_policy_update_fails() {
    let mut pump = open(vec![camera(19, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    let input = packet(19, requests[0].frame.sequence + 1, &[8, 9]);
    let mut sink = Sink::default();
    let error = pump.observe(&input, &sender, &mut sink).unwrap_err();
    assert!(matches!(
        error,
        PumpError::Response(FallResponseError::Policy(FallStageError::Policy(_)))
    ));
    assert_eq!(error.exit(), Exit::Runtime);
    assert_eq!(sink.order, vec![Kind::Decision]);
    assert_eq!(sink.decisions[0].frame, requests[0].frame);
    assert_missing(&sink.decisions[0], &[7]);
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn gap_refusal_precedes_pending_flush_and_preserves_typed_cause() {
    let mut pump = open(vec![camera(19, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    let mut input = packet(19, requests[0].frame.sequence + 1, &[7]);
    input.frame.pts_ns += 120_000_000_000;
    let mut sink = Sink::default();
    let error = pump.observe(&input, &sender, &mut sink).unwrap_err();
    assert!(matches!(
        error,
        PumpError::Response(FallResponseError::Policy(FallStageError::Gap(_)))
    ));
    assert_eq!(error.exit(), Exit::Runtime);
    assert!(sink.order.is_empty());
    assert_eq!(pump.pending_scores(), 1);
    pump.consume(refused(&requests[0]), &AtomicBool::new(false), &mut sink)
        .unwrap();
    assert_eq!(sink.decisions.len(), 1);
    assert_eq!(sink.decisions[0].frame, requests[0].frame);
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn observe_forwards_each_original_completion_but_never_invents_a_coast_receipt() {
    for coast in [false, true] {
        let mut pump = open(vec![camera(19, 1)]);
        let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
        let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
        let pending = requests[0].frame;
        let mut input = packet(19, pending.sequence + 1, &[8]);
        if coast {
            input.frame.pts_ns = pending.pts_ns;
        }
        let mut sink = Sink::default();
        pump.observe(&input, &sender, &mut sink).unwrap();
        assert_eq!(sink.decisions[0].frame, pending);
        assert_missing(&sink.decisions[0], &[7]);
        if coast {
            assert_eq!(sink.order, vec![Kind::Decision]);
        } else {
            assert_eq!(sink.order, vec![Kind::Decision, Kind::Decision]);
            assert_eq!(sink.decisions[1].frame, input.frame);
            assert_eq!(sink.decisions[1].index, input.frame.sequence as i64);
            assert_eq!(sink.decisions[1].time, input.frame.pts_ns as f64 / 1e9);
            assert_eq!(sink.decisions[1].snapshots[0].track_id, Some(8));
            assert!(sink.decisions[1].events.is_empty());
        }
        let completed = sink.decisions.len();
        pump.flush(&mut sink).unwrap();
        assert_eq!(sink.decisions.len(), completed);
        assert_eq!(pump.pending_scores(), 0);
    }
}

#[test]
fn stale_duplicate_and_unknown_frame_scores_are_not_admitted() {
    let mut pump = open(vec![camera(19, 2)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7, 8], &sender, &receiver);
    let stop = AtomicBool::new(false);
    let mut sink = Sink::default();
    let expected = score();
    let mut stale = FallResponse {
        frame: requests[0].frame,
        track_id: requests[0].track_id,
        score: Ok(expected),
    };
    stale.frame.sequence += 1000;
    let mut unknown = FallResponse {
        frame: requests[0].frame,
        track_id: 99,
        score: Ok(expected),
    };
    unknown.frame.frame_number += 1;
    for response in [stale, unknown] {
        pump.consume(response, &stop, &mut sink).unwrap();
        assert!(sink.scores.is_empty());
        assert!(sink.order.is_empty());
        assert_eq!(pump.pending_scores(), 2);
    }
    pump.consume(
        FallResponse {
            frame: requests[0].frame,
            track_id: requests[0].track_id,
            score: Ok(expected),
        },
        &stop,
        &mut sink,
    )
    .unwrap();
    assert_eq!(pump.pending_scores(), 1);
    assert!(sink.decisions.is_empty());
    pump.consume(
        FallResponse {
            frame: requests[0].frame,
            track_id: requests[0].track_id,
            score: Ok(expected),
        },
        &stop,
        &mut sink,
    )
    .unwrap();
    assert_eq!(
        sink.scores,
        vec![(requests[0].frame, requests[0].track_id, expected)]
    );
    assert_eq!(sink.order, vec![Kind::Score]);
    assert_eq!(pump.pending_scores(), 1);
    assert!(!stop.load(Ordering::SeqCst));
}

#[test]
fn accepted_track_scores_notify_before_the_final_decision() {
    let mut pump = open(vec![camera(19, 2)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7, 8], &sender, &receiver);
    let stop = AtomicBool::new(false);
    let mut sink = Sink::default();
    let expected = score();
    for request in &requests {
        pump.consume(
            FallResponse {
                frame: request.frame,
                track_id: request.track_id,
                score: Ok(expected),
            },
            &stop,
            &mut sink,
        )
        .unwrap();
    }
    assert_eq!(
        sink.scores,
        requests
            .iter()
            .map(|request| (request.frame, request.track_id, expected))
            .collect::<Vec<_>>()
    );
    assert_eq!(sink.order, vec![Kind::Score, Kind::Score, Kind::Decision]);
    assert_eq!(sink.decisions.len(), 1);
    assert_eq!(sink.decisions[0].frame, requests[0].frame);
    assert_eq!(
        sink.decisions[0]
            .snapshots
            .iter()
            .map(|snapshot| snapshot.track_id)
            .collect::<Vec<_>>(),
        requests
            .iter()
            .map(|request| Some(request.track_id))
            .collect::<Vec<_>>()
    );
    let receipt = &sink.decisions[0];
    assert_eq!(receipt.generations.len(), receipt.snapshots.len());
    for (snapshot, generation) in receipt.snapshots.iter().zip(&receipt.generations) {
        assert_eq!(generation.is_some(), snapshot.track_id.is_some());
        assert!(
            requests
                .iter()
                .any(|request| Some(request.track_id) == snapshot.track_id)
        );
    }
    assert_eq!(pump.pending_scores(), 0);
    assert!(!stop.load(Ordering::SeqCst));
}
#[test]
fn inactive_source_validates_pose_without_requests_and_refuses_a_response() {
    let mut pump = open(vec![inactive(19), camera(3, 1)]);
    assert_eq!(pump.source_count(), 2);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let mut sink = Sink::default();
    pump.observe(&packet(19, 4, &[7]), &sender, &mut sink)
        .expect("inactive pose is admitted");
    assert!(receiver.try_iter().next().is_none());
    assert!(sink.order.is_empty());
    assert_eq!(pump.pending_scores(), 0);

    let mut malformed = packet(19, 5, &[7]);
    malformed.frame.source_width = 0;
    assert_eq!(
        pump.observe(&malformed, &sender, &mut sink),
        Err(PumpError::Ingest {
            source_id: 19,
            cause: IngestRefusal::SourceSize,
        })
    );
    assert_eq!(
        pump.observe(&packet(8, 5, &[7]), &sender, &mut sink),
        Err(PumpError::UnknownSource(8))
    );
    assert!(sink.order.is_empty());

    let stop = AtomicBool::new(false);
    let valid = pump.consume(
        FallResponse {
            frame: packet(19, 4, &[7]).frame,
            track_id: 7,
            score: Ok(score()),
        },
        &stop,
        &mut sink,
    );
    assert_eq!(valid, Err(PumpError::InactiveSource(19)));
    assert_eq!(PumpError::InactiveSource(19).exit(), Exit::Runtime);
    let window = pump.consume(
        FallResponse {
            frame: packet(19, 4, &[7]).frame,
            track_id: 7,
            score: Err(FallGpuError::Window),
        },
        &stop,
        &mut sink,
    );
    assert_eq!(window, Err(PumpError::InactiveSource(19)));
    assert!(!stop.load(Ordering::SeqCst));
    assert!(sink.order.is_empty());
    assert_eq!(pump.pending_scores(), 0);
    pump.flush(&mut sink).expect("inactive flush");
    assert!(sink.order.is_empty());

    let fatal = pump
        .consume(
            FallResponse {
                frame: packet(19, 4, &[7]).frame,
                track_id: 7,
                score: Err(FallGpuError::Poisoned),
            },
            &stop,
            &mut sink,
        )
        .unwrap_err();
    assert!(matches!(
        fatal,
        PumpError::Response(FallResponseError::Accelerator(FallGpuError::Poisoned))
    ));
    assert_eq!(fatal.exit(), Exit::FatalAccelerator);
    assert!(stop.load(Ordering::SeqCst));
    assert!(sink.order.is_empty());
    assert_eq!(pump.source_count(), 2);
}

#[test]
fn disabled_domain_does_not_disturb_an_active_sources_pending_request() {
    let mut pump = open(vec![inactive(19), camera(3, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 3, &[7], &sender, &receiver);
    let request = &requests[0];
    assert_eq!(pump.pending_scores(), 1);
    let mut sink = Sink::default();
    pump.observe(&packet(19, 4, &[7]), &sender, &mut sink)
        .expect("disabled fall does not remove the media source");
    assert_eq!(pump.pending_scores(), 1);
    assert!(receiver.try_iter().next().is_none());
    assert!(sink.order.is_empty());
    let stop = AtomicBool::new(false);
    pump.consume(
        FallResponse {
            frame: request.frame,
            track_id: request.track_id,
            score: Ok(score()),
        },
        &stop,
        &mut sink,
    )
    .expect("active source retains its request");
    assert_eq!(sink.order, [Kind::Score, Kind::Decision]);
    assert_eq!(sink.scores.len(), 1);
    assert_eq!(sink.scores[0].0, request.frame);
    assert_eq!(sink.scores[0].1, request.track_id);
    assert_eq!(sink.decisions.len(), 1);
    assert_eq!(sink.decisions[0].frame, request.frame);
    assert_eq!(pump.pending_scores(), 0);
    assert!(!stop.load(Ordering::SeqCst));
}

#[test]
fn utc_half_open_window_skips_the_current_frame_and_flushes_the_original_pending() {
    // 2024-01-15 12:00:00Z is inside 12:00..13:00; 13:00:00Z is the excluded end.
    let inside = 1_705_320_000;
    let outside = inside + 3_600;
    let clock = clock_at(inside);
    let mut outside_pump = PolicyPump::new(
        vec![camera(19, 1)],
        Some(utc_window("12:00", "13:00")),
        clock.clone(),
    )
    .expect("outside roster");
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let pending_requests = awaiting(&mut outside_pump, 19, &[7], &sender, &receiver);
    let original = pending_requests[0].frame;
    let before_reads = clock.reads.load(Ordering::SeqCst);
    clock.set(outside);
    let mut sink = Sink::default();
    let current = packet(19, original.sequence + 1, &[8]);
    outside_pump
        .observe(&current, &sender, &mut sink)
        .expect("outside frame");
    assert!(receiver.try_iter().next().is_none());
    assert_eq!(sink.decisions.len(), 2);
    assert_eq!(sink.decisions[0].frame, original);
    assert_missing(&sink.decisions[0], &[7]);
    assert_eq!(sink.decisions[1].frame, current.frame);
    assert_eq!(
        sink.decisions[1].snapshots[0].reason,
        DecisionTraceReason::OutsideDetectionWindow
    );
    assert!(sink.decisions[1].events.is_empty());
    assert_eq!(outside_pump.pending_scores(), 0);
    assert_eq!(clock.reads.load(Ordering::SeqCst), before_reads + 1);
}

#[test]
fn reentry_preserves_request_cadence_against_an_ungated_control() {
    let start = 1_705_320_000;
    let clock = clock_at(start);
    let window = utc_window("12:00", "13:00");
    let mut gated =
        PolicyPump::new(vec![camera(19, 1)], Some(window), clock.clone()).expect("reentry roster");
    let mut control = open(vec![camera(19, 1)]);
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let mut gated_sink = Sink::default();
    let mut control_sink = Sink::default();
    let mut gated_requests = Vec::new();
    let mut control_requests = Vec::new();
    for sequence in 0..120 {
        let outside = (10..15).contains(&sequence);
        clock.set(if outside { start + 3600 } else { start });
        gated
            .observe(&packet(19, sequence, &[7]), &sender, &mut gated_sink)
            .expect("gated cadence");
        let emitted: Vec<_> = receiver.try_iter().collect();
        if outside {
            assert!(emitted.is_empty());
            continue; // Control sees precisely the admitted frame sequence.
        }
        gated_requests.extend(emitted.into_iter().map(|request| request.frame.sequence));
        control
            .observe(&packet(19, sequence, &[7]), &sender, &mut control_sink)
            .expect("ungated cadence");
        control_requests.extend(receiver.try_iter().map(|request| request.frame.sequence));
    }
    assert!(!gated_requests.is_empty());
    assert_eq!(gated_requests, control_requests);
    assert!(receiver.try_iter().next().is_none());
}

#[test]
fn unknown_invalid_and_disabled_sources_do_not_read_a_failing_clock() {
    let mut pump = PolicyPump::new(
        vec![inactive(19), camera(3, 1)],
        Some(utc_window("12:00", "13:00")),
        Arc::new(FailClock),
    )
    .expect("mixed roster");
    let (sender, receiver) = mpsc::sync_channel(1);
    let mut sink = Sink::default();
    assert_eq!(
        pump.observe(&packet(8, 1, &[7]), &sender, &mut sink),
        Err(PumpError::UnknownSource(8))
    );
    let mut malformed = packet(19, 1, &[7]);
    malformed.frame.source_width = 0;
    assert_eq!(
        pump.observe(&malformed, &sender, &mut sink),
        Err(PumpError::Ingest {
            source_id: 19,
            cause: IngestRefusal::SourceSize,
        })
    );
    pump.observe(&packet(19, 1, &[7]), &sender, &mut sink)
        .expect("disabled source skips the clock");
    assert!(receiver.try_iter().next().is_none());
    assert!(sink.order.is_empty());
    assert_eq!(pump.pending_scores(), 0);
}

#[test]
fn clock_out_of_range_preserves_pending_and_outside_contains_no_request() {
    let inside = 1_705_316_400;
    let clock = clock_at(inside);
    let mut pump = PolicyPump::new(
        vec![camera(19, 1)],
        Some(utc_window("11:00", "12:00")),
        clock.clone(),
    )
    .expect("pending roster");
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let requests = awaiting(&mut pump, 19, &[7], &sender, &receiver);
    assert_eq!(pump.pending_scores(), 1);
    let pending = requests[0].frame;

    // One second before Python's year 1 lower bound; pre-epoch alone is valid.
    clock.set(-62_135_596_801);
    let mut sink = Sink::default();
    let error = pump
        .observe(&packet(19, pending.sequence + 1, &[7]), &sender, &mut sink)
        .unwrap_err();
    assert_eq!(
        error,
        PumpError::Window(DetectionWindowError::ClockOutOfRange)
    );
    assert_eq!(error.exit(), Exit::Runtime);
    assert!(receiver.try_iter().next().is_none());
    assert!(sink.order.is_empty());
    assert_eq!(pump.pending_scores(), 1);

    let mut outside = PolicyPump::new(
        vec![camera(19, 1)],
        Some(utc_window("11:00", "12:00")),
        clock_at(inside + 3_600),
    )
    .expect("outside roster");
    outside
        .observe(&packet(19, 0, &[7]), &sender, &mut sink)
        .expect("outside contains");
    assert!(receiver.try_iter().next().is_none());
    assert_eq!(sink.order, [Kind::Decision]);
    assert_eq!(
        sink.decisions[0].snapshots[0].reason,
        DecisionTraceReason::OutsideDetectionWindow
    );
    assert_eq!(outside.pending_scores(), 0);

    pump.consume(
        FallResponse {
            frame: pending,
            track_id: requests[0].track_id,
            score: Err(FallGpuError::Window),
        },
        &AtomicBool::new(false),
        &mut sink,
    )
    .expect("accepted response finishes after the wall clock has moved on");
    assert_eq!(pump.pending_scores(), 0);
}
