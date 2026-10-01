//! Public completion receipts must not lose, duplicate or rebind policy updates.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, SyncSender};

use seeon_deepstream_native::FrameIdentity;
use seeon_ml_worker::msg::{FALL_REQUEST_CAPACITY, FallRequest, FallResponse};
use seeon_ml_worker::policy::fall::{DecisionUpdate, FallStage, FallStageError};
use seeon_ml_worker::policy::ingest::Frame;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider, FallProbabilities};
use seeon_worker::trace::DecisionTraceSnapshot;
use seeon_worker_runtime::fall_gpu::FallGpuError;

#[derive(Debug)]
struct Receipt {
    frame: FrameIdentity,
    index: i64,
    time: f64,
    snapshots: Vec<DecisionTraceSnapshot>,
    generations: Vec<Option<u64>>,
    events: Vec<BusinessEvent>,
}

fn collect(receipts: &mut Vec<Receipt>) -> impl for<'a> FnMut(DecisionUpdate<'a>) + '_ {
    |update| {
        receipts.push(Receipt {
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
}

fn stage(capacity: usize) -> FallStage {
    FallStage::new(decider(capacity), 1.0).expect("valid test calibration")
}

fn decider(capacity: usize) -> FallPolicyDecider {
    FallPolicyDecider::new(
        "camera-1",
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
    .expect("valid test policy")
}

fn frame(sequence: u64, time: f64, tracks: &[u64]) -> Frame {
    Frame {
        identity: FrameIdentity {
            sequence,
            pts_ns: (time * 1e9) as u64,
            pts_valid: 1,
            source_width: 640,
            source_height: 360,
            ..FrameIdentity::default()
        },
        width: 640,
        height: 360,
        live_track_ids: tracks.to_vec(),
        rows: tracks
            .iter()
            .map(|&id| (id, [0.5; 56]))
            .collect::<BTreeMap<_, _>>(),
        time_sec: Some(time),
        frame_index: sequence as i64,
    }
}

type Awaiting = (
    FallStage,
    Vec<FallRequest>,
    Frame,
    SyncSender<FallRequest>,
    Receiver<FallRequest>,
);

fn awaiting(tracks: &[u64]) -> Awaiting {
    awaiting_stage(stage(tracks.len()), tracks)
}

fn awaiting_stage(mut stage: FallStage, tracks: &[u64]) -> Awaiting {
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    for sequence in 0..120 {
        let input = frame(sequence, sequence as f64 / 10.0, tracks);
        stage
            .observe(&input, &sender, &mut |_| {})
            .expect("valid frame");
        let requests: Vec<_> = receiver.try_iter().collect();
        if !requests.is_empty() {
            assert_eq!(requests.len(), tracks.len());
            return (stage, requests, input, sender, receiver);
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

#[test]
fn generation_is_observed_before_a_later_update_evicts_the_track() {
    let mut decider = decider(1);
    decider
        .update(
            0,
            0.0,
            &BTreeMap::from([(
                1,
                FallProbabilities::new(1.0, 0.0, 0.0).expect("background"),
            )]),
            [1],
            None,
        )
        .expect("create a scored generation");
    let generation = decider
        .generation_for(1)
        .expect("scored track has generation");
    let ttl = decider.policy().parameters().track_ttl_frames;
    let stage = FallStage::new(decider, 1.0).expect("valid calibration");
    let (mut stage, _, pending, sender, _receiver) = awaiting_stage(stage, &[1]);
    let next = frame(
        pending.identity.sequence + ttl + 1,
        pending.time_sec.unwrap() + 0.1,
        &[2],
    );
    let mut receipts = Vec::new();
    stage
        .observe(&next, &sender, &mut collect(&mut receipts))
        .expect("flush then evict");
    assert_eq!(receipts.len(), 2);
    assert_eq!(receipts[0].frame, pending.identity);
    assert_eq!(receipts[0].generations, vec![Some(generation)]);
    assert_eq!(receipts[1].generations, vec![None]);
    assert_eq!(stage.decider().generation_for(1), None);
}

#[test]
fn one_observe_preserves_both_zero_event_updates_with_original_frames() {
    let (mut stage, _, pending, sender, _receiver) = awaiting(&[1]);
    let next = frame(
        pending.identity.sequence + 1,
        pending.time_sec.unwrap() + 0.1,
        &[2],
    );
    let mut receipts = Vec::new();
    let events = stage
        .observe(&next, &sender, &mut collect(&mut receipts))
        .expect("two updates");
    assert!(events.is_empty());
    assert_eq!(receipts.len(), 2);
    for (receipt, input, track) in [(&receipts[0], &pending, 1), (&receipts[1], &next, 2)] {
        assert_eq!(receipt.frame, input.identity);
        assert_eq!(receipt.index, input.frame_index);
        assert_eq!(receipt.time, input.time_sec.unwrap());
        assert_eq!(receipt.snapshots.len(), 1);
        assert_eq!(receipt.snapshots[0].track_id, Some(track));
        assert!(receipt.events.is_empty());
    }
}

#[test]
fn flush_then_coast_reports_only_the_completed_pending_frame() {
    let (mut stage, _, pending, sender, _receiver) = awaiting(&[1]);
    let coast = frame(
        pending.identity.sequence + 1,
        pending.time_sec.unwrap(),
        &[1],
    );
    let mut receipts = Vec::new();
    stage
        .observe(&coast, &sender, &mut collect(&mut receipts))
        .expect("coast");
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].frame, pending.identity);
    assert!(!stage.decider().last_update_evaluated());
    stage
        .flush(&mut collect(&mut receipts))
        .expect("empty flush");
    assert_eq!(receipts.len(), 1);
}

#[test]
fn stale_duplicate_and_partial_responses_do_not_republish_decisions() {
    let (mut stage, requests, pending, _sender, _receiver) = awaiting(&[1, 2]);
    let mut receipts = Vec::new();
    for request in &requests {
        let mut stale = refused(request);
        stale.frame.sequence += 1000;
        stage
            .consume(stale, &mut collect(&mut receipts))
            .expect("stale response");
        assert!(receipts.is_empty());
    }
    stage
        .consume(refused(&requests[0]), &mut collect(&mut receipts))
        .expect("partial response");
    stage
        .consume(refused(&requests[0]), &mut collect(&mut receipts))
        .expect("duplicate response");
    assert!(receipts.is_empty());
    stage
        .consume(refused(&requests[1]), &mut collect(&mut receipts))
        .expect("last response");
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].frame, pending.identity);
    assert_eq!(
        receipts[0]
            .snapshots
            .iter()
            .map(|s| s.track_id)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(2)]
    );
    stage
        .consume(refused(&requests[1]), &mut collect(&mut receipts))
        .expect("late duplicate");
    stage
        .flush(&mut collect(&mut receipts))
        .expect("no pending decision");
    assert_eq!(receipts.len(), 1);
}

#[test]
fn gap_error_precedes_flush_and_preserves_the_pending_response() {
    let (mut stage, requests, pending, sender, _receiver) = awaiting(&[1]);
    let gap = frame(
        pending.identity.sequence + 1,
        pending.time_sec.unwrap() + 120.0,
        &[1],
    );
    let before = stage.counters();
    let mut receipts = Vec::new();
    assert!(matches!(
        stage.observe(&gap, &sender, &mut collect(&mut receipts)),
        Err(FallStageError::Gap(_))
    ));
    assert!(receipts.is_empty());
    assert_eq!(stage.counters(), before);
    stage
        .consume(refused(&requests[0]), &mut collect(&mut receipts))
        .expect("pending response survives gap");
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].frame, pending.identity);
}

#[test]
fn earlier_success_receipt_survives_later_policy_failure() {
    let (mut stage, _, pending, sender, _receiver) = awaiting(&[1]);
    let excessive_live = frame(
        pending.identity.sequence + 1,
        pending.time_sec.unwrap() + 0.1,
        &[2, 3],
    );
    let mut receipts = Vec::new();
    assert!(matches!(
        stage.observe(&excessive_live, &sender, &mut collect(&mut receipts)),
        Err(FallStageError::Policy(_))
    ));
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].frame, pending.identity);
    assert_eq!(receipts[0].snapshots[0].track_id, Some(1));
}

#[test]
fn empty_success_is_observed_without_fabricating_snapshots_or_replaying_flush() {
    let mut stage = stage(1);
    let (sender, _receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let empty = frame(7, 1.0, &[]);
    let mut receipts = Vec::new();
    stage
        .observe(&empty, &sender, &mut collect(&mut receipts))
        .expect("empty update");
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].frame, empty.identity);
    assert!(receipts[0].snapshots.is_empty());
    assert!(receipts[0].events.is_empty());
    stage
        .flush(&mut collect(&mut receipts))
        .expect("empty flush");
    stage
        .flush(&mut collect(&mut receipts))
        .expect("repeated empty flush");
    assert_eq!(receipts.len(), 1);
}
