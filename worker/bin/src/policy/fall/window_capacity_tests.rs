use std::collections::BTreeMap;
use std::sync::mpsc;

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, PoseBbox56Row, ZERO_ROW};
use seeon_worker::trace::DecisionTraceMissingReason as Reason;

use crate::msg::FALL_REQUEST_CAPACITY;
use crate::policy::ingest::Frame;

use super::window::{RECONNECT_CAPACITY, ReconnectCapacityError, TRACK_TTL_FRAMES, Windows};
use super::{FallStage, FallStageError};

const OVERFLOW_TRACK: u64 = RECONNECT_CAPACITY as u64;

fn empty_rows() -> BTreeMap<u64, PoseBbox56Row> {
    BTreeMap::new()
}

fn advance_empty(windows: &mut Windows, updates: u64) {
    for _ in 0..updates {
        windows.update(&empty_rows(), &[]).unwrap();
    }
}

fn full_history() -> Windows {
    let mut windows = Windows::default();
    let ids: Vec<_> = (0..RECONNECT_CAPACITY as u64).collect();
    windows.update(&empty_rows(), &ids).unwrap();
    advance_empty(&mut windows, TRACK_TTL_FRAMES - 1);
    windows
}

fn full_history_with_due_expiration() -> Windows {
    let mut windows = full_history();
    windows.update(&empty_rows(), &[OVERFLOW_TRACK]).unwrap();
    advance_empty(&mut windows, TRACK_TTL_FRAMES - 1);
    windows
}

fn stage() -> FallStage {
    let decider = FallPolicyDecider::new(
        "camera-1",
        "facility-1",
        "boot-1",
        "epoch-1",
        0,
        FallPolicy::default(),
        FallCapacities {
            retained_tracks: 2,
            generation_identities: 2,
            episodes: 2,
            vote_window: 5,
        },
    )
    .expect("valid test policy");
    FallStage::new(decider, 1.0).expect("valid test calibration")
}

fn frame(sequence: u64, pts_ns: u64, tracks: &[u64]) -> Frame {
    Frame {
        identity: FrameIdentity {
            sequence,
            pts_ns,
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
        time_sec: Some(pts_ns as f64 / 1e9),
        frame_index: sequence as i64,
    }
}

#[test]
fn exact_capacity_reconnects_with_padding_at_a_due_stride() {
    let mut windows = full_history();
    advance_empty(&mut windows, 4);

    let row = [0.75; 56];
    let outcome = windows
        .update(&BTreeMap::from([(0, row)]), &[0])
        .expect("all 4096 reconnect markers are retained");
    assert_eq!(outcome.due.len(), 1);
    let (track_id, window) = outcome.due.into_iter().next().unwrap();
    assert_eq!(track_id, 0);
    assert!(
        window[..FALL_WINDOW_FRAMES - 1]
            .iter()
            .all(|entry| *entry == ZERO_ROW)
    );
    assert_eq!(window[FALL_WINDOW_FRAMES - 1], row);
}

#[test]
fn overflow_refuses_eviction_and_preserves_old_markers_and_expired_track() {
    let mut windows = full_history_with_due_expiration();
    assert_eq!(
        windows.update(&empty_rows(), &[]).unwrap_err(),
        ReconnectCapacityError
    );
    assert_eq!(
        windows.update(&empty_rows(), &[]).unwrap_err(),
        ReconnectCapacityError,
        "the expired track remains available for a later marker admission"
    );

    let first = [0.25; 56];
    let mut outcome = windows.update(&BTreeMap::from([(0, first)]), &[0]).unwrap();
    for _ in 0..4 {
        outcome = windows.update(&BTreeMap::from([(0, first)]), &[0]).unwrap();
    }
    assert_eq!(outcome.due.len(), 1, "the old marker was not forgotten");
    let (track_id, window) = outcome.due.into_iter().next().unwrap();
    assert_eq!(track_id, 0);
    assert!(
        window[..FALL_WINDOW_FRAMES - 5]
            .iter()
            .all(|entry| *entry == ZERO_ROW)
    );
    assert!(
        window[FALL_WINDOW_FRAMES - 5..]
            .iter()
            .all(|entry| *entry == first)
    );

    let mut outcome = windows.update(&empty_rows(), &[OVERFLOW_TRACK]).unwrap();
    for _ in 0..4 {
        outcome = windows.update(&empty_rows(), &[OVERFLOW_TRACK]).unwrap();
    }
    assert_eq!(
        outcome.due.len(),
        1,
        "the freed marker slot retained the previously refused expired track"
    );
    assert_eq!(outcome.due[0].0, OVERFLOW_TRACK);
}

#[test]
fn epoch_clear_recovers_capacity_and_discards_old_reconnect_markers() {
    let mut windows = full_history_with_due_expiration();
    assert_eq!(
        windows.update(&empty_rows(), &[]).unwrap_err(),
        ReconnectCapacityError
    );
    windows.clear();

    let mut outcome = windows.update(&empty_rows(), &[0]).unwrap();
    for _ in 0..4 {
        outcome = windows.update(&empty_rows(), &[0]).unwrap();
    }
    assert!(outcome.due.is_empty());
    assert_eq!(outcome.reasons.get(&0), Some(&Reason::ClassifierWarmup));
}

#[test]
fn stage_propagates_capacity_refusal_on_a_valid_row() {
    let mut stage = stage();
    stage.windows = full_history_with_due_expiration();
    let (requests, _receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let input = frame(1, 1_000_000_000, &[OVERFLOW_TRACK + 10]);

    assert_eq!(
        stage.observe(&input, &requests, &mut |_| {}).unwrap_err(),
        FallStageError::ReconnectCapacity
    );
    assert_eq!(stage.counters().resample_gap_rows, 0);
}

#[test]
fn stage_propagates_capacity_refusal_on_a_synthetic_gap_row() {
    let mut stage = stage();
    let (requests, _receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let prime = frame(1, 0, &[1]);
    stage
        .observe(&prime, &requests, &mut |_| {})
        .expect("seed the resampler");
    stage.windows = full_history_with_due_expiration();

    let gap = frame(2, 4_000_000_000, &[OVERFLOW_TRACK + 10]);
    assert_eq!(
        stage.observe(&gap, &requests, &mut |_| {}).unwrap_err(),
        FallStageError::ReconnectCapacity
    );
    assert_eq!(
        stage.counters().resample_gap_rows,
        0,
        "the error arose before the synthetic gap row could be counted"
    );
}

#[test]
fn batch_capacity_refusal_preserves_pending_decision_before_any_flush() {
    use super::decision::Pending;

    let mut stage = stage();
    let (requests, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    stage
        .observe(&frame(1, 0, &[]), &requests, &mut |_| {})
        .unwrap();
    stage.windows = full_history();
    stage
        .windows
        .update(&empty_rows(), &[OVERFLOW_TRACK])
        .unwrap();
    let pending_frame = frame(2, 1, &[1]).identity;
    stage.pending = Some(Pending {
        frame: pending_frame,
        frame_index: 2,
        time_sec: 0.0,
        live: vec![1],
        awaited: [1].into_iter().collect(),
        probabilities: BTreeMap::new(),
        reasons: BTreeMap::new(),
    });
    let mut observations = 0;
    assert_eq!(
        stage
            .observe(&frame(3, 4_000_000_000, &[]), &requests, &mut |_| {
                observations += 1;
            })
            .unwrap_err(),
        FallStageError::ReconnectCapacity
    );
    assert_eq!(observations, 0);
    assert_eq!(stage.pending.as_ref().unwrap().frame, pending_frame);
    assert!(receiver.try_recv().is_err());
    assert_eq!(stage.counters().resample_gap_rows, 0);
}
