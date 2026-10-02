//! Known outside-window frames notify without scoring or moving fall state.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver};

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::episode::BusinessEvent;
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
use seeon_worker::pose_bbox56::FALL_WINDOW_FRAMES;
use seeon_worker::temporal::DEFAULT_MAX_GAP_ROWS;
use seeon_worker::trace::DecisionTraceSnapshot;

use crate::msg::{FALL_REQUEST_CAPACITY, FallRequest};
use crate::policy::ingest::Frame;

use super::{DecisionUpdate, FallCounters, FallStage, FallStageError};

struct Receipt {
    frame: FrameIdentity,
    index: i64,
    time: f64,
    snapshots: Vec<DecisionTraceSnapshot>,
    events: Vec<BusinessEvent>,
}

fn collect(receipts: &mut Vec<Receipt>) -> impl for<'a> FnMut(DecisionUpdate<'a>) + '_ {
    |update| {
        receipts.push(Receipt {
            frame: update.frame,
            index: update.frame_index,
            time: update.time_sec,
            snapshots: update.snapshots.to_vec(),
            events: update.events.to_vec(),
        });
    }
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

fn frame(sequence: u64, time: Option<f64>, tracks: &[u64]) -> Frame {
    Frame {
        identity: FrameIdentity {
            sequence,
            pts_ns: time.map_or(0, |seconds| (seconds * 1e9) as u64),
            pts_valid: u32::from(time.is_some()),
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
        time_sec: time,
        frame_index: sequence as i64,
    }
}

fn assert_outside(receipt: &Receipt, input: &Frame, time: f64) {
    assert_eq!(receipt.frame, input.identity);
    assert_eq!(receipt.index, input.frame_index);
    assert_eq!(receipt.time, time);
    assert!(receipt.events.is_empty());
    assert_eq!(receipt.snapshots.len(), 1);
    let snapshot = &receipt.snapshots[0];
    assert_eq!(snapshot, &DecisionTraceSnapshot::outside_detection_window());
    assert_eq!(snapshot.reason.as_str(), "outside-detection-window");
    assert_eq!(snapshot.previous_state.as_str(), "not-evaluated");
    assert_eq!(snapshot.current_state.as_str(), "not-evaluated");
    assert!(!snapshot.triggered);
    assert_eq!(snapshot.track_id, None);
    assert_eq!(snapshot.bed_id, None);
    assert!(snapshot.values().is_empty());
    assert_eq!(snapshot.missing_values().len(), 1);
}

fn drain(receiver: &Receiver<FallRequest>) -> Vec<FallRequest> {
    receiver.try_iter().collect()
}

#[test]
fn successive_gated_frames_notify_actual_frames_without_moving_state() {
    let mut stage = stage();
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let before_pts = stage.last_pts_ns;
    let before_counters = stage.counters();
    let before_snapshots = stage.decider().last_trace_snapshots().to_vec();
    let first = frame(3, None, &[9]);
    let mut receipts = Vec::new();
    let events = stage
        .skip_outside_window(&first, &mut collect(&mut receipts))
        .expect("first gated frame");
    assert!(events.is_empty());
    assert_outside(&receipts[0], &first, 0.0);

    let second = frame(4, Some(1.25), &[11]);
    let events = stage
        .skip_outside_window(&second, &mut collect(&mut receipts))
        .expect("second gated frame");
    assert!(events.is_empty());
    assert_eq!(receipts.len(), 2);
    assert_outside(&receipts[1], &second, 1.25);
    assert_ne!(receipts[0].frame, receipts[1].frame);
    assert_ne!(receipts[0].index, receipts[1].index);
    assert_eq!(receipts[0].snapshots, receipts[1].snapshots);
    assert_eq!(receipts[0].snapshots[0].track_id, None);
    assert_eq!(stage.last_pts_ns, before_pts);
    assert_eq!(stage.counters(), before_counters);
    assert_eq!(stage.counters(), FallCounters::default());
    assert_eq!(stage.pending_scores(), 0);
    assert_eq!(stage.decider().last_trace_snapshots(), before_snapshots);
    assert!(stage.decider().last_trace_snapshots().is_empty());
    assert!(!stage.decider().last_update_evaluated());
    assert_eq!(stage.decider().track_id_switch_absorbed_total(), 0);
    assert!(drain(&receiver).is_empty());
    let _ = sender;
}

#[test]
fn gated_frame_completes_pending_under_its_original_frame_first() {
    let mut stage = stage();
    let (sender, receiver) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let mut pending = None;
    for sequence in 0..120_u64 {
        let input = frame(sequence, Some(sequence as f64 / 10.0), &[1]);
        stage
            .observe(&input, &sender, &mut |_| {})
            .expect("warmup frame");
        if stage.pending_scores() == 1 {
            pending = Some(input);
            break;
        }
    }
    let pending = pending.expect("fixed sequence produced a due window");
    let request = drain(&receiver);
    assert_eq!(request.len(), 1);
    assert_eq!(request[0].frame, pending.identity);
    assert_eq!(request[0].track_id, 1);
    assert_eq!(request[0].window.len(), FALL_WINDOW_FRAMES);
    let pts = stage.last_pts_ns;
    let gaps = stage.counters().resample_gap_rows;
    let missing = stage.counters().missing_observations;
    let gated = frame(pending.identity.sequence + 1, Some(9.5), &[7]);
    let mut receipts = Vec::new();
    let events = stage
        .skip_outside_window(&gated, &mut collect(&mut receipts))
        .expect("flush then gate");
    assert!(events.is_empty());
    assert_eq!(receipts.len(), 2);
    assert_eq!(receipts[0].frame, pending.identity);
    assert_eq!(receipts[0].index, pending.frame_index);
    assert_eq!(receipts[0].time, pending.time_sec.unwrap());
    assert_eq!(receipts[0].snapshots.len(), 1);
    assert_eq!(receipts[0].snapshots[0].track_id, Some(1));
    assert!(receipts[0].events.is_empty());
    assert_eq!(
        stage.decider().last_trace_snapshots(),
        receipts[0].snapshots
    );
    assert_ne!(
        receipts[0].snapshots[0],
        DecisionTraceSnapshot::outside_detection_window()
    );
    assert_outside(&receipts[1], &gated, 9.5);
    assert_ne!(receipts[1].snapshots, receipts[0].snapshots);
    assert_ne!(receipts[1].frame, receipts[0].frame);
    assert!(receipts[1].events.is_empty());
    assert_eq!(stage.pending_scores(), 0);
    assert_eq!(stage.last_pts_ns, pts);
    assert_eq!(stage.counters().resample_gap_rows, gaps);
    assert_eq!(stage.counters().missing_observations, missing + 1);
    assert!(drain(&receiver).is_empty());
    let _ = sender;
}

#[test]
fn reentry_follows_existing_pts_rule_without_an_invented_reset() {
    let mut gated = stage();
    let mut control = stage();
    let (gated_tx, gated_rx) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let (control_tx, control_rx) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    let active = frame(1, Some(1.0), &[1]);
    gated
        .observe(&active, &gated_tx, &mut |_| {})
        .expect("active frame");
    control
        .observe(&active, &control_tx, &mut |_| {})
        .expect("control active frame");
    assert!(drain(&gated_rx).is_empty());
    assert!(drain(&control_rx).is_empty());
    assert_eq!(gated.last_pts_ns, Some(1_000_000_000));
    assert_eq!(control.last_pts_ns, gated.last_pts_ns);
    assert_eq!(gated.counters(), control.counters());
    assert_eq!(gated.pending_scores(), control.pending_scores());

    let outside = frame(8, Some(1.2), &[4]);
    let mut receipts = Vec::new();
    gated
        .skip_outside_window(&outside, &mut collect(&mut receipts))
        .expect("known outside frame");
    assert_eq!(receipts.len(), 1);
    assert_outside(&receipts[0], &outside, 1.2);
    assert_eq!(gated.last_pts_ns, Some(1_000_000_000));
    assert!(drain(&gated_rx).is_empty());

    let mut compared = 0_u32;
    for step in 2_u64..=40 {
        let time = 1.0 + (step - 1) as f64 * 0.2;
        let input = frame(step, Some(time), &[1]);
        control
            .observe(&input, &control_tx, &mut |_| {})
            .expect("control resumed frame");
        gated
            .observe(&input, &gated_tx, &mut |_| {})
            .expect("gated stage resumes on the same frame");
        let control_requests = drain(&control_rx);
        let gated_requests = drain(&gated_rx);
        assert_eq!(gated_requests.len(), control_requests.len());
        for (left, right) in gated_requests.iter().zip(&control_requests) {
            assert_eq!(left.frame, right.frame);
            assert_eq!(left.frame, input.identity);
            assert_eq!(left.track_id, right.track_id);
            assert_eq!(left.track_id, 1);
            assert_eq!(left.window.as_ref(), right.window.as_ref());
            compared += 1;
        }
        assert_eq!(gated.last_pts_ns, control.last_pts_ns);
        assert_eq!(gated.counters(), control.counters());
        assert_eq!(gated.pending_scores(), control.pending_scores());
    }
    assert!(
        compared >= 1,
        "resumed stages must emit a real comparable request"
    );
    let counters = gated.counters();
    let far = frame(90, Some(1.0 + 39.0 * 0.2 + 120.0), &[4]);
    let error = gated
        .observe(&far, &gated_tx, &mut |_| {})
        .expect_err("large forward re-entry uses resampler.push");
    match error {
        FallStageError::Gap(gap) => {
            assert!(gap.gap_rows > i128::from(gap.max_gap_rows));
            assert_eq!(gap.max_gap_rows, DEFAULT_MAX_GAP_ROWS);
        }
        other => panic!("expected the existing PTS gap, got {other:?}"),
    }
    assert_eq!(gated.counters(), counters);
    assert_eq!(gated.pending_scores(), control.pending_scores());
    assert!(drain(&gated_rx).is_empty());
    assert_ne!(gated.last_pts_ns, control.last_pts_ns);
}
