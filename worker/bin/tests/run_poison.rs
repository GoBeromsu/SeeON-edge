//! T27/c2: accelerator poison must not become an ordinary missing observation.
//! Oracle: approved Stage4 exit table; real FallStage counters and stop ownership.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use seeon_deepstream_native::{FrameIdentity, StateError};
use seeon_ml_worker::exit::Exit;
use seeon_ml_worker::msg::{FALL_REQUEST_CAPACITY, FallRequest, FallResponse};
use seeon_ml_worker::policy::fall::FallStage;
use seeon_ml_worker::policy::ingest::Frame;
use seeon_ml_worker::run::policy::{FallResponseError, consume_fall_response};
use seeon_worker::fall::{FallCapacities, FallPolicy, FallPolicyDecider};
use seeon_worker::trace::{DecisionTraceMissingReason, DecisionTraceValueName};
use seeon_worker_runtime::cpu::fall::{ErrorKind, FallCpuError};
use seeon_worker_runtime::evidence::EvidenceError;
use seeon_worker_runtime::fall_gpu::FallGpuError;

fn awaiting_score() -> (FallStage, FallRequest) {
    let decider = FallPolicyDecider::new(
        "camera-1",
        "facility-1",
        "boot-1",
        "epoch-1",
        0,
        FallPolicy::default(),
        FallCapacities {
            retained_tracks: 1,
            generation_identities: 1,
            episodes: 1,
            vote_window: 5,
        },
    )
    .expect("valid policy");
    let mut stage = FallStage::new(decider, 1.0).expect("valid calibration");
    let (sender, requests) = mpsc::sync_channel(FALL_REQUEST_CAPACITY);
    // Deterministic synthetic frame inputs, not wall-clock waits or test retries.
    for index in 0..120_u64 {
        let identity = FrameIdentity {
            sequence: index,
            pts_ns: index * 100_000_000,
            pts_valid: 1,
            source_width: 640,
            source_height: 360,
            ..FrameIdentity::default()
        };
        let frame = Frame {
            identity,
            width: 640,
            height: 360,
            live_track_ids: vec![1],
            rows: BTreeMap::from([(1, [0.5; 56])]),
            time_sec: Some(index as f64 / 10.0),
            frame_index: index as i64,
        };
        stage
            .observe(&frame, &sender, &mut |_| {})
            .expect("valid synthetic frame");
        match requests.try_recv() {
            Ok(request) => return (stage, request),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => panic!("live request sender lost"),
        }
    }
    panic!("synthetic sequence did not produce a due window");
}

#[test]
fn fatal_accelerator_results_stop_before_policy_consumption() {
    for error in [
        FallGpuError::Poisoned,
        FallGpuError::Native(StateError::ExecutionFailed),
        FallGpuError::Output,
        FallGpuError::Evidence(EvidenceError::NoDeviceToHost),
    ] {
        let (mut stage, request) = awaiting_score();
        let before = stage.counters();
        let stop = AtomicBool::new(false);
        let response = FallResponse {
            frame: request.frame,
            track_id: request.track_id,
            score: Err(error.into()),
        };
        let refusal = consume_fall_response(&mut stage, response, &stop, &mut |_| {})
            .expect_err("fatal accelerator result must refuse continuation");
        assert_eq!(refusal, FallResponseError::Accelerator(error));
        assert_eq!(refusal.exit(), Exit::FatalAccelerator);
        assert!(stop.load(Ordering::SeqCst));
        assert_eq!(stage.counters(), before);
    }
}

#[test]
fn fatal_cpu_results_stop_before_policy_consumption_even_for_stale_frames() {
    for error in [
        FallCpuError::Native(ErrorKind::Execution),
        FallCpuError::Output,
        FallCpuError::Poisoned,
    ] {
        for stale in [false, true] {
            let (mut stage, request) = awaiting_score();
            let before = stage.counters();
            let pending = stage.pending_scores();
            let snapshots = stage.decider().last_trace_snapshots().to_vec();
            let evaluated = stage.decider().last_update_evaluated();
            let generation = stage.decider().generation_for(request.track_id);
            let fallen = stage.decider().is_fallen(request.track_id);
            let switches = stage.decider().track_id_switch_absorbed_total();
            let stop = AtomicBool::new(false);
            let mut response = FallResponse {
                frame: request.frame,
                track_id: request.track_id,
                score: Err(error.into()),
            };
            if stale {
                response.frame.sequence += 1000;
            }
            let mut observations = 0;
            let refusal =
                consume_fall_response(&mut stage, response, &stop, &mut |_| observations += 1)
                    .expect_err("fatal CPU result must refuse continuation before stale routing");
            assert_eq!(refusal, FallResponseError::Cpu(error));
            assert_eq!(refusal.exit(), Exit::Runtime);
            assert!(stop.load(Ordering::SeqCst));
            assert_eq!(observations, 0);
            assert_eq!(stage.counters(), before);
            assert_eq!(pending, 1);
            assert_eq!(stage.pending_scores(), pending);
            assert_eq!(stage.decider().last_trace_snapshots(), snapshots);
            assert_eq!(stage.decider().last_update_evaluated(), evaluated);
            assert_eq!(stage.decider().generation_for(request.track_id), generation);
            assert_eq!(stage.decider().is_fallen(request.track_id), fallen);
            assert_eq!(stage.decider().track_id_switch_absorbed_total(), switches);
        }
    }
}

#[test]
fn window_refusal_remains_a_missing_observation_without_stopping() {
    let (mut stage, request) = awaiting_score();
    let before = stage.counters();
    let stop = AtomicBool::new(false);
    let response = FallResponse {
        frame: request.frame,
        track_id: request.track_id,
        score: Err(FallGpuError::Window.into()),
    };
    consume_fall_response(&mut stage, response, &stop, &mut |_| {})
        .expect("recoverable input refusal");
    assert!(!stop.load(Ordering::SeqCst));
    assert_eq!(
        stage.counters().missing_observations,
        before.missing_observations + 1
    );
    assert_eq!(stage.counters().stale_responses, before.stale_responses);
}

#[test]
fn cpu_window_refusal_completes_a_missing_observation_without_stopping() {
    let (mut stage, request) = awaiting_score();
    let before = stage.counters();
    let stop = AtomicBool::new(false);
    let response = FallResponse {
        frame: request.frame,
        track_id: request.track_id,
        score: Err(FallCpuError::Window.into()),
    };
    let mut observations = 0;
    let events = consume_fall_response(&mut stage, response, &stop, &mut |update| {
        observations += 1;
        assert_eq!(update.frame, request.frame);
        assert!(update.events.is_empty());
        assert_eq!(update.snapshots.len(), 1);
        assert_eq!(update.snapshots[0].track_id, Some(request.track_id));
        assert_eq!(
            update.snapshots[0]
                .missing_values()
                .get(&DecisionTraceValueName::FallTransitionProbability),
            Some(&DecisionTraceMissingReason::AdapterReturnedNoData)
        );
    })
    .expect("recoverable CPU input refusal");
    assert!(events.is_empty());
    assert_eq!(observations, 1);
    assert!(!stop.load(Ordering::SeqCst));
    assert_eq!(stage.pending_scores(), 0);
    assert_eq!(
        stage.counters().missing_observations,
        before.missing_observations + 1
    );
    assert_eq!(stage.counters().stale_responses, before.stale_responses);
}
