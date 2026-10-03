//! Fall owner reply boundary. Fatal state is independent of bounded transport.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};

use seeon_worker_runtime::fall_gpu::{FallGpuError, FallScore};

use crate::exit::Exit;
use crate::inference::State;
use crate::msg::{FallRequest, FallResponse};

pub(super) fn send(
    request: &FallRequest,
    score: Result<FallScore, FallGpuError>,
    responses: &SyncSender<FallResponse>,
    state: &State,
    stop: &AtomicBool,
) -> bool {
    let fatal = matches!(score, Err(error) if error != FallGpuError::Window);
    if fatal {
        // Publish the cause before stop: supervision may observe stop immediately.
        state.fail(Exit::FatalAccelerator);
        stop.store(true, Ordering::SeqCst);
    }
    let response = FallResponse {
        frame: request.frame,
        track_id: request.track_id,
        score: score.map(Into::into).map_err(Into::into),
    };
    let connected = !matches!(
        responses.try_send(response),
        Err(TrySendError::Disconnected(_))
    );
    connected && !fatal
}

#[cfg(test)]
mod tests {
    // Synthetic GPU result values exercise transport, not native GPU execution.
    use super::*;
    use crate::msg::{FALL_RESPONSE_CAPACITY, FallInferenceError, FallScore as ReplyScore};
    use seeon_deepstream_native::{FrameIdentity, GpuMetrics, StateError};
    use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, ZERO_ROW};
    use seeon_worker_runtime::evidence::{
        AcceleratorEvidence, EngineDigest, EvidenceError, Precision,
    };
    use std::sync::mpsc::{self, TryRecvError};

    fn request(sequence: u64) -> FallRequest {
        FallRequest {
            frame: FrameIdentity {
                sequence,
                ..FrameIdentity::default()
            },
            track_id: 71,
            window: Box::new([ZERO_ROW; FALL_WINDOW_FRAMES]),
        }
    }

    fn success() -> FallScore {
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
            logit: 2.0,
            evidence: AcceleratorEvidence::from_delta(
                &before,
                &after,
                2,
                EngineDigest::new([7; 32]),
                Precision::Fp32,
            )
            .unwrap(),
        }
    }

    fn fatal_errors() -> [FallGpuError; 4] {
        [
            FallGpuError::Native(StateError::ExecutionFailed),
            FallGpuError::Output,
            FallGpuError::Evidence(EvidenceError::NoDeviceToHost),
            FallGpuError::Poisoned,
        ]
    }

    #[test]
    fn full_queue_cannot_erase_any_fatal_error_or_evict_prior_replies() {
        for error in fatal_errors() {
            let state = State::default();
            let stop = AtomicBool::new(false);
            let (tx, rx) = mpsc::sync_channel(FALL_RESPONSE_CAPACITY);
            let score = success();
            for sequence in 0..FALL_RESPONSE_CAPACITY {
                assert!(send(
                    &request(sequence as u64),
                    Ok(score),
                    &tx,
                    &state,
                    &stop
                ));
            }
            assert!(!send(&request(1000), Err(error), &tx, &state, &stop));
            assert_eq!(state.failure(), Some(Exit::FatalAccelerator));
            assert!(stop.load(Ordering::SeqCst));
            for sequence in 0..FALL_RESPONSE_CAPACITY {
                let response = rx.try_recv().unwrap();
                assert_eq!(
                    (response.frame.sequence, response.track_id),
                    (sequence as u64, 71)
                );
                assert_eq!(response.score, Ok(ReplyScore::TensorRt(score)));
            }
            assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
            assert_eq!(state.failure(), Some(Exit::FatalAccelerator));
        }
    }

    #[test]
    fn disconnected_requester_cannot_hide_fatal_or_invent_success_failure() {
        for result in fatal_errors()
            .map(Err)
            .into_iter()
            .chain([Ok(success()), Err(FallGpuError::Window)])
        {
            let state = State::default();
            let stop = AtomicBool::new(false);
            let (tx, rx) = mpsc::sync_channel(1);
            drop(rx);
            assert!(!send(&request(7), result, &tx, &state, &stop));
            let fatal = matches!(result, Err(error) if error != FallGpuError::Window);
            assert_eq!(state.failure(), fatal.then_some(Exit::FatalAccelerator));
            assert_eq!(stop.load(Ordering::SeqCst), fatal);
        }
    }

    #[test]
    fn window_refusal_and_success_keep_owner_reusable_even_when_full() {
        let state = State::default();
        let stop = AtomicBool::new(false);
        let (tx, rx) = mpsc::sync_channel(1);
        assert!(send(
            &request(3),
            Err(FallGpuError::Window),
            &tx,
            &state,
            &stop
        ));
        assert!(send(
            &request(4),
            Err(FallGpuError::Window),
            &tx,
            &state,
            &stop
        ));
        assert!(send(&request(5), Ok(success()), &tx, &state, &stop));
        let refusal = rx.try_recv().unwrap();
        assert_eq!(refusal.frame.sequence, 3);
        assert_eq!(
            refusal.score,
            Err(FallInferenceError::TensorRt(FallGpuError::Window))
        );
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
        assert!(send(&request(6), Ok(success()), &tx, &state, &stop));
        assert_eq!(
            rx.try_recv().unwrap().score,
            Ok(ReplyScore::TensorRt(success()))
        );
        assert_eq!(state.failure(), None);
        assert!(!stop.load(Ordering::SeqCst));
    }

    #[test]
    fn delivered_fatal_reply_still_terminates_owner_and_retains_exact_error() {
        for error in fatal_errors() {
            let state = State::default();
            let stop = AtomicBool::new(false);
            let (tx, rx) = mpsc::sync_channel(1);
            assert!(!send(&request(9), Err(error), &tx, &state, &stop));
            assert_eq!(
                rx.try_recv().unwrap().score,
                Err(FallInferenceError::TensorRt(error))
            );
            assert_eq!(state.failure(), Some(Exit::FatalAccelerator));
            assert!(stop.load(Ordering::SeqCst));
        }
    }
}
