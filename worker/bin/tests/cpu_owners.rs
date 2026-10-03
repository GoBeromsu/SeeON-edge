//! Actual ORT CPU actors: thread ownership, readiness, bounded channels and failure
//! retention. These are not CUDA numerical qualification or full camera-loop tests.

use seeon_deepstream_native::FrameIdentity;
use seeon_ml_worker::{
    exit::Exit,
    inference::{
        self, Owner, Runtime,
        cpu::{self, CapturedModel},
    },
    msg::{
        BedRequest, FALL_RESPONSE_CAPACITY, FallRequest, FallScore, ONESHOT_CAPACITY,
        StoredPoseRequest,
    },
    poll::poll_until,
    seam::{Clock, SystemClock},
};
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, ZERO_ROW};
use seeon_worker_runtime::cpu::{
    Threads, bed::BedCpuError, fall::FallCpuError, stored_pose::StoredPoseCpuError,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

fn required(name: &str) -> PathBuf {
    std::env::var_os(name).expect(name).into()
}
fn captured(onnx: Vec<u8>) -> CapturedModel {
    CapturedModel {
        runtime_library: required("SEEON_TEST_ORT_RUNTIME"),
        onnx: onnx.into(),
    }
}
fn model(role: &str) -> CapturedModel {
    let (path, hash) = match role {
        "fall" => (
            "fall/pose-bbox56-gru/model.onnx",
            "258ae9d9460534e659bf97af4bc55a083c830fa190ff6c8ed347db6bdbf32163",
        ),
        "bed" => (
            "bed/yolo26l-seg.onnx",
            "de1c081df29936d6cf42329ec53b4d22f1e02f13bf1ea3b769907dc1d95a3d86",
        ),
        "pose" => (
            "pose/yolo26n-pose.onnx",
            "724ae1b1b4420cea85f98ac7c5ccabf3d56229f7c25390475e09b7056e3b9526",
        ),
        _ => panic!("unknown role"),
    };
    let bytes = fs::read(required("SEEON_TEST_ORT_MODELS").join(path)).unwrap();
    let actual: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(actual, hash);
    captured(bytes)
}
fn ready<R>(owner: &Owner<R>, threads: Threads, outputs: u32) {
    assert_eq!(
        owner.readiness.recv_timeout(Duration::from_secs(60)),
        Ok(Ok(()))
    );
    let Some(Runtime::OnnxRuntimeCpu(info)) = owner.state.runtime() else {
        panic!("actual CPU runtime facts required")
    };
    assert_eq!(info.runtime_version, "1.29.0");
    assert_eq!(info.threads, threads);
    assert_eq!(
        (info.abi_version, info.input_count, info.output_count),
        (1, 1, outputs)
    );
    assert_eq!(owner.state.failure(), None);
    assert_eq!(
        owner.readiness.recv_timeout(Duration::from_secs(1)),
        Err(mpsc::RecvTimeoutError::Disconnected)
    );
}
fn finish<R>(owner: Owner<R>, stop: &AtomicBool) {
    stop.store(true, Ordering::SeqCst);
    let clock = SystemClock::new();
    poll_until(
        &clock,
        clock.monotonic() + Duration::from_secs(10),
        "CPU thread stop",
        || owner.thread.is_finished(),
    )
    .unwrap();
    assert_eq!(
        inference::join(
            owner.thread,
            &clock,
            clock.monotonic() + Duration::from_secs(1)
        ),
        Ok(())
    );
}
fn window(sequence: u64) -> FallRequest {
    FallRequest {
        frame: FrameIdentity {
            sequence,
            ..FrameIdentity::default()
        },
        track_id: 71,
        window: Box::new([ZERO_ROW; FALL_WINDOW_FRAMES]),
    }
}
fn send<R>(owner: &Owner<R>, request: R) {
    let clock = SystemClock::new();
    let mut pending = Some(request);
    poll_until(
        &clock,
        clock.monotonic() + Duration::from_secs(10),
        "CPU request capacity",
        || match owner.requests.try_send(pending.take().unwrap()) {
            Ok(()) => true,
            Err(mpsc::TrySendError::Full(request)) => {
                pending = Some(request);
                false
            }
            Err(mpsc::TrySendError::Disconnected(_)) => panic!("owner disconnected"),
        },
    )
    .unwrap();
}

#[test]
#[ignore = "requires pinned CPU models, ORT library and telemetry opt-out"]
fn actual_fall_actor_preserves_identity_and_recovers_after_input_refusal() {
    let stop = Arc::new(AtomicBool::new(false));
    let (owner, responses) = cpu::spawn_fall(model("fall"), Arc::clone(&stop)).unwrap();
    ready(&owner, Threads::Default, 1);
    let mut bad = window(1);
    bad.window[0][0] = f32::NAN;
    send(&owner, bad);
    assert!(matches!(
        responses
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .score,
        Err(seeon_ml_worker::msg::FallInferenceError::Cpu(
            FallCpuError::Window
        ))
    ));
    for sequence in [2, 3] {
        send(&owner, window(sequence));
        let response = responses.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!((response.frame.sequence, response.track_id), (sequence, 71));
        assert!(matches!(response.score, Ok(FallScore::Cpu(value)) if value.is_finite()));
    }
    assert!(!stop.load(Ordering::SeqCst));
    assert_eq!(owner.state.failure(), None);
    finish(owner, &stop);
}

#[test]
#[ignore = "requires pinned CPU models, ORT library and telemetry opt-out"]
fn actual_image_actors_recover_input_and_return_cpu_outputs() {
    let stop = Arc::new(AtomicBool::new(false));
    let bed = cpu::spawn_bed(model("bed"), Arc::clone(&stop)).unwrap();
    ready(&bed, Threads::Single, 2);
    for valid in [false, true] {
        let (reply, replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
        send(
            &bed,
            BedRequest {
                rgb: if valid {
                    vec![0; 640 * 360 * 3]
                } else {
                    vec![]
                },
                width: 640,
                height: 360,
                reply,
            },
        );
        let result = replies.recv_timeout(Duration::from_secs(30)).unwrap();
        if valid {
            let output = result.unwrap();
            assert!(output.evidence.is_none());
            assert_eq!(
                (output.detections.len(), output.protos.len()),
                (300 * 38, 32 * 320 * 320)
            );
            assert!(
                output
                    .detections
                    .iter()
                    .chain(&output.protos)
                    .all(|value| value.is_finite())
            );
            assert_eq!(
                (
                    output.letterbox.source_width,
                    output.letterbox.source_height
                ),
                (640, 360)
            );
        } else {
            assert!(matches!(
                result,
                Err(seeon_ml_worker::msg::BedInferenceError::Cpu(
                    BedCpuError::Input(_)
                ))
            ));
        }
    }
    assert_eq!(bed.state.failure(), None);
    finish(bed, &stop);
    stop.store(false, Ordering::SeqCst);
    let pose = cpu::spawn_stored_pose(model("pose"), 0.25, Arc::clone(&stop)).unwrap();
    ready(&pose, Threads::Single, 1);
    for valid in [false, true] {
        let (reply, replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
        send(
            &pose,
            StoredPoseRequest {
                rgb: if valid {
                    vec![0; 640 * 360 * 3]
                } else {
                    vec![]
                },
                width: 640,
                height: 360,
                reply,
            },
        );
        let result = replies.recv_timeout(Duration::from_secs(30)).unwrap();
        if valid {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result,
                Err(seeon_ml_worker::msg::StoredPoseInferenceError::Cpu(
                    StoredPoseCpuError::Input(_)
                ))
            ));
        }
    }
    assert_eq!(pose.state.failure(), None);
    finish(pose, &stop);
}

#[test]
#[ignore = "requires ORT library and telemetry opt-out"]
fn invalid_onnx_refuses_readiness_without_runtime_claims() {
    let stop = Arc::new(AtomicBool::new(false));
    let (owner, _) = cpu::spawn_fall(captured(vec![1, 2, 3]), Arc::clone(&stop)).unwrap();
    assert_eq!(
        owner.readiness.recv_timeout(Duration::from_secs(10)),
        Ok(Err(Exit::Runtime))
    );
    assert!(owner.state.runtime().is_none());
    assert_eq!(owner.state.failure(), Some(Exit::Runtime));
    finish(owner, &stop);
}

// A real small ONNX graph: warm-up sum=0 yields 1; sum=1 yields infinity.
// This forces post-readiness output rejection without a production test seam.
fn output_fault_model() -> CapturedModel {
    let output = Command::new(required("SEEON_TEST_PYTHON")).args(["-c", r#"
import sys
from onnx import TensorProto as T, helper as h
nodes = [h.make_node('ReduceSum', ['window', 'axes'], ['sum'], keepdims=0),
         h.make_node('Sub', ['one', 'sum'], ['denominator']),
         h.make_node('Reciprocal', ['denominator'], ['value']),
         h.make_node('Unsqueeze', ['value', 'unsqueeze'], ['84'])]
g = h.make_graph(nodes, 'fatal-after-warmup', [h.make_tensor_value_info('window', T.FLOAT, [1,30,56])],
                 [h.make_tensor_value_info('84', T.FLOAT, [1,1])],
                 [h.make_tensor('axes', T.INT64, [2], [1,2]), h.make_tensor('one', T.FLOAT, [1], [1]), h.make_tensor('unsqueeze', T.INT64, [1], [1])])
m = h.make_model(g, opset_imports=[h.make_opsetid('',17)], ir_version=9)
import onnx
onnx.checker.check_model(m)
sys.stdout.buffer.write(m.SerializeToString())
"#]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    captured(output.stdout)
}

#[test]
#[ignore = "requires Python ONNX, ORT library and telemetry opt-out"]
fn fatal_output_survives_a_full_response_queue_and_stops_the_actor() {
    let stop = Arc::new(AtomicBool::new(false));
    let (owner, responses) = cpu::spawn_fall(output_fault_model(), Arc::clone(&stop)).unwrap();
    ready(&owner, Threads::Default, 1);
    for sequence in 0..FALL_RESPONSE_CAPACITY {
        send(&owner, window(sequence as u64));
    }
    let mut fatal = window(FALL_RESPONSE_CAPACITY as u64);
    fatal.window[0][0] = 1.0;
    send(&owner, fatal);
    let clock = SystemClock::new();
    poll_until(
        &clock,
        clock.monotonic() + Duration::from_secs(10),
        "retained CPU failure",
        || owner.state.failure().is_some(),
    )
    .unwrap();
    assert_eq!(owner.state.failure(), Some(Exit::Runtime));
    poll_until(
        &clock,
        clock.monotonic() + Duration::from_secs(10),
        "failed CPU owner exit",
        || owner.thread.is_finished(),
    )
    .unwrap();
    assert!(stop.load(Ordering::SeqCst));
    for sequence in 0..FALL_RESPONSE_CAPACITY {
        let response = responses.try_recv().unwrap();
        assert_eq!(response.frame.sequence, sequence as u64);
        assert_eq!(response.score, Ok(FallScore::Cpu(1.0)));
    }
    assert!(matches!(
        responses.try_recv(),
        Err(mpsc::TryRecvError::Disconnected)
    ));
    assert_eq!(
        owner.state.failure(),
        Some(Exit::Runtime),
        "draining cannot erase the fatal result"
    );
    finish(owner, &stop);
}
