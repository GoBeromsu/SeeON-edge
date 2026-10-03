//! Native CPU composition only; not accelerator or full camera-loop acceptance.

use super::*;
use crate::msg::FallScore;
use crate::seam::SystemClock;
use seeon_deepstream_native::FrameIdentity;
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, PoseBbox56Row, ZERO_ROW};
use seeon_worker_runtime::cpu::Threads;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn required(name: &str) -> PathBuf {
    let value = std::env::var_os(name).expect(name);
    assert!(!value.is_empty(), "{name} must be a nonblank path");
    value.into()
}

fn model(root: &Path, path: &str, hash: &str) -> Arc<[u8]> {
    let bytes = fs::read(root.join(path)).expect("pinned CPU model");
    let actual: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(actual, hash, "{path}");
    bytes.into()
}

fn assets() -> (CpuModels, Box<[PoseBbox56Row; FALL_WINDOW_FRAMES]>) {
    let runtime_library = required("SEEON_TEST_ORT_RUNTIME");
    assert!(runtime_library.is_file(), "actual ORT library required");
    let root = required("SEEON_TEST_ORT_MODELS");
    let fixtures = required("SEEON_TEST_ORT_FIXTURES");
    let bytes = fs::read(fixtures.join("fall/windows.f32")).expect("fall fixture windows");
    let mut window = Box::new([ZERO_ROW; FALL_WINDOW_FRAMES]);
    let window_bytes = FALL_WINDOW_FRAMES * ZERO_ROW.len() * 4;
    assert_eq!(bytes.len() % window_bytes, 0, "whole fixture windows");
    let first = bytes.get(..window_bytes).expect("first recorded window");
    for (value, chunk) in window.iter_mut().flatten().zip(first.chunks_exact(4)) {
        *value = f32::from_le_bytes(chunk.try_into().expect("four bytes"));
        assert!(value.is_finite(), "finite fixture value");
    }
    (
        CpuModels {
            runtime_library,
            fall: model(
                &root,
                "fall/pose-bbox56-gru/model.onnx",
                "258ae9d9460534e659bf97af4bc55a083c830fa190ff6c8ed347db6bdbf32163",
            ),
            bed: model(
                &root,
                "bed/yolo26l-seg.onnx",
                "de1c081df29936d6cf42329ec53b4d22f1e02f13bf1ea3b769907dc1d95a3d86",
            ),
            stored_pose: model(
                &root,
                "pose/yolo26n-pose.onnx",
                "724ae1b1b4420cea85f98ac7c5ccabf3d56229f7c25390475e09b7056e3b9526",
            ),
        },
        window,
    )
}

fn empty() -> ModelOwners {
    ModelOwners {
        fall: None,
        bed: None,
        stored_pose: None,
        fall_responses: None,
        stop: Arc::new(AtomicBool::new(false)),
        shutdown: Arc::new(ShutdownDeadline::new(Duration::from_secs(10)).unwrap()),
        lease: None,
        states: [None, None, None],
    }
}

fn policy() -> BootPolicy {
    BootPolicy {
        device_ordinal: 0,
        stored_pose_threshold: 0.25,
        deployed_batch: None,
        readiness_budget: Duration::from_secs(180),
    }
}

fn observed(models: &ModelOwners, role: ModelRole, threads: Threads, outputs: u32) -> Runtime {
    let runtime = models.runtime(role).expect("retained native runtime facts");
    let Runtime::OnnxRuntimeCpu(info) = runtime else {
        panic!("CPU runtime required for {role:?}")
    };
    assert_eq!(info.runtime_version, "1.29.0");
    assert_eq!(info.threads, threads);
    assert_eq!(
        (info.abi_version, info.input_count, info.output_count),
        (1, 1, outputs)
    );
    runtime.clone()
}

fn reply(models: &ModelOwners, window: &[PoseBbox56Row; FALL_WINDOW_FRAMES], sequence: u64) {
    let request = FallRequest {
        frame: FrameIdentity {
            sequence,
            ..FrameIdentity::default()
        },
        track_id: 71,
        window: Box::new(*window),
    };
    assert!(models.fall().unwrap().requests.try_send(request).is_ok());
    let response = models
        .fall_responses()
        .unwrap()
        .recv_timeout(Duration::from_secs(10))
        .expect("actual CPU fall response");
    assert_eq!((response.frame.sequence, response.track_id), (sequence, 71));
    let score = response.score.expect("actual CPU logit");
    assert!(matches!(score, FallScore::Cpu(value) if value.is_finite()));
    assert!(
        score.accelerator().is_none(),
        "CPU is not accelerator evidence"
    );
}

fn close(models: &mut ModelOwners, clock: &SystemClock) {
    let errors = models.close(clock);
    assert!(
        errors.is_empty(),
        "all native joins must succeed: {errors:?}"
    );
    assert!(models.fall().is_none());
    assert!(models.bed().is_none());
    assert!(models.stored_pose().is_none());
    assert!(!models.has_live_threads());
}

#[test]
#[ignore = "requires pinned CPU models, ORT library, fixtures and telemetry opt-out"]
fn actual_cpu_composition_warms_all_roles_and_retains_runtime_after_close() {
    let (inputs, window) = assets();
    let clock = SystemClock::new();
    let mut models = empty();
    models
        .start(ModelInputs::OnnxRuntimeCpu(&inputs), policy(), &clock)
        .unwrap();
    drop(inputs);
    let facts = [
        (
            ModelRole::Fall,
            observed(&models, ModelRole::Fall, Threads::Default, 1),
        ),
        (
            ModelRole::Bed,
            observed(&models, ModelRole::Bed, Threads::Single, 2),
        ),
        (
            ModelRole::StoredPose,
            observed(&models, ModelRole::StoredPose, Threads::Single, 1),
        ),
    ];
    assert_eq!(
        [
            models.fall().unwrap().thread.thread().name(),
            models.bed().unwrap().thread.thread().name(),
            models.stored_pose().unwrap().thread.thread().name(),
        ],
        [Some("cpu-fall"), Some("cpu-bed"), Some("cpu-stored-pose")]
    );
    assert!(models.has_live_threads());
    assert_eq!(models.failure(), None);
    reply(&models, &window, 47);
    close(&mut models, &clock);
    assert_eq!(models.failure(), None);
    for (role, runtime) in facts {
        assert_eq!(models.runtime(role), Some(&runtime), "{role:?} after join");
    }
}

#[test]
#[ignore = "requires pinned CPU models, ORT library, fixtures and telemetry opt-out"]
fn actual_cpu_partial_start_failure_retains_fall_until_explicit_cleanup() {
    let (inputs, window) = assets();
    for bed in [Arc::<[u8]>::from(vec![1, 2, 3]), Arc::clone(&inputs.fall)] {
        let broken = CpuModels {
            runtime_library: inputs.runtime_library.clone(),
            fall: Arc::clone(&inputs.fall),
            bed,
            stored_pose: Arc::clone(&inputs.stored_pose),
        };
        let clock = SystemClock::new();
        let mut models = empty();
        assert_eq!(
            models.start(ModelInputs::OnnxRuntimeCpu(&broken), policy(), &clock),
            Err(ModelStartError {
                model: ModelRole::Bed,
                kind: StartKind::Refused(Exit::Runtime),
            })
        );
        let fall_runtime = observed(&models, ModelRole::Fall, Threads::Default, 1);
        assert!(!models.fall().unwrap().thread.is_finished());
        assert_eq!(models.fall().unwrap().state.failure(), None);
        assert_eq!(models.bed().unwrap().state.failure(), Some(Exit::Runtime));
        assert!(models.runtime(ModelRole::Bed).is_none());
        assert!(models.stored_pose().is_none(), "no later owner or fallback");
        assert!(models.states[2].is_none());
        assert!(models.runtime(ModelRole::StoredPose).is_none());
        assert!(!models.stop.load(Ordering::SeqCst));
        assert!(models.shutdown.deadline().is_none());
        assert_eq!(models.failure(), Some(Exit::Runtime));
        reply(&models, &window, 48);
        close(&mut models, &clock);
        assert_eq!(models.runtime(ModelRole::Fall), Some(&fall_runtime));
        assert_eq!(models.states[0].as_ref().unwrap().failure(), None);
        assert_eq!(
            models.states[1].as_ref().unwrap().failure(),
            Some(Exit::Runtime)
        );
        assert!(models.runtime(ModelRole::Bed).is_none());
        assert!(models.runtime(ModelRole::StoredPose).is_none());
        assert_eq!(models.failure(), Some(Exit::Runtime));
        assert_eq!(
            models.failure(),
            Some(Exit::Runtime),
            "non-consuming after joins"
        );
    }
}
