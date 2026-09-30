//! GPU model owners against real TensorRT engines (design §2.3, §2.4 step 4):
//! readiness after one verified warm-up, refusal on an unopenable engine, and
//! a stop flag that ends the owner before its join deadline.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use seeon_deepstream_native::FrameIdentity;
use seeon_ml_worker::exit::Exit;
use seeon_ml_worker::gpu::owners::{self, Owner};
use seeon_ml_worker::msg::{BedRequest, FallRequest, ONESHOT_CAPACITY, StoredPoseRequest};
use seeon_ml_worker::seam::{Clock, SystemClock};
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, ZERO_ROW};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest};
use sha2::{Digest, Sha256};

static GPU: Mutex<()> = Mutex::new(());

const DEVICE: i32 = 0;
/// Product default `person_threshold` (`worker/runtime/worker.py` L975).
const PERSON_THRESHOLD: f64 = 0.25;
/// Engine deserialisation plus one warm-up; an upper bound, not a pause.
const READY_WAIT: Duration = Duration::from_secs(120);
const REPLY_WAIT: Duration = Duration::from_secs(30);
const JOIN_BUDGET: Duration = Duration::from_secs(5);
const FRAME_WIDTH: i64 = 640;
const FRAME_HEIGHT: i64 = 360;

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn env_path(var: &str) -> PathBuf {
    let value = std::env::var_os(var).unwrap_or_else(|| panic!("{var} is required"));
    let text = value
        .to_str()
        .unwrap_or_else(|| panic!("{var} must be UTF-8"));
    assert!(
        !text.trim().is_empty() && !text.contains('\0'),
        "{var} must be a nonblank path"
    );
    PathBuf::from(value)
}

fn digest_of(path: &Path) -> EngineDigest {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    EngineDigest::new(Sha256::digest(bytes).into())
}

fn black_frame() -> Vec<u8> {
    vec![0; 640 * 360 * 3]
}

fn fall_request(sequence: u64) -> FallRequest {
    FallRequest {
        frame: FrameIdentity {
            sequence,
            ..FrameIdentity::default()
        },
        track_id: 7,
        window: Box::new([ZERO_ROW; FALL_WINDOW_FRAMES]),
    }
}

/// `call_seq` counts cumulative attempts: the warm-up plus this call.
fn assert_first_served_receipt(evidence: &AcceleratorEvidence, digest: EngineDigest) {
    assert_eq!(evidence.call_seq(), 2, "one warm-up, then this call");
    assert_eq!(evidence.engine_sha256(), digest);
    assert_eq!(evidence.device_ordinal(), DEVICE);
}

fn join_before_deadline(thread: std::thread::JoinHandle<()>) {
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + JOIN_BUDGET;
    assert_eq!(owners::join(thread, &clock, deadline), Ok(()));
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_FALL_ENGINE, SEEON_TEST_BED_ENGINE and SEEON_TEST_STORED_POSE_ENGINE"]
fn each_owner_is_ready_once_and_its_first_reply_follows_one_warmup() {
    let _gpu = gpu_lock();
    let fall_engine = env_path("SEEON_TEST_FALL_ENGINE");
    let bed_engine = env_path("SEEON_TEST_BED_ENGINE");
    let pose_engine = env_path("SEEON_TEST_STORED_POSE_ENGINE");
    let (fall_digest, bed_digest) = (digest_of(&fall_engine), digest_of(&bed_engine));
    let pose_digest = digest_of(&pose_engine);
    let stop = Arc::new(AtomicBool::new(false));

    let (fall, answers) =
        owners::spawn_fall(fall_engine, DEVICE, fall_digest, stop.clone()).expect("spawn gpu-fall");
    assert_eq!(fall.readiness.recv_timeout(READY_WAIT), Ok(Ok(())));
    let bed =
        owners::spawn_bed(bed_engine, DEVICE, bed_digest, stop.clone()).expect("spawn gpu-bed");
    assert_eq!(bed.readiness.recv_timeout(READY_WAIT), Ok(Ok(())));
    let pose = owners::spawn_stored_pose(
        pose_engine,
        DEVICE,
        pose_digest,
        PERSON_THRESHOLD,
        stop.clone(),
    )
    .expect("spawn gpu-stored-pose");
    assert_eq!(pose.readiness.recv_timeout(READY_WAIT), Ok(Ok(())));

    fall.requests
        .send(fall_request(41))
        .expect("fall request queued");
    let response = answers.recv_timeout(REPLY_WAIT).expect("fall reply");
    assert_eq!(response.frame.sequence, 41);
    assert_eq!(response.track_id, 7);
    let score = response.score.expect("fall score");
    assert_first_served_receipt(&score.evidence, fall_digest);

    let (reply, replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let request = BedRequest {
        rgb: black_frame(),
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        reply,
    };
    bed.requests.send(request).expect("bed request queued");
    let output = replies
        .recv_timeout(REPLY_WAIT)
        .expect("bed reply")
        .expect("bed output");
    assert_first_served_receipt(&output.evidence, bed_digest);

    let (reply, replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let request = StoredPoseRequest {
        rgb: black_frame(),
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        reply,
    };
    pose.requests
        .send(request)
        .expect("stored-pose request queued");
    replies
        .recv_timeout(REPLY_WAIT)
        .expect("stored-pose reply")
        .expect("stored-pose boxes");

    stop.store(true, Ordering::SeqCst);
    for (thread, readiness) in [
        (fall.thread, fall.readiness),
        (bed.thread, bed.readiness),
        (pose.thread, pose.readiness),
    ] {
        join_before_deadline(thread);
        assert_eq!(
            readiness.try_recv(),
            Err(TryRecvError::Disconnected),
            "no second readiness"
        );
    }
}

fn assert_refused<R>(what: &str, owner: Owner<R>, request: R) {
    assert_eq!(
        owner.readiness.recv_timeout(READY_WAIT),
        Ok(Err(Exit::FatalAccelerator)),
        "{what}"
    );
    join_before_deadline(owner.thread);
    let sent = owner.requests.try_send(request);
    assert!(
        matches!(sent, Err(TrySendError::Disconnected(_))),
        "{what} serves nothing"
    );
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_FALL_ENGINE, SEEON_TEST_BED_ENGINE and SEEON_TEST_STORED_POSE_ENGINE"]
fn an_unopenable_engine_is_fatal_and_nothing_is_served() {
    let _gpu = gpu_lock();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gpu_owners_open_failure");
    match fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => panic!("clear {}: {error}", dir.display()),
    }
    fs::create_dir_all(&dir).expect("temp dir");
    let garbage = dir.join("garbage.engine");
    fs::write(&garbage, [0x5a_u8; 16]).expect("garbage engine");
    let digest = digest_of(&garbage);
    let stop = Arc::new(AtomicBool::new(false));

    for engine in [dir.join("missing.engine"), garbage] {
        let (fall, _answers) = owners::spawn_fall(engine.clone(), DEVICE, digest, stop.clone())
            .expect("spawn gpu-fall");
        assert_refused("gpu-fall", fall, fall_request(1));

        let bed =
            owners::spawn_bed(engine.clone(), DEVICE, digest, stop.clone()).expect("spawn gpu-bed");
        let (reply, _replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
        let request = BedRequest {
            rgb: black_frame(),
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
            reply,
        };
        assert_refused("gpu-bed", bed, request);

        let pose =
            owners::spawn_stored_pose(engine, DEVICE, digest, PERSON_THRESHOLD, stop.clone())
                .expect("spawn gpu-stored-pose");
        let (reply, _replies) = mpsc::sync_channel(ONESHOT_CAPACITY);
        let request = StoredPoseRequest {
            rgb: black_frame(),
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
            reply,
        };
        assert_refused("gpu-stored-pose", pose, request);
    }
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_FALL_ENGINE, SEEON_TEST_BED_ENGINE and SEEON_TEST_STORED_POSE_ENGINE"]
fn the_stop_flag_ends_a_ready_owner_before_its_join_deadline() {
    let _gpu = gpu_lock();
    let engine = env_path("SEEON_TEST_FALL_ENGINE");
    let digest = digest_of(&engine);
    let stop = Arc::new(AtomicBool::new(false));

    let (fall, _answers) =
        owners::spawn_fall(engine, DEVICE, digest, stop.clone()).expect("spawn gpu-fall");
    assert_eq!(fall.readiness.recv_timeout(READY_WAIT), Ok(Ok(())));
    // The request side stays open, so only the flag can end the loop.
    let requests = fall.requests;

    stop.store(true, Ordering::SeqCst);
    join_before_deadline(fall.thread);
    drop(requests);
}
