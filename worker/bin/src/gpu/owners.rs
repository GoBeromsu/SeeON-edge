//! Thread-confined GPU model owners (design §2.3, §2.4 step 4).
//!
//! Each spawn opens its model inside its own thread, runs exactly one warm-up,
//! checks the receipt, reports readiness once on a `sync_channel(1)`, then
//! serves requests until its stop flag is set or every request sender is gone.
//! A model is opened only when its spawn is called.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use seeon_deepstream_native::GpuModel;
use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, ZERO_ROW};
use seeon_worker_runtime::bed_gpu::BedGpu;
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};
use seeon_worker_runtime::fall_gpu::FallGpu;
use seeon_worker_runtime::stored_pose::StoredPoseGpu;

use crate::exit::Exit;
use crate::msg::{
    BedOutput, BedRequest, FALL_REQUEST_CAPACITY, FALL_RESPONSE_CAPACITY, FallRequest,
    FallResponse, GPU_REQUEST_CAPACITY, ONESHOT_CAPACITY, Readiness, StoredPoseRequest,
};
use crate::poll::{POLL_INTERVAL, Timeout, poll_until};
use crate::seam::Clock;

/// Warm-up frame of the bed and stored-pose owners: one black 640x360 RGB
/// frame, as the Python warm-up.
const WARMUP_WIDTH: i64 = 640;
const WARMUP_HEIGHT: i64 = 360;
const WARMUP_RGB_BYTES: usize = 640 * 360 * 3;

/// A spawned owner thread, its one readiness report and its request queue.
pub struct Owner<R> {
    pub thread: JoinHandle<()>,
    pub readiness: Receiver<Readiness>,
    pub requests: SyncSender<R>,
}

/// Why an owner thread did not end cleanly within its join deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JoinError {
    Timeout(Timeout),
    Panicked,
}

/// gpu-fall: scores pose-bbox56 windows; answers on the returned receiver.
pub fn spawn_fall(
    engine: PathBuf,
    device_ordinal: i32,
    digest: EngineDigest,
    stop: Arc<AtomicBool>,
) -> io::Result<(Owner<FallRequest>, Receiver<FallResponse>)> {
    let (requests, queue) = mpsc::sync_channel::<FallRequest>(FALL_REQUEST_CAPACITY);
    let (responses, answers) = mpsc::sync_channel(FALL_RESPONSE_CAPACITY);
    let (ready, readiness) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let thread = thread::Builder::new()
        .name("gpu-fall".to_owned())
        .spawn(move || {
            let Some(mut fall) = report(ready, warm_fall(&engine, device_ordinal, digest)) else {
                return;
            };
            serve(&stop, &queue, |request| {
                let response = FallResponse {
                    frame: request.frame,
                    track_id: request.track_id,
                    score: fall.score(request.window.as_slice()),
                };
                // A full response queue drops this score; a gone policy ends the owner.
                !matches!(
                    responses.try_send(response),
                    Err(TrySendError::Disconnected(_))
                )
            });
        })?;
    Ok((
        Owner {
            thread,
            readiness,
            requests,
        },
        answers,
    ))
}

/// gpu-bed: answers each frame with an owned copy of the raw bed outputs.
pub fn spawn_bed(
    engine: PathBuf,
    device_ordinal: i32,
    digest: EngineDigest,
    stop: Arc<AtomicBool>,
) -> io::Result<Owner<BedRequest>> {
    let (requests, queue) = mpsc::sync_channel::<BedRequest>(GPU_REQUEST_CAPACITY);
    let (ready, readiness) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let thread = thread::Builder::new()
        .name("gpu-bed".to_owned())
        .spawn(move || {
            let Some(mut bed) = report(ready, warm_bed(&engine, device_ordinal, digest)) else {
                return;
            };
            serve(&stop, &queue, |request| {
                let output = bed
                    .infer(&request.rgb, request.width, request.height)
                    .map(|raw| BedOutput {
                        detections: raw.detections.to_vec(),
                        protos: raw.protos.to_vec(),
                        letterbox: raw.letterbox,
                        evidence: raw.evidence,
                    });
                // The requester may have given up; its reply is then dropped.
                let _ = request.reply.try_send(output);
                true
            });
        })?;
    Ok(Owner {
        thread,
        readiness,
        requests,
    })
}

/// gpu-stored-pose: answers each frame with the person boxes at `threshold`.
pub fn spawn_stored_pose(
    engine: PathBuf,
    device_ordinal: i32,
    digest: EngineDigest,
    threshold: f64,
    stop: Arc<AtomicBool>,
) -> io::Result<Owner<StoredPoseRequest>> {
    let (requests, queue) = mpsc::sync_channel::<StoredPoseRequest>(GPU_REQUEST_CAPACITY);
    let (ready, readiness) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let thread = thread::Builder::new()
        .name("gpu-stored-pose".to_owned())
        .spawn(move || {
            let warm = warm_stored_pose(&engine, device_ordinal, digest, threshold);
            let Some(mut pose) = report(ready, warm) else {
                return;
            };
            serve(&stop, &queue, |request| {
                let boxes = pose.infer(&request.rgb, request.width, request.height);
                // The requester may have given up; its reply is then dropped.
                let _ = request.reply.try_send(boxes);
                true
            });
        })?;
    Ok(Owner {
        thread,
        readiness,
        requests,
    })
}

/// Waits until `thread` has ended, up to the absolute monotonic `deadline`.
pub fn join(
    thread: JoinHandle<()>,
    clock: &dyn Clock,
    deadline: Duration,
) -> Result<(), JoinError> {
    poll_until(clock, deadline, "gpu owner exit", || thread.is_finished())
        .map_err(JoinError::Timeout)?;
    thread.join().map_err(|_| JoinError::Panicked)
}

fn warm_fall(engine: &Path, device_ordinal: i32, digest: EngineDigest) -> Option<FallGpu> {
    let model = GpuModel::open(engine, device_ordinal).ok()?;
    let mut fall = FallGpu::new(model, device_ordinal, digest);
    let warm = fall.score(&[ZERO_ROW; FALL_WINDOW_FRAMES]).ok()?;
    receipt_matches(&warm.evidence, device_ordinal, digest).then_some(fall)
}

fn warm_bed(engine: &Path, device_ordinal: i32, digest: EngineDigest) -> Option<BedGpu> {
    let model = GpuModel::open(engine, device_ordinal).ok()?;
    let mut bed = BedGpu::new(model, device_ordinal, digest);
    let frame = vec![0u8; WARMUP_RGB_BYTES];
    let warm = bed.infer(&frame, WARMUP_WIDTH, WARMUP_HEIGHT).ok()?;
    let matches = receipt_matches(&warm.evidence, device_ordinal, digest);
    matches.then_some(bed)
}

/// `StoredPoseGpu` returns no receipt, so its one warm-up call is bracketed
/// by the model counters instead.
fn warm_stored_pose(
    engine: &Path,
    device_ordinal: i32,
    digest: EngineDigest,
    threshold: f64,
) -> Option<StoredPoseGpu> {
    let model = GpuModel::open(engine, device_ordinal).ok()?;
    let mut pose = StoredPoseGpu::new(model, threshold).ok()?;
    let before = pose.metrics().ok()?;
    let frame = vec![0u8; WARMUP_RGB_BYTES];
    pose.infer(&frame, WARMUP_WIDTH, WARMUP_HEIGHT).ok()?;
    let after = pose.metrics().ok()?;
    AcceleratorEvidence::from_delta(&before, &after, device_ordinal, digest, Precision::Fp32)
        .ok()
        .map(|_receipt| pose)
}

fn receipt_matches(
    evidence: &AcceleratorEvidence,
    device_ordinal: i32,
    digest: EngineDigest,
) -> bool {
    evidence.device_ordinal() == device_ordinal && evidence.engine_sha256() == digest
}

/// Sends the one readiness report and closes the channel. A warmed owner is
/// served only when its report was delivered.
fn report<T>(ready: SyncSender<Readiness>, owner: Option<T>) -> Option<T> {
    let readiness = match owner {
        Some(_) => Ok(()),
        None => Err(Exit::FatalAccelerator),
    };
    ready.try_send(readiness).ok()?;
    owner
}

/// Serves until the stop flag is set or every request sender is gone;
/// `handle` returns false once its answer side is gone for good.
fn serve<R>(stop: &AtomicBool, queue: &Receiver<R>, mut handle: impl FnMut(R) -> bool) {
    while !stop.load(Ordering::SeqCst) {
        match queue.recv_timeout(POLL_INTERVAL) {
            Ok(request) => {
                if !handle(request) {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}
