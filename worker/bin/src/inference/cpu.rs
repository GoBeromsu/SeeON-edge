//! Explicit ORT CPU actors. Captured model bytes are opened, warmed, served and
//! dropped on the same thread. No engine fallback or accelerator receipt exists.

use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;

use seeon_worker::pose_bbox56::{FALL_WINDOW_FRAMES, ZERO_ROW};
use seeon_worker_runtime::cpu::bed::{BedCpu, BedCpuError};
use seeon_worker_runtime::cpu::fall::{FallCpu, FallCpuError};
use seeon_worker_runtime::cpu::stored_pose::{StoredPoseCpu, StoredPoseCpuError};
use seeon_worker_runtime::cpu::{Info, Model, Threads};

use super::{Owner, Runtime, State};
use crate::exit::Exit;
use crate::msg::{
    BedOutput, BedRequest, FALL_REQUEST_CAPACITY, FALL_RESPONSE_CAPACITY, FallRequest,
    FallResponse, FallScore, GPU_REQUEST_CAPACITY, ONESHOT_CAPACITY, Readiness, StoredPoseRequest,
};
use crate::poll::POLL_INTERVAL;

/// Composition supplies admitted immutable bytes and the image-owned ORT library.
/// This is not an environment/configuration surface and does not reopen ONNX paths.
pub struct CapturedModel {
    pub runtime_library: PathBuf,
    pub onnx: Arc<[u8]>,
}

pub fn spawn_fall(
    model: CapturedModel,
    stop: Arc<AtomicBool>,
) -> io::Result<(Owner<FallRequest>, Receiver<FallResponse>)> {
    let (responses, answers) = mpsc::sync_channel(FALL_RESPONSE_CAPACITY);
    let owner = launch(
        "cpu-fall",
        FALL_REQUEST_CAPACITY,
        move |queue, ready, state| {
            let warmed = warm(model, Threads::Default, |model| {
                let mut fall = FallCpu::new(model);
                fall.score(&[ZERO_ROW; FALL_WINDOW_FRAMES])
                    .map_err(|_| Exit::Runtime)?;
                Ok(fall)
            });
            let Some(mut fall) = report(ready, warmed, &state) else {
                return;
            };
            serve(&stop, &queue, |request: FallRequest| {
                let result = fall.score(request.window.as_slice());
                let fatal = matches!(result, Err(error) if error != FallCpuError::Window);
                latch(fatal, &state, &stop);
                let response = FallResponse {
                    frame: request.frame,
                    track_id: request.track_id,
                    score: result.map(FallScore::Cpu).map_err(Into::into),
                };
                let connected = !matches!(
                    responses.try_send(response),
                    Err(TrySendError::Disconnected(_))
                );
                connected && !fatal
            });
        },
    )?;
    Ok((owner, answers))
}

pub fn spawn_bed(model: CapturedModel, stop: Arc<AtomicBool>) -> io::Result<Owner<BedRequest>> {
    launch(
        "cpu-bed",
        GPU_REQUEST_CAPACITY,
        move |queue, ready, state| {
            let warmed = warm(model, Threads::Single, |model| {
                let mut bed = BedCpu::new(model);
                bed.infer(&vec![0; 640 * 360 * 3], 640, 360)
                    .map_err(|_| Exit::Runtime)?;
                Ok(bed)
            });
            let Some(mut bed) = report(ready, warmed, &state) else {
                return;
            };
            serve(&stop, &queue, |request: BedRequest| {
                let result = bed.infer(&request.rgb, request.width, request.height);
                let fatal = matches!(result, Err(error) if !matches!(error, BedCpuError::Input(_)));
                latch(fatal, &state, &stop);
                let output = result
                    .map(|raw| BedOutput {
                        detections: raw.detections.to_vec(),
                        protos: raw.protos.to_vec(),
                        letterbox: raw.letterbox,
                        evidence: None,
                    })
                    .map_err(Into::into);
                let _ = request.reply.try_send(output);
                !fatal
            });
        },
    )
}

pub fn spawn_stored_pose(
    model: CapturedModel,
    threshold: f64,
    stop: Arc<AtomicBool>,
) -> io::Result<Owner<StoredPoseRequest>> {
    launch(
        "cpu-stored-pose",
        GPU_REQUEST_CAPACITY,
        move |queue, ready, state| {
            let warmed = warm(model, Threads::Single, |model| {
                let mut pose = StoredPoseCpu::new(model, threshold).map_err(|_| Exit::Config)?;
                pose.infer(&vec![0; 640 * 360 * 3], 640, 360)
                    .map_err(|_| Exit::Runtime)?;
                Ok(pose)
            });
            let Some(mut pose) = report(ready, warmed, &state) else {
                return;
            };
            serve(&stop, &queue, |request: StoredPoseRequest| {
                let result = pose.infer(&request.rgb, request.width, request.height);
                let fatal =
                    matches!(result, Err(error) if !matches!(error, StoredPoseCpuError::Input(_)));
                latch(fatal, &state, &stop);
                let _ = request.reply.try_send(result.map_err(Into::into));
                !fatal
            });
        },
    )
}

fn launch<R: Send + 'static>(
    name: &str,
    capacity: usize,
    run: impl FnOnce(Receiver<R>, SyncSender<Readiness>, Arc<State>) + Send + 'static,
) -> io::Result<Owner<R>> {
    let (requests, queue) = mpsc::sync_channel(capacity);
    let (ready, readiness) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let state = Arc::new(State::default());
    let actor_state = Arc::clone(&state);
    let thread = thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || run(queue, ready, actor_state))?;
    Ok(Owner {
        thread,
        readiness,
        requests,
        state,
    })
}

fn warm<T>(
    model: CapturedModel,
    threads: Threads,
    warmup: impl FnOnce(Model) -> Result<T, Exit>,
) -> Result<(T, Info), Exit> {
    let model =
        Model::open(&model.runtime_library, &model.onnx, threads).map_err(|_| Exit::Runtime)?;
    let info = model.info().clone();
    Ok((warmup(model)?, info))
}

fn report<T>(
    ready: SyncSender<Readiness>,
    warmed: Result<(T, Info), Exit>,
    state: &State,
) -> Option<T> {
    match warmed {
        Ok((owner, info)) => {
            if !state.started(Runtime::OnnxRuntimeCpu(info)) {
                return None;
            }
            ready.try_send(Ok(())).ok()?;
            Some(owner)
        }
        Err(exit) => {
            state.fail(exit);
            let _ = ready.try_send(Err(exit));
            None
        }
    }
}

fn latch(fatal: bool, state: &State, stop: &AtomicBool) {
    if fatal {
        state.fail(Exit::Runtime);
        stop.store(true, Ordering::SeqCst);
    }
}

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
