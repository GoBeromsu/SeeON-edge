//! The media thread body (design §2.3): it owns the `!Send` Pass 1
//! `MediaOwner`, reports readiness once, and hands every polled packet over
//! with `try_send`, counting what a full channel drops. Stop, a fatal status
//! or a failed call ends the loop; `release` then stops, reaps and closes
//! only when no record slot is left reserved.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use seeon_deepstream_native::{MediaConfig, MediaError, MediaOwner, MediaPoll, MediaResult};

use super::diagnostics::Diagnostics;
use super::release::{drain_records, release};
use super::{COMMAND_CAPACITY, Command};
use crate::exit::Exit;
use crate::gpu::owners::JoinError;
use crate::msg::{ONESHOT_CAPACITY, PosePacket, PreviewPacket, Readiness, RecordReceipt};
use crate::poll::{POLL_INTERVAL, poll_until};
use crate::seam::Clock;

/// Pose packets taken from one source per loop turn, so one busy camera
/// cannot starve the others, the commands or the stop flag.
const POSE_BURST: usize = 16;

pub struct MediaParams {
    /// Moved into the thread: `MediaOwner::open` copies it there.
    pub config: MediaConfig,
    /// The open budget, the native stop deadline and the record reaping
    /// budget after stop, in milliseconds.
    pub shutdown_budget_ms: u32,
    pub stop: Arc<AtomicBool>,
    pub clock: Arc<dyn Clock>,
    /// One channel for every camera, capacity `POSE_PER_CAMERA` x cameras.
    pub pose_tx: SyncSender<PosePacket>,
    /// Capacity `PREVIEW_CAPACITY`.
    pub preview_tx: SyncSender<PreviewPacket>,
    /// Capacity `RECORD_CAPACITY`.
    pub record_tx: SyncSender<RecordReceipt>,
    /// Receiver of a `sync_channel(COMMAND_CAPACITY)`.
    pub commands: Receiver<Command>,
    pub diagnostics: Arc<Diagnostics>,
}

/// Starts the media thread. Readiness is `Err(Exit::Config)` for a refused
/// configuration and `Err(Exit::Runtime)` for a failed open or start.
pub fn spawn(params: MediaParams) -> io::Result<(JoinHandle<()>, Receiver<Readiness>)> {
    let (ready, readiness) = mpsc::sync_channel(ONESHOT_CAPACITY);
    let thread = thread::Builder::new()
        .name("media".to_owned())
        .spawn(move || run(&params, &ready))?;
    Ok((thread, readiness))
}

/// Waits until the media thread has ended, up to the absolute monotonic
/// `deadline`.
pub fn join(
    thread: JoinHandle<()>,
    clock: &dyn Clock,
    deadline: Duration,
) -> Result<(), JoinError> {
    poll_until(clock, deadline, "media owner exit", || thread.is_finished())
        .map_err(JoinError::Timeout)?;
    thread.join().map_err(|_| JoinError::Panicked)
}

fn run(params: &MediaParams, ready: &SyncSender<Readiness>) {
    let sources: Vec<u32> = params
        .config
        .sources
        .iter()
        .map(|source| source.source_id)
        .collect();
    let mut owner = match MediaOwner::open(&params.config, params.shutdown_budget_ms) {
        Ok(owner) => owner,
        Err(error) => {
            // A refused open holds nothing, so there is nothing to release.
            let exit = match error {
                MediaError::InvalidArgument(_) => Exit::Config,
                _ => Exit::Runtime,
            };
            ready.try_send(Err(exit)).ok();
            return;
        }
    };
    let started = owner
        .start()
        .is_ok_and(|status| status.result == MediaResult::Ok);
    let readiness = if started { Ok(()) } else { Err(Exit::Runtime) };
    let delivered = ready.try_send(readiness).is_ok();
    if started && delivered {
        serve(&mut owner, params, &sources);
    }
    release(owner, params);
}

/// Runs until the stop flag, a fatal status or a failed owner call.
fn serve(owner: &mut MediaOwner, params: &MediaParams, sources: &[u32]) {
    while !params.stop.load(Ordering::SeqCst) {
        execute_commands(owner, params);
        let polled = sources
            .iter()
            .enumerate()
            .try_for_each(|(camera, &source_id)| poll_poses(owner, params, camera, source_id));
        if polled.is_err() || drain_previews(owner, params).is_err() {
            return;
        }
        if drain_records(owner, params).is_err() {
            return;
        }
        match owner.read_status() {
            Ok(status) => {
                params.diagnostics.record_status(&status);
                if status.result == MediaResult::Fatal {
                    return;
                }
            }
            Err(_) => return,
        }
        params.clock.pause(POLL_INTERVAL);
    }
}

/// Executes at most `COMMAND_CAPACITY` queued commands. A reply nobody
/// waits for any more is counted; the command itself has already run.
fn execute_commands(owner: &mut MediaOwner, params: &MediaParams) {
    for _ in 0..COMMAND_CAPACITY {
        let Ok(command) = params.commands.try_recv() else {
            return;
        };
        let delivered = match command {
            Command::RecordStart {
                source_id,
                binding,
                request_id,
                lookback_seconds,
                forward_seconds,
                reply,
            } => {
                let started = owner.record_start(
                    source_id,
                    binding,
                    request_id,
                    lookback_seconds,
                    forward_seconds,
                );
                reply.try_send(started).is_ok()
            }
            Command::RecordStop { ticket, reply } => {
                reply.try_send(owner.record_stop(&ticket)).is_ok()
            }
            Command::Preview {
                source_id,
                binding,
                request_id,
                draw_objects,
                timeout_ms,
                reply,
            } => {
                let requested =
                    owner.request_preview(source_id, binding, request_id, draw_objects, timeout_ms);
                reply.try_send(requested).is_ok()
            }
        };
        if !delivered {
            params
                .diagnostics
                .update(|snapshot| snapshot.replies_dropped += 1);
        }
    }
}

fn poll_poses(
    owner: &mut MediaOwner,
    params: &MediaParams,
    camera: usize,
    source_id: u32,
) -> Result<(), MediaError> {
    for _ in 0..POSE_BURST {
        let MediaPoll::Ready(packet) = owner.poll_pose(source_id)? else {
            return Ok(());
        };
        if params.pose_tx.try_send(packet).is_err() {
            params.diagnostics.count_handoff_drop(camera);
        }
    }
    Ok(())
}

fn drain_previews(owner: &mut MediaOwner, params: &MediaParams) -> Result<(), MediaError> {
    while let MediaPoll::Ready(packet) = owner.poll_preview()? {
        if params.preview_tx.try_send(packet).is_err() {
            params
                .diagnostics
                .update(|snapshot| snapshot.previews_dropped += 1);
        }
    }
    Ok(())
}
