//! The media thread body (design §2.3): it owns the `!Send` Pass 1
//! `MediaOwner`, reports readiness once, and hands every polled packet over
//! with `try_send`, counting what a full channel drops. Stop, a fatal status
//! or a failed call ends the loop; `release` then stops, reaps and closes
//! only when no record slot is left reserved and root permits close. A panic
//! in start or the loop also goes through `release` before it resumes.
//!
//! Owner termination is sticky on `Snapshot.failure` before readiness is
//! sent and before `release` publishes shutdown. That field is the actual
//! Rust exit, not a native diagnostic or a native-close fact. An explicit
//! stop leaves it `None`. A known `FatalAccelerator` is never replaced.

use std::io;
use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use seeon_deepstream_native::{MediaConfig, MediaError, MediaOwner, MediaPoll, MediaResult};

use super::diagnostics::Diagnostics;
use super::record_start;
use super::record_stop;
use super::release::{drain_records, release};
use super::shutdown::ShutdownControl;
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
    /// Native opening budget only; release uses the shared shutdown control.
    pub open_budget_ms: u32,
    pub shutdown: Arc<ShutdownControl>,
    /// Media-component stop only; GPU owners must remain alive for draining.
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
    let mut owner = match MediaOwner::open(&params.config, params.open_budget_ms) {
        Ok(owner) => ManuallyDrop::new(owner),
        Err(error) => {
            eprintln!("ml-worker: media open refused: {error}");
            // A refused open holds nothing, so there is nothing to release.
            let exit = match error {
                MediaError::InvalidArgument(_) => Exit::Config,
                _ => Exit::Runtime,
            };
            // Store the actual refusal before readiness. Open holds nothing,
            // so stop/finalization/closed stay false.
            params.diagnostics.update(|snapshot| {
                snapshot.open_refused = true;
                snapshot.failure = Some(exit);
            });
            ready.try_send(Err(exit)).ok();
            return;
        }
    };
    // Guard ownership immediately after open, including start and cleanup:
    // native Drop must never bypass the reserved-slot/root-permission gate.
    let mut failure = None;
    let mut reported = false;
    let served = panic::catch_unwind(AssertUnwindSafe(|| {
        let started = match owner.start() {
            Ok(status) if status.result == MediaResult::Ok => true,
            Ok(status) => {
                eprintln!("ml-worker: media start refused: {:?}", status.result);
                failure = Some(Exit::Runtime);
                false
            }
            Err(error) => {
                eprintln!("ml-worker: media start refused: {error}");
                failure = Some(Exit::Runtime);
                false
            }
        };
        // Failed start is sticky before readiness, so root cannot observe
        // stop while the cause is still absent.
        if let Some(exit) = failure {
            record_failure(params, exit);
        }
        let readiness = if started { Ok(()) } else { Err(Exit::Runtime) };
        reported = true;
        let delivered = ready.try_send(readiness).is_ok();
        if started && delivered {
            failure = serve(&mut owner, params, &sources).err();
        }
    }));
    if served.is_err() && failure.is_none() {
        failure = Some(Exit::Runtime);
    }
    if let Some(exit) = failure {
        record_failure(params, exit);
    }
    if !reported {
        ready.try_send(Err(Exit::Runtime)).ok();
    }
    // Root observes finalization diagnostics before joining; joining first
    // would block the root that must grant close permission. The terminal
    // cause is already stored, so shutdown publication cannot look clean.
    release(owner, params);
    if let Err(payload) = served {
        panic::resume_unwind(payload);
    }
}

/// Explicit stop is `Ok`. A failed pose, preview, or record poll, or a
/// failed status read, is `Runtime`. A successful status whose result is
/// fatal is `FatalAccelerator`, matching the existing root media-fatal policy.
fn serve(owner: &mut MediaOwner, params: &MediaParams, sources: &[u32]) -> Result<(), Exit> {
    let mut record_requests = record_start::Sequence::new();
    while !params.stop.load(Ordering::SeqCst) {
        execute_commands(owner, params, &mut record_requests);
        let polled = sources
            .iter()
            .enumerate()
            .try_for_each(|(camera, &source_id)| poll_poses(owner, params, camera, source_id));
        if polled.is_err() || drain_previews(owner, params).is_err() {
            return Err(Exit::Runtime);
        }
        if drain_records(owner, params).is_err() {
            return Err(Exit::Runtime);
        }
        match owner.read_status() {
            Ok(status) => {
                params.diagnostics.record_status(&status);
                if status.result == MediaResult::Fatal {
                    return Err(Exit::FatalAccelerator);
                }
            }
            Err(_) => return Err(Exit::Runtime),
        }
        params.clock.pause(POLL_INTERVAL);
    }
    Ok(())
}

/// Records the actual owner exit without replacing a known accelerator fault.
fn record_failure(params: &MediaParams, exit: Exit) {
    params.diagnostics.update(|snapshot| {
        if snapshot.failure.is_none() || exit == Exit::FatalAccelerator {
            snapshot.failure = Some(exit);
        }
    });
}

/// Executes at most `COMMAND_CAPACITY` queued commands. A reply nobody
/// waits for any more is counted; the command itself has already run.
fn execute_commands(
    owner: &mut MediaOwner,
    params: &MediaParams,
    record_requests: &mut record_start::Sequence,
) {
    for _ in 0..COMMAND_CAPACITY {
        let Ok(command) = params.commands.try_recv() else {
            return;
        };
        let delivered = match command {
            Command::RecordStart {
                source_id,
                binding,
                lookback_seconds,
                forward_seconds,
                reply,
            } => {
                let started = record_start::start(
                    owner,
                    record_requests,
                    source_id,
                    binding,
                    lookback_seconds,
                    forward_seconds,
                );
                reply.try_send(started).is_ok()
            }
            Command::RecordStop { ticket, reply } => {
                reply.try_send(record_stop::stop(owner, &ticket)).is_ok()
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::{self, TryRecvError};
    use std::time::Duration;

    use seeon_deepstream_native::{
        MediaArgument, MediaBinding, MediaConfig, MediaError, MediaOwner, SourceConfig,
    };

    use super::{COMMAND_CAPACITY, MediaParams, join, spawn};
    use crate::exit::Exit;
    use crate::media::diagnostics::{CameraCounters, Diagnostics, Snapshot};
    use crate::media::shutdown::ShutdownControl;
    use crate::msg::{POSE_PER_CAMERA, PREVIEW_CAPACITY, RECORD_CAPACITY};
    use crate::poll::poll_until;
    use crate::seam::{Clock, SystemClock};
    use crate::shutdown::ShutdownDeadline;

    /// Values sit inside adapter prevalidation ranges. Budget 0 is rejected
    /// first, before native construction, so this refusal needs no GPU.
    fn zero_budget_config() -> MediaConfig {
        MediaConfig {
            sources: vec![SourceConfig {
                source_id: 0,
                binding: MediaBinding {
                    token: 1,
                    generation: 1,
                    epoch: 1,
                },
                uri: "file:///not-a-native-open.mp4".into(),
                record_prefix: "camera".into(),
            }],
            infer_config_path: "infer.txt".into(),
            tracker_config_path: "tracker.yml".into(),
            tracker_library_path: "tracker.so".into(),
            record_directory: "records".into(),
            record_cache_seconds: 30,
            record_capacity: 4,
            mux_width: 640,
            mux_height: 360,
            mux_batch_timeout_us: 40000,
            mux_live_source: false,
            tracker_width: 960,
            tracker_height: 544,
            queue_max_buffers: 4,
            preview_enabled: false,
            max_preview_bytes: 0,
            allow_file_uris: true,
            rtsp_reconnect_interval_sec: 0,
        }
    }

    struct JoinOnDrop {
        thread: Option<std::thread::JoinHandle<()>>,
        clock: Arc<SystemClock>,
    }

    impl Drop for JoinOnDrop {
        fn drop(&mut self) {
            let Some(thread) = self.thread.take() else {
                return;
            };
            let deadline = self
                .clock
                .monotonic()
                .saturating_add(Duration::from_secs(2));
            let _ = join(thread, self.clock.as_ref(), deadline);
        }
    }

    #[test]
    fn zero_open_budget_refuses_without_native_finalization() {
        let config = zero_budget_config();
        assert!(matches!(
            MediaOwner::open(&config, 0),
            Err(MediaError::InvalidArgument(MediaArgument::ShutdownBudget))
        ));

        let clock = Arc::new(SystemClock::new());
        let shutdown = Arc::new(ShutdownControl::new(Arc::new(
            ShutdownDeadline::new(Duration::from_secs(2)).expect("shutdown budget"),
        )));
        let diagnostics = Arc::new(Diagnostics::new(config.sources.len()));
        let (pose_tx, _pose_rx) = mpsc::sync_channel(POSE_PER_CAMERA);
        let (preview_tx, _preview_rx) = mpsc::sync_channel(PREVIEW_CAPACITY);
        let (record_tx, _record_rx) = mpsc::sync_channel(RECORD_CAPACITY);
        let (_command_tx, commands) = mpsc::sync_channel(COMMAND_CAPACITY);
        let media_clock: Arc<dyn Clock> = clock.clone();
        let (thread, readiness) = spawn(MediaParams {
            config,
            open_budget_ms: 0,
            shutdown,
            stop: Arc::new(AtomicBool::new(false)),
            clock: media_clock,
            pose_tx,
            preview_tx,
            record_tx,
            commands,
            diagnostics: Arc::clone(&diagnostics),
        })
        .expect("media thread spawns");
        let mut owner = JoinOnDrop {
            thread: Some(thread),
            clock: Arc::clone(&clock),
        };

        let mut observed = None;
        let deadline = clock.monotonic().saturating_add(Duration::from_secs(2));
        poll_until(
            clock.as_ref(),
            deadline,
            "media readiness",
            || match readiness.try_recv() {
                Ok(value) => {
                    observed = Some(value);
                    true
                }
                Err(TryRecvError::Empty) => false,
                Err(TryRecvError::Disconnected) => {
                    panic!("media thread ended without a readiness report")
                }
            },
        )
        .expect("readiness arrives under the canonical poll");
        assert_eq!(observed, Some(Err(Exit::Config)));
        assert_eq!(
            diagnostics.snapshot(),
            Snapshot {
                open_refused: true,
                failure: Some(Exit::Config),
                cameras: vec![CameraCounters::default()],
                ..Snapshot::default()
            }
        );

        let thread = owner.thread.take().expect("thread still owned");
        let join_deadline = clock.monotonic().saturating_add(Duration::from_secs(2));
        join(thread, clock.as_ref(), join_deadline).expect("media thread joins");
    }

    /// Invalid source configuration preserves Config before readiness,
    /// without manufacturing native stop or close.
    #[test]
    fn empty_source_refusal_preserves_config_before_readiness() {
        let mut config = zero_budget_config();
        config.sources.clear();
        assert!(matches!(
            MediaOwner::open(&config, 1),
            Err(MediaError::InvalidArgument(MediaArgument::Sources))
        ));

        let clock = Arc::new(SystemClock::new());
        let shutdown = Arc::new(ShutdownControl::new(Arc::new(
            ShutdownDeadline::new(Duration::from_secs(2)).expect("shutdown budget"),
        )));
        let diagnostics = Arc::new(Diagnostics::new(0));
        let (pose_tx, _pose_rx) = mpsc::sync_channel(POSE_PER_CAMERA);
        let (preview_tx, _preview_rx) = mpsc::sync_channel(PREVIEW_CAPACITY);
        let (record_tx, _record_rx) = mpsc::sync_channel(RECORD_CAPACITY);
        let (_command_tx, commands) = mpsc::sync_channel(COMMAND_CAPACITY);
        let (thread, readiness) = spawn(MediaParams {
            config,
            open_budget_ms: 1,
            shutdown,
            stop: Arc::new(AtomicBool::new(false)),
            clock: Arc::clone(&clock) as Arc<dyn Clock>,
            pose_tx,
            preview_tx,
            record_tx,
            commands,
            diagnostics: Arc::clone(&diagnostics),
        })
        .expect("media thread spawns");
        let mut owner = JoinOnDrop {
            thread: Some(thread),
            clock: Arc::clone(&clock),
        };

        let mut observed = None;
        let deadline = clock.monotonic().saturating_add(Duration::from_secs(2));
        poll_until(
            clock.as_ref(),
            deadline,
            "media readiness",
            || match readiness.try_recv() {
                Ok(value) => {
                    observed = Some(value);
                    true
                }
                Err(TryRecvError::Empty) => false,
                Err(TryRecvError::Disconnected) => {
                    panic!("media thread ended without a readiness report")
                }
            },
        )
        .expect("readiness arrives under the canonical poll");
        assert_eq!(observed, Some(Err(Exit::Config)));
        let snapshot = diagnostics.snapshot();
        assert_eq!(snapshot.failure, Some(Exit::Config));
        assert!(snapshot.open_refused);
        assert!(!snapshot.finalization_started);
        assert!(!snapshot.finalization_complete);
        assert!(!snapshot.stopped);
        assert!(!snapshot.closed);
        assert!(!snapshot.close_withheld);
        assert!(snapshot.state.is_none());
        assert!(!snapshot.fatal);

        let thread = owner.thread.take().expect("thread still owned");
        join(
            thread,
            clock.as_ref(),
            clock.monotonic().saturating_add(Duration::from_secs(2)),
        )
        .expect("media thread joins");
    }

    #[test]
    fn known_accelerator_failure_is_not_replaced() {
        let diagnostics = Arc::new(Diagnostics::new(0));
        let shutdown = Arc::new(ShutdownControl::new(Arc::new(
            ShutdownDeadline::new(Duration::from_secs(2)).expect("shutdown budget"),
        )));
        let (pose_tx, _pose_rx) = mpsc::sync_channel(1);
        let (preview_tx, _preview_rx) = mpsc::sync_channel(1);
        let (record_tx, _record_rx) = mpsc::sync_channel(1);
        let (_command_tx, commands) = mpsc::sync_channel(1);
        let params = MediaParams {
            config: zero_budget_config(),
            open_budget_ms: 1,
            shutdown,
            stop: Arc::new(AtomicBool::new(true)),
            clock: Arc::new(SystemClock::new()),
            pose_tx,
            preview_tx,
            record_tx,
            commands,
            diagnostics: Arc::clone(&diagnostics),
        };
        assert_eq!(diagnostics.snapshot().failure, None);
        diagnostics.update(|snapshot| snapshot.failure = Some(Exit::FatalAccelerator));
        super::record_failure(&params, Exit::Runtime);
        super::record_failure(&params, Exit::Config);
        assert_eq!(diagnostics.snapshot().failure, Some(Exit::FatalAccelerator));
        assert!(!diagnostics.snapshot().open_refused);
        assert!(!diagnostics.snapshot().finalization_started);
        assert!(!diagnostics.snapshot().stopped);
        assert!(!diagnostics.snapshot().closed);
    }
}
