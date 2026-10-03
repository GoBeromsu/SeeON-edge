//! Shared 25s shutdown. Media stop and finalize precede policy drain, delivery
//! flush and GPU joins. Native close is granted only after those joins succeed.
//! A live media handle is retained on timeout until this shutdown call returns.
//! `!closed` exits 1 and retains the lease until process exit.

use std::sync::Arc;
use std::time::Duration;

use crate::exit::Exit;
use crate::poll::poll_until;
use crate::run::Booted;
use crate::run::boot::startup::cutoff;
use crate::run::models::CleanupError;
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;

use super::output::Session;
use super::runtime::{self, RunOutcome};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunExit {
    Clean,
    Runtime,
    Config,
    Fatal,
}

impl RunExit {
    pub const fn exit(self) -> Exit {
        match self {
            Self::Clean => Exit::CleanShutdown,
            Self::Runtime => Exit::Runtime,
            Self::Config => Exit::Config,
            Self::Fatal => Exit::FatalAccelerator,
        }
    }
}

pub fn shutdown(
    mut session: Session,
    mut booted: Booted,
    clock: &dyn Clock,
    deadline: &Arc<ShutdownDeadline>,
    outcome: RunOutcome,
    mut sink: runtime::LiveSink,
) -> RunExit {
    // Even an unrepresentable deadline must quiesce producers and request
    // media stop. Failure still withholds native close and retains the lease.
    session.request_media_stop();
    if deadline.request_at(clock.monotonic()).is_err() {
        booted.models.retain_lease_until_process_exit();
        return unsafe_exit(&outcome, owner_failure(&session, &booted));
    }
    let finalized = wait_finalized(&session, clock, deadline);
    let policy_drained = drain_policy(&mut session, clock, &mut sink);
    let outputs = stop_outputs(
        session.exporter.as_mut(),
        [session.heartbeat.as_mut(), session.status.as_mut()],
        session.sender.as_mut(),
        clock,
        deadline,
    );
    let gpu_errors = booted.models.close(clock);
    let closed = close_after_gpu_join(&gpu_errors, || {
        grant_and_join(&mut session, clock, deadline)
    });
    let overrun = !finalized
        || !policy_drained
        || outputs.is_err()
        || cutoff(deadline, clock).is_err()
        || !gpu_errors.is_empty();
    if !closed || overrun {
        booted.models.retain_lease_until_process_exit();
        return unsafe_exit(&outcome, owner_failure(&session, &booted));
    }
    // The owner can fail after the initial stop observation. Read its final
    // cause after the join; clean native release does not erase that failure.
    terminal_exit(&outcome, owner_failure(&session, &booted))
}

pub(super) fn drain_policy(
    session: &mut Session,
    clock: &dyn Clock,
    sink: &mut runtime::LiveSink,
) -> bool {
    let flushed = session.pump.flush(sink);
    let delivered = runtime::apply_sink(session, clock, sink);
    let recorded = session.publications.drain_records(clock);
    // Finalization precedes this drain. Late or previously queued contributors
    // cannot be declared complete merely because no further receipt arrived.
    let recordings_complete = session.publications.recordings_complete();
    if !recordings_complete {
        eprintln!("ml-worker: shutdown refused: recording attribution remains unpublished");
    }
    flushed.is_ok() && delivered.is_ok() && recorded.is_ok() && recordings_complete
}

fn wait_finalized(session: &Session, clock: &dyn Clock, deadline: &ShutdownDeadline) -> bool {
    wait_with_deadline(clock, deadline, "media finalization", || {
        session.media.as_ref().is_none_or(|media| {
            let state = media.diagnostics.snapshot();
            state.open_refused || state.finalization_complete
        })
    })
    .is_some()
}

fn wait_with_deadline(
    clock: &dyn Clock,
    deadline: &ShutdownDeadline,
    what: &'static str,
    mut finished: impl FnMut() -> bool,
) -> Option<Duration> {
    let upper_bound = cutoff(deadline, clock).ok()?;
    let mut expired = false;
    poll_until(clock, upper_bound, what, || {
        expired = cutoff(deadline, clock).is_err();
        expired || finished()
    })
    .ok()?;
    if expired {
        return None;
    }
    cutoff(deadline, clock).ok()
}

fn close_after_gpu_join(errors: &[CleanupError], close: impl FnOnce() -> bool) -> bool {
    // A failed GPU join must not even invoke the native-close permission path.
    errors.is_empty() && close()
}

fn grant_and_join(session: &mut Session, clock: &dyn Clock, deadline: &ShutdownDeadline) -> bool {
    if cutoff(deadline, clock).is_err() {
        return false;
    }
    let Some(media) = session.media.as_mut() else {
        return true;
    };
    // A refused open has no native owner to close, but still needs its actual
    // thread join. Otherwise ShutdownControl rereads after the time sample.
    if !media.diagnostics.snapshot().open_refused
        && media.shutdown.permit_close(clock.monotonic()).is_err()
    {
        return false;
    }
    if !join_media(&mut media.thread, clock, deadline) {
        return false;
    }
    closed_after_join(session)
}

fn join_media(
    thread: &mut Option<std::thread::JoinHandle<()>>,
    clock: &dyn Clock,
    deadline: &ShutdownDeadline,
) -> bool {
    let Some(handle) = thread.as_ref() else {
        // This one-shot shutdown requires ownership of the media join result;
        // absence is not evidence that native teardown completed successfully.
        return false;
    };
    if wait_with_deadline(clock, deadline, "media owner exit", || handle.is_finished()).is_none() {
        return false;
    }
    // As with GPU cleanup, take only an already-finished native owner. Do not
    // enter a second deadline wait through the consuming low-level join.
    let joined = thread.take().is_some_and(|handle| handle.join().is_ok());
    joined && cutoff(deadline, clock).is_ok()
}

fn closed_after_join(session: &Session) -> bool {
    session.media.as_ref().is_none_or(|media| {
        let state = media.diagnostics.snapshot();
        state.open_refused || state.closed
    })
}

fn stop_outputs(
    exporter: Option<&mut crate::run::exporter::Handle>,
    mut telemetry: [Option<&mut crate::telemetry::LoopHandle>; 2],
    sender: Option<&mut crate::run::delivery::Handle>,
    clock: &dyn Clock,
    deadline: &ShutdownDeadline,
) -> Result<(), super::output::OutputError> {
    // Every owner must receive stop even when the first join fails.
    let exporter_stop = exporter.as_ref().map(|exporter| exporter.request_stop());
    for handle in telemetry.iter_mut().flatten() {
        handle.request_stop();
    }
    if let Some(sender) = sender.as_ref() {
        sender.request_stop();
    }
    let mut error = None;
    if let Some(exporter) = exporter {
        let joined = wait_with_deadline(clock, deadline, "record exporter", || {
            exporter.is_finished()
        })
        .ok_or(crate::run::exporter::JoinError::Timeout)
        .and_then(|limit| exporter.join(clock, limit));
        match joined {
            Ok(report) if report.finished && !report.pending => {}
            Ok(report) => {
                eprintln!(
                    "ml-worker: record exporter shutdown incomplete finished={} pending={} failures={}",
                    report.finished,
                    report.pending,
                    report.failures.len()
                );
                error = Some(super::output::OutputError::Records);
            }
            Err(reason) => {
                eprintln!(
                    "ml-worker: record exporter shutdown failed reason={reason:?} idle_wait_bound={:?}",
                    exporter_stop.map(|stop| stop.idle_wait_bound)
                );
                error = Some(super::output::OutputError::Records);
            }
        }
    }
    if let Some(sender) = sender {
        let joined =
            wait_with_deadline(clock, deadline, "delivery sender", || sender.is_finished())
                .ok_or(crate::run::delivery::JoinError::NotFinished)
                .and_then(|_| sender.join());
        if let Err(reason) = joined {
            eprintln!("ml-worker: delivery sender shutdown failed reason={reason:?}");
            if error.is_none() {
                error = Some(super::output::OutputError::Delivery);
            }
        }
    }
    for (name, handle) in ["heartbeat", "status"].into_iter().zip(telemetry) {
        let Some(handle) = handle else { continue };
        let joined = wait_with_deadline(clock, deadline, "telemetry loop", || !handle.is_alive())
            .ok_or(crate::telemetry::JoinError::Timeout)
            .and_then(|limit| handle.join(clock, limit));
        if let Err(reason) = joined {
            eprintln!("ml-worker: telemetry shutdown failed owner={name} reason={reason:?}");
            if error.is_none() {
                error = Some(super::output::OutputError::Join);
            }
        }
    }
    if cutoff(deadline, clock).is_err() && error.is_none() {
        eprintln!("ml-worker: output shutdown exceeded the shared deadline");
        error = Some(super::output::OutputError::Join);
    }
    error.map_or(Ok(()), Err)
}

fn owner_failure(session: &Session, booted: &Booted) -> Option<Exit> {
    let media = session
        .media
        .as_ref()
        .and_then(|media| media.diagnostics.snapshot().failure);
    if media == Some(Exit::FatalAccelerator) {
        media
    } else {
        booted.models.failure().or(media)
    }
}

fn terminal_exit(outcome: &RunOutcome, media: Option<Exit>) -> RunExit {
    let primary = match outcome {
        RunOutcome::Failed(error) => error.exit(),
        RunOutcome::Restart(_) | RunOutcome::Stopped => Exit::CleanShutdown,
    };
    if primary == Exit::FatalAccelerator || media == Some(Exit::FatalAccelerator) {
        return RunExit::Fatal;
    }
    let exit = if primary == Exit::CleanShutdown {
        media.unwrap_or(primary)
    } else {
        primary
    };
    match exit {
        Exit::CleanShutdown => RunExit::Clean,
        Exit::Config => RunExit::Config,
        _ => RunExit::Runtime,
    }
}

fn unsafe_exit(outcome: &RunOutcome, media: Option<Exit>) -> RunExit {
    if terminal_exit(outcome, media) == RunExit::Fatal {
        RunExit::Fatal
    } else {
        RunExit::Runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::JoinError;

    #[test]
    fn model_faults_prevent_clean_exit_without_hiding_accelerator_failure() {
        let failed = RunOutcome::Failed(runtime::RuntimeError::ModelOwner(Exit::Runtime));
        assert_eq!(terminal_exit(&failed, None), RunExit::Runtime);
        assert_eq!(
            terminal_exit(&failed, Some(Exit::FatalAccelerator)),
            RunExit::Fatal
        );
        assert_eq!(
            terminal_exit(&RunOutcome::Stopped, Some(Exit::Runtime)),
            RunExit::Runtime
        );
    }

    fn requested(clock: &dyn Clock) -> ShutdownDeadline {
        let deadline = ShutdownDeadline::new(Duration::from_secs(2)).unwrap();
        deadline.request_at(clock.monotonic()).unwrap();
        deadline
    }

    #[test]
    fn earlier_observation_preempts_ready_condition_and_completion() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct ClockThatPublishes {
            deadline: Arc<ShutdownDeadline>,
            reads: AtomicUsize,
            on_read: bool,
        }
        impl Clock for ClockThatPublishes {
            fn monotonic(&self) -> Duration {
                if self.reads.fetch_add(1, Ordering::SeqCst) == 1 && self.on_read {
                    self.deadline.request_at(Duration::ZERO).unwrap();
                }
                Duration::from_secs(26)
            }
            fn wall(&self) -> std::time::SystemTime {
                std::time::UNIX_EPOCH
            }
            fn pause(&self, _: Duration) {
                panic!("an expired deadline must not wait");
            }
        }
        for on_read in [true, false] {
            let deadline = Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).unwrap());
            assert_eq!(
                deadline.request_at(Duration::from_secs(10)).unwrap(),
                Duration::from_secs(35)
            );
            let clock = ClockThatPublishes {
                deadline: Arc::clone(&deadline),
                reads: AtomicUsize::new(0),
                on_read,
            };
            let calls = std::cell::Cell::new(0);
            let result = wait_with_deadline(&clock, &deadline, "late observation", || {
                calls.set(calls.get() + 1);
                deadline.request_at(Duration::ZERO).unwrap();
                true
            });
            assert!(
                result.is_none(),
                "the captured 35-second limit is not authority"
            );
            assert_eq!(calls.get(), usize::from(!on_read));
            assert_eq!(deadline.deadline(), Some(Duration::from_secs(25)));
        }
    }

    #[test]
    fn timeout_maps_to_runtime_not_clean() {
        assert_eq!(unsafe_exit(&RunOutcome::Stopped, None), RunExit::Runtime);
        assert_eq!(RunExit::Runtime.exit(), Exit::Runtime);
    }

    #[test]
    fn owner_failure_after_stop_cannot_become_clean_release() {
        for (failure, expected) in [
            (Exit::Config, RunExit::Config),
            (Exit::Runtime, RunExit::Runtime),
            (Exit::FatalAccelerator, RunExit::Fatal),
        ] {
            assert_eq!(terminal_exit(&RunOutcome::Stopped, Some(failure)), expected);
            assert_eq!(expected.exit(), failure);
        }
        assert_eq!(terminal_exit(&RunOutcome::Stopped, None), RunExit::Clean);
    }

    #[test]
    fn accelerator_failure_survives_later_errors_and_unsafe_cleanup() {
        let fatal = RunOutcome::Failed(runtime::RuntimeError::MediaFatal);
        let runtime = RunOutcome::Failed(runtime::RuntimeError::MediaOwner(Exit::Runtime));
        assert_eq!(terminal_exit(&fatal, Some(Exit::Runtime)), RunExit::Fatal);
        assert_eq!(
            terminal_exit(&runtime, Some(Exit::FatalAccelerator)),
            RunExit::Fatal
        );
        assert_eq!(unsafe_exit(&fatal, None), RunExit::Fatal);
        assert_eq!(
            unsafe_exit(&RunOutcome::Stopped, Some(Exit::FatalAccelerator)),
            RunExit::Fatal
        );
        assert_eq!(
            unsafe_exit(&RunOutcome::Stopped, Some(Exit::Config)),
            RunExit::Runtime
        );
    }

    #[test]
    fn gpu_join_failure_blocks_native_close() {
        let called = std::cell::Cell::new(false);
        let failed_join = CleanupError::Owner {
            model: crate::run::ModelRole::Fall,
            error: JoinError::Panicked,
        };
        assert!(!close_after_gpu_join(&[failed_join], || {
            called.set(true);
            true
        }));
        assert!(!called.get());
        assert!(close_after_gpu_join(&[], || {
            called.set(true);
            true
        }));
        assert!(called.get());
        assert!(!close_after_gpu_join(&[], || false));
    }

    #[test]
    fn failed_first_output_does_not_leave_the_other_owner_running() {
        use crate::seam::SystemClock;
        use crate::telemetry::{self, Publish, Schedule, SendError};
        struct Panics;
        impl Publish for Panics {
            fn publish(&mut self) -> Result<(), SendError> {
                panic!("owned publisher failure");
            }
        }
        struct Announces(std::sync::mpsc::Sender<()>);
        impl Publish for Announces {
            fn publish(&mut self) -> Result<(), SendError> {
                self.0.send(()).unwrap();
                Ok(())
            }
        }
        let clock = SystemClock::new();
        let (started, running) = std::sync::mpsc::channel();
        let mut failed = telemetry::spawn("shutdown-failed", Panics, Schedule::HEARTBEAT).unwrap();
        let mut live =
            telemetry::spawn("shutdown-live", Announces(started), Schedule::HEARTBEAT).unwrap();
        running.recv_timeout(Duration::from_secs(2)).unwrap();
        let limit = clock.monotonic() + Duration::from_secs(2);
        poll_until(&clock, limit, "owned publisher failure", || {
            !failed.is_alive()
        })
        .unwrap();
        let deadline = requested(&clock);
        let result = stop_outputs(
            None,
            [Some(&mut failed), Some(&mut live)],
            None,
            &clock,
            &deadline,
        );
        let alive_at_return = live.is_alive();
        // Cleanup is unconditional, including when the counterfactual omits stop.
        live.request_stop();
        live.join(&clock, clock.monotonic() + Duration::from_secs(2))
            .unwrap();
        assert!(matches!(
            result,
            Err(super::super::output::OutputError::Join)
        ));
        assert!(
            !alive_at_return,
            "a failed first join must not bypass the second stop/join"
        );
        assert_eq!(live.attempts(), 1);
    }

    #[test]
    fn timed_out_media_handle_remains_owned_for_a_later_join() {
        use crate::seam::SystemClock;
        let clock = SystemClock::new();
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let (finished, done) = std::sync::mpsc::channel();
        let mut thread = Some(std::thread::spawn(move || {
            let _ = gate.recv_timeout(Duration::from_secs(2));
            finished.send(()).unwrap();
        }));
        let expired = ShutdownDeadline::new(Duration::from_nanos(1)).unwrap();
        expired.request_at(Duration::ZERO).unwrap();
        let first = join_media(&mut thread, &clock, &expired);
        let retained = thread.as_ref().is_some_and(|handle| !handle.is_finished());
        drop(release);
        done.recv_timeout(Duration::from_secs(2)).unwrap();
        let joined = join_media(&mut thread, &clock, &requested(&clock));
        assert!(!first);
        assert!(retained, "timeout must not detach a running media owner");
        assert!(joined);
        assert!(thread.is_none());
    }

    #[test]
    fn finished_exporter_with_pending_failure_gap_is_not_clean_shutdown() {
        use crate::records::builder::{Frame, Stream, sdk_frame_record};
        use crate::records::{Lanes, Provenance};
        use crate::relay::RelayClient;
        use crate::seam::SystemClock;
        // Own the loopback endpoint but never answer: the real transport's
        // bounded read times out, leaving the actual exporter failure gap.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let client = RelayClient::new(
            &format!("http://{}", listener.local_addr().unwrap()),
            "test-token",
            Duration::from_millis(50),
        )
        .unwrap();
        let lanes = Arc::new(Lanes::new(4).unwrap());
        let record = sdk_frame_record(
            &Stream {
                camera_id: "shutdown-test-camera".into(),
                worker_boot_id: "shutdown-test-boot".into(),
                source_generation: 1,
                stream_epoch: 1,
            },
            Frame {
                frame_seq: 7,
                source_pts_ns: Some(700),
            },
            1_700_000_000_000_000_000,
            None,
            7,
        )
        .unwrap();
        assert!(lanes.try_emit(record));
        let provenance = Provenance {
            worker_build_revision: "test-build".into(),
            worker_image_digest: "test-image".into(),
            model_digest: "test-model".into(),
            calibration_digest: "test-calibration".into(),
            preprocessing_identity: "test-preprocessing".into(),
            policy_identity: "test-policy".into(),
            config_digest: "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
                .into(),
        };
        let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
        let mut exporter = crate::run::exporter::spawn(
            Arc::clone(&lanes),
            client,
            provenance,
            1,
            10,
            Arc::clone(&clock),
        )
        .unwrap();
        let observed = poll_until(
            clock.as_ref(),
            clock.monotonic() + Duration::from_secs(2),
            "actual export failure",
            || !exporter.report().failures.is_empty(),
        );
        let result = stop_outputs(
            Some(&mut exporter),
            [None, None],
            None,
            clock.as_ref(),
            &requested(clock.as_ref()),
        );
        observed.unwrap();
        assert!(exporter.report().finished);
        assert!(exporter.report().pending);
        assert!(lanes.has_work());
        assert!(matches!(
            result,
            Err(super::super::output::OutputError::Records)
        ));
    }
}
