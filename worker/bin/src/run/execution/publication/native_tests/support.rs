//! Native media-thread lifecycle and shutdown guard.
//! Release code does not reach this module.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use seeon_deepstream_native::MediaBinding;

use super::super::{PublicationConfig, Publications, channels, open};
use super::fixture::{evidence_dir, media_config, prepared_record_dir};
use super::wait::poll_until;
use crate::media::diagnostics::Diagnostics;
use crate::media::owner::{self, MediaParams};
use crate::media::shutdown::ShutdownControl;
use crate::msg::{POSE_PER_CAMERA, PREVIEW_CAPACITY, Readiness};
use crate::seam::{Clock, SystemClock};
use crate::shutdown::ShutdownDeadline;

pub(super) const SOURCE_ID: u32 = 0;
pub(super) const CAMERA_ID: &str = "cam-native-receipt";
pub(super) const FACILITY_ID: &str = "fac-native-receipt";
pub(super) const EVENT_IDENTITY: &str = "00000000-0000-4000-8000-000000000071";
pub(super) const BOOT_ID: &str = "00000000-0000-4000-8000-000000000099";
const STOP_MS: u32 = 5_000;
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(25);

pub(super) static ACTUAL_MEDIA: Mutex<()> = Mutex::new(());

pub(super) struct Owned(PathBuf);

impl Drop for Owned {
    fn drop(&mut self) {
        eprintln!("NATIVE_PUBLICATION_RETAINED={}", self.0.display());
    }
}

pub(super) struct Started {
    pub publications: Publications,
    pub receipt_tx: std::sync::mpsc::SyncSender<crate::msg::RecordReceipt>,
    pub record_dir: PathBuf,
    pub clock: Arc<SystemClock>,
    _owned: Owned,
    media: Option<MediaGuard>,
}

pub(super) struct MediaGuard {
    stop: Arc<AtomicBool>,
    deadline: Arc<ShutdownDeadline>,
    shutdown: Arc<ShutdownControl>,
    diagnostics: Arc<Diagnostics>,
    thread: Option<JoinHandle<()>>,
    clock: Arc<SystemClock>,
    requested: bool,
}

impl Drop for MediaGuard {
    fn drop(&mut self) {
        if self.thread.is_none() {
            return;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.request_stop();
            self.finish(self.shutdown_deadline());
        }));
        if result.is_err() {
            eprintln!("native recording cleanup failed; refusing detached-owner success");
            std::process::exit(1);
        }
    }
}

impl MediaGuard {
    pub(super) fn request_stop(&mut self) {
        if self.requested {
            return;
        }
        let now = self.clock.monotonic();
        self.deadline
            .request_at(now)
            .expect("shared 25s shutdown cutoff");
        self.stop.store(true, Ordering::SeqCst);
        self.requested = true;
    }

    pub(super) fn shutdown_deadline(&self) -> Instant {
        let end = self.deadline.deadline().expect("shared shutdown deadline");
        let now = self.clock.monotonic();
        Instant::now() + end.saturating_sub(now)
    }

    pub(super) fn finish(&mut self, overall: Instant) {
        if self.thread.is_none() {
            return;
        }
        let cutoff = overall.min(self.shutdown_deadline());
        // A refused open never enters release, so waiting for finalization
        // would burn the shared 25s cutoff and hide the original error.
        poll_until(cutoff, "media finalization deadline", || {
            let snapshot = self.diagnostics.snapshot();
            (snapshot.finalization_complete
                || snapshot.open_refused
                || self
                    .thread
                    .as_ref()
                    .expect("owned media thread")
                    .is_finished())
            .then_some(())
        });
        if !self.diagnostics.snapshot().open_refused
            && !self
                .thread
                .as_ref()
                .expect("owned media thread")
                .is_finished()
        {
            let now = self.clock.monotonic();
            assert!(
                self.shutdown.close_permitted(now) || self.shutdown.permit_close(now).is_ok(),
                "close permission is not valid before the shared deadline"
            );
        }
        poll_until(cutoff, "media thread completion deadline", || {
            self.thread
                .as_ref()
                .expect("owned media thread")
                .is_finished()
                .then_some(())
        });
        let join_at = self
            .deadline
            .deadline()
            .expect("original shutdown deadline");
        let thread = self.thread.take().expect("finished media thread");
        owner::join(thread, self.clock.as_ref(), join_at).expect("media thread join");
        assert!(self.clock.monotonic() < join_at, "late media join");
    }

    pub(super) fn diagnostics(&self) -> Arc<Diagnostics> {
        Arc::clone(&self.diagnostics)
    }
}

impl Started {
    pub(super) fn open(overall: Instant) -> Self {
        let record_dir = prepared_record_dir();
        let owned = Owned(evidence_dir("native-publication"));
        eprintln!("NATIVE_PUBLICATION_RETAINED={}", owned.0.display());
        let state_dir = owned.0.join("state");
        let store_root = owned.0.join("store");
        fs::create_dir(&state_dir).expect("owned state directory");
        let binding = MediaBinding {
            token: 73,
            generation: 7,
            epoch: 11,
        };
        let cameras = [(
            SOURCE_ID,
            binding,
            CAMERA_ID.to_owned(),
            FACILITY_ID.to_owned(),
        )];
        let (command_tx, command_rx, record_tx, record_rx) = channels();
        let receipt_tx = record_tx.clone();
        let clock = Arc::new(SystemClock::new());
        let publications = open(
            PublicationConfig {
                boot_id: BOOT_ID,
                state_dir: &state_dir,
                record_dir: &record_dir,
                store_root: &store_root,
                cameras: &cameras,
                config_version: 1,
                manifest_sha: None,
            },
            command_tx,
            record_rx,
            Arc::clone(&clock) as Arc<dyn Clock>,
        )
        .expect("open production publications");
        let stop = Arc::new(AtomicBool::new(false));
        let deadline = Arc::new(ShutdownDeadline::new(SHUTDOWN_BUDGET).expect("budget"));
        let shutdown = Arc::new(ShutdownControl::new(Arc::clone(&deadline)));
        let diagnostics = Arc::new(Diagnostics::new(1));
        let (pose_tx, _pose_rx) = std::sync::mpsc::sync_channel(POSE_PER_CAMERA);
        let (preview_tx, _preview_rx) = std::sync::mpsc::sync_channel(PREVIEW_CAPACITY);
        let (thread, readiness) = owner::spawn(MediaParams {
            config: media_config(record_dir.clone(), binding),
            open_budget_ms: STOP_MS,
            shutdown: Arc::clone(&shutdown),
            stop: Arc::clone(&stop),
            clock: Arc::clone(&clock) as Arc<dyn Clock>,
            pose_tx,
            preview_tx,
            record_tx,
            commands: command_rx,
            diagnostics: Arc::clone(&diagnostics),
        })
        .expect("spawn production media thread");
        let media = MediaGuard {
            stop,
            deadline,
            shutdown,
            diagnostics,
            thread: Some(thread),
            clock: Arc::clone(&clock),
            requested: false,
        };
        wait_ready(&readiness, &media, overall);
        Self {
            publications,
            receipt_tx,
            record_dir,
            clock,
            media: Some(media),
            _owned: owned,
        }
    }

    pub(super) fn shutdown_deadline(&self) -> Instant {
        self.media.as_ref().expect("media").shutdown_deadline()
    }

    pub(super) fn request_stop(&mut self) {
        self.publications.quiesce();
        self.media.as_mut().expect("media").request_stop();
    }

    pub(super) fn finish(&mut self, overall: Instant) {
        self.media.as_mut().expect("media").finish(overall);
    }

    pub(super) fn diagnostics(&self) -> Arc<Diagnostics> {
        self.media.as_ref().expect("media").diagnostics()
    }
}

impl Drop for Started {
    fn drop(&mut self) {
        self.request_stop();
        drop(self.media.take());
    }
}

fn wait_ready(readiness: &Receiver<Readiness>, media: &MediaGuard, overall: Instant) {
    let ready = poll_until(
        overall.min(Instant::now() + Duration::from_secs(15)),
        "media readiness deadline",
        || readiness.try_recv().ok(),
    );
    ready.expect("production media thread admitted the RTSP source");
    poll_until(
        overall.min(Instant::now() + Duration::from_secs(15)),
        "actual linked video frame deadline",
        || {
            let snapshot = media.diagnostics.snapshot();
            assert!(
                !snapshot.fatal && snapshot.failure.is_none(),
                "{snapshot:?}"
            );
            snapshot.cameras.first().and_then(|camera| {
                (camera.video_linked == 1 && camera.published_frames > 0).then_some(())
            })
        },
    );
}
