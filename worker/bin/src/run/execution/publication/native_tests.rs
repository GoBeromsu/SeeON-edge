//! One deterministically admitted event, recorded by the production media
//! thread and published by `Publications::drain_records`.
//!
//! Not a model-to-event trigger, a worker SIGTERM, or durable PostgreSQL.
//! Detection time comes from production staging. Ticket, duration, path, and
//! media come only from the receipt the media thread converts during shutdown
//! finalization. The recorder cap is 120s, so this test does not wait for a
//! spontaneous receipt.

#[path = "native_tests/assertions.rs"]
mod assertions;
#[path = "native_tests/fixture.rs"]
mod fixture;
#[path = "native_tests/published.rs"]
mod published;
#[path = "native_tests/receipt.rs"]
mod receipt;
#[path = "native_tests/support.rs"]
mod support;
#[path = "native_tests/wait.rs"]
mod wait;

use std::process;
use std::sync::PoisonError;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use assertions::admit;
use support::{ACTUAL_MEDIA, Started};

/// Ready (15) + linked frame (15) + growing MP4 (15) + shared shutdown (25),
/// plus slack so a slow legitimate phase cannot hard-exit mid-shutdown.
const OVERALL: Duration = Duration::from_secs(75);

#[test]
#[ignore = "requires actual GPU/SDK, isolated synthetic RTSP and a new owned output directory"]
fn deterministic_native_recording_publishes_actual_clip_and_closes() {
    let _turn = ACTUAL_MEDIA.lock().unwrap_or_else(PoisonError::into_inner);
    let overall = Instant::now() + OVERALL;
    std::thread::scope(|scope| {
        let (finished, completion) = mpsc::channel();
        scope.spawn(move || {
            match completion.recv_timeout(overall.saturating_duration_since(Instant::now())) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    eprintln!(
                        "deterministic_native_recording_publishes_actual_clip_and_closes: overall deadline"
                    );
                    process::exit(1);
                }
            }
        });
        publish_actual(overall);
        finished
            .send(())
            .expect("watchdog exited before publication finished");
    });
}

fn publish_actual(overall: Instant) {
    let mut started = Started::open(overall);
    let admitted = admit(&mut started);
    let grown = assertions::growing_mp4(
        &started.record_dir,
        overall.min(Instant::now() + Duration::from_secs(15)),
    );
    // One stop. Quiesce first so shutdown cannot start another recording.
    // The shared 25s cutoff starts here; finalization drains the real receipt.
    started.request_stop();
    let receipt_deadline = started.shutdown_deadline();
    let observed = receipt::capture(&mut started, receipt_deadline, &grown);
    let sealed = receipt::file_identity(&observed);
    let retained = observed.ticket;
    let duration_ms = observed.duration_ms;
    receipt::assert_ticket(&admitted.ticket, &observed.ticket);
    receipt::forward_once(&mut started, observed);
    published::assert_published(
        &started.publications,
        &started.record_dir,
        &admitted,
        &retained,
        duration_ms,
        &sealed,
    );
    started.finish(overall);
    let snapshot = started.diagnostics().snapshot();
    assert!(snapshot.finalization_started, "{snapshot:?}");
    assert!(snapshot.finalization_complete, "{snapshot:?}");
    assert!(snapshot.stopped, "media thread did not stop: {snapshot:?}");
    assert!(snapshot.closed, "media thread did not close: {snapshot:?}");
    assert!(!snapshot.close_withheld, "{snapshot:?}");
    assert!(snapshot.failure.is_none(), "{snapshot:?}");
    assert_eq!(snapshot.receipts_dropped, 0, "{snapshot:?}");
    assert_eq!(snapshot.replies_dropped, 0, "{snapshot:?}");
    assert_eq!(snapshot.records_reserved, 0, "{snapshot:?}");
    assert!(Instant::now() < overall, "overall deadline");
}
