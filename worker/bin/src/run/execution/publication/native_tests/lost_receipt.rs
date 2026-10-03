//! A genuine terminal receipt cannot reach its disconnected Rust receiver.
//! The real media owner flags Runtime before the unchanged 150s recorder
//! timeout, while native stop/reap/close and the owned join still finish.
//!
//! Logical retention is checked only through event attribution and recorder
//! state/pending/counters; private ticket/contributors are not observable here.
//! No measured codec/duration, whole-runtime, lease, crash, or ENOSPC claim.

use std::fs;
use std::io::ErrorKind;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::PoisonError;
use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};

use super::assertions::{self, GrownFile};
use super::support::{ACTUAL_MEDIA, FACILITY_ID, Started};
use crate::clips::recorder::{
    Boundary, CAP_SECONDS, Counters, NATIVE_COMPLETION_GRACE_SECONDS, State,
};
use crate::exit::Exit;
use crate::seam::Clock;

#[test]
#[ignore = "requires actual GPU/SDK, isolated synthetic RTSP and a new owned output directory"]
fn genuine_native_dropped_terminal_receipt_latches_runtime_failure_and_closes() {
    let _turn = ACTUAL_MEDIA.lock().unwrap_or_else(PoisonError::into_inner);
    let overall = Instant::now() + super::OVERALL;
    std::thread::scope(|scope| {
        let (done, completion) = mpsc::channel();
        scope.spawn(move || {
            if let Err(RecvTimeoutError::Timeout) =
                completion.recv_timeout(overall.saturating_duration_since(Instant::now()))
            {
                eprintln!("genuine native dropped receipt: overall deadline");
                std::process::exit(1);
            }
        });
        lose_actual_receipt(overall);
        done.send(()).expect("watchdog exited early");
    });
}

fn lose_actual_receipt(overall: Instant) {
    // Started supplies the real SystemClock and the existing shared 25s guard.
    let mut started = Started::open(overall);
    let admission_begin = started.clock.monotonic();
    let admitted = assertions::admit(&mut started);
    assert_eq!(started.publications.recorders.len(), 1);
    let logical_before = logical_observation(&started);
    assert_eq!(logical_before.0, State::Recording);
    assert_eq!(logical_before.2, 0, "admitted alert must not be pending");
    let events_before = started.publications.events.clone();
    assert_eq!(events_before.len(), 1);
    let event = events_before
        .get(admitted.event_ref)
        .expect("admitted event");
    assert_eq!(event.identity, admitted.event_ref);
    assert_eq!(event.camera_id, admitted.camera_id);
    assert_eq!(event.facility_id, FACILITY_ID);
    let queue_before = started
        .publications
        .queue
        .entries()
        .expect("admitted queue");
    assert!(
        queue_before
            .iter()
            .all(|entry| { entry.get("kind").and_then(serde_json::Value::as_str) != Some("CLIP") })
    );
    let clips = started.publications.store.root().join("clips");
    assert_absent(&clips);
    assert_absent(started.publications.sidecars.directory());

    let grown = assertions::growing_mp4(
        &started.record_dir,
        overall.min(Instant::now() + Duration::from_secs(15)),
    );
    let recording_path = grown_path(&started.record_dir, &grown);
    eprintln!(
        "NATIVE_LOST_RECEIPT_GROWN admitted_ticket={:?} path={} dev={} ino={} len={}",
        admitted.ticket,
        recording_path.display(),
        grown.dev,
        grown.ino,
        grown.len
    );
    let before = started.diagnostics().snapshot();
    assert!(
        !before.open_refused && !before.fatal && before.failure.is_none(),
        "{before:?}"
    );
    assert_eq!(before.receipts_dropped, 0, "{before:?}");
    assert!(
        matches!(
            started.publications.records.try_recv(),
            Err(TryRecvError::Empty)
        ),
        "receipt arrived before disconnecting its actual receiver"
    );

    // Disconnect the ORIGINAL production receiver before the ordinary stop.
    // The replacement has no sender; no RecordReceipt is injected or rewritten.
    let (unused_sender, disconnected_receiver) = mpsc::channel();
    drop(unused_sender);
    let original_receiver =
        std::mem::replace(&mut started.publications.records, disconnected_receiver);
    drop(original_receiver);
    assert!(matches!(
        started.publications.records.try_recv(),
        Err(TryRecvError::Disconnected)
    ));
    started.request_stop();
    // finish waits for real finalization, grants the existing close gate, and
    // joins the owned thread before the original shared shutdown deadline.
    started.finish(overall);
    let status = started.diagnostics().snapshot();
    assert!(!status.open_refused && !status.fatal, "{status:?}");
    assert_eq!(status.failure, Some(Exit::Runtime), "{status:?}");
    assert!(
        status.finalization_started && status.finalization_complete,
        "{status:?}"
    );
    assert!(
        status.stopped && status.closed && !status.close_withheld,
        "{status:?}"
    );
    assert_eq!(
        (
            status.receipts_dropped,
            status.replies_dropped,
            status.records_reserved
        ),
        (1, 0, 0),
        "{status:?}"
    );

    let receipt_observation_cutoff = started.clock.monotonic();
    let receipt_budget =
        Duration::from_secs(u64::from(CAP_SECONDS + NATIVE_COMPLETION_GRACE_SECONDS));
    assert!(
        receipt_observation_cutoff.saturating_sub(admission_begin) < receipt_budget,
        "owner loss must be flagged before the unchanged recorder timeout"
    );
    assert!(
        !started
            .publications
            .drain_records(started.clock.as_ref())
            .expect("drain disconnected receiver"),
        "lost receipt must not publish a clip"
    );
    started
        .publications
        .tick_recorders(receipt_observation_cutoff)
        .expect("real observation is before the original receipt deadline");
    // These observations are NOT exact private-ticket/contributor proof, nor
    // does channel disappearance prove SDK ownership release.
    assert_eq!(logical_observation(&started), logical_before);
    assert!(started.publications.recorders[0].is_quiesced());
    assert_eq!(
        started.publications.events, events_before,
        "event attribution changed"
    );
    assert_eq!(
        started
            .publications
            .queue
            .entries()
            .expect("retained queue"),
        queue_before,
        "loss synthesized or removed a delivery entry"
    );
    assert_absent(&clips);
    assert_absent(started.publications.sidecars.directory());
    assert!(Instant::now() < overall, "overall deadline");
    drop(started);

    // Retain the genuine grown inode after receiver and owned-guard drop.
    // Without its receipt, neither codec nor duration nor READY is asserted.
    let retained = fs::symlink_metadata(&recording_path).expect("lost-receipt MP4 retained");
    assert!(
        retained.is_file(),
        "retained recording is not a regular file"
    );
    assert_eq!((retained.dev(), retained.ino()), (grown.dev, grown.ino));
    assert!(
        retained.len() >= grown.len,
        "genuine grown file was truncated"
    );
    eprintln!(
        "NATIVE_LOST_RECEIPT_RETAINED path={} dev={} ino={} len={} diagnostics={status:?} codec_and_duration=unmeasured",
        recording_path.display(),
        retained.dev(),
        retained.ino(),
        retained.len()
    );
}

fn logical_observation(started: &Started) -> (State, Boundary, usize, Counters) {
    let recorder = &started.publications.recorders[0];
    (
        recorder.state(),
        recorder.boundary(),
        recorder.pending(),
        recorder.counters(),
    )
}

fn grown_path(record_dir: &Path, grown: &GrownFile) -> PathBuf {
    let paths: Vec<_> = fs::read_dir(record_dir)
        .expect("record directory")
        .map(|entry| entry.expect("record entry").path())
        .filter(|path| {
            let meta = fs::symlink_metadata(path).expect("growing file metadata");
            meta.is_file()
                && path.extension().is_some_and(|extension| extension == "mp4")
                && (meta.dev(), meta.ino()) == (grown.dev, grown.ino)
        })
        .collect();
    let [path] = paths.as_slice() else {
        panic!("grown native inode must name exactly one MP4: {paths:?}");
    };
    path.clone()
}

fn assert_absent(path: &Path) {
    let error = fs::symlink_metadata(path).expect_err("unexpected synthetic publication");
    assert_eq!(
        error.kind(),
        ErrorKind::NotFound,
        "{}: {error}",
        path.display()
    );
}
