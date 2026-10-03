//! One actual native receipt, captured before production drain.

use std::fs;
use std::os::unix::fs::MetadataExt;
use std::sync::mpsc::TryRecvError;
use std::time::Instant;

use seeon_deepstream_native::{MediaResult, RecordTicket};

use super::assertions::GrownFile;
use super::support::Started;
use super::wait::poll_until;

pub(super) fn capture(
    started: &mut Started,
    deadline: Instant,
    grown: &GrownFile,
) -> crate::msg::RecordReceipt {
    let observed = poll_until(deadline, "native receipt deadline", || {
        match started.publications.records.try_recv() {
            Ok(receipt) => Some(receipt),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => panic!("native receipt channel disconnected"),
        }
    });
    assert_eq!(
        (observed.result, observed.error, observed.contains_video),
        (MediaResult::Ok, 0, true),
        "native receipt is not READY-grade"
    );
    let sealed = file_identity(&observed);
    eprintln!(
        "NATIVE_PUBLICATION_RECEIPT result={:?} error={} duration_ms={} contains_video={} file={:?} dev={} ino={} len={}",
        observed.result,
        observed.error,
        observed.duration_ms,
        observed.contains_video,
        observed.filename,
        sealed.dev,
        sealed.ino,
        sealed.len
    );
    assert_eq!(
        (sealed.dev, sealed.ino),
        (grown.dev, grown.ino),
        "native receipt does not name the grown MP4"
    );
    assert!(sealed.len >= grown.len, "sealed file shrank");
    observed
}

pub(super) fn file_identity(receipt: &crate::msg::RecordReceipt) -> GrownFile {
    let path = receipt.directory.join(&receipt.filename);
    let meta = fs::symlink_metadata(&path).expect("native receipt file");
    assert!(meta.is_file(), "native receipt does not name a file");
    GrownFile {
        dev: meta.dev(),
        ino: meta.ino(),
        len: meta.len(),
    }
}

/// A start reply may legitimately have `session_valid == 0`. Production
/// diagnostics do not retain current-source status, so the receipt is the
/// session and the admitted ticket is never rewritten.
pub(super) fn assert_ticket(admitted: &RecordTicket, observed: &RecordTicket) {
    assert_eq!(
        (
            observed.source_id,
            observed.binding,
            observed.request_id,
            observed.coalesced
        ),
        (admitted.source_id, admitted.binding, admitted.request_id, 0),
        "native receipt is not the admitted recording"
    );
    assert_eq!(observed.session_valid, 1, "native receipt has no session");
    assert_ne!(observed.session_id, u32::MAX, "native session is unset");
    if admitted.session_valid == 1 {
        assert_ne!(admitted.session_id, u32::MAX);
        assert_eq!(observed.session_id, admitted.session_id);
    }
}

pub(super) fn forward_once(started: &mut Started, observed: crate::msg::RecordReceipt) {
    poll_until(
        started.shutdown_deadline(),
        "native producer finalization",
        || {
            started
                .diagnostics()
                .snapshot()
                .finalization_complete
                .then_some(())
        },
    );
    assert!(
        matches!(
            started.publications.records.try_recv(),
            Err(TryRecvError::Empty)
        ),
        "extra native receipt before forwarding the observed receipt"
    );
    started
        .receipt_tx
        .send(observed)
        .expect("return actual receipt unchanged");
    assert!(
        started
            .publications
            .drain_records(started.clock.as_ref())
            .expect("production drain refused the native receipt"),
        "drain published nothing"
    );
    assert!(
        matches!(
            started.publications.records.try_recv(),
            Err(TryRecvError::Empty)
        ),
        "unseen or duplicate native receipt"
    );
    assert_eq!(
        started.publications.recorders[0].state(),
        crate::clips::recorder::State::Idle,
        "receipt was not consumed by the production recorder"
    );
}
