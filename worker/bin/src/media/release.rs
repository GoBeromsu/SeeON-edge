//! Release of the media owner (design §2.3). Stop cancels open recordings;
//! their slots stay reserved until the completions are read. Native close
//! over a reserved slot corrupts the process heap, so close runs only after
//! a proven stop and a status read that shows no slot reserved. Otherwise the
//! handle is leaked: the root then exits nonzero instead of reopening media.

use std::time::Duration;

use seeon_deepstream_native::{MediaError, MediaOwner, MediaPoll, MediaResult};

use super::owner::MediaParams;
use crate::poll::poll_until;

/// Reading a completion is what retires its native record slot, so every
/// completion is read even when `record_tx` cannot take the receipt.
pub(super) fn drain_records(
    owner: &mut MediaOwner,
    params: &MediaParams,
) -> Result<(), MediaError> {
    while let MediaPoll::Ready(receipt) = owner.poll_record()? {
        if params.record_tx.try_send(receipt).is_err() {
            params
                .diagnostics
                .update(|snapshot| snapshot.receipts_dropped += 1);
        }
    }
    Ok(())
}

/// Every path out of the media thread after a successful open comes here.
/// One deadline, taken before stop, bounds the whole release: the native stop
/// and the reaping after it share `shutdown_budget_ms`, so release never
/// takes two budgets. Reaping continues until a status read shows no slot
/// reserved or that deadline passes; that last read, not an inference, gates
/// close.
pub(super) fn release(mut owner: MediaOwner, params: &MediaParams) {
    let clock = params.clock.as_ref();
    let budget = Duration::from_millis(u64::from(params.shutdown_budget_ms));
    let deadline = clock.monotonic() + budget;
    let stopped = owner
        .stop(params.shutdown_budget_ms)
        .is_ok_and(|status| status.result == MediaResult::Ok);
    let mut reserved = None;
    // A timeout leaves the last reserved count in `reserved` and in the
    // diagnostics, which is where the outcome is observed. `poll_until`
    // evaluates the condition once even when stop used the whole budget.
    let _ = poll_until(clock, deadline, "media record release", || {
        drain_records(&mut owner, params).is_err()
            || match owner.read_status() {
                Ok(status) => {
                    params.diagnostics.record_status(&status);
                    reserved = Some(status.records_reserved);
                    status.records_reserved == 0
                }
                Err(_) => true,
            }
    });
    let closable = stopped && reserved == Some(0);
    let closed = closable
        && owner
            .close()
            .is_ok_and(|status| status.result == MediaResult::Ok);
    params.diagnostics.update(|snapshot| {
        snapshot.stopped = stopped;
        snapshot.closed = closed;
        snapshot.close_withheld = !closable;
    });
    if !closed {
        // `MediaOwner::drop` would stop and close outside this gate. Leaking
        // the handle is safe; the process ends without closing it.
        std::mem::forget(owner);
    }
}
