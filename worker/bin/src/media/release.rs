//! Release of the media owner (design §2.3). Stop cancels open recordings;
//! their slots stay reserved until the completions are read. Native close
//! over a reserved slot corrupts the process heap, so close runs only after
//! a proven stop, a status read showing no slot reserved, and root permission
//! before the shared deadline. Otherwise retain the handle until process exit.

use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};

use seeon_deepstream_native::{MediaError, MediaOwner, MediaPoll, MediaResult};

use super::owner::MediaParams;
use crate::poll::poll_until;

/// Reading a completion retires its native record slot. While time remains,
/// read completions even when `record_tx` cannot take the receipt.
pub(super) fn drain_records(
    owner: &mut MediaOwner,
    params: &MediaParams,
) -> Result<(), MediaError> {
    loop {
        if expired(params) {
            return Ok(());
        }
        let MediaPoll::Ready(receipt) = owner.poll_record()? else {
            return Ok(());
        };
        if params.record_tx.try_send(receipt).is_err() {
            params
                .diagnostics
                .update(|snapshot| snapshot.receipts_dropped += 1);
        }
    }
}

fn expired(params: &MediaParams) -> bool {
    let now = params.clock.monotonic();
    params.shutdown.deadline().is_some_and(|end| now >= end)
}

/// Every path out of the media thread after a successful open comes here.
/// Ownership enters already guarded: even a panicking Clock or cleanup must
/// retain the handle, not run native Drop outside the explicit close gate.
/// The shared deadline bounds admission and waits, not an already-blocked
/// native call, which this layer cannot interrupt.
pub(super) fn release(mut owner: ManuallyDrop<MediaOwner>, params: &MediaParams) {
    params.diagnostics.update(|snapshot| {
        snapshot.finalization_started = true;
        snapshot.close_withheld = true;
    });
    let clock = params.clock.as_ref();
    let mut stopped = false;
    let mut reserved = None;
    let finalized = panic::catch_unwind(AssertUnwindSafe(|| {
        let deadline = params.shutdown.begin(clock.monotonic())?;
        if let Some(remaining) = params.shutdown.remaining_ms(clock.monotonic()) {
            stopped = owner
                .stop(remaining)
                .is_ok_and(|status| status.result == MediaResult::Ok);
        }
        let _ = poll_until(clock, deadline, "media record release", || {
            if expired(params) || drain_records(&mut owner, params).is_err() || expired(params) {
                return true;
            }
            match owner.read_status() {
                Ok(status) => {
                    params.diagnostics.record_status(&status);
                    reserved = Some(status.records_reserved);
                    status.records_reserved == 0
                }
                Err(_) => true,
            }
        });
        Ok::<_, crate::shutdown::DeadlineError>(deadline)
    }));
    // Publish the attempt and its safety outcome BEFORE any permission wait.
    // Root must drain policy/delivery and close GPU owners before granting it.
    params.diagnostics.update(|snapshot| {
        snapshot.stopped = stopped;
        snapshot.finalization_complete = true;
    });
    let deadline = match finalized {
        Ok(Ok(deadline)) => deadline,
        // No usable observation: retain the owner without inventing stop time.
        Ok(Err(_)) => return,
        Err(payload) => panic::resume_unwind(payload),
    };
    if !stopped || reserved != Some(0) {
        return;
    }
    let permitted = poll_until(clock, deadline, "media close permission", || {
        expired(params) || params.shutdown.close_permitted(clock.monotonic())
    });
    if permitted.is_err() || !params.shutdown.close_permitted(clock.monotonic()) {
        return;
    }
    let closed = owner
        .close()
        .is_ok_and(|status| status.result == MediaResult::Ok);
    params.diagnostics.update(|snapshot| {
        snapshot.closed = closed;
        snapshot.close_withheld = false;
    });
    if closed {
        // Native close cleared its handle. Only this proven-success path may
        // dispose Rust buffers normally; every other path intentionally leaks.
        drop(ManuallyDrop::into_inner(owner));
    }
}
