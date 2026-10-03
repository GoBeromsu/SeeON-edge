//! Release of the media owner (design §2.3). Native stop finalizes a started
//! recording through stop-sr within the first half of this call's remaining
//! budget, then reserves the rest for graph teardown. A later stop cannot
//! move that split. A queued recording that never started is cancelled, and
//! its slot stays reserved until the completion is read. Native close
//! over a reserved slot corrupts the process heap, so close runs only after
//! a proven stop, a status read showing no slot reserved, and root permission
//! before the shared deadline. Otherwise retain the handle until process exit.

use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::time::Duration;

use seeon_deepstream_native::{MediaError, MediaOwner, MediaPoll, MediaResult};

use super::owner::MediaParams;
use super::shutdown::ShutdownControl;
use crate::poll::poll_until;
use crate::seam::Clock;
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

/// The captured `begin()` bound only caps a pause. Each check samples the
/// clock before the shared deadline, so an earlier observation published
/// during the wait refuses admission before any pause toward that stale bound.
fn close_permission_admitted(
    clock: &dyn Clock,
    control: &ShutdownControl,
    bound: Duration,
) -> bool {
    let admitted = poll_until(clock, bound, "media close permission", || {
        let now = clock.monotonic();
        control.deadline().is_some_and(|end| now >= end)
            || control.close_permitted(clock.monotonic())
    });
    // Predicate acceptance is not admission: the following sample can find the
    // shared deadline already shortened past the granted permission.
    admitted.is_ok() && control.close_permitted(clock.monotonic())
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
    if !close_permission_admitted(clock, &params.shutdown, deadline) {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::shutdown::ShutdownDeadline;

    const NOW: Duration = Duration::from_secs(26);
    const STALE_BOUND: Duration = Duration::from_secs(35);
    const EARLIER: Duration = Duration::from_secs(25);

    struct ScriptClock {
        shared: Arc<ShutdownDeadline>,
        control: ShutdownControl,
        now: Duration,
        reads: AtomicUsize,
        pauses: AtomicUsize,
        grant_during_pause: bool,
        shorten_on_read: Option<usize>,
    }

    impl ScriptClock {
        fn begun_at_ten(grant_during_pause: bool, shorten_on_read: Option<usize>) -> Self {
            let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).unwrap());
            let control = ShutdownControl::new(Arc::clone(&shared));
            assert_eq!(control.begin(Duration::from_secs(10)).unwrap(), STALE_BOUND);
            Self {
                control,
                shared,
                now: NOW,
                reads: AtomicUsize::new(0),
                pauses: AtomicUsize::new(0),
                grant_during_pause,
                shorten_on_read,
            }
        }
    }

    impl Clock for ScriptClock {
        fn monotonic(&self) -> Duration {
            let read = self.reads.fetch_add(1, Ordering::SeqCst);
            if self.shorten_on_read == Some(read) {
                assert_eq!(self.shared.deadline(), Some(STALE_BOUND));
                assert_eq!(self.shared.request_at(Duration::ZERO).unwrap(), EARLIER);
            }
            self.now
        }

        fn wall(&self) -> std::time::SystemTime {
            std::time::UNIX_EPOCH
        }

        fn pause(&self, limit: Duration) {
            let pauses = self.pauses.fetch_add(1, Ordering::SeqCst);
            if pauses > 0 {
                panic!("permission poll paused past the one expected transition: {limit:?}");
            }
            if self.grant_during_pause {
                assert_eq!(self.shared.deadline(), Some(STALE_BOUND));
                assert!(self.control.permit_close(self.now).is_ok());
                assert!(self.control.close_permitted(self.now));
                assert_eq!(self.shared.request_at(Duration::ZERO).unwrap(), EARLIER);
            }
        }
    }

    #[test]
    fn earlier_observation_during_permission_poll_refuses_before_stale_bound() {
        let clock = ScriptClock::begun_at_ten(true, None);
        assert_eq!(clock.shared.deadline(), Some(STALE_BOUND));
        assert!(
            !close_permission_admitted(&clock, &clock.control, STALE_BOUND),
            "a permit granted while the old deadline is live is not admission at 26s"
        );
        assert_eq!(clock.shared.deadline(), Some(EARLIER));
        assert_eq!(clock.pauses.load(Ordering::SeqCst), 1);
        assert!(!clock.control.close_permitted(NOW));
    }

    #[test]
    fn post_poll_sample_refuses_permission_shortened_after_acceptance() {
        let clock = ScriptClock::begun_at_ten(false, Some(2));
        assert!(clock.control.permit_close(NOW).is_ok());
        assert!(
            !close_permission_admitted(&clock, &clock.control, STALE_BOUND),
            "permission accepted inside the wait must fail the following sample"
        );
        assert_eq!(clock.shared.deadline(), Some(EARLIER));
        assert_eq!(clock.pauses.load(Ordering::SeqCst), 0);
        assert!(!clock.control.close_permitted(NOW));
    }

    #[test]
    fn first_sample_shortening_to_exact_expiry_refuses_without_a_pause() {
        let mut clock = ScriptClock::begun_at_ten(false, Some(0));
        clock.now = EARLIER;
        clock
            .control
            .permit_close(clock.now)
            .expect("old bound is live");
        assert!(!close_permission_admitted(
            &clock,
            &clock.control,
            STALE_BOUND
        ));
        assert_eq!(clock.shared.deadline(), Some(clock.now));
        assert_eq!(clock.pauses.load(Ordering::SeqCst), 0);
        assert!(!clock.control.close_permitted(clock.now));
    }

    #[test]
    fn genuine_permission_before_live_deadline_is_admitted() {
        let clock = ScriptClock::begun_at_ten(false, None);
        assert!(clock.control.permit_close(NOW).is_ok());
        assert!(close_permission_admitted(
            &clock,
            &clock.control,
            STALE_BOUND
        ));
        assert_eq!(clock.shared.deadline(), Some(STALE_BOUND));
        assert_eq!(clock.pauses.load(Ordering::SeqCst), 0);
        assert!(clock.control.close_permitted(NOW));
    }
}
