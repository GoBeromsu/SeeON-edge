//! Bounded telemetry shutdown. Stop disconnects the backoff channel; it does
//! not cancel a publish already in flight. Join waits only until the caller's
//! deadline and keeps the thread on timeout. Drop requests stop and then
//! detaches an unjoined thread, so a dropped handle cannot be joined later.

use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use std::time::{Duration, SystemTime};

use seeon_ml_worker::seam::{Clock, SystemClock};
use seeon_ml_worker::telemetry::{JoinError, LoopHandle, Publish, Schedule, SendError, spawn};

const HOUR: Duration = Duration::from_secs(3_600);
const SAFETY: Duration = Duration::from_secs(5);

/// An expired deadline must not pause or wait for a blocked publisher.
struct FrozenClock;

impl Clock for FrozenClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    fn pause(&self, _limit: Duration) {
        panic!("an expired join must not pause");
    }
}

/// `publish` reports entry, waits for one release, then reports completion.
/// The wait is the in-flight proof: the loop cannot finish while it is held.
struct Hold {
    entered: SyncSender<()>,
    release: Mutex<Receiver<()>>,
    finished: SyncSender<()>,
}

impl Publish for Hold {
    fn publish(&mut self) -> Result<(), SendError> {
        self.entered.try_send(()).expect("entry delivered");
        self.release
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .recv()
            .expect("publisher released");
        self.finished.try_send(()).expect("publish finished");
        Ok(())
    }
}

/// Reports entry, then panics. The entry is delivered before the panic.
struct PanicAfterEntry {
    entered: SyncSender<()>,
}

impl Publish for PanicAfterEntry {
    fn publish(&mut self) -> Result<(), SendError> {
        self.entered.try_send(()).expect("entry delivered");
        panic!("injected telemetry publisher failure");
    }
}

struct Held {
    handle: LoopHandle,
    entered: Receiver<()>,
    release: SyncSender<()>,
    finished: Receiver<()>,
}

fn hold(schedule: Schedule) -> Held {
    let (entered_tx, entered) = mpsc::sync_channel(1);
    let (release, release_rx) = mpsc::sync_channel(1);
    let (finished_tx, finished) = mpsc::sync_channel(1);
    let handle = spawn(
        "telemetry-shutdown",
        Hold {
            entered: entered_tx,
            release: Mutex::new(release_rx),
            finished: finished_tx,
        },
        schedule,
    )
    .expect("loop starts");
    Held {
        handle,
        entered,
        release,
        finished,
    }
}

fn hour() -> Schedule {
    Schedule {
        interval: HOUR,
        initial_backoff: HOUR,
        max_backoff: HOUR,
    }
}

fn entered(held: &Held) {
    held.entered.recv_timeout(SAFETY).expect("publish entered");
}

/// One publish finishes, then stop during the one-hour backoff ends the
/// thread. Public counters stay at that one attempt.
#[test]
fn stop_wakes_a_long_backoff_without_another_publish() {
    let mut held = hold(hour());
    entered(&held);
    held.release.send(()).expect("release");
    held.finished
        .recv_timeout(SAFETY)
        .expect("publish finished");
    assert_eq!(held.handle.attempts(), 1);
    held.handle.request_stop();
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    assert_eq!(held.handle.join(&clock, deadline), Ok(()));
    assert!(!held.handle.is_alive());
    assert_eq!(held.handle.attempts(), 1);
    assert_eq!(held.handle.successes(), 1);
}

/// While publish is blocked, a frozen clock makes the timeout deterministic.
/// Releasing the publisher lets a later real-clock join succeed.
#[test]
fn inflight_publish_times_out_and_rejoin_succeeds_after_release() {
    let mut held = hold(hour());
    entered(&held);
    held.handle.request_stop();
    assert_eq!(
        held.handle.join(&FrozenClock, Duration::ZERO),
        Err(JoinError::Timeout)
    );
    assert!(held.handle.is_alive(), "timeout keeps the loop thread");
    assert_eq!(held.handle.attempts(), 1);
    assert_eq!(held.handle.successes(), 0);
    held.release.send(()).expect("release");
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    assert_eq!(held.handle.join(&clock, deadline), Ok(()));
    assert!(!held.handle.is_alive());
    assert_eq!(held.handle.successes(), 1);
    assert_eq!(held.handle.attempts(), 1);
}

/// Entry is signalled, then the publisher panics. Both joins return the
/// remembered typed failure.
#[test]
fn publisher_panic_is_a_typed_join_failure() {
    let (entered_tx, entered) = mpsc::sync_channel(1);
    let mut handle = spawn(
        "telemetry-panic",
        PanicAfterEntry {
            entered: entered_tx,
        },
        hour(),
    )
    .expect("loop starts");
    entered.recv_timeout(SAFETY).expect("publish entered");
    handle.request_stop();
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    assert_eq!(handle.join(&clock, deadline), Err(JoinError::Panicked));
    assert_eq!(handle.join(&clock, deadline), Err(JoinError::Panicked));
    assert!(!handle.is_alive());
}

/// Requesting stop twice, then joining twice, stays on the remembered result.
#[test]
fn repeated_stop_and_join_are_safe() {
    let mut held = hold(hour());
    entered(&held);
    held.release.send(()).expect("release");
    held.finished
        .recv_timeout(SAFETY)
        .expect("publish finished");
    held.handle.request_stop();
    held.handle.request_stop();
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    assert_eq!(held.handle.join(&clock, deadline), Ok(()));
    assert_eq!(held.handle.join(&clock, deadline), Ok(()));
    assert_eq!(held.handle.successes(), 1);
    assert_eq!(held.handle.attempts(), 1);
    assert!(!held.handle.is_alive());
}

/// Drop returns while publish is blocked. The helper drops the handle and
/// reports that before this thread releases the publisher. The release guard
/// runs on every exit, including a failed assertion.
#[test]
fn drop_returns_while_publish_is_blocked() {
    let held = hold(hour());
    entered(&held);
    let handle = held.handle;
    let (dropped_tx, dropped) = mpsc::sync_channel(1);
    let helper = thread::spawn(move || {
        drop(handle);
        dropped_tx.send(()).expect("drop reported");
    });
    let release = held.release;
    let mut guard = ReleaseGuard {
        release: Some(move || {
            let _ = release.send(());
        }),
    };
    let reported = dropped.recv_timeout(SAFETY);
    guard.release_now();
    assert_eq!(
        reported,
        Ok(()),
        "drop returned before the publisher was released"
    );
    held.finished
        .recv_timeout(SAFETY)
        .expect("publisher finished");
    helper.join().expect("helper");
}

struct ReleaseGuard<F: FnOnce()> {
    release: Option<F>,
}

impl<F: FnOnce()> ReleaseGuard<F> {
    fn release_now(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

impl<F: FnOnce()> Drop for ReleaseGuard<F> {
    fn drop(&mut self) {
        self.release_now();
    }
}
