//! T3: `poll_until` against the D6 deadline rule (design §0.1, §7.1) with a
//! fake clock; no real time passes.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use seeon_ml_worker::poll::{POLL_INTERVAL, Timeout, poll_until};
use seeon_ml_worker::seam::Clock;

/// Time moves only when `pause` is called, by exactly the requested limit.
struct FakeClock {
    now: Mutex<Duration>,
    pauses: Mutex<Vec<Duration>>,
}

impl FakeClock {
    fn at(start: Duration) -> Self {
        Self {
            now: Mutex::new(start),
            pauses: Mutex::new(Vec::new()),
        }
    }

    fn now(&self) -> Duration {
        *self.now.lock().expect("clock lock")
    }

    fn pauses(&self) -> Vec<Duration> {
        self.pauses.lock().expect("pauses lock").clone()
    }
}

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        self.now()
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + self.now()
    }

    fn pause(&self, limit: Duration) {
        // A zero pause at the deadline would spin forever on this clock.
        assert!(limit > Duration::ZERO, "pause of zero at {:?}", self.now());
        assert!(
            limit <= POLL_INTERVAL,
            "pause {limit:?} longer than one poll interval"
        );
        *self.now.lock().expect("clock lock") += limit;
        self.pauses.lock().expect("pauses lock").push(limit);
    }
}

/// `f` answers `results` in order and then `false`; it refuses to be called
/// more than `limit` times so a missed deadline fails instead of hanging.
fn answers(results: &[bool], limit: usize) -> (impl FnMut() -> bool, Rc<Cell<usize>>) {
    let calls = Rc::new(Cell::new(0));
    let seen = calls.clone();
    let results = results.to_vec();
    let f = move || {
        let call = seen.get();
        assert!(call < limit, "f called more than {limit} times");
        seen.set(call + 1);
        results.get(call).copied().unwrap_or(false)
    };
    (f, calls)
}

#[test]
fn ready_before_the_deadline_returns_ok() {
    let clock = FakeClock::at(Duration::from_secs(1));
    let (f, calls) = answers(&[false, false, true], 100);
    let result = poll_until(&clock, Duration::from_secs(2), "bed ready", f);
    assert_eq!(result, Ok(()));
    assert_eq!(calls.get(), 3);
    assert_eq!(clock.pauses().len(), 2);
    assert!(clock.now() < Duration::from_secs(2));
}

#[test]
fn condition_is_checked_before_any_pause() {
    let clock = FakeClock::at(Duration::from_secs(5));
    let (f, calls) = answers(&[true], 100);
    assert_eq!(
        poll_until(&clock, Duration::from_secs(6), "bed ready", f),
        Ok(())
    );
    assert_eq!(calls.get(), 1);
    assert!(clock.pauses().is_empty());
}

#[test]
fn times_out_exactly_at_the_deadline() {
    let start = Duration::from_secs(1);
    let deadline = start + Duration::from_millis(25);
    let clock = FakeClock::at(start);
    let (f, calls) = answers(&[], 100);
    let result = poll_until(&clock, deadline, "stored pose ready", f);
    assert_eq!(
        result,
        Err(Timeout {
            what: "stored pose ready"
        })
    );
    assert_eq!(
        clock.now(),
        deadline,
        "time stops at the deadline, never past it"
    );
    assert_eq!(clock.pauses().iter().sum::<Duration>(), deadline - start);
    assert_eq!(
        calls.get(),
        clock.pauses().len() + 1,
        "f is checked after every pause"
    );
}

#[test]
fn deadline_already_reached_checks_once_and_times_out() {
    for (start, deadline) in [(3_000, 3_000), (3_000, 2_000)] {
        let clock = FakeClock::at(Duration::from_millis(start));
        let (f, calls) = answers(&[], 100);
        let result = poll_until(&clock, Duration::from_millis(deadline), "fall ready", f);
        assert_eq!(
            result,
            Err(Timeout { what: "fall ready" }),
            "start {start} deadline {deadline}"
        );
        assert_eq!(calls.get(), 1, "start {start} deadline {deadline}");
        assert!(
            clock.pauses().is_empty(),
            "start {start} deadline {deadline}"
        );
    }
}
