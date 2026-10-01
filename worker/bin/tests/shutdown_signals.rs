//! Standalone Stage4 signal/deadline proofs, not root/media/boot adoption.
//! Small budgets prove bounded behavior, not exact 25s system qualification.
//! Real clocks/signals are confined to owned subprocess boundary tests; pure
//! deadline tests use explicit observations. No sleeps synchronize tests.

use std::io::{Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use seeon_ml_worker::seam::{Clock, SystemClock};
use seeon_ml_worker::shutdown::{
    DeadlineError, MAX_SHUTDOWN_BUDGET, ShutdownDeadline, SignalShutdown, register,
};

const SELECTED: &[u8] = b"\nSEEON_SHUTDOWN_SELECTED\n";
const COMPLETED: &[u8] = b"\nSEEON_SHUTDOWN_COMPLETED\n";
const ARMED: &[u8] = b"\nSEEON_SHUTDOWN_ARMED\n";
const ATEXIT: &[u8] = b"\nSEEON_SHUTDOWN_ATEXIT\n";
const CHILD_CASE: &str = "SEEON_SHUTDOWN_SIGNAL_CHILD";
const CONTAINMENT: Duration = Duration::from_secs(20);
const SHORT_BUDGET: Duration = Duration::from_millis(250);

#[test]
fn deadline_rejects_zero_and_above_cap_budgets() {
    // Approved shutdown contract, independent of the production cap constant.
    let cap = Duration::from_secs(25);
    assert_eq!(
        ShutdownDeadline::new(Duration::ZERO).expect_err("zero"),
        DeadlineError::InvalidBudget
    );
    assert_eq!(
        ShutdownDeadline::new(cap + Duration::from_nanos(1)).expect_err("above cap"),
        DeadlineError::InvalidBudget
    );
    assert!(ShutdownDeadline::new(Duration::from_nanos(1)).is_ok());
    let accepted = ShutdownDeadline::new(cap).expect("cap");
    assert_eq!(accepted.deadline(), None);
    assert_eq!(accepted.requested_at(), None);
}

#[test]
fn deadline_rejects_unrepresentable_values_without_recording_a_replacement() {
    let deadline = ShutdownDeadline::new(Duration::from_nanos(1)).expect("budget");
    for (sample, error) in [
        (Duration::new(u64::MAX, 1), DeadlineError::TimestampOverflow),
        (
            Duration::from_nanos(u64::MAX),
            DeadlineError::DeadlineOverflow,
        ),
        (
            Duration::from_nanos(u64::MAX - 1),
            DeadlineError::DeadlineOverflow,
        ),
    ] {
        assert_eq!(deadline.request_at(sample), Err(error));
        assert_eq!(deadline.deadline(), None);
        assert_eq!(deadline.requested_at(), None);
    }
    let last_valid = Duration::from_nanos(u64::MAX - 2);
    assert_eq!(
        deadline
            .request_at(last_valid)
            .expect("largest valid observation"),
        Duration::from_nanos(u64::MAX - 1)
    );
    assert_eq!(deadline.requested_at(), Some(last_valid));
    assert_eq!(
        deadline.request_at(Duration::MAX),
        Err(DeadlineError::TimestampOverflow)
    );
    assert_eq!(
        deadline.deadline(),
        Some(Duration::from_nanos(u64::MAX - 1))
    );
    assert_eq!(deadline.requested_at(), Some(last_valid));

    let sum_overflow = ShutdownDeadline::new(Duration::from_secs(1)).expect("budget");
    assert_eq!(
        sum_overflow.request_at(Duration::from_nanos(u64::MAX - 5)),
        Err(DeadlineError::DeadlineOverflow)
    );
    assert_eq!(sum_overflow.deadline(), None);
    assert_eq!(sum_overflow.requested_at(), None);
}

#[test]
fn earlier_sample_published_after_later_sample_shortens_deadline() {
    let budget = Duration::from_secs(25);
    let shared = ShutdownDeadline::new(budget).expect("budget");
    let later = Duration::from_secs(11);
    let earlier = Duration::from_secs(3);
    assert_eq!(
        shared.request_at(later).expect("later publication"),
        later + budget
    );
    assert_eq!(
        shared.request_at(earlier).expect("earlier publication"),
        earlier + budget
    );
    assert_eq!(shared.deadline(), Some(earlier + budget));
    assert_eq!(shared.requested_at(), Some(earlier));
}

#[test]
fn repeated_requests_do_not_renew_or_extend_deadline() {
    let budget = Duration::from_millis(40);
    let shared = ShutdownDeadline::new(budget).expect("budget");
    // Zero is a real observation, not the unrequested sentinel.
    let first = Duration::ZERO;
    let fixed = shared.request_at(first).expect("first");
    assert_eq!(shared.request_at(first).expect("duplicate"), fixed);
    assert_eq!(
        shared.request_at(Duration::from_secs(100)).expect("later"),
        fixed
    );
    assert_eq!(shared.deadline(), Some(budget));
    assert_eq!(shared.requested_at(), Some(first));
}

#[test]
fn dynamic_read_observes_deadline_shortened_after_first_return() {
    let budget = Duration::from_secs(2);
    let shared = Arc::new(ShutdownDeadline::new(budget).expect("budget"));
    let reader = Arc::clone(&shared);
    let returned = shared
        .request_at(Duration::from_secs(9))
        .expect("first return");
    assert_eq!(returned, Duration::from_secs(11));
    shared
        .request_at(Duration::from_secs(4))
        .expect("shortening publication");
    assert_eq!(reader.deadline(), Some(Duration::from_secs(6)));
    assert_eq!(reader.requested_at(), Some(Duration::from_secs(4)));
    assert!(reader.deadline().expect("shortened") < returned);
}

struct OwnedChild {
    child: Child,
}

impl OwnedChild {
    fn spawn(case: &str) -> Self {
        let child = Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", case, "--nocapture", "--test-threads=1"])
            .env(CHILD_CASE, case)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("owned shutdown child");
        Self { child }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn await_child(mut owned: OwnedChild, case: &str) -> (ExitStatus, Vec<u8>) {
    let started = Instant::now();
    let mut report_pipe = owned.child.stdout.take().expect("child report");
    let report_handle = thread::spawn(move || {
        let mut report = Vec::new();
        report_pipe.read_to_end(&mut report).expect("child report");
        report
    });
    let status = loop {
        if let Some(status) = owned.child.try_wait().expect("child status") {
            break status;
        }
        if started.elapsed() >= CONTAINMENT {
            owned.child.kill().expect("kill only owned child");
            owned.child.wait().expect("reap owned child");
            let report = report_handle.join().expect("report reader");
            panic!(
                "{case} exceeded containment: {}",
                String::from_utf8_lossy(&report)
            );
        }
        thread::yield_now();
    };
    (status, report_handle.join().expect("report reader"))
}

fn marker(value: &[u8]) {
    let mut stdout = std::io::stdout().lock();
    if stdout
        .write_all(value)
        .and_then(|()| stdout.flush())
        .is_err()
    {
        signal_hook::low_level::exit(2);
    }
}

fn contains(report: &[u8], value: &[u8]) -> bool {
    report.windows(value.len()).any(|window| window == value)
}

fn owned_case(case: &str, forced: bool, action: impl FnOnce()) {
    if std::env::var(CHILD_CASE).ok().as_deref() == Some(case) {
        marker(SELECTED);
        action();
        marker(COMPLETED);
        // Finish the oracle before libtest's own shutdown can consume the budget.
        signal_hook::low_level::exit(0);
    }
    let (status, report) = await_child(OwnedChild::spawn(case), case);
    let diagnostic = String::from_utf8_lossy(&report);
    assert!(
        contains(&report, SELECTED),
        "{case}: empty exact selection: {diagnostic}"
    );
    assert_eq!(
        status.code(),
        Some(i32::from(forced)),
        "{case}: {diagnostic}"
    );
    if forced {
        assert!(
            contains(&report, ARMED),
            "{case}: never reached armed oracle: {diagnostic}"
        );
        assert!(!contains(&report, COMPLETED), "{case}: ordinary completion");
        if case == "blocking_atexit_still_forces_exit" {
            assert!(contains(&report, ATEXIT), "{case}: never entered atexit");
        }
    } else {
        assert!(
            contains(&report, COMPLETED),
            "{case}: oracle never completed: {diagnostic}"
        );
    }
}

fn raise_own_signal(signal: i32) {
    // Only called within owned_case's selected child. raise targets this thread,
    // not the parent PID or a process group, and returns after the handler.
    signal_hook::low_level::raise(signal).expect("owned child self signal");
}

fn block_root() -> ! {
    loop {
        thread::park();
    }
}

#[test]
fn registration_rejects_a_duplicate_without_changing_deadline() {
    owned_case(
        "registration_rejects_a_duplicate_without_changing_deadline",
        false,
        || {
            let clock = Arc::new(SystemClock::new());
            let first = register(Arc::clone(&clock), MAX_SHUTDOWN_BUDGET).expect("register");
            let requested = first.request_shutdown().expect("ordinary trigger");
            let shared = first.shared_deadline();
            drop(first);
            let duplicate = register(clock, MAX_SHUTDOWN_BUDGET);
            assert_eq!(
                duplicate.err().expect("duplicate rejected").kind(),
                std::io::ErrorKind::AlreadyExists
            );
            assert_eq!(shared.deadline(), Some(requested));
        },
    );
}

#[test]
fn ordinary_trigger_uses_concrete_clock() {
    owned_case("ordinary_trigger_uses_concrete_clock", false, || {
        let clock = Arc::new(SystemClock::new());
        let shutdown = register(Arc::clone(&clock), MAX_SHUTDOWN_BUDGET).expect("register");
        let before = clock.monotonic();
        let deadline = shutdown.request_shutdown().expect("ordinary trigger");
        let after = clock.monotonic();
        let requested = shutdown
            .shared_deadline()
            .requested_at()
            .expect("requested");
        assert!(requested >= before && requested <= after);
        assert_eq!(deadline, requested + MAX_SHUTDOWN_BUDGET);
        assert!(shutdown.requested());
        assert_eq!(shutdown.request_shutdown().expect("repeat"), deadline);
    });
}

fn signal_capture(first_signal: i32, second_signal: i32) {
    let clock = Arc::new(SystemClock::new());
    let before = clock.monotonic();
    let shutdown = register(Arc::clone(&clock), MAX_SHUTDOWN_BUDGET).expect("register");
    let registered = clock.monotonic();
    let anchor_window = registered - before;
    // The concrete-clock boundary requires an actual elapsed-time condition:
    // separate receipt from setup enough to reject a handler returning only C0.
    // No sleep or assumed scheduler delay is used as a synchronization event.
    let receipt_boundary = registered + anchor_window + Duration::from_millis(1);
    while clock.monotonic() < receipt_boundary {
        thread::yield_now();
    }
    let before_signal = clock.monotonic();
    raise_own_signal(first_signal);
    let after = clock.monotonic();
    assert!(shutdown.requested());
    let shared = shutdown.shared_deadline();
    let requested = shared.requested_at().expect("signal receipt");
    // Conservative anchoring can subtract at most the measured setup window.
    assert!(requested >= before_signal.saturating_sub(anchor_window) && requested <= after);
    let first = shared.deadline().expect("signal deadline");
    assert_eq!(first, requested + MAX_SHUTDOWN_BUDGET);
    for _ in 0..3 {
        raise_own_signal(second_signal);
        assert_eq!(shared.deadline(), Some(first), "re-signal renewed budget");
    }
}

#[test]
fn sigterm_then_sigint_capture_without_renewal() {
    owned_case("sigterm_then_sigint_capture_without_renewal", false, || {
        signal_capture(signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT);
    });
}

#[test]
fn sigint_then_sigterm_capture_without_renewal() {
    owned_case("sigint_then_sigterm_capture_without_renewal", false, || {
        signal_capture(signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM);
    });
}

#[test]
fn watchdog_exits_when_root_never_polls() {
    owned_case("watchdog_exits_when_root_never_polls", true, || {
        let _shutdown = register(Arc::new(SystemClock::new()), SHORT_BUDGET).expect("register");
        raise_own_signal(signal_hook::consts::SIGTERM);
        marker(ARMED);
        // No requested(), deadline(), or request_shutdown() call by the root.
        block_root();
    });
}

#[test]
fn dropped_handle_after_arming_still_forces_exit() {
    owned_case(
        "dropped_handle_after_arming_still_forces_exit",
        true,
        || {
            let shutdown = register(Arc::new(SystemClock::new()), SHORT_BUDGET).expect("register");
            shutdown.request_shutdown().expect("arm before drop");
            drop(shutdown);
            marker(ARMED);
            block_root();
        },
    );
}

#[test]
fn dropped_handle_before_signal_retains_hooks_and_watchdog() {
    owned_case(
        "dropped_handle_before_signal_retains_hooks_and_watchdog",
        true,
        || {
            let shutdown = register(Arc::new(SystemClock::new()), SHORT_BUDGET).expect("register");
            drop(shutdown);
            raise_own_signal(signal_hook::consts::SIGINT);
            marker(ARMED);
            block_root();
        },
    );
}

// Test-only libc fixture: no production syscall ABI or dependency is added.
static EXIT_SHUTDOWN: OnceLock<SignalShutdown> = OnceLock::new();

extern "C" fn block_cleanup() {
    marker(ATEXIT);
    let Some(shutdown) = EXIT_SHUTDOWN.get() else {
        signal_hook::low_level::exit(2);
    };
    if shutdown.request_shutdown().is_err() {
        signal_hook::low_level::exit(2);
    }
    marker(ARMED);
    block_root();
}

#[test]
fn blocking_atexit_still_forces_exit() {
    owned_case("blocking_atexit_still_forces_exit", true, || {
        unsafe extern "C" {
            fn atexit(handler: extern "C" fn()) -> i32;
        }
        // SAFETY: owned child only; static, ABI-correct callback never unwinds.
        assert_eq!(unsafe { atexit(block_cleanup) }, 0, "test atexit fixture");
        let shutdown = register(Arc::new(SystemClock::new()), SHORT_BUDGET).expect("register");
        assert!(EXIT_SHUTDOWN.set(shutdown).is_ok());
        // Request inside the callback proves cleanup is actually blocking before
        // the budget starts. A watchdog using process::exit hangs on exit locks.
        std::process::exit(0);
    });
}
