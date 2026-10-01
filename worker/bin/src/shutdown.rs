//! Process-lifetime signal receipt and an independent shutdown deadline.
//!
//! Linux's `SystemClock` (Rust `Instant`) and CLOCK_MONOTONIC must share a
//! rate. Capture C0 before K0: S = C0 + (Ks - K0) conservatively consumes
//! anchor setup time rather than extending the budget. S timestamps handler
//! receipt, not the sender's kill time or the kernel's pending-signal time.
//! Install once, before native/GPU work and competing signal registrations.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustix::time::{ClockId, DynamicClockId, clock_gettime_dynamic};
use signal_hook::consts::{SIGINT, SIGTERM};

use crate::seam::{Clock, SystemClock};

#[forbid(unsafe_code)]
mod deadline;
#[forbid(unsafe_code)]
mod watchdog;

pub use deadline::{DeadlineError, MAX_SHUTDOWN_BUDGET, ShutdownDeadline};

#[cfg(not(all(target_os = "linux", target_has_atomic = "64")))]
compile_error!("shutdown requires Linux and lock-free 64-bit atomics");

static REGISTERED: AtomicBool = AtomicBool::new(false);

/// Dropping this handle does not unregister hooks or disarm the watchdog.
/// Successful registration lasts until actual process exit, including atexit.
pub struct SignalShutdown {
    clock: Arc<SystemClock>,
    deadline: Arc<ShutdownDeadline>,
}

impl SignalShutdown {
    /// Consumers must reread this shared state; earlier receipts can shorten it.
    pub fn shared_deadline(&self) -> Arc<ShutdownDeadline> {
        Arc::clone(&self.deadline)
    }

    pub fn requested(&self) -> bool {
        self.deadline.deadline().is_some()
    }

    /// Ordinary-thread trigger in the same concrete SystemClock coordinates.
    pub fn request_shutdown(&self) -> Result<Duration, DeadlineError> {
        self.deadline.request_at(self.clock.monotonic())
    }
}

/// Prepare enforcement before installing either signal hook.
///
/// Validation, clock, or thread-spawn failure leaves no hooks and permits retry.
/// Once the watchdog exists the registration slot is permanently consumed,
/// even on hook-install failure: signal-hook documents that an error may still
/// have installed the action. Retaining enforcement covers those requests.
/// Treat such an error as startup failure, not permission to continue startup.
pub fn register(clock: Arc<SystemClock>, budget: Duration) -> io::Result<SignalShutdown> {
    let deadline = Arc::new(
        ShutdownDeadline::new(budget)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?,
    );
    if REGISTERED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "shutdown already registered",
        ));
    }
    let anchor = match prepare(&clock, &deadline) {
        Ok(anchor) => anchor,
        Err(error) => {
            REGISTERED.store(false, Ordering::Release);
            return Err(error);
        }
    };
    for signal in [SIGTERM, SIGINT] {
        install(signal, anchor, Arc::clone(&deadline))?;
    }
    Ok(SignalShutdown { clock, deadline })
}

#[derive(Clone, Copy)]
struct Anchor {
    system: Duration,
    kernel: Duration,
}

fn prepare(clock: &Arc<SystemClock>, deadline: &Arc<ShutdownDeadline>) -> io::Result<Anchor> {
    let system = clock.monotonic(); // C0 FIRST; never reverse these reads.
    // This primes the exact rustix dynamic/vDSO path used by the handler.
    let kernel = kernel_now().ok_or_else(|| io::Error::other("monotonic clock unavailable"))?;
    watchdog::start(Arc::clone(clock), Arc::clone(deadline))?;
    Ok(Anchor { system, kernel })
}

fn kernel_now() -> Option<Duration> {
    let time = clock_gettime_dynamic(DynamicClockId::Known(ClockId::Monotonic)).ok()?;
    let secs = u64::try_from(time.tv_sec).ok()?;
    let nanos = u32::try_from(time.tv_nsec).ok()?;
    if nanos >= 1_000_000_000 {
        return None;
    }
    Some(Duration::new(secs, nanos))
}

impl Anchor {
    fn receipt(self) -> Option<Duration> {
        self.system
            .checked_add(kernel_now()?.checked_sub(self.kernel)?)
    }
}

fn install(signal: i32, anchor: Anchor, deadline: Arc<ShutdownDeadline>) -> io::Result<()> {
    // SAFETY: the 'static closure owns its Arc and Copy anchor; invocation only
    // borrows them (no clone/drop/allocation). Before any hook is installed,
    // prepare primes rustix 1.1.5's exact dynamic clock path: its initialized
    // Linux vDSO/syscall branch returns errors, unlike clock_gettime's assert.
    // Receipt uses checked local arithmetic and request_at uses only lock-free
    // AtomicU64 fetch_min. No Clock trait call, locks, formatting, logging,
    // allocation, channel sends, panics, or cleanup occurs in the handler.
    // Failure goes directly to signal-hook's async-signal-safe _exit wrapper.
    #[allow(unsafe_code)]
    unsafe {
        signal_hook::low_level::register(signal, move || {
            let Some(observed) = anchor.receipt() else {
                signal_hook::low_level::exit(1);
            };
            if deadline.request_at(observed).is_err() {
                signal_hook::low_level::exit(1);
            }
        })?;
    }
    Ok(())
}
