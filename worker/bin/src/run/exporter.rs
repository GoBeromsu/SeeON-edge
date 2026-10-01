//! Exporter ownership, independent of policy/media stop. Request stop only
//! AFTER policy producers drain. The frozen lane wait has no notification or
//! cancellation hook: stop cannot wake it before its configured flush deadline.

use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::poll::poll_until;
use crate::records::id::ContractError;
use crate::records::{Exporter, ExporterError, Lanes, Provenance, RELAY_TIMEOUT, Receipt};
use crate::relay::RelayClient;
use crate::relay::wire::DeliveryFailure;
use crate::seam::Clock;

#[derive(Debug)]
pub enum SpawnError {
    Provenance(ContractError),
    TransportTimeout,
    Settings(ExporterError),
    Thread(io::Error),
}

impl fmt::Display for SpawnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("execution-record exporter could not start")
    }
}
impl std::error::Error for SpawnError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinError {
    Timeout,
    Panicked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StopRequest {
    pub newly_requested: bool,
    /// No active wake is available. Excludes a flush already doing bounded I/O.
    pub idle_wait_bound: Duration,
}

/// Histories retain the existing exporter's bounded newest-first-in-time tail.
/// `finished` proves thread work ended, NOT that every record was accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub finished: bool,
    pub pending: bool,
    pub receipts: Vec<Receipt>,
    pub failures: Vec<DeliveryFailure>,
}

/// A timeout retains the native handle for a later join with the same common
/// deadline (or an explicitly extended owner deadline). Drop requests stop but
/// DETACHES an unjoined thread; it is never evidence of completed shutdown.
#[must_use = "retain the exporter owner until a successful bounded join"]
pub struct Handle {
    thread: Option<JoinHandle<Report>>,
    stop: Arc<AtomicBool>,
    history: Arc<Mutex<Report>>,
    flush: Duration,
    joined: Option<Result<Report, JoinError>>,
}

impl Handle {
    pub fn request_stop(&self) -> StopRequest {
        StopRequest {
            newly_requested: !self.stop.swap(true, Ordering::AcqRel),
            idle_wait_bound: self.flush,
        }
    }

    /// Snapshot of completed flushes, including failures, while still running.
    pub fn report(&self) -> Report {
        lock(&self.history).clone()
    }

    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// `deadline` is absolute in the supplied clock's monotonic domain. This
    /// never adds a private shutdown budget and never consumes a timed-out owner.
    pub fn join(&mut self, clock: &dyn Clock, deadline: Duration) -> Result<Report, JoinError> {
        if let Some(joined) = &self.joined {
            return joined.clone();
        }
        poll_until(clock, deadline, "execution-record exporter", || {
            self.is_finished()
        })
        .map_err(|_| JoinError::Timeout)?;
        let result = match self.thread.take() {
            Some(thread) => thread.join().map_err(|_| JoinError::Panicked),
            None => Err(JoinError::Panicked),
        };
        self.joined = Some(result.clone());
        result
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

/// Caller resolves and validates ALL provenance, including the canonical
/// redacted config digest. Validation here is defense in depth, not generation.
/// Use the approved bounded relay transport; no new retry layer is installed.
pub fn spawn(
    lanes: Arc<Lanes>,
    client: RelayClient,
    provenance: Provenance,
    batch_max: u64,
    flush_ms: u64,
    clock: Arc<dyn Clock>,
) -> Result<Handle, SpawnError> {
    provenance.validate().map_err(SpawnError::Provenance)?;
    if client.timeout().is_zero() || client.timeout() > RELAY_TIMEOUT {
        return Err(SpawnError::TransportTimeout);
    }
    let exporter = Exporter::new(Arc::clone(&lanes), client, provenance, batch_max, flush_ms)
        .map_err(SpawnError::Settings)?;
    let stop = Arc::new(AtomicBool::new(false));
    let history = Arc::new(Mutex::new(report(&exporter, &lanes, false)));
    let thread_stop = Arc::clone(&stop);
    let thread_history = Arc::clone(&history);
    let thread = thread::Builder::new()
        .name("records-exporter".to_owned())
        .spawn(move || {
            run(
                exporter,
                &lanes,
                clock.as_ref(),
                &thread_stop,
                &thread_history,
            )
        })
        .map_err(SpawnError::Thread)?;
    Ok(Handle {
        thread: Some(thread),
        stop,
        history,
        flush: Duration::from_millis(flush_ms),
        joined: None,
    })
}

fn run(
    mut exporter: Exporter,
    lanes: &Lanes,
    clock: &dyn Clock,
    stop: &AtomicBool,
    history: &Mutex<Report>,
) -> Report {
    while !stop.load(Ordering::Acquire) {
        if let Some(backoff) = exporter.failure_backoff() {
            let deadline = clock.monotonic().saturating_add(backoff);
            if poll_until(clock, deadline, "execution-record backoff", || {
                stop.load(Ordering::Acquire)
            })
            .is_ok()
            {
                break;
            }
        }
        if exporter.wait_for_work(clock) {
            exporter.flush_once(clock);
            *lock(history) = report(&exporter, lanes, false);
        }
    }
    // Producers are already drained. Finish successful batches, but never spin
    // on an export-failed gap during shutdown; leave it visible in the lanes.
    while lanes.has_work() && exporter.failure_backoff().is_none() {
        exporter.flush_once(clock);
        *lock(history) = report(&exporter, lanes, false);
    }
    let final_report = report(&exporter, lanes, true);
    *lock(history) = final_report.clone();
    final_report
}

fn report(exporter: &Exporter, lanes: &Lanes, finished: bool) -> Report {
    Report {
        finished,
        pending: lanes.has_work(),
        receipts: exporter.receipts(),
        failures: exporter.failures(),
    }
}

fn lock(history: &Mutex<Report>) -> MutexGuard<'_, Report> {
    match history.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}
