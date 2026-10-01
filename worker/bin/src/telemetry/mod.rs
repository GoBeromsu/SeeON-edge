//! Telemetry (plan row L70): the relay heartbeat (Python
//! `worker.py:_RelayHeartbeat`), the runtime-status publisher (Python
//! `runtime_status_sender.py`) and the GPU probe behind its `gpu` block.
//! [`spawn`] runs one [`Publish`] on its own thread; a relay failure is
//! counted and backed off, and never ends the loop. Stop disconnects the
//! channel the backoff `recv_timeout` already observes, so a long backoff
//! wakes immediately. It does not cancel a publish already in flight.
//! [`LoopHandle::join`] waits only until the caller's deadline and keeps the
//! thread when that deadline passes. Drop requests stop and then detaches an
//! unjoined thread; a dropped handle cannot be joined later.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::json::JsonError;
use crate::poll::poll_until;
use crate::relay::TransportError;
use crate::seam::Clock;

pub mod gpu;
pub mod heartbeat;
pub mod status;

/// When the loop publishes next: `interval` after a success, and
/// `initial_backoff * 2^(failures - 1)` capped at `max_backoff` after the
/// `failures`-th consecutive failure (Python `_backoff_delay`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub interval: Duration,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Schedule {
    /// Python `RuntimeStatusSender` defaults: publish every 5 s, back off
    /// from 1 s to 30 s.
    pub const STATUS: Self = Self {
        interval: Duration::from_secs(5),
        initial_backoff: Duration::from_secs(1),
        max_backoff: Duration::from_secs(30),
    };
    /// Python `heartbeat_interval_sec` default: one attempt every 30 s,
    /// whatever the outcome of the last one.
    pub const HEARTBEAT: Self = Self {
        interval: Duration::from_secs(30),
        initial_backoff: Duration::from_secs(30),
        max_backoff: Duration::from_secs(30),
    };

    /// The pause before the next attempt after `consecutive_failures`.
    pub fn delay(&self, consecutive_failures: u32) -> Duration {
        if consecutive_failures == 0 {
            return self.interval;
        }
        let factor = 1u32
            .checked_shl(consecutive_failures - 1)
            .unwrap_or(u32::MAX);
        self.initial_backoff
            .checked_mul(factor)
            .map_or(self.max_backoff, |delay| delay.min(self.max_backoff))
    }
}

/// Why a telemetry body could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum PayloadError {
    /// The named id is empty; the backend requires `min_length=1`.
    BlankId(&'static str),
    /// The named number is negative or not finite; the backend requires `ge=0`.
    Range(&'static str),
    /// The body did not serialise.
    Json(JsonError),
}

impl From<JsonError> for PayloadError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

/// Why one publish attempt failed; each counts as one loop failure.
#[derive(Clone, Debug, PartialEq)]
pub enum SendError {
    /// No HTTP response (Python: the transport raised).
    Transport(TransportError),
    /// A response outside 2xx.
    Status(u16),
    /// A 2xx runtime-status response without `accepted: true` and a
    /// non-negative integer `generation`.
    Malformed,
    /// The body could not be built.
    Payload(PayloadError),
}

impl fmt::Display for SendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => write!(f, "relay transport failed: {error}"),
            Self::Status(status) => write!(f, "relay answered HTTP {status}"),
            Self::Malformed => f.write_str("relay response was not an acceptance"),
            Self::Payload(error) => write!(f, "telemetry body refused: {error:?}"),
        }
    }
}

impl std::error::Error for SendError {}

impl From<TransportError> for SendError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}

impl From<PayloadError> for SendError {
    fn from(error: PayloadError) -> Self {
        Self::Payload(error)
    }
}

/// One publish attempt of a telemetry loop.
pub trait Publish: Send + 'static {
    fn publish(&mut self) -> Result<(), SendError>;
}

#[derive(Default)]
struct Counters {
    attempts: AtomicU64,
    failures: AtomicU64,
    successes: AtomicU64,
}

/// Why a telemetry thread did not end cleanly within its join deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JoinError {
    Timeout,
    Panicked,
}

/// The running loop: its counters, liveness and stop switch.
pub struct LoopHandle {
    counters: Arc<Counters>,
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
    joined: Option<Result<(), JoinError>>,
}

impl LoopHandle {
    /// Attempts made so far.
    pub fn attempts(&self) -> u64 {
        self.counters.attempts.load(Ordering::SeqCst)
    }

    /// Failed attempts so far (Python `failure_count`); never reset.
    pub fn failures(&self) -> u64 {
        self.counters.failures.load(Ordering::SeqCst)
    }

    /// Successful attempts so far.
    pub fn successes(&self) -> u64 {
        self.counters.successes.load(Ordering::SeqCst)
    }

    /// Whether the loop thread is still running.
    pub fn is_alive(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
    }

    /// Disconnects the stop channel. Idempotent; does not wait, and does not
    /// cancel a publish already inside the relay client.
    pub fn request_stop(&mut self) {
        drop(self.stop.take());
    }

    /// `deadline` is absolute in the supplied clock's monotonic domain.
    /// A timeout retains the native handle for a later join. A finished join
    /// is remembered, so a repeated call does not join twice.
    pub fn join(&mut self, clock: &dyn Clock, deadline: Duration) -> Result<(), JoinError> {
        if let Some(joined) = self.joined {
            return joined;
        }
        poll_until(clock, deadline, "telemetry loop", || {
            self.thread.as_ref().is_none_or(JoinHandle::is_finished)
        })
        .map_err(|_| JoinError::Timeout)?;
        let result = match self.thread.take() {
            Some(thread) => thread.join().map_err(|_| JoinError::Panicked),
            None => Err(JoinError::Panicked),
        };
        self.joined = Some(result);
        result
    }
}

impl Drop for LoopHandle {
    /// Requests stop, then drops the native handle. `JoinHandle`'s own drop
    /// detaches an unjoined thread; this is not evidence the loop has ended,
    /// and nothing can rejoin it afterwards.
    fn drop(&mut self) {
        self.request_stop();
    }
}

/// Starts `publisher` on a thread named `name`, publishing at once and then
/// on `schedule` until the handle is stopped or dropped. The backoff is the
/// stop channel's `recv_timeout`: disconnect wakes it immediately, and the
/// delay is unchanged.
pub fn spawn<P: Publish>(
    name: &str,
    mut publisher: P,
    schedule: Schedule,
) -> std::io::Result<LoopHandle> {
    let counters = Arc::new(Counters::default());
    let (stop, stopped) = mpsc::channel::<()>();
    let shared = Arc::clone(&counters);
    let thread = thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            let mut consecutive = 0u32;
            loop {
                shared.attempts.fetch_add(1, Ordering::SeqCst);
                match publisher.publish() {
                    Ok(()) => {
                        consecutive = 0;
                        shared.successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(_) => {
                        consecutive = consecutive.saturating_add(1);
                        shared.failures.fetch_add(1, Ordering::SeqCst);
                    }
                }
                match stopped.recv_timeout(schedule.delay(consecutive)) {
                    Err(RecvTimeoutError::Timeout) => {}
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        })?;
    Ok(LoopHandle {
        counters,
        stop: Some(stop),
        thread: Some(thread),
        joined: None,
    })
}
