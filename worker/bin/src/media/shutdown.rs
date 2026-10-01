//! Root-owned close permission and one absolute monotonic shutdown deadline.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::shutdown::{DeadlineError, ShutdownDeadline};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutdownError {
    NotStarted,
    Expired,
}

impl fmt::Display for ShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NotStarted => "media shutdown has not started",
            Self::Expired => "media shutdown deadline has expired",
        })
    }
}

impl std::error::Error for ShutdownError {}

/// All callers share the root/signal deadline and monotonic clock origin.
/// Permission never implies stop and is effective only before the current deadline.
pub struct ShutdownControl {
    deadline: Arc<ShutdownDeadline>,
    permitted: AtomicBool,
}

impl ShutdownControl {
    pub fn new(deadline: Arc<ShutdownDeadline>) -> Self {
        Self {
            deadline,
            permitted: AtomicBool::new(false),
        }
    }

    /// Media EOF/failure may initiate shutdown; earlier observations published
    /// later can still shorten the returned deadline.
    pub fn begin(&self, now: Duration) -> Result<Duration, DeadlineError> {
        self.deadline.request_at(now)
    }

    pub fn deadline(&self) -> Option<Duration> {
        self.deadline.deadline()
    }

    /// Only the root grants this, after draining policy/delivery and closing
    /// GPU owners. Repeated grants do not renew either permission or time.
    pub fn permit_close(&self, now: Duration) -> Result<(), ShutdownError> {
        let deadline = self.deadline().ok_or(ShutdownError::NotStarted)?;
        if now >= deadline {
            return Err(ShutdownError::Expired);
        }
        self.permitted.store(true, Ordering::Release);
        Ok(())
    }

    pub fn close_permitted(&self, now: Duration) -> bool {
        self.permitted.load(Ordering::Acquire)
            && self.deadline().is_some_and(|deadline| now < deadline)
    }

    /// Whole milliseconds only: zero must never reach a native timeout.
    /// Remaining spans are capped at the largest representable native timeout.
    pub fn remaining_ms(&self, now: Duration) -> Option<u32> {
        let deadline = self.deadline()?;
        let milliseconds = deadline.checked_sub(now)?.as_millis();
        if milliseconds == 0 {
            return None;
        }
        Some(milliseconds.min(u128::from(u32::MAX)) as u32)
    }
}
