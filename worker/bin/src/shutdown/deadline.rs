//! One atomic absolute deadline; request time is derived, never separately raced.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub const MAX_SHUTDOWN_BUDGET: Duration = Duration::from_secs(25);
const UNREQUESTED: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeadlineError {
    InvalidBudget,
    TimestampOverflow,
    DeadlineOverflow,
}

impl fmt::Display for DeadlineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidBudget => "shutdown budget must be positive and at most 25 seconds",
            Self::TimestampOverflow => "shutdown observation exceeds integer nanosecond range",
            Self::DeadlineOverflow => "shutdown deadline overflows or collides with sentinel",
        })
    }
}

impl std::error::Error for DeadlineError {}

#[derive(Debug)]
pub struct ShutdownDeadline {
    budget_ns: u64,
    deadline_ns: AtomicU64,
}

impl ShutdownDeadline {
    pub fn new(budget: Duration) -> Result<Self, DeadlineError> {
        if budget.is_zero() || budget > MAX_SHUTDOWN_BUDGET {
            return Err(DeadlineError::InvalidBudget);
        }
        let budget_ns =
            u64::try_from(budget.as_nanos()).map_err(|_| DeadlineError::InvalidBudget)?;
        Ok(Self {
            budget_ns,
            deadline_ns: AtomicU64::new(UNREQUESTED),
        })
    }

    /// Record a real observation in the consumer's common monotonic coordinates.
    /// Earlier observations published later shorten the deadline. The returned
    /// value is current at publication; consumers must reread before deciding.
    /// Errors leave state untouched, never substitute a later observation.
    pub fn request_at(&self, observed_at: Duration) -> Result<Duration, DeadlineError> {
        let observed_ns =
            u64::try_from(observed_at.as_nanos()).map_err(|_| DeadlineError::TimestampOverflow)?;
        let candidate = observed_ns
            .checked_add(self.budget_ns)
            .filter(|value| *value != UNREQUESTED)
            .ok_or(DeadlineError::DeadlineOverflow)?;
        let previous = self.deadline_ns.fetch_min(candidate, Ordering::AcqRel);
        Ok(Duration::from_nanos(previous.min(candidate)))
    }

    pub fn deadline(&self) -> Option<Duration> {
        let value = self.deadline_ns.load(Ordering::Acquire);
        (value != UNREQUESTED).then(|| Duration::from_nanos(value))
    }

    pub fn requested_at(&self) -> Option<Duration> {
        self.deadline()?
            .checked_sub(Duration::from_nanos(self.budget_ns))
    }
}
