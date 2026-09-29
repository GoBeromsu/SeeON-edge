//! The one deadline-bounded condition loop (D6). The pause between checks
//! goes through the `Clock` seam and never decides an outcome.

use std::fmt;
use std::time::Duration;

use crate::seam::Clock;

/// Longest pause between two condition checks.
pub const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// `what` names the awaited condition; no other detail is carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Timeout {
    pub what: &'static str,
}

impl fmt::Display for Timeout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "timed out waiting for {}", self.what)
    }
}
impl std::error::Error for Timeout {}

/// Checks `f` until it returns true or `clock.monotonic()` reaches `deadline`.
/// `f` is checked before the first pause, and no pause runs past the deadline.
pub fn poll_until(
    clock: &dyn Clock,
    deadline: Duration,
    what: &'static str,
    mut f: impl FnMut() -> bool,
) -> Result<(), Timeout> {
    loop {
        if f() {
            return Ok(());
        }
        let now = clock.monotonic();
        if now < deadline {
            clock.pause((deadline - now).min(POLL_INTERVAL));
        } else {
            return Err(Timeout { what });
        }
    }
}
