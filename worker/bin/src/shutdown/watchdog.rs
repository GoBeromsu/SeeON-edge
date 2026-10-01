//! Independent process-lifetime enforcement, not a root/media join timeout.

use std::io;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::seam::{Clock, SystemClock};

use super::ShutdownDeadline;

// Bounded polling revisits a deadline shortened while asleep. This is not a
// hard-real-time guarantee: scheduling, SIGSTOP, suspend and kernel _exit
// descriptor cleanup can delay termination beyond the user-space budget.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

pub(super) fn start(clock: Arc<SystemClock>, deadline: Arc<ShutdownDeadline>) -> io::Result<()> {
    let thread = thread::Builder::new()
        .name("shutdown-watchdog".into())
        .spawn(move || {
            loop {
                let wait = match deadline.deadline() {
                    Some(absolute) => {
                        let now = clock.monotonic();
                        if now >= absolute {
                            signal_hook::low_level::exit(1);
                        }
                        absolute.saturating_sub(now).min(POLL_INTERVAL)
                    }
                    None => POLL_INTERVAL,
                };
                // Never hold a root/media lock, wait for an owner, or renew a budget.
                clock.pause(wait);
            }
        })?;
    // Detach; the thread owns clock/deadline independently of the public handle.
    // There is deliberately no cancellation, Drop shutdown, or "complete" API.
    drop(thread);
    Ok(())
}
