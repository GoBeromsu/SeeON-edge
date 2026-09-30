//! Python `worker/runtime/config/restart.py` L15-85: the restart directive,
//! its tracker and the interval-gated check. The generation is
//! `restart_epoch` (`restart.py:25`). Any strict advance of the
//! `(generation, version, registry)` tuple yields a directive, so a bump of
//! `restart_epoch`, `config_version` or `registry_version` alone restarts.
//! The caller exits with `Exit::CleanShutdown` on a directive.

use std::time::Duration;

use crate::exit::Exit;
use crate::seam::Clock;

/// Python `RestartDirective(generation, version, registry)`, ordered as a
/// tuple; the same triple names the LKG revision file.
pub use crate::config::lkg::Directive as RestartDirective;

/// Python `make_restart_check(poll_interval_sec=60.0)`.
pub const RESTART_POLL_INTERVAL: Duration = Duration::from_secs(60);

/// The process exit a restart directive asks for.
pub const RESTART_EXIT: Exit = Exit::CleanShutdown;

/// Python `RestartDirectiveTracker`.
#[derive(Clone, Debug)]
pub struct RestartTracker {
    current: RestartDirective,
}

impl RestartTracker {
    /// Starts at the directive the process booted with.
    pub fn new(boot: RestartDirective) -> Self {
        Self { current: boot }
    }

    /// The highest directive observed so far.
    pub fn current(&self) -> RestartDirective {
        self.current
    }

    /// Python `observe`: a candidate at or below the current directive is
    /// ignored (so a higher LKG never loops restarts); a higher candidate
    /// becomes current and is returned as the directive to restart for.
    pub fn observe(&mut self, candidate: RestartDirective) -> Option<RestartDirective> {
        if candidate <= self.current {
            return None;
        }
        self.current = candidate;
        Some(candidate)
    }
}

/// Python `make_restart_check`: pulls at most once per interval; a failed
/// pull (`None`) yields no directive.
#[derive(Clone, Debug)]
pub struct RestartCheck {
    tracker: RestartTracker,
    interval: Duration,
    last_checked: Option<Duration>,
}

impl RestartCheck {
    /// `interval` is normally `RESTART_POLL_INTERVAL`.
    pub fn new(boot: RestartDirective, interval: Duration) -> Self {
        Self {
            tracker: RestartTracker::new(boot),
            interval,
            last_checked: None,
        }
    }

    /// The directive to restart for, if any. `pull` runs only when the
    /// interval has passed since the last pull (the first call always
    /// pulls, as Python starts at `-interval`).
    pub fn check<F>(&mut self, clock: &dyn Clock, pull: F) -> Option<RestartDirective>
    where
        F: FnOnce() -> Option<RestartDirective>,
    {
        let now = clock.monotonic();
        if let Some(last) = self.last_checked
            && now.saturating_sub(last) < self.interval
        {
            return None;
        }
        self.last_checked = Some(now);
        self.tracker.observe(pull()?)
    }

    /// The tracker, for the current directive.
    pub fn tracker(&self) -> &RestartTracker {
        &self.tracker
    }
}
