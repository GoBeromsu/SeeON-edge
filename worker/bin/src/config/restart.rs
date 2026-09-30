//! Python `worker/runtime/config/restart.py` L15-85: the restart directive,
//! its tracker and the interval-gated check. The generation is
//! `restart_epoch` (`restart.py:25`). A changed epoch or a changed camera
//! roster yields a directive (design row 3: `MediaConfig.sources` is fixed
//! at open, so a roster change restarts and never hot-swaps); a directive
//! that only moves `config_version` or `registry_version` yields none. The
//! caller exits with `Exit::CleanShutdown` on a directive.

use std::time::Duration;

use crate::exit::Exit;
use crate::relay::cameras::{RuntimeCamera, WorkerConfigPayload};
use crate::seam::Clock;

/// Python `RestartDirective(generation, version, registry)`, ordered as a
/// tuple; the same triple names the LKG revision file.
pub use crate::config::lkg::Directive as RestartDirective;

/// Python `make_restart_check(poll_interval_sec=60.0)`.
pub const RESTART_POLL_INTERVAL: Duration = Duration::from_secs(60);

/// The process exit a restart directive asks for.
pub const RESTART_EXIT: Exit = Exit::CleanShutdown;

/// The camera roster the media sources were opened with: sorted
/// `(camera_id, rtsp_url)` pairs of the runtime cameras.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Roster(Vec<(String, String)>);

impl Roster {
    /// The roster of already-resolved runtime cameras.
    pub fn from_cameras(cameras: &[RuntimeCamera]) -> Self {
        let mut pairs: Vec<(String, String)> = cameras
            .iter()
            .map(|camera| (camera.camera_id.clone(), camera.rtsp_url.clone()))
            .collect();
        pairs.sort();
        pairs.dedup();
        Self(pairs)
    }

    /// The roster a pulled config would open; `None` when its cameras do
    /// not resolve (Python `to_worker_config` raises).
    pub fn from_config(config: &WorkerConfigPayload) -> Option<Self> {
        config
            .runtime_cameras()
            .ok()
            .map(|cameras| Self::from_cameras(&cameras))
    }

    /// The `(camera_id, rtsp_url)` pairs, sorted.
    pub fn pairs(&self) -> &[(String, String)] {
        &self.0
    }
}

/// Python `RestartDirectiveTracker`, narrowed to epoch and roster changes.
#[derive(Clone, Debug)]
pub struct RestartTracker {
    current: RestartDirective,
    boot_roster: Roster,
}

impl RestartTracker {
    /// Starts at the directive and roster the process booted with.
    pub fn new(boot: RestartDirective, boot_roster: Roster) -> Self {
        Self {
            current: boot,
            boot_roster,
        }
    }

    /// The highest directive observed so far.
    pub fn current(&self) -> RestartDirective {
        self.current
    }

    /// Python `observe`: a candidate at or below the current directive is
    /// ignored (so a higher LKG never loops restarts). A higher candidate
    /// becomes current; it restarts when its `restart_epoch` differs or its
    /// roster differs from the boot roster.
    pub fn observe(
        &mut self,
        candidate: RestartDirective,
        roster: &Roster,
    ) -> Option<RestartDirective> {
        if candidate <= self.current {
            return None;
        }
        let epoch_changed = candidate.generation != self.current.generation;
        self.current = candidate;
        (epoch_changed || *roster != self.boot_roster).then_some(candidate)
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
    pub fn new(boot: RestartDirective, boot_roster: Roster, interval: Duration) -> Self {
        Self {
            tracker: RestartTracker::new(boot, boot_roster),
            interval,
            last_checked: None,
        }
    }

    /// The directive to restart for, if any. `pull` runs only when the
    /// interval has passed since the last pull (the first call always
    /// pulls, as Python starts at `-interval`).
    pub fn check<F>(&mut self, clock: &dyn Clock, pull: F) -> Option<RestartDirective>
    where
        F: FnOnce() -> Option<(RestartDirective, Roster)>,
    {
        let now = clock.monotonic();
        if let Some(last) = self.last_checked
            && now.saturating_sub(last) < self.interval
        {
            return None;
        }
        self.last_checked = Some(now);
        let (candidate, roster) = pull()?;
        self.tracker.observe(candidate, &roster)
    }

    /// The tracker, for the current directive.
    pub fn tracker(&self) -> &RestartTracker {
        &self.tracker
    }
}
