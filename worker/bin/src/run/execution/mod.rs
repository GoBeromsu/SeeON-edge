//! Production run orchestration. Parent registers this module and calls
//! [`execute`] from Run/default dispatch. `Ok` is a completed lifecycle, mapped
//! by the caller to `Exit::CleanShutdown`. Config pull precedes the GPU lease.
//! An empty admitted roster is legal and still runs model warm gates.
//! Delivery egress is owned independently of policy execution.

mod lifecycle;
mod output;
mod publication;
mod recording;
mod runtime;

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cli::Flags;
use crate::config::env::Env;
use crate::config::pull::{self, PullError};
use crate::exit::Exit;
use crate::run::boot::{BootError, boot};
use crate::run::event_payload::is_uuid;
use crate::run::settings::{BootPolicy, Settings, SettingsError};
use crate::run::status::{BootStatusContext, ReportContextError, ReportOutcome};
use crate::seam::{Clock, IdSource, RandomIds, SystemClock};
use crate::shutdown::{self, MAX_SHUTDOWN_BUDGET};

use lifecycle::RunExit;
use output::OutputError;
use runtime::RuntimeError;

/// Baked production relay. Python `--config` is refused, so this is the only URL.
pub const RELAY_URL: &str = "http://ml-api:8000";
const READINESS_BUDGET: Duration = Duration::from_secs(30);
/// Image stored-pose operating point. Pass 3 owns its request path.
const STORED_POSE_THRESHOLD: f64 = 0.25;

/// A refused or incomplete production run. Display never includes a URI or token.
#[derive(Debug)]
pub enum RunFailure {
    Settings(SettingsError),
    Clock,
    Identity,
    Release(PullError),
    Pull(PullError),
    Report(ReportContextError),
    Signal,
    Boot { error: BootError, exit: Exit },
    Output(OutputError),
    Runtime(RuntimeError),
    Shutdown(RunExit),
}

impl RunFailure {
    pub fn exit(&self) -> Exit {
        match self {
            Self::Settings(error) => error.exit(),
            Self::Clock | Self::Identity | Self::Signal => Exit::Runtime,
            Self::Release(error) | Self::Pull(error) => error.exit(),
            Self::Report(error) => error.exit(),
            Self::Boot { exit, .. } => *exit,
            Self::Output(error) => error.exit(),
            Self::Runtime(error) => error.exit(),
            Self::Shutdown(exit) => exit.exit(),
        }
    }
}

impl fmt::Display for RunFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Settings(_) => formatter.write_str("worker settings refused"),
            Self::Clock => formatter.write_str("worker clock is not representable"),
            Self::Identity => formatter.write_str("worker boot identity is not a canonical uuid"),
            Self::Release(error) => write!(formatter, "release identity refused: {error:?}"),
            Self::Pull(error) => write!(formatter, "worker config pull refused: {error:?}"),
            Self::Report(_) => formatter.write_str("boot status context refused"),
            Self::Signal => formatter.write_str("shutdown signal registration failed"),
            Self::Boot {
                error: BootError::Failed(failure),
                ..
            } => {
                write!(
                    formatter,
                    "boot refused: reason={} report={}",
                    failure.reason.wire(),
                    failure.report
                )
            }
            Self::Boot {
                error: BootError::Stopped(_),
                ..
            } => formatter.write_str("boot stopped before model owners"),
            Self::Output(error) => write!(formatter, "production output refused: {error}"),
            Self::Runtime(error) => write!(formatter, "production runtime refused: {error}"),
            Self::Shutdown(exit) => write!(formatter, "worker stopped with error: {exit:?}"),
        }
    }
}

impl fmt::Display for ReportOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted { generation } => write!(formatter, "accepted generation {generation}"),
            Self::Transport => formatter.write_str("transport"),
            Self::Status(status) => write!(formatter, "status {status}"),
            Self::Malformed => formatter.write_str("malformed"),
            Self::Payload => formatter.write_str("payload"),
            Self::NoFacility => formatter.write_str("no facility"),
        }
    }
}

/// Runs the real lifecycle. Success is returned only after shutdown completes.
/// A boot failure has already consumed its single report, including `NoFacility`.
pub fn execute(env: Env, flags: Flags) -> Result<(), RunFailure> {
    let mut settings =
        Settings::from_flags(env, flags, production_policy()).map_err(RunFailure::Settings)?;
    let clock = Arc::new(SystemClock::new());
    let started_at = unix_seconds(clock.wall()).ok_or(RunFailure::Clock)?;
    let boot_id = RandomIds.uuid4().map_err(|_| RunFailure::Identity)?;
    if !is_uuid(&boot_id) {
        return Err(RunFailure::Identity);
    }
    let token = settings
        .relay_token()
        .map_err(SettingsError::Environment)
        .map_err(RunFailure::Settings)?;
    pull::check_release_identity(RELAY_URL).map_err(RunFailure::Release)?;
    let config = pull::pull_startup_config(
        RELAY_URL,
        token,
        settings.state_dir(),
        std::path::Path::new(crate::config::windows::ZONEINFO_DIR),
    )
    .map_err(RunFailure::Pull)?;
    let report = BootStatusContext::for_config(RELAY_URL, token, &config, started_at)
        .map_err(RunFailure::Report)?;
    let signals = shutdown::register(Arc::clone(&clock), MAX_SHUTDOWN_BUDGET)
        .map_err(|_| RunFailure::Signal)?;
    let deadline = signals.shared_deadline();
    // Empty roster has no deployed source-batch requirement.
    settings.policy.deployed_batch = deployed_batch(&config);
    let booted =
        boot(settings, report, clock.as_ref(), Arc::clone(&deadline)).map_err(|error| {
            let exit = error.exit(clock.as_ref());
            RunFailure::Boot { error, exit }
        })?;
    let prepared = output::prepare(
        &booted,
        &config,
        &boot_id,
        Arc::clone(&clock) as Arc<dyn Clock>,
        Arc::clone(&deadline),
    );
    let mut session = match prepared {
        Ok(session) => session,
        Err(failure) => {
            return failed_preparation(failure, booted, clock.as_ref(), &deadline);
        }
    };
    let (outcome, sink) = runtime::run(
        &mut session,
        &booted,
        clock.as_ref(),
        &deadline,
        config.directive,
    );
    match lifecycle::shutdown(session, booted, clock.as_ref(), &deadline, outcome, sink) {
        RunExit::Clean => Ok(()),
        other => Err(RunFailure::Shutdown(other)),
    }
}

fn failed_preparation(
    failure: output::PrepareFailure,
    mut booted: crate::run::Booted,
    clock: &dyn Clock,
    deadline: &Arc<shutdown::ShutdownDeadline>,
) -> Result<(), RunFailure> {
    let output::PrepareFailure { error, session } = failure;
    eprintln!("ml-worker: output startup refused: {error}");
    let cleaned = if let Some(session) = session {
        // No policy turn has run; this is the real initial empty sink, not a
        // replacement for a sink containing accepted work.
        let sink = runtime::LiveSink::new(session.publications.clock());
        lifecycle::shutdown(
            *session,
            booted,
            clock,
            deadline,
            runtime::RunOutcome::Stopped,
            sink,
        )
    } else {
        // Static output admission failed before any output owner was started.
        let requested = deadline.request_at(clock.monotonic()).is_ok();
        let errors = booted.models.close(clock);
        if !requested
            || !errors.is_empty()
            || crate::run::boot::startup::cutoff(deadline, clock).is_err()
        {
            booted.models.retain_lease_until_process_exit();
            RunExit::Runtime
        } else {
            RunExit::Clean
        }
    };
    if cleaned != RunExit::Clean {
        return Err(RunFailure::Shutdown(cleaned));
    }
    if error.exit() == Exit::CleanShutdown {
        Ok(())
    } else {
        Err(RunFailure::Output(error))
    }
}

fn production_policy() -> BootPolicy {
    BootPolicy {
        device_ordinal: 0,
        stored_pose_threshold: STORED_POSE_THRESHOLD,
        deployed_batch: None,
        readiness_budget: READINESS_BUDGET,
    }
}
fn deployed_batch(config: &crate::config::pull::PulledConfig) -> Option<i128> {
    batch_for(config.cameras.len())
}

fn unix_seconds(at: SystemTime) -> Option<f64> {
    let seconds = at.duration_since(UNIX_EPOCH).ok()?.as_secs_f64();
    seconds.is_finite().then_some(seconds)
}

fn batch_for(cameras: usize) -> Option<i128> {
    match cameras {
        0 => None,
        count => i128::try_from(count).ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumed_no_facility_report_is_named_without_reposting() {
        assert_eq!(ReportOutcome::NoFacility.to_string(), "no facility");
        let text = format!("{}", ReportOutcome::NoFacility);
        assert!(!text.contains("http"));
    }

    #[test]
    fn missing_config_keeps_the_original_pull_exit() {
        assert_eq!(RunFailure::Pull(PullError::NoConfig).exit(), Exit::Config);
        assert_eq!(
            RunFailure::Release(PullError::FormatMismatch).exit(),
            Exit::RefuseToStart
        );
    }
    #[test]
    fn empty_roster_leaves_deployed_batch_unset() {
        assert_eq!(batch_for(0), None);
        assert_eq!(batch_for(3), Some(3));
    }
}
