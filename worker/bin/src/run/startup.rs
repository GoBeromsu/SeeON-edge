//! Startup cancellation and cleanup use the process-wide shutdown authority.
//! Native calls are not cancellable here; the independent watchdog owns exit.

use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::TryRecvError;
use std::time::Duration;

use super::super::ModelRole;
use super::super::models::{CleanupError, ModelOwners, ModelStartError, StartKind};
use crate::exit::Exit;
use crate::gpu::owners::{JoinError, Owner};
use crate::poll::poll_until;
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;

/// Cancellation is not a model failure and never sends a boot-failure report.
/// Retain partial owners until the root acts on this outcome.
pub struct BootStopped {
    pub cleanup: Vec<CleanupError>,
    pub owners: Option<Box<ModelOwners>>,
    shutdown: Arc<ShutdownDeadline>,
}

impl BootStopped {
    /// Recheck at exit: a late, earlier signal receipt can revoke clean shutdown.
    pub fn exit(&self, clock: &dyn Clock) -> Exit {
        if self.cleanup.is_empty() && cutoff(&self.shutdown, clock).is_ok() {
            Exit::CleanShutdown
        } else {
            Exit::Runtime
        }
    }
}

impl fmt::Debug for BootStopped {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BootStopped")
            .field("cleanup", &self.cleanup)
            .finish_non_exhaustive()
    }
}

pub(super) fn stopped(
    mut owners: Option<ModelOwners>,
    clock: &dyn Clock,
    shutdown: Arc<ShutdownDeadline>,
) -> Box<BootStopped> {
    let cleanup = cleanup(owners.as_mut(), clock, &shutdown);
    Box::new(BootStopped {
        cleanup,
        owners: owners.map(Box::new),
        shutdown,
    })
}

pub(super) fn cleanup(
    owners: Option<&mut ModelOwners>,
    clock: &dyn Clock,
    shutdown: &ShutdownDeadline,
) -> Vec<CleanupError> {
    if let Some(owners) = owners {
        return owners.close(clock);
    }
    match shutdown.request_at(clock.monotonic()) {
        Err(error) => vec![CleanupError::Control(error)],
        Ok(_) => cutoff(shutdown, clock).err().into_iter().collect(),
    }
}

/// Read time before the authority so publication during a clock read is visible.
pub(crate) fn cutoff(
    shutdown: &ShutdownDeadline,
    clock: &dyn Clock,
) -> Result<Duration, CleanupError> {
    let now = clock.monotonic();
    match shutdown.deadline() {
        Some(deadline) if now < deadline => Ok(deadline),
        Some(_) => Err(CleanupError::Expired),
        None => Err(CleanupError::NotRequested),
    }
}

pub(crate) fn wait<R>(
    owner: Option<&Owner<R>>,
    model: ModelRole,
    shutdown: &ShutdownDeadline,
    clock: &dyn Clock,
    readiness_deadline: Duration,
) -> Result<(), ModelStartError> {
    let error = |kind| ModelStartError { model, kind };
    let Some(owner) = owner else {
        return Err(error(StartKind::ReadinessClosed));
    };
    let mut result = None;
    let bounded = poll_until(clock, readiness_deadline, "GPU model readiness", || {
        let now = clock.monotonic();
        result = if shutdown.deadline().is_some() {
            Some(Err(StartKind::Stopped))
        } else if now >= readiness_deadline {
            Some(Err(StartKind::Deadline))
        } else {
            match owner.readiness.try_recv() {
                Ok(ready) => Some(ready.map_err(StartKind::Refused)),
                Err(TryRecvError::Disconnected) => Some(Err(StartKind::ReadinessClosed)),
                Err(TryRecvError::Empty) => None,
            }
        };
        result.is_some()
    });
    // A refusal already observed is a genuine model failure, even if a signal
    // is published immediately afterwards. Do not relabel it as cancellation.
    if let Some(Err(kind @ (StartKind::Refused(_) | StartKind::ReadinessClosed))) = result {
        return Err(error(kind));
    }
    if shutdown.deadline().is_some() {
        return Err(error(StartKind::Stopped));
    }
    match (bounded, result) {
        (Ok(()), Some(result)) => result.map_err(error),
        (Err(_), _) => Err(error(StartKind::Deadline)),
        (Ok(()), None) => Err(error(StartKind::ReadinessClosed)),
    }
}

pub(crate) fn close_one<R>(
    owner: &mut Option<Owner<R>>,
    model: ModelRole,
    clock: &dyn Clock,
    shutdown: &ShutdownDeadline,
) -> Result<(), CleanupError> {
    let upper_bound = cutoff(shutdown, clock)?;
    let Some(current) = owner.as_ref() else {
        return Ok(());
    };
    let mut expired = None;
    let bounded = poll_until(clock, upper_bound, "GPU owner cleanup", || {
        // The poll endpoint is only an upper bound, never the authority.
        expired = cutoff(shutdown, clock).err();
        expired.is_some() || current.thread.is_finished()
    });
    if let Some(error) = expired {
        return Err(error);
    }
    bounded.map_err(|error| CleanupError::Owner {
        model,
        error: JoinError::Timeout(error),
    })?;
    cutoff(shutdown, clock)?;
    // The low-level owners::join consumes live handles on timeout. Only take a
    // finished handle and join directly, with no second captured-deadline wait.
    if let Some(finished) = owner.take() {
        finished.thread.join().map_err(|_| CleanupError::Owner {
            model,
            error: JoinError::Panicked,
        })?;
    }
    cutoff(shutdown, clock)?;
    Ok(())
}
