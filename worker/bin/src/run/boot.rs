//! Boot orchestration in the approved order: settings, lease, identity, CUDA, owners.

use std::fmt;
use std::io;
use std::sync::Arc;

use super::models::{CleanupError, ModelOwners, ModelStartError, StartKind};
use super::settings::{self, Settings};
use super::status::{BootReason, BootStatusContext, ReportOutcome};
use super::{Admitted, IdentityError};
use crate::exit::Exit;
use crate::gpu::lease::{self, LeaseError};
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;
use crate::telemetry::gpu::GpuStatus;

#[path = "startup.rs"]
pub(super) mod startup;
pub use startup::BootStopped;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseFailure {
    Unavailable,
    CreateDir(io::ErrorKind),
    Open(rustix::io::Errno),
    Lock(rustix::io::Errno),
}

impl From<LeaseError> for LeaseFailure {
    fn from(error: LeaseError) -> Self {
        match error {
            LeaseError::Unavailable { .. } => Self::Unavailable,
            LeaseError::CreateDir { kind, .. } => Self::CreateDir(kind),
            LeaseError::Open { errno, .. } => Self::Open(errno),
            LeaseError::Lock { errno, .. } => Self::Lock(errno),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BootCause {
    Lease(LeaseFailure),
    Identity(IdentityError),
    CudaUnavailable,
    Model(ModelStartError),
}

/// The failure has already attempted its one report. Never repost it in main.
/// `owners` retains partial startup and the lease, including timed-out threads.
/// Parent must terminate the process, not reopen models after a failed boot.
pub struct BootFailure {
    pub reason: BootReason,
    pub cause: BootCause,
    pub gpu: GpuStatus,
    pub report: ReportOutcome,
    pub cleanup: Vec<CleanupError>,
    pub owners: Option<Box<ModelOwners>>,
    shutdown: Arc<ShutdownDeadline>,
}

impl BootFailure {
    pub fn exit(&self, clock: &dyn Clock) -> Exit {
        if self.cleanup.is_empty() && startup::cutoff(&self.shutdown, clock).is_ok() {
            self.reason.exit()
        } else {
            Exit::Runtime
        }
    }
}

impl fmt::Debug for BootFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BootFailure")
            .field("reason", &self.reason)
            .field("cause", &self.cause)
            .field("report", &self.report)
            .field("cleanup", &self.cleanup)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub enum BootError {
    Failed(Box<BootFailure>),
    Stopped(Box<BootStopped>),
}

impl BootError {
    pub fn exit(&self, clock: &dyn Clock) -> Exit {
        match self {
            Self::Failed(failure) => failure.exit(clock),
            Self::Stopped(stopped) => stopped.exit(clock),
        }
    }
}

/// Successful boot keeps all warmed owners, channels and lease alive.
/// The parent must recheck Flow batch admission against the eventual pulled roster.
pub struct Booted {
    pub settings: Settings,
    pub admitted: Admitted,
    pub gpu: GpuStatus,
    pub models: ModelOwners,
}

/// Runs steps 2–4 using step-1 validated settings and reporting context.
/// Genuine failures probe real CUDA state and make exactly one 2s status attempt.
/// Cancellation makes no failure report and does not enter native diagnostics.
/// The clock must share the signal watchdog's monotonic origin.
pub fn boot(
    settings: Settings,
    report: BootStatusContext,
    clock: &dyn Clock,
    shutdown: Arc<ShutdownDeadline>,
) -> Result<Booted, BootError> {
    let policy = settings.policy();
    if shutdown.deadline().is_some() {
        return Err(BootError::Stopped(startup::stopped(None, clock, shutdown)));
    }
    let lease = match lease::acquire(settings.state_dir()) {
        Ok(lease) => lease,
        Err(error) => {
            return Err(fail(
                BootCause::Lease(error.into()),
                report,
                None,
                None,
                clock,
                &shutdown,
                policy.device_ordinal,
            ));
        }
    };
    let mut models = ModelOwners::new(lease, Arc::clone(&shutdown));
    if shutdown.deadline().is_some() {
        return Err(BootError::Stopped(startup::stopped(
            Some(models),
            clock,
            shutdown,
        )));
    }
    let admitted = match settings::admit(&settings) {
        Ok(admitted) => admitted,
        Err(error) => {
            return Err(fail(
                BootCause::Identity(error),
                report,
                None,
                Some(models),
                clock,
                &shutdown,
                policy.device_ordinal,
            ));
        }
    };
    if shutdown.deadline().is_some() {
        return Err(BootError::Stopped(startup::stopped(
            Some(models),
            clock,
            shutdown,
        )));
    }
    let gpu = GpuStatus::probe(policy.device_ordinal, clock);
    if !gpu.cuda_context_ok {
        return Err(fail(
            BootCause::CudaUnavailable,
            report,
            Some(gpu),
            Some(models),
            clock,
            &shutdown,
            policy.device_ordinal,
        ));
    }
    if shutdown.deadline().is_some() {
        return Err(BootError::Stopped(startup::stopped(
            Some(models),
            clock,
            shutdown,
        )));
    }
    if let Err(error) = models.start(&admitted.engines, policy, clock) {
        if error.kind == StartKind::Stopped {
            return Err(BootError::Stopped(startup::stopped(
                Some(models),
                clock,
                shutdown,
            )));
        }
        return Err(fail(
            BootCause::Model(error),
            report,
            Some(gpu),
            Some(models),
            clock,
            &shutdown,
            policy.device_ordinal,
        ));
    }
    if shutdown.deadline().is_some() {
        return Err(BootError::Stopped(startup::stopped(
            Some(models),
            clock,
            shutdown,
        )));
    }
    Ok(Booted {
        settings,
        admitted,
        gpu,
        models,
    })
}

fn fail(
    cause: BootCause,
    context: BootStatusContext,
    gpu: Option<GpuStatus>,
    mut owners: Option<ModelOwners>,
    clock: &dyn Clock,
    shutdown: &Arc<ShutdownDeadline>,
    device_ordinal: i32,
) -> BootError {
    // Publish the failure observation before cleanup, native diagnostics or HTTP.
    let cleanup = startup::cleanup(owners.as_mut(), clock, shutdown);
    let gpu = match gpu {
        Some(gpu) => gpu,
        None => GpuStatus::probe(device_ordinal, clock),
    };
    let reason = match &cause {
        BootCause::Lease(_) => BootReason::GpuLease,
        BootCause::Identity(_) => BootReason::EngineIdentity,
        BootCause::CudaUnavailable => BootReason::CudaUnavailable,
        BootCause::Model(_) => BootReason::EngineOpen,
    };
    let report = context.send(reason, &gpu);
    BootError::Failed(Box::new(BootFailure {
        reason,
        cause,
        gpu,
        report,
        cleanup,
        owners: owners.map(Box::new),
        shutdown: Arc::clone(shutdown),
    }))
}
