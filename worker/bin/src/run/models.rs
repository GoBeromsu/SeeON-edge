//! Real model owners and bounded partial-start cleanup. Inputs must already be admitted.

#[path = "model_inputs.rs"]
mod inputs;
pub use inputs::{CpuModels, ModelInputs};

#[cfg(test)]
#[path = "model_cpu_tests.rs"]
mod cpu_tests;

#[cfg(test)]
#[path = "model_state_tests.rs"]
mod state_tests;

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use super::ModelRole;
use super::boot::startup::{close_one, cutoff, wait};
use super::settings::BootPolicy;
use crate::exit::Exit;
use crate::gpu::lease::GpuLease;
use crate::inference::{JoinError, Owner, Runtime, State};
use crate::msg::{BedRequest, FallRequest, FallResponse, StoredPoseRequest};
use crate::seam::Clock;
use crate::shutdown::{DeadlineError, ShutdownDeadline};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartKind {
    Spawn(io::ErrorKind),
    Refused(Exit),
    ReadinessClosed,
    Deadline,
    Stopped,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelStartError {
    pub model: ModelRole,
    pub kind: StartKind,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CleanupError {
    Control(DeadlineError),
    NotRequested,
    Expired,
    Owner { model: ModelRole, error: JoinError },
}

/// Holds request channels, fall responses, native owner threads and the GPU lease.
/// Call `close` before dropping, then retain this object until media has closed.
/// A timed-out owner is retained for the caller; it cannot safely be killed in Rust.
pub struct ModelOwners {
    fall: Option<Owner<FallRequest>>,
    bed: Option<Owner<BedRequest>>,
    stored_pose: Option<Owner<StoredPoseRequest>>,
    fall_responses: Option<Receiver<FallResponse>>,
    stop: Arc<AtomicBool>,
    shutdown: Arc<ShutdownDeadline>,
    lease: Option<GpuLease>,
    states: [Option<Arc<State>>; 3],
}

impl ModelOwners {
    pub(crate) fn new(lease: GpuLease, shutdown: Arc<ShutdownDeadline>) -> Self {
        Self {
            fall: None,
            bed: None,
            stored_pose: None,
            fall_responses: None,
            stop: Arc::new(AtomicBool::new(false)),
            shutdown,
            lease: Some(lease),
            states: [None, None, None],
        }
    }

    pub fn fall(&self) -> Option<&Owner<FallRequest>> {
        self.fall.as_ref()
    }
    pub fn bed(&self) -> Option<&Owner<BedRequest>> {
        self.bed.as_ref()
    }
    pub fn stored_pose(&self) -> Option<&Owner<StoredPoseRequest>> {
        self.stored_pose.as_ref()
    }
    pub fn fall_responses(&self) -> Option<&Receiver<FallResponse>> {
        self.fall_responses.as_ref()
    }

    /// Retained after joins: shutdown must not erase a fault or loaded runtime identity.
    pub fn failure(&self) -> Option<Exit> {
        let mut first = None;
        for state in self.states.iter().flatten() {
            if let Some(exit) = state.failure() {
                if exit == Exit::FatalAccelerator {
                    return Some(exit);
                }
                first.get_or_insert(exit);
            }
        }
        first
    }

    pub fn runtime(&self, role: ModelRole) -> Option<&Runtime> {
        let index = match role {
            ModelRole::Fall => 0,
            ModelRole::Bed => 1,
            ModelRole::StoredPose => 2,
        };
        self.states[index].as_ref()?.runtime()
    }

    pub(crate) fn start(
        &mut self,
        inputs: ModelInputs<'_>,
        policy: BootPolicy,
        clock: &dyn Clock,
    ) -> Result<(), ModelStartError> {
        let deadline = clock.monotonic().saturating_add(policy.readiness_budget);
        let spawn_error = |model, error: io::Error| ModelStartError {
            model,
            kind: StartKind::Spawn(error.kind()),
        };
        self.before_spawn(ModelRole::Fall, clock, deadline)?;
        let (owner, responses) = inputs
            .spawn_fall(policy, Arc::clone(&self.stop))
            .map_err(|error| spawn_error(ModelRole::Fall, error))?;
        self.states[0] = Some(Arc::clone(&owner.state));
        self.fall = Some(owner);
        self.fall_responses = Some(responses);
        wait(
            self.fall.as_ref(),
            ModelRole::Fall,
            &self.shutdown,
            clock,
            deadline,
        )?;
        self.before_spawn(ModelRole::Bed, clock, deadline)?;
        self.bed = Some(
            inputs
                .spawn_bed(policy, Arc::clone(&self.stop))
                .map_err(|error| spawn_error(ModelRole::Bed, error))?,
        );
        self.states[1] = self.bed.as_ref().map(|owner| Arc::clone(&owner.state));
        wait(
            self.bed.as_ref(),
            ModelRole::Bed,
            &self.shutdown,
            clock,
            deadline,
        )?;
        self.before_spawn(ModelRole::StoredPose, clock, deadline)?;
        self.stored_pose = Some(
            inputs
                .spawn_stored_pose(policy, Arc::clone(&self.stop))
                .map_err(|error| spawn_error(ModelRole::StoredPose, error))?,
        );
        self.states[2] = self
            .stored_pose
            .as_ref()
            .map(|owner| Arc::clone(&owner.state));
        wait(
            self.stored_pose.as_ref(),
            ModelRole::StoredPose,
            &self.shutdown,
            clock,
            deadline,
        )
    }

    fn before_spawn(
        &self,
        model: ModelRole,
        clock: &dyn Clock,
        deadline: Duration,
    ) -> Result<(), ModelStartError> {
        let now = clock.monotonic();
        let kind = if self.shutdown.deadline().is_some() {
            StartKind::Stopped
        } else if now >= deadline {
            StartKind::Deadline
        } else {
            return Ok(());
        };
        Err(ModelStartError { model, kind })
    }

    /// Publish this observation before cleanup; never renew an existing cutoff.
    /// Timeout keeps live handles and the lease. A control error preserves both.
    pub fn close(&mut self, clock: &dyn Clock) -> Vec<CleanupError> {
        if let Err(error) = self.shutdown.request_at(clock.monotonic()) {
            self.retain_lease_until_process_exit();
            return vec![CleanupError::Control(error)];
        }
        self.stop.store(true, Ordering::SeqCst);
        if let Err(error) = cutoff(&self.shutdown, clock) {
            return vec![error];
        }
        let results = [
            close_one(&mut self.fall, ModelRole::Fall, clock, &self.shutdown),
            close_one(&mut self.bed, ModelRole::Bed, clock, &self.shutdown),
            close_one(
                &mut self.stored_pose,
                ModelRole::StoredPose,
                clock,
                &self.shutdown,
            ),
        ];
        let mut errors: Vec<_> = results.into_iter().filter_map(Result::err).collect();
        if let Err(error) = cutoff(&self.shutdown, clock)
            && !errors.contains(&error)
        {
            errors.push(error);
        }
        errors
    }

    /// Root calls this when media has not proved closed, even after GPU joins.
    /// The OS releases the lock at process exit; repeated calls are harmless.
    pub fn retain_lease_until_process_exit(&mut self) {
        if let Some(lease) = self.lease.take() {
            std::mem::forget(lease);
        }
    }

    /// True means process exit is required unless a later bounded close succeeds.
    pub fn has_live_threads(&self) -> bool {
        self.fall
            .as_ref()
            .is_some_and(|owner| !owner.thread.is_finished())
            || self
                .bed
                .as_ref()
                .is_some_and(|owner| !owner.thread.is_finished())
            || self
                .stored_pose
                .as_ref()
                .is_some_and(|owner| !owner.thread.is_finished())
    }
}

impl Drop for ModelOwners {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Never admit a second process while a hung native owner remains live.
        // On this fatal path the process owns the lease until OS teardown.
        if self.has_live_threads() {
            self.retain_lease_until_process_exit();
        }
    }
}
