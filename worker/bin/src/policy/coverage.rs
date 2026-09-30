//! Policy-input observation coverage for one camera (G16): the latest actual
//! input and at most one open gap, ported from
//! `worker/runtime/flow/observation_coverage.py`. Results are in-process
//! values only; they never reach the wire (B10, X9) and nothing is logged.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use crate::seam::Clock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationIdentity {
    pub worker_boot_id: String,
    pub camera_id: String,
    pub source_generation: u64,
    pub stream_epoch: u64,
}

/// One policy input as the pump sees it, before a host time is attached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataObservation {
    pub identity: ObservationIdentity,
    pub seq: u64,
    pub source_pts_ns: Option<u64>,
    pub native_publish_sequence: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActualObservation {
    pub identity: ObservationIdentity,
    pub seq: u64,
    pub source_pts_ns: Option<u64>,
    pub native_publish_sequence: u64,
    pub host_time: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObservationGap {
    pub identity: ObservationIdentity,
    pub last_actual: Option<ActualObservation>,
    pub loss_detected_host_time: f64,
}

/// `None` in a duration or gap field means "unknown", never zero.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationRecovery {
    pub gap: ObservationGap,
    pub next_actual: ActualObservation,
    pub host_observation_duration: Option<f64>,
    pub source_duration_ns: Option<u64>,
    pub native_publish_sequence_gap: Option<u64>,
}

/// The observation belongs to another boot, camera, generation or epoch than
/// the expected binding. Nothing was changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForeignIdentity;

impl fmt::Display for ForeignIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("observation identity does not match the expected source binding")
    }
}
impl std::error::Error for ForeignIdentity {}

pub struct ObservationCoverage {
    expected: ObservationIdentity,
    clock: Arc<dyn Clock>,
    last_actual: Option<ActualObservation>,
    open_gap: Option<ObservationGap>,
}

impl ObservationCoverage {
    pub fn new(expected: ObservationIdentity, clock: Arc<dyn Clock>) -> Self {
        Self {
            expected,
            clock,
            last_actual: None,
            open_gap: None,
        }
    }

    pub fn last_actual(&self) -> Option<&ActualObservation> {
        self.last_actual.as_ref()
    }

    pub fn open_gap(&self) -> Option<&ObservationGap> {
        self.open_gap.as_ref()
    }

    pub fn rebind(&mut self, expected: ObservationIdentity) {
        self.expected = expected;
    }

    /// Records one actual input and closes the open gap, if any. `host_time`
    /// defaults to the clock's monotonic seconds.
    pub fn observe(
        &mut self,
        observation: MetadataObservation,
        host_time: Option<f64>,
    ) -> Result<Option<ObservationRecovery>, ForeignIdentity> {
        let host_time = host_time.unwrap_or_else(|| self.now());
        if observation.identity != self.expected {
            return Err(ForeignIdentity);
        }
        let actual = ActualObservation {
            identity: observation.identity,
            seq: observation.seq,
            source_pts_ns: observation.source_pts_ns,
            native_publish_sequence: observation.native_publish_sequence,
            host_time,
        };
        let recovery = self.open_gap.take().map(|gap| close_gap(gap, &actual));
        self.last_actual = Some(actual);
        Ok(recovery)
    }

    /// Opens a gap against the expected binding; `None` while one is open.
    pub fn detect_gap(&mut self, host_time: Option<f64>) -> Option<ObservationGap> {
        if self.open_gap.is_some() {
            return None;
        }
        let gap = ObservationGap {
            identity: self.expected.clone(),
            last_actual: self.last_actual.clone(),
            loss_detected_host_time: host_time.unwrap_or_else(|| self.now()),
        };
        self.open_gap = Some(gap.clone());
        Some(gap)
    }

    fn now(&self) -> f64 {
        self.clock.monotonic().as_secs_f64()
    }
}

fn close_gap(gap: ObservationGap, actual: &ActualObservation) -> ObservationRecovery {
    let previous = gap.last_actual.as_ref();
    let same_boot = previous.filter(|p| {
        p.identity.worker_boot_id == actual.identity.worker_boot_id
            && p.identity.camera_id == actual.identity.camera_id
    });
    let same_epoch = same_boot.filter(|p| {
        p.identity.source_generation == actual.identity.source_generation
            && p.identity.stream_epoch == actual.identity.stream_epoch
    });
    let source_duration_ns = same_epoch.and_then(|p| {
        let (before, after) = (p.source_pts_ns?, actual.source_pts_ns?);
        after.checked_sub(before)
    });
    let host_observation_duration = same_boot
        .filter(|p| actual.host_time.partial_cmp(&p.host_time) != Some(Ordering::Less))
        .map(|p| actual.host_time - p.host_time);
    let native_publish_sequence_gap = publish_sequence_gap(previous, actual);
    ObservationRecovery {
        gap,
        next_actual: actual.clone(),
        host_observation_duration,
        source_duration_ns,
        native_publish_sequence_gap,
    }
}

/// Frames the native side published but this consumer never saw; `None`
/// across an identity change or when the sequence did not advance.
fn publish_sequence_gap(
    previous: Option<&ActualObservation>,
    actual: &ActualObservation,
) -> Option<u64> {
    let previous = previous.filter(|p| p.identity == actual.identity)?;
    let skipped = actual
        .native_publish_sequence
        .checked_sub(previous.native_publish_sequence)?;
    skipped.checked_sub(1)
}
