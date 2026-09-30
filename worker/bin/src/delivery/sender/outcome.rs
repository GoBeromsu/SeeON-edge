//! What a sender pass reports: one outcome per `run_once` step and why the
//! pass ended.

use crate::delivery::QueueError;
use crate::delivery::sender::BodyError;
use crate::delivery::snapshot::MediaEntryError;
use crate::relay::wire::DeliveryFailure;

/// Why an entry could not become a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidEntry {
    UnknownKind,
    Body(BodyError),
    Media(MediaEntryError),
}

impl From<BodyError> for InvalidEntry {
    fn from(error: BodyError) -> Self {
        Self::Body(error)
    }
}

impl From<MediaEntryError> for InvalidEntry {
    fn from(error: MediaEntryError) -> Self {
        Self::Media(error)
    }
}

/// What one `run_once` step did to its entry.
#[derive(Clone, Debug, PartialEq)]
pub enum EntryOutcome {
    /// The relay acknowledged (2xx, or 409 with a matching receipt); removed.
    Acknowledged,
    /// Moved to the dead-letter directory under `status`.
    DeadLettered { status: u16 },
    /// Dead-lettering failed or retention is full; the entry stays.
    RetentionFull { status: u16 },
    /// The entry stays; `counted` says whether an attempt was spent.
    Failed {
        failure: DeliveryFailure,
        counted: bool,
    },
    /// CAMERA_MAPPING_MISSING clip, held back for `OPERATOR_BLOCKED_RETRY`.
    OperatorBlocked { failure: DeliveryFailure },
    /// The entry cannot become a request; an attempt was spent.
    Invalid(InvalidEntry),
    /// Clip export is disabled by the backend; the clip stays.
    ClipExportDisabled,
    /// The relay acknowledged but removing the file failed; it stays.
    AckRemovalFailed,
}

impl EntryOutcome {
    pub fn is_acknowledged(&self) -> bool {
        matches!(self, Self::Acknowledged)
    }
}

/// Why a pass ended.
#[derive(Debug)]
pub enum DrainStop {
    /// Nothing is due.
    Idle,
    /// The last step did not acknowledge; wait `SENDER_IDLE_WAIT`.
    Unacknowledged,
    /// `MAX_ACCEPTED_ENTRIES` steps acknowledged; call again without waiting.
    StepLimit,
    /// The queue could not be listed; entries stay durable.
    Queue(QueueError),
}

#[derive(Debug)]
pub struct DrainSummary {
    /// `(entry_id, outcome)` per step, in order.
    pub outcomes: Vec<(String, EntryOutcome)>,
    pub stop: DrainStop,
}

impl DrainSummary {
    fn last_failure(&self) -> Option<&DeliveryFailure> {
        match self.outcomes.last() {
            Some((_, EntryOutcome::Failed { failure, .. }))
            | Some((_, EntryOutcome::OperatorBlocked { failure })) => Some(failure),
            _ => None,
        }
    }

    /// True when the pass ended on a transport failure (no HTTP status).
    pub fn relay_unreachable(&self) -> bool {
        self.last_failure().is_some_and(|failure| {
            failure.status_code.is_none() && failure.transport_error.is_some()
        })
    }

    /// The relay's `Retry-After` on the failure that ended the pass.
    pub fn retry_after_seconds(&self) -> Option<f64> {
        self.last_failure()
            .and_then(|failure| failure.retry_after_seconds)
    }
}
