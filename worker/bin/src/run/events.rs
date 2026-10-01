//! Already-admitted event -> durable queue -> synchronous execution receipt.
//! Incident identity, recorder startup and sealed publication belong to the caller.

use std::fmt;

use seeon_worker::episode::BusinessEvent;

use super::event_payload::{captured_at, is_uuid, observed_at, payload};
use crate::delivery::{AdmissionResult, DeliveryQueue, QueueError};
use crate::policy::emit::{EmitError, Stager};
use crate::records::Record;
use crate::records::builder::{Admission, Frame, Stream, event_delivery_record};
use crate::records::id::ContractError;
use crate::seam::Clock;

#[derive(Debug)]
pub enum EventDeliveryError {
    /// Only canonical, already-admitted UUIDs may cross this boundary.
    Identity,
    /// The clock cannot be represented by the record's unsigned nanoseconds.
    WallTime,
    Record(ContractError),
    CameraMismatch,
    /// A stager configured for another camera/facility must not relabel an event.
    StagerIdentity,
    Stage(EmitError),
    Queue(QueueError),
    Refused(AdmissionResult),
}

impl fmt::Display for EventDeliveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => f.write_str("event identity must be an admitted canonical UUID"),
            Self::WallTime => f.write_str("event wall time is outside the record envelope"),
            Self::Record(error) => write!(f, "event delivery record: {error}"),
            Self::CameraMismatch => f.write_str("event camera does not match trigger stream"),
            Self::StagerIdentity => f.write_str("stager identity does not match admitted event"),
            Self::Stage(error) => write!(f, "event staging: {error}"),
            // Do not interpolate raw filesystem errors or configured paths.
            Self::Queue(_) => f.write_str("event delivery queue operation failed"),
            Self::Refused(result) => write!(
                f,
                "event delivery admission failed: {}",
                result.fault.map_or("None", |fault| fault.as_str())
            ),
        }
    }
}

impl std::error::Error for EventDeliveryError {}

impl From<ContractError> for EventDeliveryError {
    fn from(error: ContractError) -> Self {
        Self::Record(error)
    }
}

/// Recorder input, returned only after durable acceptance and its receipt.
#[derive(Debug)]
pub struct StagedEvent<'event> {
    pub event_ref: &'event str,
    pub detected_at: String,
    pub admission: AdmissionResult,
}

/// Borrows the real per-camera stager, queue and injected wall clock.
/// There is no transport, mutable journal, identity minting or retry here.
pub struct EventDelivery<'a> {
    clock: &'a dyn Clock,
    stager: &'a Stager,
    queue: &'a DeliveryQueue,
}

impl<'a> EventDelivery<'a> {
    pub fn new(clock: &'a dyn Clock, stager: &'a Stager, queue: &'a DeliveryQueue) -> Self {
        Self {
            clock,
            stager,
            queue,
        }
    }

    /// `event` already has its admitted UUID; `stream` and `frame` are the
    /// original trigger, not the frame current when an async score completes.
    /// The mandatory observer is synchronous, non-reentrant and receipt-only.
    /// Route it to the execution lanes; it must not decide admission or panic.
    /// On error the root owns recovery: never regenerate this immutable entry
    /// with a new timestamp after uncertain queue I/O. No retry occurs here.
    pub fn stage<'event>(
        &self,
        event: &'event BusinessEvent,
        stream: &Stream,
        frame: Frame,
        observer: &mut dyn FnMut(Record),
    ) -> Result<StagedEvent<'event>, EventDeliveryError> {
        if event.camera_id != stream.camera_id {
            return Err(EventDeliveryError::CameraMismatch);
        }
        if !is_uuid(&event.identity) {
            return Err(EventDeliveryError::Identity);
        }
        let (observed_at_ns, detected_at) = captured_at(self.clock)?;
        let mut admission = Admission {
            edge_event_id: event.identity.clone(),
            event_type: event.event_type.clone(),
            domain: event.domain.clone(),
            admitted: false,
            reason: None,
        };
        // Validate a not-yet-admitted record before queue side effects. It is
        // not an observed refusal; only the actual result is sent below.
        event_delivery_record(stream, frame, observed_at_ns, &admission)?;
        let result = self.admit(event, &detected_at);
        // The delivery receipt observes the outcome, not the earlier detection.
        let observed_at_ns = observed_at(self.clock)?;
        match result {
            Ok(result) if result.accepted => {
                admission.admitted = result.accepted;
                admission.reason = result.fault.map(|fault| fault.as_str().to_owned());
                observer(event_delivery_record(
                    stream,
                    frame,
                    observed_at_ns,
                    &admission,
                )?);
                Ok(StagedEvent {
                    event_ref: &event.identity,
                    detected_at,
                    admission: result,
                })
            }
            other => {
                let error = match other {
                    Ok(result) => EventDeliveryError::Refused(result),
                    Err(error) => error,
                };
                admission.admitted = false;
                admission.reason = Some(error.receipt_reason());
                observer(event_delivery_record(
                    stream,
                    frame,
                    observed_at_ns,
                    &admission,
                )?);
                Err(error)
            }
        }
    }

    fn admit(
        &self,
        event: &BusinessEvent,
        detected_at: &str,
    ) -> Result<AdmissionResult, EventDeliveryError> {
        let entry = self
            .stager
            .event_entry(&payload(event, detected_at))
            .map_err(EventDeliveryError::Stage)?;
        if entry.fields().camera_id != event.camera_id
            || entry.fields().facility_id != event.facility_id
        {
            return Err(EventDeliveryError::StagerIdentity);
        }
        self.queue
            .try_admit(&entry.into())
            .map_err(EventDeliveryError::Queue)
    }
}

impl EventDeliveryError {
    /// Flow's exception-class reason convention, without raw configuration or
    /// filesystem values. Policy refusals retain the queue's exact fault token.
    fn receipt_reason(&self) -> String {
        match self {
            Self::Refused(result) => result
                .fault
                .map_or("None", |fault| fault.as_str())
                .to_owned(),
            Self::Stage(error) => format!("ValueError: {error}"),
            Self::Queue(QueueError::Io(error)) => match error.kind() {
                std::io::ErrorKind::NotFound => "FileNotFoundError".to_owned(),
                std::io::ErrorKind::PermissionDenied => "PermissionError".to_owned(),
                std::io::ErrorKind::NotADirectory => "NotADirectoryError".to_owned(),
                std::io::ErrorKind::IsADirectory => "IsADirectoryError".to_owned(),
                std::io::ErrorKind::AlreadyExists => "FileExistsError".to_owned(),
                kind => format!("OSError: {kind:?}"),
            },
            Self::Queue(QueueError::Json(error)) => format!("ValueError: {error}"),
            Self::Queue(QueueError::InvalidEntryId(error)) => format!("ValueError: {error}"),
            Self::Queue(QueueError::CorruptEntry) => "ValueError: corrupt queue entry".to_owned(),
            Self::Queue(QueueError::UnrecoverableName) => {
                "ValueError: invalid retained entry name".to_owned()
            }
            _ => format!("ValueError: {self}"),
        }
    }
}
