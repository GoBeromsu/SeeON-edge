//! Already-admitted event -> durable queue -> synchronous execution receipt.
//! Incident identity, recorder startup and sealed publication belong to the caller.

use std::fmt;

use seeon_worker::episode::BusinessEvent;

use super::event_payload::{captured_at, is_uuid, observed_at, payload, reject_non_scalar_audit};

use crate::delivery::{AdmissionResult, DeliveryQueue, QueueError};
use crate::policy::emit::{EmitError, Payload, Stager};
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
    /// `event_sink._event_audit`: an object or array audit value.
    Audit,
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
            Self::Audit => f.write_str("event audit must be scalar"),
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

/// Frozen event payload and the original trigger, captured once.
/// Queue admission has not started. `staged` is the exact payload a later
/// `stage` call rebuilds through the real stager, even if the wall clock moves.
/// `accepted` is private progress written only after `try_admit` returns an
/// accepted result, before any receipt clock or record work. Payload, detection
/// time, frame, and stream stay frozen.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedEvent {
    detected_at: String,
    staged: Payload,
    camera_id: String,
    facility_id: String,
    stream: Stream,
    frame: Frame,
    admission: Admission,
    accepted: Option<AdmissionResult>,
}

/// Recorder input, returned only after durable acceptance and its receipt.
/// Owns `event_ref` so the caller can retain the successful receipt after this
/// borrow ends. Receipts are copied from the queue result, never invented.
#[derive(Clone, Debug, PartialEq)]
pub struct StagedEvent {
    pub event_ref: String,
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

    /// Capture `detected_at` and the original trigger once. Identity, camera,
    /// scalar audit, and the initial clock/record check all happen before any
    /// queue effect. Stager validation stays in `stage`, so a wrong stager
    /// still emits its refusal receipt after this preparation succeeds.
    pub fn prepare(
        &self,
        event: &BusinessEvent,
        stream: &Stream,
        frame: Frame,
        audit: Option<&Payload>,
    ) -> Result<PreparedEvent, EventDeliveryError> {
        if event.camera_id != stream.camera_id {
            return Err(EventDeliveryError::CameraMismatch);
        }
        if !is_uuid(&event.identity) {
            return Err(EventDeliveryError::Identity);
        }
        reject_non_scalar_audit(audit)?;
        let (observed_at_ns, detected_at) = captured_at(self.clock)?;
        let staged = payload(event, &detected_at, audit)?;
        let admission = Admission {
            edge_event_id: event.identity.clone(),
            event_type: event.event_type.clone(),
            domain: event.domain.clone(),
            admitted: false,
            reason: None,
        };
        // Validate a not-yet-admitted record before queue side effects. It is
        // not an observed refusal; only the actual result is sent by `stage`.
        event_delivery_record(stream, frame, observed_at_ns, &admission)?;
        Ok(PreparedEvent {
            detected_at,
            staged,
            camera_id: event.camera_id.clone(),
            facility_id: event.facility_id.clone(),
            stream: stream.clone(),
            frame,
            admission,
            accepted: None,
        })
    }

    /// Rebuild the frozen payload through the current stager, then admit it.
    /// Repeating this call does not read the clock for a new detection time.
    /// The receipt clock is read only after stager validation and admission.
    /// The mandatory observer is synchronous, non-reentrant and receipt-only.
    /// Route it to the execution lanes; it must not decide admission or panic.
    /// A receipt-clock or receipt-record failure after durable acceptance is
    /// not a reason to rebuild the payload. The actual accepted result is
    /// retained on this prepared value before that work. A later retry still
    /// validates the current stager and its camera/facility binding, then
    /// reuses that result instead of calling `try_admit` again.
    pub fn stage(
        &self,
        prepared: &mut PreparedEvent,
        observer: &mut dyn FnMut(Record),
    ) -> Result<StagedEvent, EventDeliveryError> {
        let result = self.admit(prepared);
        if let Ok(result) = &result
            && result.accepted
        {
            prepared.accepted = Some(*result);
        }
        let observed_at_ns = observed_at(self.clock)?;
        let mut admission = prepared.admission.clone();
        match result {
            Ok(result) if result.accepted => {
                admission.admitted = result.accepted;
                admission.reason = result.fault.map(|fault| fault.as_str().to_owned());
                observer(event_delivery_record(
                    &prepared.stream,
                    prepared.frame,
                    observed_at_ns,
                    &admission,
                )?);
                Ok(StagedEvent {
                    event_ref: prepared.admission.edge_event_id.clone(),
                    detected_at: prepared.detected_at.clone(),
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
                    &prepared.stream,
                    prepared.frame,
                    observed_at_ns,
                    &admission,
                )?);
                Err(error)
            }
        }
    }

    fn admit(&self, prepared: &PreparedEvent) -> Result<AdmissionResult, EventDeliveryError> {
        let entry = self
            .stager
            .event_entry(&prepared.staged)
            .map_err(EventDeliveryError::Stage)?;
        if entry.fields().camera_id != prepared.camera_id
            || entry.fields().facility_id != prepared.facility_id
        {
            return Err(EventDeliveryError::StagerIdentity);
        }
        // The current stager still binds this event. Reuse the retained
        // admission instead of publishing again, even if its queue entry is gone.
        if let Some(accepted) = prepared.accepted {
            return Ok(accepted);
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
