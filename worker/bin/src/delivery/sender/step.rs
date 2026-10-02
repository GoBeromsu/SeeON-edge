//! One Python `run_once` step on a selected entry: `_send`, then the
//! acknowledge, dead-letter, defer and attempt bookkeeping of its outcome.

use serde_json::Value;

use super::{EXHAUSTED_STATUS, OPERATOR_BLOCKED_RETRY};
use crate::delivery::sender::{
    BodyError, EntryOutcome, InvalidEntry, MAX_ENTRY_ATTEMPTS, SenderState, clip_request,
    event_body,
};
use crate::delivery::snapshot::MediaEntryError;
use crate::delivery::snapshot::{self, MediaSendError};
use crate::delivery::{DeliveryQueue, EntryKind};
use crate::relay::client::{DeadlineClient, RequestError};
use crate::relay::wire::{
    DeliveryDisposition, DeliveryFailure, DeliveryFailureCode, parse_clip_result,
    parse_event_result,
};
use crate::seam::Clock;

pub(super) struct Pass<'a> {
    pub(super) queue: &'a DeliveryQueue,
    pub(super) client: &'a DeadlineClient<'a>,
    pub(super) clock: &'a dyn Clock,
    pub(super) clip_export_enabled: bool,
}

/// A relay acknowledgement, or the failure that replaced it.
enum SendError {
    Cutoff,
    Invalid(InvalidEntry),
}

impl From<BodyError> for SendError {
    fn from(error: BodyError) -> Self {
        Self::Invalid(error.into())
    }
}

impl From<MediaEntryError> for SendError {
    fn from(error: MediaEntryError) -> Self {
        Self::Invalid(error.into())
    }
}
type Sent = Result<(), DeliveryFailure>;

impl Pass<'_> {
    pub(super) fn step(
        &self,
        state: &mut SenderState,
        entry: &Value,
        id: &str,
        reset_deferred: bool,
    ) -> EntryOutcome {
        if self.client.check().is_err() {
            return EntryOutcome::Cutoff;
        }
        let kind = entry
            .get("kind")
            .and_then(Value::as_str)
            .and_then(EntryKind::from_wire);
        if kind == Some(EntryKind::Clip) && !self.clip_export_enabled {
            if self.client.check().is_err() {
                return EntryOutcome::Cutoff;
            }
            admit(state, reset_deferred);
            state.defer(id);
            return EntryOutcome::ClipExportDisabled;
        }
        if state.attempts(id) >= MAX_ENTRY_ATTEMPTS {
            if self.client.check().is_err() {
                return EntryOutcome::Cutoff;
            }
            admit(state, reset_deferred);
            return self.retain(state, id, EXHAUSTED_STATUS, true);
        }
        let Some(kind) = kind else {
            if self.client.check().is_err() {
                return EntryOutcome::Cutoff;
            }
            admit(state, reset_deferred);
            return invalid(state, id, InvalidEntry::UnknownKind);
        };
        // Snapshot entries are sent whatever the relay answered for their
        // event (accepted or accepted_local), as Python `_send` does
        // (`evidence_export_client.py` L282-316).
        match self.send(kind, entry) {
            Err(SendError::Cutoff) => EntryOutcome::Cutoff,
            Err(SendError::Invalid(reason)) => {
                if self.client.check().is_err() {
                    return EntryOutcome::Cutoff;
                }
                admit(state, reset_deferred);
                invalid(state, id, reason)
            }
            Ok(Err(failure)) => self.fail(state, id, kind, failure, reset_deferred),
            Ok(Ok(())) => {
                if self.client.check().is_err() {
                    return EntryOutcome::Cutoff;
                }
                admit(state, reset_deferred);
                state.forget_attempts(id);
                self.acknowledge(state, id)
            }
        }
    }

    /// Python `_send`. Local cutoff is the outer error.
    fn send(&self, kind: EntryKind, entry: &Value) -> Result<Sent, SendError> {
        if self.client.check().is_err() {
            return Err(SendError::Cutoff);
        }
        match kind {
            EntryKind::Event => self.send_event(entry),
            EntryKind::Clip => self.send_clip(entry),
            EntryKind::SnapshotAttachment | EntryKind::SnapshotDisposition => {
                match snapshot::send_media(self.client, kind, entry, self.clock) {
                    Ok(sent) => Ok(sent),
                    Err(MediaSendError::Cutoff) => Err(SendError::Cutoff),
                    Err(MediaSendError::Invalid(error)) => Err(error.into()),
                }
            }
        }
    }

    fn send_event(&self, entry: &Value) -> Result<Sent, SendError> {
        let body = event_body(entry)?;
        let edge_event_id = match entry.get("edge_event_id") {
            None => return Err(BodyError::MissingField("edge_event_id").into()),
            Some(value) => value
                .as_str()
                .ok_or(BodyError::WrongType("edge_event_id"))?,
        };
        if self.client.check().is_err() {
            return Err(SendError::Cutoff);
        }
        let result = http_result(self.client.post_alert(&body))?;
        if self.client.check().is_err() {
            return Err(SendError::Cutoff);
        }
        Ok(parse_event_result(result, edge_event_id, self.clock).map(|_| ()))
    }

    fn send_clip(&self, entry: &Value) -> Result<Sent, SendError> {
        let request = match clip_request(entry) {
            Err(BodyError::InvalidEventIdentity) => {
                return Ok(Err(invalid_event_payload()));
            }
            other => other?,
        };
        if self.client.check().is_err() {
            return Err(SendError::Cutoff);
        }
        let path = request.path();
        let result = http_result(self.client.put_json(&path, &request.body))?;
        if self.client.check().is_err() {
            return Err(SendError::Cutoff);
        }
        let receipt =
            parse_clip_result(result, &request.clip_id, request.state_version, self.clock);
        Ok(receipt.map(|_| ()))
    }

    /// Python `run_once`'s `DeliveryFailure` branch.
    fn fail(
        &self,
        state: &mut SenderState,
        id: &str,
        kind: EntryKind,
        failure: DeliveryFailure,
        reset_deferred: bool,
    ) -> EntryOutcome {
        if kind == EntryKind::Clip
            && failure.code == DeliveryFailureCode::CameraMappingMissing.as_str()
        {
            let until = self.clock.monotonic() + OPERATOR_BLOCKED_RETRY;
            if self.client.check().is_err() {
                return EntryOutcome::Cutoff;
            }
            admit(state, reset_deferred);
            state.defer(id);
            state.block(id, until);
            return EntryOutcome::OperatorBlocked { failure };
        }
        if failure.disposition == DeliveryDisposition::Permanent
            && let Some(status) = failure.status_code
            && (400..500).contains(&status)
            && (kind == EntryKind::Clip || status == 422)
        {
            if self.client.check().is_err() {
                return EntryOutcome::Cutoff;
            }
            admit(state, reset_deferred);
            return self.retain(state, id, status, false);
        }
        if self.client.check().is_err() {
            return EntryOutcome::Cutoff;
        }
        admit(state, reset_deferred);
        state.defer(id);
        let counted = failure.disposition != DeliveryDisposition::Retry;
        if counted {
            state.count_attempt(id);
        }
        EntryOutcome::Failed { failure, counted }
    }

    /// Dead-letter under `status`. A false or failed result only defers
    /// selection: cleanup may already have changed the queue or retention.
    fn retain(
        &self,
        state: &mut SenderState,
        id: &str,
        status: u16,
        exhausted: bool,
    ) -> EntryOutcome {
        match self.queue.dead_letter(id, status) {
            Ok(true) => {
                if exhausted {
                    state.forget_attempts(id);
                }
                state.undefer(id);
                EntryOutcome::DeadLettered { status }
            }
            Ok(false) => {
                eprintln!(
                    "ml-worker: dead-letter of queue entry {id} under status {status} was not \
                     confirmed: the entry was missing before retention or the retention area is \
                     full; local queue state may already have changed"
                );
                state.defer(id);
                EntryOutcome::RetentionDeferred { status }
            }
            Err(error) => {
                eprintln!(
                    "ml-worker: dead-letter operation for queue entry {id} under status {status} \
                     failed: {error}; local queue state may already have changed"
                );
                state.defer(id);
                EntryOutcome::RetentionDeferred { status }
            }
        }
    }

    fn acknowledge(&self, state: &mut SenderState, id: &str) -> EntryOutcome {
        match self.queue.acknowledge(id) {
            Ok(_) => {
                state.undefer(id);
                state.unblock(id);
                EntryOutcome::Acknowledged
            }
            Err(error) => {
                eprintln!(
                    "ml-worker: relay accepted queue entry {id}, but queue cleanup failed: \
                     {error}; local queue state may already have changed"
                );
                state.defer(id);
                EntryOutcome::AckRemovalFailed
            }
        }
    }
}
fn admit(state: &mut SenderState, reset_deferred: bool) {
    if reset_deferred {
        state.commit_deferred_reset();
    }
}

fn invalid(state: &mut SenderState, id: &str, reason: InvalidEntry) -> EntryOutcome {
    state.defer(id);
    state.count_attempt(id);
    EntryOutcome::Invalid(reason)
}

/// `send_clip`'s answer when `_local_event_identity` refuses the entry.
fn invalid_event_payload() -> DeliveryFailure {
    DeliveryFailure {
        disposition: DeliveryDisposition::Permanent,
        code: "INVALID_EVENT_PAYLOAD".to_owned(),
        status_code: None,
        retry_after_seconds: None,
        transport_error: None,
    }
}
fn http_result(
    result: Result<crate::relay::wire::Response, RequestError>,
) -> Result<crate::relay::wire::HttpResult, SendError> {
    match result {
        Ok(response) => Ok(Ok(response)),
        Err(RequestError::Cutoff) => Err(SendError::Cutoff),
        Err(RequestError::Transport(error)) => Ok(Err(DeliveryFailure::from(error))),
    }
}
