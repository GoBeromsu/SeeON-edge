//! One Python `run_once` step on a selected entry: `_send`, then the
//! acknowledge, dead-letter, defer and attempt bookkeeping of its outcome.

use serde_json::Value;

use super::{EXHAUSTED_STATUS, OPERATOR_BLOCKED_RETRY};
use crate::delivery::sender::{
    BodyError, EntryOutcome, InvalidEntry, MAX_ENTRY_ATTEMPTS, SenderState, clip_request,
    event_body,
};
use crate::delivery::snapshot;
use crate::delivery::{DeliveryQueue, EntryKind};
use crate::relay::RelayClient;
use crate::relay::wire::{
    DeliveryDisposition, DeliveryFailure, DeliveryFailureCode, parse_clip_result,
    parse_event_result,
};
use crate::seam::Clock;

pub(super) struct Pass<'a> {
    pub(super) queue: &'a DeliveryQueue,
    pub(super) client: &'a RelayClient,
    pub(super) clock: &'a dyn Clock,
    pub(super) clip_export_enabled: bool,
}

/// A relay acknowledgement, or the failure that replaced it.
type Sent = Result<(), DeliveryFailure>;

impl Pass<'_> {
    pub(super) fn step(&self, state: &mut SenderState, entry: &Value, id: &str) -> EntryOutcome {
        let kind = entry
            .get("kind")
            .and_then(Value::as_str)
            .and_then(EntryKind::from_wire);
        if kind == Some(EntryKind::Clip) && !self.clip_export_enabled {
            state.defer(id);
            return EntryOutcome::ClipExportDisabled;
        }
        if state.attempts(id) >= MAX_ENTRY_ATTEMPTS {
            return self.retain(state, id, EXHAUSTED_STATUS, true);
        }
        let Some(kind) = kind else {
            return invalid(state, id, InvalidEntry::UnknownKind);
        };
        // Snapshot entries are sent whatever the relay answered for their
        // event (accepted or accepted_local), as Python `_send` does
        // (`evidence_export_client.py` L282-316).
        match self.send(kind, entry) {
            Err(reason) => invalid(state, id, reason),
            Ok(Err(failure)) => self.fail(state, id, kind, failure),
            Ok(Ok(())) => {
                state.forget_attempts(id);
                self.acknowledge(state, id)
            }
        }
    }

    /// Python `_send`.
    fn send(&self, kind: EntryKind, entry: &Value) -> Result<Sent, InvalidEntry> {
        match kind {
            EntryKind::Event => {
                let body = event_body(entry)?;
                let edge_event_id = match entry.get("edge_event_id") {
                    None => return Err(BodyError::MissingField("edge_event_id").into()),
                    Some(value) => value
                        .as_str()
                        .ok_or(BodyError::WrongType("edge_event_id"))?,
                };
                let result = self.client.post_alert(&body).map_err(DeliveryFailure::from);
                Ok(parse_event_result(result, edge_event_id, self.clock).map(|_| ()))
            }
            EntryKind::Clip => {
                let request = match clip_request(entry) {
                    Err(BodyError::InvalidEventIdentity) => {
                        return Ok(Err(invalid_event_payload()));
                    }
                    other => other?,
                };
                let path = request.path();
                let result = self
                    .client
                    .put_json(&path, &request.body)
                    .map_err(DeliveryFailure::from);
                let receipt =
                    parse_clip_result(result, &request.clip_id, request.state_version, self.clock);
                Ok(receipt.map(|_| ()))
            }
            EntryKind::SnapshotAttachment | EntryKind::SnapshotDisposition => {
                Ok(snapshot::send_media(self.client, kind, entry, self.clock)?)
            }
        }
    }

    /// Python `run_once`'s `DeliveryFailure` branch.
    fn fail(
        &self,
        state: &mut SenderState,
        id: &str,
        kind: EntryKind,
        failure: DeliveryFailure,
    ) -> EntryOutcome {
        if kind == EntryKind::Clip
            && failure.code == DeliveryFailureCode::CameraMappingMissing.as_str()
        {
            state.defer(id);
            state.block(id, self.clock.monotonic() + OPERATOR_BLOCKED_RETRY);
            return EntryOutcome::OperatorBlocked { failure };
        }
        if failure.disposition == DeliveryDisposition::Permanent
            && let Some(status) = failure.status_code
            && (400..500).contains(&status)
            && (kind == EntryKind::Clip || status == 422)
        {
            return self.retain(state, id, status, false);
        }
        state.defer(id);
        let counted = failure.disposition != DeliveryDisposition::Retry;
        if counted {
            state.count_attempt(id);
        }
        EntryOutcome::Failed { failure, counted }
    }

    /// Python `_retain`: dead-letter under `status`; any failure keeps it.
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
                    "ml-worker: queue entry {id} was not dead-lettered under status {status}: \
                     it is missing or the retention area is full; it stays queued"
                );
                state.defer(id);
                EntryOutcome::RetentionFull { status }
            }
            Err(error) => {
                eprintln!(
                    "ml-worker: queue entry {id} could not be dead-lettered under status \
                     {status}: {error}; it stays queued"
                );
                state.defer(id);
                EntryOutcome::RetentionFull { status }
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
                    "ml-worker: queue entry {id} was acknowledged by the relay but could not \
                     be removed: {error}; it stays queued"
                );
                state.defer(id);
                EntryOutcome::AckRemovalFailed
            }
        }
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
