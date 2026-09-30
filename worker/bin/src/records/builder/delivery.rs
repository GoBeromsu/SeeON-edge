//! `event.delivery` (admission and sender attempts) and `backend.acceptance`
//! (`worker/pipeline/diagnostics/emit_delivery.py`).

use std::path::{Component, Path};

use super::{Draft, Frame, PRODUCER_BACKEND, PRODUCER_EVENT, Stream, make_record, optional_text};
use crate::json::Json;
use crate::records::id::{ContractError, int, optional, text};
use crate::records::wire::{PROCESS_SCOPE, Record, RecordKind};

/// Stream-scoped queue admission, observed against the triggering frame.
/// `admitted` must be the durable queue's own proof of acceptance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub edge_event_id: String,
    pub event_type: String,
    pub domain: String,
    pub admitted: bool,
    pub reason: Option<String>,
}

/// `DELIVERY_ATTEMPT_OUTCOMES`: the sender's own dispositions, never Hub
/// acceptance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptOutcome {
    RetryTransient,
    RetryCounted,
    RefusedRetained,
    RefusedRetentionFull,
    ExhaustedRetained,
    ExhaustedRetentionFull,
    AckRemovalDeferred,
}

impl AttemptOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RetryTransient => "retry-transient",
            Self::RetryCounted => "retry-counted",
            Self::RefusedRetained => "refused-retained",
            Self::RefusedRetentionFull => "refused-retention-full",
            Self::ExhaustedRetained => "exhausted-retained",
            Self::ExhaustedRetentionFull => "exhausted-retention-full",
            Self::AckRemovalDeferred => "ack-removal-deferred",
        }
    }
}

/// One sender disposition for a queued event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    pub edge_event_id: String,
    pub outcome: AttemptOutcome,
    pub attempt: u64,
    pub max_attempts: u64,
    pub failure_class: Option<String>,
    pub status_code: Option<u16>,
    pub retained: Option<bool>,
    /// Only the final path component reaches the wire.
    pub dead_letter_dir: Option<String>,
    /// `"EVENT"` for the event queue.
    pub queue_kind: String,
}

/// A Backend receipt for one queued event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Acceptance {
    pub edge_event_id: String,
    pub status: String,
    pub hub_event_id: String,
}

/// `Path(value).name`: the last component, `""` for a root or `.`.
fn path_name(value: &str) -> String {
    match Path::new(value).components().next_back() {
        Some(Component::Normal(name)) => name.to_string_lossy().into_owned(),
        Some(Component::ParentDir) => "..".to_owned(),
        _ => String::new(),
    }
}

/// A process-scoped stream: the observing boot with `PROCESS_SCOPE`
/// generation and epoch, because a queue entry outlives its origin stream.
fn process_stream(camera_id: &str, observing_boot_id: &str) -> Stream {
    Stream {
        camera_id: camera_id.to_owned(),
        worker_boot_id: observing_boot_id.to_owned(),
        source_generation: PROCESS_SCOPE,
        stream_epoch: PROCESS_SCOPE,
    }
}

/// `event_delivery_record`: the admission joins its event by
/// `causal_unit_id == edge_event_id`.
pub fn event_delivery_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    admission: &Admission,
) -> Result<Record, ContractError> {
    let draft = Draft {
        record_kind: RecordKind::EventDelivery,
        producer: PRODUCER_EVENT,
        causal_unit_id: admission.edge_event_id.clone(),
        outcome: if admission.admitted {
            "admitted"
        } else {
            "refused"
        }
        .to_owned(),
        payload: vec![
            text("edge_event_id", &admission.edge_event_id),
            text("event_type", &admission.event_type),
            text("domain", &admission.domain),
            text("queue", "delivery"),
            optional_text("reason", admission.reason.as_deref()),
        ],
    };
    make_record(stream, observed_at_ns, Some(frame), draft)
}

/// `delivery_attempt_record`: process-scoped, stamped with the boot that
/// observed the attempt.
pub fn delivery_attempt_record(
    camera_id: &str,
    observing_boot_id: &str,
    observed_at_ns: u64,
    attempt: &Attempt,
) -> Result<Record, ContractError> {
    let dead_letter_dir = attempt.dead_letter_dir.as_deref().map(path_name);
    let draft = Draft {
        record_kind: RecordKind::EventDelivery,
        producer: PRODUCER_EVENT,
        causal_unit_id: attempt.edge_event_id.clone(),
        outcome: attempt.outcome.as_str().to_owned(),
        payload: vec![
            text("edge_event_id", &attempt.edge_event_id),
            int("attempt", attempt.attempt),
            int("max_attempts", attempt.max_attempts),
            optional_text("failure_class", attempt.failure_class.as_deref()),
            optional(
                "status_code",
                attempt.status_code.map(|code| Json::Int(i128::from(code))),
            ),
            optional("retained", attempt.retained.map(Json::Bool)),
            optional_text("dead_letter_dir", dead_letter_dir.as_deref()),
            text("queue_kind", &attempt.queue_kind),
        ],
    };
    let stream = process_stream(camera_id, observing_boot_id);
    make_record(&stream, observed_at_ns, None, draft)
}

/// `backend_acceptance_record`: process-scoped; origin fields are null,
/// never fabricated.
pub fn backend_acceptance_record(
    camera_id: &str,
    observing_boot_id: &str,
    observed_at_ns: u64,
    acceptance: &Acceptance,
) -> Result<Record, ContractError> {
    let status = acceptance.status.as_str();
    let outcome = match status {
        "accepted" => "hub-accepted",
        other => other,
    };
    let hub_accepted = status == "accepted" && !acceptance.hub_event_id.is_empty();
    let draft = Draft {
        record_kind: RecordKind::BackendAcceptance,
        producer: PRODUCER_BACKEND,
        causal_unit_id: acceptance.edge_event_id.clone(),
        outcome: outcome.to_owned(),
        payload: vec![
            text("edge_event_id", &acceptance.edge_event_id),
            text("status", status),
            text("hub_event_id", &acceptance.hub_event_id),
            (
                "accepted_local".to_owned(),
                Json::Bool(status == "accepted_local"),
            ),
            ("hub_accepted".to_owned(), Json::Bool(hub_accepted)),
            text("observing_boot_id", observing_boot_id),
            optional("origin_boot_id", None),
            optional("origin_source_generation", None),
            optional("origin_stream_epoch", None),
        ],
    };
    let stream = process_stream(camera_id, observing_boot_id);
    make_record(&stream, observed_at_ns, None, draft)
}
