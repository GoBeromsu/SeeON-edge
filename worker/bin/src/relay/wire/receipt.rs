//! Python `_event_receipt` and `_clip_receipt`
//! (`shared/events/evidence_http_transport.py`): the success bodies the relay
//! returns for an event POST and a clip upload, read field by field with the
//! same type rules as Python's `isinstance` checks.

use serde_json::Value;

use super::failure::parse_json_object;

/// `EventReceipt.status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventStatus {
    /// Accepted and pushed upstream; `event_id` is the upstream id.
    Accepted,
    /// Stored by the edge backend, which will never push it upstream.
    AcceptedLocal,
}

impl EventStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::AcceptedLocal => "accepted_local",
        }
    }
}

/// Python `EventReceipt(status, edge_event_id, event_id)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventReceipt {
    pub status: EventStatus,
    pub edge_event_id: String,
    /// Empty for `AcceptedLocal`.
    pub event_id: String,
}

/// `ClipReceipt.state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipState {
    Ready,
    Unavailable,
    Expired,
}

impl ClipState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "READY",
            Self::Unavailable => "UNAVAILABLE",
            Self::Expired => "EXPIRED",
        }
    }

    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "READY" => Some(Self::Ready),
            "UNAVAILABLE" => Some(Self::Unavailable),
            "EXPIRED" => Some(Self::Expired),
            _ => None,
        }
    }
}

/// Python `ClipReceipt(clip_id, state, state_version, sha256, size_bytes)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipReceipt {
    pub clip_id: String,
    pub state: ClipState,
    pub state_version: i64,
    pub sha256: Option<String>,
    pub size_bytes: Option<i64>,
}

/// Python `_event_receipt`: `None` when the body is not a receipt.
pub fn event_receipt(body: &[u8]) -> Option<EventReceipt> {
    let payload = parse_json_object(body);
    let Some(Value::String(edge_event_id)) = payload.get("edge_event_id") else {
        return None;
    };
    match payload.get("status") {
        Some(Value::String(status)) if status == "accepted_local" => Some(EventReceipt {
            status: EventStatus::AcceptedLocal,
            edge_event_id: edge_event_id.clone(),
            event_id: String::new(),
        }),
        Some(Value::String(status)) if status == "accepted" => match payload.get("event_id") {
            Some(Value::String(event_id)) if !event_id.is_empty() => Some(EventReceipt {
                status: EventStatus::Accepted,
                edge_event_id: edge_event_id.clone(),
                event_id: event_id.clone(),
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Python `_clip_receipt`: `None` when the body is not a receipt.
pub fn clip_receipt(body: &[u8]) -> Option<ClipReceipt> {
    let payload = parse_json_object(body);
    let Some(Value::String(clip_id)) = payload.get("clip_id") else {
        return None;
    };
    let state = match payload.get("state") {
        Some(Value::String(state)) => ClipState::from_wire(state)?,
        _ => return None,
    };
    let state_version = python_int(payload.get("state_version")?)?;
    let sha256 = match payload.get("sha256") {
        None | Some(Value::Null) => None,
        Some(Value::String(sha256)) => Some(sha256.clone()),
        Some(_) => return None,
    };
    let size_bytes = match payload.get("size_bytes") {
        None | Some(Value::Null) => None,
        Some(value) => Some(python_int(value)?),
    };
    Some(ClipReceipt {
        clip_id: clip_id.clone(),
        state,
        state_version,
        sha256,
        size_bytes,
    })
}

/// Python `isinstance(value, int)`: JSON integers and, because `bool` is an
/// `int` subclass, `true`/`false` as 1/0. Integers outside `i64` are refused.
fn python_int(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(flag) => Some(i64::from(*flag)),
        Value::Number(number) => number.as_i64(),
        _ => None,
    }
}
