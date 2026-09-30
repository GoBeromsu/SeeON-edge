//! Request bodies built from queue entries: Python `_payload` (the alert
//! body with its decision trace rejoined as `audit`), `_clip_claim`, and the
//! JSON payload `EvidenceExportClient.send_clip` PUTs.

use serde_json::{Map, Value};

use super::base64::decode_base64;

/// Python `MAX_LOCAL_EVENT_PAYLOAD_BYTES` for the clip's local identity.
pub const MAX_LOCAL_EVENT_PAYLOAD_BYTES: usize = 512 * 1024;

/// Why an entry could not become a request: the Python sender raised inside
/// `_send`, deferred the entry and counted an attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyError {
    /// `KeyError`: a required field is absent.
    MissingField(&'static str),
    /// `TypeError`/`ValueError` from `int()` or `str()` on a wrongly typed field.
    WrongType(&'static str),
    /// `binascii.Error`: the field is not padded standard base64.
    Base64(&'static str),
    /// `UnicodeDecodeError`: the decoded bytes are not ASCII.
    NotAscii(&'static str),
    /// `json.JSONDecodeError`: the decoded text is not JSON.
    NotJson(&'static str),
    /// `TypeError`: a truthy trace cannot be merged into a non-object body.
    AuditTarget,
    /// `ValueError` from `ClipLocalState(...)`.
    LocalState,
    /// `ValueError` from `EvidenceReasonCode(...)`.
    UnavailableReason,
    /// The clip id cannot be one URL path segment (`clips/{clip_id}`).
    ClipIdSegment,
    /// `send_clip`'s `_local_event_identity` refused camera or facility id;
    /// Python answers `DeliveryFailure(PERMANENT, "INVALID_EVENT_PAYLOAD")`.
    InvalidEventIdentity,
}

/// Python `ClipLocalState`.
const LOCAL_STATES: [&str; 4] = ["AWAITING_FINALIZE", "VERIFIED", "UNAVAILABLE", "CORRUPT"];
/// Python `EvidenceReasonCode`.
const REASONS: [&str; 7] = [
    "ENCODER_FAILED",
    "NO_FRAMES",
    "FINALIZE_FAILED",
    "STREAM_EPOCH_MISMATCH",
    "INTERRUPTED_FINALIZE",
    "MISSING",
    "CORRUPT",
];
const TEXT_MEDIA_FIELDS: [&str; 6] = [
    "sha256",
    "mime_type",
    "codec",
    "clip_start_at",
    "clip_end_at",
    "finalized_at",
];
const INT_MEDIA_FIELDS: [&str; 2] = ["size_bytes", "duration_ms"];
/// The media fields `send_clip` copies into a READY payload, in its order.
const READY_FIELDS: [&str; 8] = [
    "sha256",
    "size_bytes",
    "mime_type",
    "codec",
    "duration_ms",
    "clip_start_at",
    "clip_end_at",
    "finalized_at",
];

/// Python `_payload` for an EVENT entry: decode `values_b64` and
/// `decision_trace_b64`, merge a truthy trace as `audit`, encode compactly
/// with sorted keys.
pub fn event_body(entry: &Value) -> Result<Vec<u8>, BodyError> {
    let values = decoded_json(entry, "values_b64")?;
    let trace = decoded_json(entry, "decision_trace_b64")?;
    let body = if truthy(&trace) {
        let Value::Object(mut body) = values else {
            return Err(BodyError::AuditTarget);
        };
        body.insert("audit".to_owned(), trace);
        Value::Object(body)
    } else {
        values
    };
    Ok(body.to_string().into_bytes())
}

/// One clip PUT: `api/v1/relay/clips/{clip_id}` with `body`, answered by a
/// receipt that must carry `clip_id` at `state_version`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipRequest {
    pub clip_id: String,
    pub state_version: i64,
    pub body: Vec<u8>,
}

impl ClipRequest {
    pub fn path(&self) -> String {
        format!("api/v1/relay/clips/{}", self.clip_id)
    }
}

/// Python `_clip_claim` followed by `send_clip`'s payload.
pub fn clip_request(entry: &Value) -> Result<ClipRequest, BodyError> {
    let local_state = text(entry, "local_state")?;
    if !LOCAL_STATES.contains(&local_state) {
        return Err(BodyError::LocalState);
    }
    let reason = optional_text(entry, "unavailable_reason")?;
    if reason.is_some_and(|reason| !REASONS.contains(&reason)) {
        return Err(BodyError::UnavailableReason);
    }
    let clip_id = text(entry, "clip_id")?;
    if clip_id.is_empty() || !clip_id.bytes().all(is_segment_byte) {
        return Err(BodyError::ClipIdSegment);
    }
    let state_version = field(entry, "state_version")?
        .as_i64()
        .ok_or(BodyError::WrongType("state_version"))?;
    optional_text(entry, "media_reference")?;
    for name in TEXT_MEDIA_FIELDS {
        optional_text(entry, name)?;
    }
    for name in INT_MEDIA_FIELDS {
        let value = field(entry, name)?;
        if !value.is_null() && value.as_i64().is_none() {
            return Err(BodyError::WrongType(name));
        }
    }
    let event_refs = event_refs(entry)?;
    let (camera_id, facility_id) =
        local_identity(field(entry, "camera_id")?, field(entry, "facility_id")?)?;
    let mut payload = Map::new();
    let ready = local_state == "VERIFIED";
    let state = if ready { "READY" } else { "UNAVAILABLE" };
    payload.insert("state".to_owned(), Value::from(state));
    payload.insert("camera_id".to_owned(), Value::from(camera_id));
    payload.insert("facility_id".to_owned(), Value::from(facility_id));
    payload.insert("event_refs".to_owned(), Value::Array(event_refs));
    payload.insert("state_version".to_owned(), Value::from(state_version));
    if ready {
        for name in READY_FIELDS {
            payload.insert(name.to_owned(), field(entry, name)?.clone());
        }
    } else {
        let backend = if reason == Some("CORRUPT") {
            "CORRUPT"
        } else {
            "CAPTURE_FAILED"
        };
        payload.insert("reason".to_owned(), Value::from(backend));
    }
    Ok(ClipRequest {
        clip_id: clip_id.to_owned(),
        state_version,
        body: Value::Object(payload).to_string().into_bytes(),
    })
}

/// Python `_local_event_identity`: both ids are non-blank strings and their
/// compact JSON stays within `MAX_LOCAL_EVENT_PAYLOAD_BYTES`.
fn local_identity<'a>(
    camera: &'a Value,
    facility: &'a Value,
) -> Result<(&'a str, &'a str), BodyError> {
    let (Some(camera_id), Some(facility_id)) = (camera.as_str(), facility.as_str()) else {
        return Err(BodyError::InvalidEventIdentity);
    };
    let blank = |value: &str| value.trim().is_empty();
    let encoded = serde_json::json!({"camera_id": camera_id, "facility_id": facility_id});
    if blank(camera_id)
        || blank(facility_id)
        || encoded.to_string().len() > MAX_LOCAL_EVENT_PAYLOAD_BYTES
    {
        return Err(BodyError::InvalidEventIdentity);
    }
    Ok((camera_id, facility_id))
}

fn event_refs(entry: &Value) -> Result<Vec<Value>, BodyError> {
    let Value::Array(values) = field(entry, "event_ids")? else {
        return Err(BodyError::WrongType("event_ids"));
    };
    if values.iter().all(Value::is_string) {
        Ok(values.clone())
    } else {
        Err(BodyError::WrongType("event_ids"))
    }
}

/// RFC 3986 `pchar` without percent-encoding: the clip id travels unescaped.
fn is_segment_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"-._~!$&'()*+,;=:@".contains(&byte)
}

fn field<'a>(entry: &'a Value, name: &'static str) -> Result<&'a Value, BodyError> {
    entry.get(name).ok_or(BodyError::MissingField(name))
}

fn text<'a>(entry: &'a Value, name: &'static str) -> Result<&'a str, BodyError> {
    field(entry, name)?
        .as_str()
        .ok_or(BodyError::WrongType(name))
}

fn optional_text<'a>(entry: &'a Value, name: &'static str) -> Result<Option<&'a str>, BodyError> {
    match field(entry, name)? {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        _ => Err(BodyError::WrongType(name)),
    }
}

fn decoded_json(entry: &Value, name: &'static str) -> Result<Value, BodyError> {
    let bytes = decode_base64(text(entry, name)?).ok_or(BodyError::Base64(name))?;
    if !bytes.is_ascii() {
        return Err(BodyError::NotAscii(name));
    }
    serde_json::from_slice(&bytes).map_err(|_| BodyError::NotJson(name))
}

/// Python truthiness of a decoded JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}
