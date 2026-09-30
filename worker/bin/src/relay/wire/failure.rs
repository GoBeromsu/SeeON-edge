//! Python `classify_http_failure`, `_failure_code`, `_header_value`,
//! `parse_json_object` and `_describe_exception`
//! (`shared/events/evidence_http_transport.py`): how a relay response that is
//! not a matching receipt, or no response at all, becomes a failure.

use serde_json::{Map, Value};

use super::retry_after::retry_after;
use crate::seam::Clock;

/// Response bodies larger than this are treated as an empty JSON object.
pub const MAX_RESPONSE_BYTES: usize = 65_536;
/// Transport error descriptions are cut to this many characters.
pub const MAX_TRANSPORT_ERROR_CHARS: usize = 200;

const RETRY_STATUSES: [u16; 5] = [401, 403, 408, 425, 429];
const COMPATIBILITY_STATUSES: [u16; 2] = [404, 405];
const UTF8_BOM: &[u8] = b"\xEF\xBB\xBF";

/// Python `DeliveryDisposition`: what the sender does with a failed delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryDisposition {
    Retry,
    Permanent,
    Compatibility,
}

impl DeliveryDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Retry => "RETRY",
            Self::Permanent => "PERMANENT",
            Self::Compatibility => "COMPATIBILITY",
        }
    }
}

/// Python `DeliveryFailureCode`: the machine-readable `detail.code` values a
/// relay error body may carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryFailureCode {
    CameraMappingMissing,
}

impl DeliveryFailureCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CameraMappingMissing => "CAMERA_MAPPING_MISSING",
        }
    }

    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "CAMERA_MAPPING_MISSING" => Some(Self::CameraMappingMissing),
            _ => None,
        }
    }
}

/// Python `DeliveryFailure`.
#[derive(Clone, Debug, PartialEq)]
pub struct DeliveryFailure {
    pub disposition: DeliveryDisposition,
    pub code: String,
    pub status_code: Option<u16>,
    pub retry_after_seconds: Option<f64>,
    pub transport_error: Option<String>,
}

impl DeliveryFailure {
    /// A 2xx response whose body is not the expected receipt.
    pub fn malformed_receipt() -> Self {
        Self::bare(DeliveryDisposition::Retry, "MALFORMED_RECEIPT")
    }

    /// No HTTP response at all: `kind` names the error, `message` describes it.
    pub fn network(kind: &str, message: &str) -> Self {
        Self {
            transport_error: Some(describe_transport_error(kind, message)),
            ..Self::bare(DeliveryDisposition::Retry, "NETWORK")
        }
    }

    fn bare(disposition: DeliveryDisposition, code: &str) -> Self {
        Self {
            disposition,
            code: code.to_owned(),
            status_code: None,
            retry_after_seconds: None,
            transport_error: None,
        }
    }
}

/// Python `_describe_exception`: `"<kind>: <message>"`, cut to
/// `MAX_TRANSPORT_ERROR_CHARS` characters with a trailing ellipsis.
pub fn describe_transport_error(kind: &str, message: &str) -> String {
    let text = format!("{kind}: {message}");
    if text.chars().count() <= MAX_TRANSPORT_ERROR_CHARS {
        return text;
    }
    let mut cut: String = text.chars().take(MAX_TRANSPORT_ERROR_CHARS - 1).collect();
    cut.push('\u{2026}');
    cut
}

/// Python `classify_http_failure`. `headers` are in wire order; the first
/// `Retry-After` (any case) is used. `clock.wall()` is "now" for date hints.
pub fn classify_http_failure(
    status: u16,
    headers: &[(String, String)],
    body: Option<&[u8]>,
    clock: &dyn Clock,
) -> DeliveryFailure {
    let code = failure_code(body);
    let disposition = if code == Some(DeliveryFailureCode::CameraMappingMissing) {
        DeliveryDisposition::Retry
    } else if COMPATIBILITY_STATUSES.contains(&status) {
        DeliveryDisposition::Compatibility
    } else if RETRY_STATUSES.contains(&status) || (500..=599).contains(&status) {
        DeliveryDisposition::Retry
    } else {
        DeliveryDisposition::Permanent
    };
    DeliveryFailure {
        disposition,
        code: code.map_or_else(|| format!("HTTP_{status}"), |c| c.as_str().to_owned()),
        status_code: Some(status),
        retry_after_seconds: retry_after(header_value(headers, "Retry-After"), clock.wall()),
        transport_error: None,
    }
}

/// Python `_failure_code`: `detail.code` from a JSON error body, when it is
/// a known `DeliveryFailureCode`.
pub fn failure_code(body: Option<&[u8]>) -> Option<DeliveryFailureCode> {
    let payload = parse_json_object(body?);
    let Some(Value::Object(detail)) = payload.get("detail") else {
        return None;
    };
    let Some(Value::String(code)) = detail.get("code") else {
        return None;
    };
    DeliveryFailureCode::from_wire(code)
}

/// Python `_header_value`: the first header whose name matches, ignoring
/// ASCII case.
pub(super) fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// Python `parse_json_object`: the body as a JSON object, or an empty object
/// when it is too large, not JSON, or not an object. A UTF-8 byte order mark
/// is skipped, as Python's `json.loads` does for bytes.
pub fn parse_json_object(body: &[u8]) -> Map<String, Value> {
    if body.len() > MAX_RESPONSE_BYTES {
        return Map::new();
    }
    let text = body.strip_prefix(UTF8_BOM).unwrap_or(body);
    match serde_json::from_slice::<Value>(text) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}
