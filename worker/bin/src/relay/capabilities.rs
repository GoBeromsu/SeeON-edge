//! Python `probe_capabilities` (`shared/events/evidence_export_client.py`)
//! and `parse_capabilities` (`shared/events/evidence_http_transport.py`):
//! ask the relay which evidence features the backend supports for a camera.

use crate::relay::wire::{DeliveryFailure, classify_http_failure, parse_json_object};
use crate::relay::{RelayClient, Response};
use crate::seam::Clock;
use serde_json::Value;

/// The backend features the relay reports. `event_idempotency` is always 1
/// once parsed; `clip_export` is 0 or 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub event_idempotency: u8,
    pub clip_export: u8,
}

impl BackendCapabilities {
    /// True when the backend accepts clip PUTs.
    pub fn clip_export_enabled(self) -> bool {
        self.clip_export == 1
    }
}

/// Why the capabilities could not be used.
#[derive(Clone, Debug, PartialEq)]
pub enum CapabilityError {
    /// `event_idempotency` is missing or is not the integer 1.
    EventIdempotency,
    /// `clip_export` is missing or is not the integer 0 or 1.
    ClipExport,
    /// No usable response: a transport failure or a non-2xx status.
    Delivery(DeliveryFailure),
}

impl CapabilityError {
    /// The failure Python returns in its place: a refused body is a
    /// malformed receipt (retry), a delivery failure is itself.
    pub fn failure(&self) -> DeliveryFailure {
        match self {
            Self::EventIdempotency | Self::ClipExport => DeliveryFailure::malformed_receipt(),
            Self::Delivery(failure) => failure.clone(),
        }
    }
}

/// Python `parse_capabilities`: the body must be a JSON object whose
/// `event_idempotency` is 1 and whose `clip_export` is 0 or 1. Only JSON
/// integers are accepted (Python would also accept `true` and `1.0`).
pub fn parse_capabilities(body: &[u8]) -> Result<BackendCapabilities, CapabilityError> {
    let payload = parse_json_object(body);
    let event_idempotency = match integer(payload.get("event_idempotency")) {
        Some(1) => 1,
        _ => return Err(CapabilityError::EventIdempotency),
    };
    let clip_export = match integer(payload.get("clip_export")) {
        Some(value @ (0 | 1)) => value,
        _ => return Err(CapabilityError::ClipExport),
    };
    Ok(BackendCapabilities {
        event_idempotency,
        clip_export,
    })
}

/// Python `probe_capabilities`: GET the capabilities for `camera_id` with
/// the client's token and timeout. A transport failure is returned as is, a
/// non-2xx is classified from its status and headers only (no body), and a
/// 2xx body is parsed.
pub fn probe_capabilities(
    client: &RelayClient,
    camera_id: &str,
    clock: &dyn Clock,
) -> Result<BackendCapabilities, CapabilityError> {
    let response: Response = client
        .get_capabilities(camera_id)
        .map_err(|error| CapabilityError::Delivery(error.into()))?;
    if !(200..300).contains(&response.status) {
        return Err(CapabilityError::Delivery(classify_http_failure(
            response.status,
            &response.headers,
            None,
            clock,
        )));
    }
    parse_capabilities(&response.body)
}

fn integer(value: Option<&Value>) -> Option<u8> {
    match value {
        Some(Value::Number(number)) => number.as_u64().and_then(|n| u8::try_from(n).ok()),
        _ => None,
    }
}
