//! Flow payload and checked wall-time capture; no admission or persistence.

use std::time::UNIX_EPOCH;

use seeon_worker::episode::BusinessEvent;

use super::events::EventDeliveryError;
use crate::clips::time::Utc;
use crate::json::Json;
use crate::policy::emit::Payload;
use crate::seam::Clock;

pub(super) fn observed_at(clock: &dyn Clock) -> Result<u64, EventDeliveryError> {
    let elapsed = clock
        .wall()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| EventDeliveryError::WallTime)?;
    u64::try_from(elapsed.as_nanos()).map_err(|_| EventDeliveryError::WallTime)
}

pub(super) fn captured_at(clock: &dyn Clock) -> Result<(u64, String), EventDeliveryError> {
    let nanos = observed_at(clock)?;
    let micros = i64::try_from(nanos / 1_000).map_err(|_| EventDeliveryError::WallTime)?;
    let detected_at = Utc::from_micros(micros).iso_micros();
    // datetime.isoformat() omits fractional seconds when microseconds are zero.
    let detected_at = match detected_at.strip_suffix(".000000Z") {
        Some(seconds) => format!("{seconds}Z"),
        None => detected_at,
    };
    Ok((nanos, detected_at))
}

/// Validate the admitted spelling without minting, normalizing or remapping it.
/// Domain episode keys are not UUIDs and must be rejected, not substituted.
pub(super) fn is_uuid(identity: &str) -> bool {
    identity.len() == 36
        && identity.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')
            }
        })
}

/// `worker/pipeline/output/event_sink.py:emit_for_frame` payload.
///
/// `audit` is the already-resolved scalar mapping. `None` omits the key; this
/// function never fabricates config, manifest, or model fields. Only an object
/// or array is refused, before any durable side effect. Python accepts
/// `str | int | float | bool | None`.
pub(super) fn payload(
    event: &BusinessEvent,
    detected_at: &str,
    audit: Option<&Payload>,
) -> Result<Payload, EventDeliveryError> {
    let text = |key: &str, value: &str| (key.to_owned(), Json::Str(value.to_owned()));
    let mut evidence = vec![
        text("domain", &event.domain),
        text("identity", &event.identity),
        ("time_sec".to_owned(), Json::Float(event.time_sec)),
    ];
    if let Some(person_id) = event.person_id {
        evidence.push(("person_id".to_owned(), Json::Int(i128::from(person_id))));
    }
    if let Some(bed_id) = event.bed_id {
        evidence.push(("bed_id".to_owned(), Json::Int(i128::from(bed_id))));
    }
    let mut body = vec![
        text("edge_event_id", &event.identity),
        text("event_type", &event.event_type),
        (
            "probability".to_owned(),
            event.probability.map_or(Json::Null, Json::Float),
        ),
        text("detected_at", detected_at),
        text("camera_id", &event.camera_id),
        text("facility_id", &event.facility_id),
        ("evidence".to_owned(), Json::Object(evidence)),
    ];
    if let Some(audit) = audit {
        reject_non_scalar_audit(Some(audit))?;
        body.push(("audit".to_owned(), Json::Object(audit.clone())));
    }
    Ok(body)
}

pub(super) fn reject_non_scalar_audit(audit: Option<&Payload>) -> Result<(), EventDeliveryError> {
    let Some(audit) = audit else {
        return Ok(());
    };
    if audit.iter().any(|(_, value)| !is_event_scalar(value)) {
        return Err(EventDeliveryError::Audit);
    }
    Ok(())
}

fn is_event_scalar(value: &Json) -> bool {
    matches!(
        value,
        Json::Null | Json::Bool(_) | Json::Int(_) | Json::Float(_) | Json::Str(_)
    )
}
