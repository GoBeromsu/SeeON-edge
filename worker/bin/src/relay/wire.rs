//! The response side of Python `shared/events/evidence_http_transport.py`:
//! an HTTP result (a response, or a transport failure) becomes a receipt or a
//! `DeliveryFailure`. Pure: no I/O, and "now" comes from the `Clock` seam.
//! `parse_capabilities` belongs to stage 3 and is not here.

use crate::seam::Clock;

mod failure;
mod receipt;
mod retry_after;

pub use failure::{
    DeliveryDisposition, DeliveryFailure, DeliveryFailureCode, MAX_RESPONSE_BYTES,
    MAX_TRANSPORT_ERROR_CHARS, classify_http_failure, describe_transport_error, failure_code,
    parse_json_object,
};
pub use receipt::{ClipReceipt, ClipState, EventReceipt, EventStatus, clip_receipt, event_receipt};
pub use retry_after::{MAX_RETRY_AFTER_SECONDS, retry_after};

/// One HTTP response as the relay client read it: status, headers in wire
/// order (lower-case names, Latin-1 values), and at most
/// `MAX_RESPONSE_BYTES + 1` body bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// A response, or the failure that stood in for one (Python `_HttpResult`).
pub type HttpResult = Result<Response, DeliveryFailure>;

/// Python `parse_event_result`: a 2xx or 409 whose receipt names
/// `expected_edge_event_id` is an acknowledgement; any other 2xx is a
/// malformed receipt; everything else is classified.
pub fn parse_event_result(
    result: HttpResult,
    expected_edge_event_id: &str,
    clock: &dyn Clock,
) -> Result<EventReceipt, DeliveryFailure> {
    let response = result?;
    match event_receipt(&response.body) {
        Some(receipt)
            if receipt.edge_event_id == expected_edge_event_id && is_ack(response.status) =>
        {
            Ok(receipt)
        }
        _ => Err(unmatched(&response, clock)),
    }
}

/// Python `parse_clip_result`: as `parse_event_result`, and the receipt's
/// `state_version` must equal `expected_state_version`, except that an
/// `EXPIRED` receipt at a newer version also acknowledges.
pub fn parse_clip_result(
    result: HttpResult,
    expected_clip_id: &str,
    expected_state_version: i64,
    clock: &dyn Clock,
) -> Result<ClipReceipt, DeliveryFailure> {
    let response = result?;
    match clip_receipt(&response.body) {
        Some(receipt)
            if receipt.clip_id == expected_clip_id
                && version_matches(&receipt, expected_state_version)
                && is_ack(response.status) =>
        {
            Ok(receipt)
        }
        _ => Err(unmatched(&response, clock)),
    }
}

fn version_matches(receipt: &ClipReceipt, expected: i64) -> bool {
    receipt.state_version == expected
        || (receipt.state == ClipState::Expired && receipt.state_version >= expected)
}

/// 409 means the relay already holds this delivery: an acknowledgement.
fn is_ack(status: u16) -> bool {
    (200..300).contains(&status) || status == 409
}

fn unmatched(response: &Response, clock: &dyn Clock) -> DeliveryFailure {
    if (200..300).contains(&response.status) {
        return DeliveryFailure::malformed_receipt();
    }
    classify_http_failure(
        response.status,
        &response.headers,
        Some(&response.body),
        clock,
    )
}
