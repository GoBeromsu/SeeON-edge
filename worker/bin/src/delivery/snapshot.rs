//! Snapshot attachment and disposition delivery: Python
//! `EvidenceExportClient.send_snapshot_attachment` / `send_snapshot_disposition`
//! (`shared/events/evidence_export_client.py`) with `_media_payload`
//! (`evidence_sender.py`) and `_parse_relay_acceptance`.

use serde_json::Value;

use crate::delivery::EntryKind;
use crate::relay::RelayClient;
use crate::relay::wire::{DeliveryFailure, HttpResult, classify_http_failure, parse_json_object};
use crate::seam::Clock;

/// Relay path for `SNAPSHOT_ATTACHMENT` entries (POST, no leading slash).
pub const SNAPSHOT_ATTACHMENTS_PATH: &str = "api/v1/relay/snapshot-attachments";
/// Relay path for `SNAPSHOT_DISPOSITION` entries (POST, no leading slash).
pub const SNAPSHOT_DISPOSITIONS_PATH: &str = "api/v1/relay/snapshot-dispositions";

/// A queued entry that cannot become a snapshot request (Python raises).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaEntryError {
    /// The entry is not a JSON object.
    NotAnObject,
    /// The entry kind is not a snapshot attachment or disposition.
    NotMedia,
}

/// The relay path for a media entry kind; `None` for EVENT and CLIP.
pub fn media_path(kind: EntryKind) -> Option<&'static str> {
    match kind {
        EntryKind::SnapshotAttachment => Some(SNAPSHOT_ATTACHMENTS_PATH),
        EntryKind::SnapshotDisposition => Some(SNAPSHOT_DISPOSITIONS_PATH),
        EntryKind::Event | EntryKind::Clip => None,
    }
}

/// Python `_media_payload`: every member of the entry except `entry_id` and
/// `kind`, encoded as compact JSON.
pub fn media_body(entry: &Value) -> Result<Vec<u8>, MediaEntryError> {
    let object = entry.as_object().ok_or(MediaEntryError::NotAnObject)?;
    let mut body = object.clone();
    body.remove("entry_id");
    body.remove("kind");
    serde_json::to_vec(&Value::Object(body)).map_err(|_| MediaEntryError::NotAnObject)
}

/// Python `_parse_relay_acceptance` (`evidence_export_client.py`
/// L447-457): a transport failure passes through, a non-2xx is classified
/// from its status and headers only (Python passes no body, so a failure
/// `code` in the body is not read), and a 2xx must carry
/// `"status": "accepted"`.
pub fn parse_relay_acceptance(
    result: HttpResult,
    clock: &dyn Clock,
) -> Result<(), DeliveryFailure> {
    let response = result?;
    if !(200..300).contains(&response.status) {
        return Err(classify_http_failure(
            response.status,
            &response.headers,
            None,
            clock,
        ));
    }
    match parse_json_object(&response.body).get("status") {
        Some(Value::String(status)) if status == "accepted" => Ok(()),
        _ => Err(DeliveryFailure::malformed_receipt()),
    }
}

/// POST one queued snapshot attachment or disposition. The outer `Err` is an
/// entry that cannot be sent at all; the inner result is the relay's verdict.
pub fn send_media(
    client: &RelayClient,
    kind: EntryKind,
    entry: &Value,
    clock: &dyn Clock,
) -> Result<Result<(), DeliveryFailure>, MediaEntryError> {
    let path = media_path(kind).ok_or(MediaEntryError::NotMedia)?;
    let body = media_body(entry)?;
    let result = client.post_json(path, &body).map_err(DeliveryFailure::from);
    Ok(parse_relay_acceptance(result, clock))
}
