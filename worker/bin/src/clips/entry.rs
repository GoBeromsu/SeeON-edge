//! Derived clip records: flow metadata from a sealed extension, and the
//! delivery entry announcing a terminal clip.

use std::collections::BTreeMap;

use crate::delivery::{ClipEntry, ClipFields, EntryError};

use super::manifest::{
    ClipMetadata, Extension, MIME_TYPE, ManifestError, TERMINAL_STATE_VERSION, Terminal,
    media_reference,
};
use super::time::Utc;

/// A flow clip starts this long before its earliest contributor.
pub const FLOW_LOOKBACK_MILLIS: i64 = 15_000;
/// The recorder that seals flow clips (Python `FlowClipPublisher`).
pub const FLOW_ENCODER: &str = "deepstream-smart-record";

/// The fields a flow clip takes from its earliest contributor's event
/// (Python `BusinessEvent`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContributorEvent {
    pub camera_id: String,
    pub facility_id: String,
    pub domain: String,
    pub event_type: String,
}

/// Flow metadata as Python `FlowClipPublisher._publish` derives it. The
/// earliest contributor's event fixes camera, facility, domain and event
/// type; the clip starts `FLOW_LOOKBACK_MILLIS` before that contributor,
/// lasts the sealed duration and is finalized no earlier than it ends.
/// `events` is keyed by `event_ref`.
pub fn flow_metadata(
    clip_id: &str,
    events: &BTreeMap<String, ContributorEvent>,
    mut extension: Extension,
    encoder: &str,
    now: Utc,
) -> Result<ClipMetadata, ManifestError> {
    extension
        .contributors
        .sort_by_key(|contributor| contributor.detected_at);
    let lookup = |event_ref: &str| events.get(event_ref).ok_or(ManifestError::MissingEvent);
    let (event, earliest) = match extension.contributors.first() {
        Some(primary) => (lookup(&primary.event_ref)?, primary.detected_at),
        None => return Err(ManifestError::Blank("contributors")),
    };
    for contributor in &extension.contributors {
        if lookup(&contributor.event_ref)?.camera_id != event.camera_id {
            return Err(ManifestError::SpansCameras);
        }
    }
    for contributor in &extension.contributors {
        if lookup(&contributor.event_ref)?.facility_id != event.facility_id {
            return Err(ManifestError::SpansFacilities);
        }
    }
    if extension.duration_ms <= 0 {
        return Err(ManifestError::NonPositiveDuration);
    }
    let start = earliest.plus_millis(-FLOW_LOOKBACK_MILLIS);
    let end = start.plus_millis(extension.duration_ms);
    Ok(ClipMetadata {
        clip_id: clip_id.to_owned(),
        camera_id: event.camera_id.clone(),
        facility_id: event.facility_id.clone(),
        domain: event.domain.clone(),
        event_type: event.event_type.clone(),
        event_refs: extension
            .contributors
            .iter()
            .map(|c| c.event_ref.clone())
            .collect(),
        detected_at: earliest,
        started_at: start,
        clip_start_at: start,
        clip_end_at: end,
        finalized_at: now.max(end),
        duration_ms: extension.duration_ms,
        encoder: encoder.to_owned(),
        truncation_reasons: Vec::new(),
        extension: Some(extension),
    })
}

/// The delivery entry announcing a terminal clip.
pub fn clip_entry(meta: &ClipMetadata, terminal: &Terminal) -> Result<ClipEntry, EntryError> {
    let mut fields = ClipFields {
        clip_id: meta.clip_id.clone(),
        event_ids: meta.event_refs.clone(),
        camera_id: meta.camera_id.clone(),
        facility_id: meta.facility_id.clone(),
        state_version: TERMINAL_STATE_VERSION,
        clip_start_at: Some(meta.clip_start_at.iso_millis()),
        clip_end_at: Some(meta.clip_end_at.iso_millis()),
        finalized_at: Some(meta.finalized_at.iso_millis()),
        ..ClipFields::default()
    };
    match terminal {
        Terminal::Ready(media) => {
            fields.local_state = "VERIFIED".to_owned();
            fields.media_reference = Some(media_reference(&meta.clip_id));
            fields.sha256 = Some(media.sha256.clone());
            fields.size_bytes = Some(media.size_bytes);
            fields.mime_type = Some(MIME_TYPE.to_owned());
            fields.codec = Some(media.codec.clone());
            fields.duration_ms = Some(media.duration_ms);
        }
        Terminal::Unavailable { reason_code, .. } => {
            fields.local_state = "UNAVAILABLE".to_owned();
            fields.unavailable_reason = Some(reason_code.clone());
        }
    }
    ClipEntry::new(fields)
}
