//! Entry constructors with the Python `__post_init__` rules: a refusal
//! names the first failing field, checked in Python order.

use super::serial::{DeliveryEntry, keyed_id};
use super::{
    ClipFields, EntryError, EntryKind, EventFields, SnapshotAttachmentFields,
    SnapshotDispositionFields,
};

// `shared/events/envelope_limits.py`.
const ENTRY_ID_MAX: usize = 128;
const EDGE_EVENT_ID_MAX: usize = 128;
const EVENT_TYPE_MAX: usize = 32;
const DETECTED_AT_MAX: usize = 64;
const CAMERA_ID_MAX: usize = 128;
const FACILITY_ID_MAX: usize = 128;
const DECISION_TRACE_BYTES_MAX: usize = 16 * 1024;
const VALUES_BYTES_MAX: usize = 32 * 1024;
const SNAPSHOT_ID_MAX: usize = 128;
const SHA256_MAX: usize = 64;
const MEDIA_REFERENCE_MAX: usize = 1024;
const MIME_TYPE_MAX: usize = 128;
const DISPOSITION_MAX: usize = 64;
const DISPOSITION_REASON_MAX: usize = 1024;

macro_rules! entry_type {
    ($entry:ident, $fields:ident, $variant:ident) => {
        /// A validated entry; its fields are immutable once admitted.
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $entry {
            fields: $fields,
        }

        impl $entry {
            pub fn entry_id(&self) -> &str {
                &self.fields.entry_id
            }

            pub fn kind(&self) -> EntryKind {
                EntryKind::$variant
            }

            pub fn fields(&self) -> &$fields {
                &self.fields
            }
        }

        impl From<$entry> for DeliveryEntry {
            fn from(entry: $entry) -> Self {
                Self::$variant(entry)
            }
        }
    };
}

entry_type!(EventEntry, EventFields, Event);
entry_type!(ClipEntry, ClipFields, Clip);
entry_type!(
    SnapshotAttachmentEntry,
    SnapshotAttachmentFields,
    SnapshotAttachment
);
entry_type!(
    SnapshotDispositionEntry,
    SnapshotDispositionFields,
    SnapshotDisposition
);

impl EventEntry {
    pub fn new(mut fields: EventFields) -> Result<Self, EntryError> {
        if fields.entry_id.is_empty() {
            fields.entry_id = format!("event-{}", fields.edge_event_id);
        }
        text(&fields.entry_id, ENTRY_ID_MAX, "entry_id")?;
        text(&fields.edge_event_id, EDGE_EVENT_ID_MAX, "edge_event_id")?;
        text(&fields.event_type, EVENT_TYPE_MAX, "event_type")?;
        text(&fields.detected_at, DETECTED_AT_MAX, "detected_at")?;
        text(&fields.camera_id, CAMERA_ID_MAX, "camera_id")?;
        text(&fields.facility_id, FACILITY_ID_MAX, "facility_id")?;
        within(
            fields.decision_trace.len(),
            DECISION_TRACE_BYTES_MAX,
            "decision_trace",
        )?;
        within(fields.values.len(), VALUES_BYTES_MAX, "values")?;
        refuse_if(
            fields.shed_detail_keys.iter().any(String::is_empty),
            "shed_detail_keys",
        )?;
        Ok(Self { fields })
    }
}

impl SnapshotAttachmentEntry {
    pub fn new(mut fields: SnapshotAttachmentFields) -> Result<Self, EntryError> {
        if fields.entry_id.is_empty() {
            let parts = [&*fields.edge_event_id, &fields.snapshot_id, &fields.sha256];
            fields.entry_id = keyed_id("attachment", &parts)?;
        }
        text(&fields.entry_id, ENTRY_ID_MAX, "entry_id")?;
        text(&fields.edge_event_id, EDGE_EVENT_ID_MAX, "edge_event_id")?;
        text(&fields.snapshot_id, SNAPSHOT_ID_MAX, "snapshot_id")?;
        text(&fields.sha256, SHA256_MAX, "sha256")?;
        text(
            &fields.media_reference,
            MEDIA_REFERENCE_MAX,
            "media_reference",
        )?;
        text(&fields.mime_type, MIME_TYPE_MAX, "mime_type")?;
        refuse_if(fields.size_bytes < 0, "size_bytes")?;
        Ok(Self { fields })
    }
}

impl ClipEntry {
    pub fn new(mut fields: ClipFields) -> Result<Self, EntryError> {
        if fields.entry_id.is_empty() {
            fields.entry_id = keyed_id("clip", &[&fields.clip_id])?;
        }
        text(&fields.entry_id, ENTRY_ID_MAX, "entry_id")?;
        text(&fields.clip_id, EDGE_EVENT_ID_MAX, "clip_id")?;
        text(&fields.camera_id, CAMERA_ID_MAX, "camera_id")?;
        text(&fields.facility_id, FACILITY_ID_MAX, "facility_id")?;
        refuse_if(fields.event_ids.is_empty(), "event_ids")?;
        for event_id in &fields.event_ids {
            text(event_id, EDGE_EVENT_ID_MAX, "event_id")?;
        }
        let verified = fields.local_state == "VERIFIED";
        refuse_if(
            !verified && fields.local_state != "UNAVAILABLE",
            "local_state",
        )?;
        refuse_if(fields.state_version < 1, "state_version")?;
        if !verified {
            let reason = fields.unavailable_reason.as_deref();
            refuse_if(reason.is_none(), "unavailable_reason")?;
            text(
                reason.unwrap_or_default(),
                DISPOSITION_REASON_MAX,
                "unavailable_reason",
            )?;
            return Ok(Self { fields });
        }
        let required = [
            (
                "media_reference",
                &fields.media_reference,
                MEDIA_REFERENCE_MAX,
            ),
            ("sha256", &fields.sha256, SHA256_MAX),
            ("mime_type", &fields.mime_type, MIME_TYPE_MAX),
            ("codec", &fields.codec, MIME_TYPE_MAX),
            ("clip_start_at", &fields.clip_start_at, DETECTED_AT_MAX),
            ("clip_end_at", &fields.clip_end_at, DETECTED_AT_MAX),
            ("finalized_at", &fields.finalized_at, DETECTED_AT_MAX),
        ];
        for (field, value, maximum) in required {
            text(value.as_deref().unwrap_or_default(), maximum, field)?;
        }
        refuse_if(fields.size_bytes.is_none_or(|size| size <= 0), "size_bytes")?;
        refuse_if(fields.duration_ms.is_none_or(|ms| ms <= 0), "duration_ms")?;
        refuse_if(fields.unavailable_reason.is_some(), "unavailable_reason")?;
        Ok(Self { fields })
    }
}

impl SnapshotDispositionEntry {
    pub fn new(mut fields: SnapshotDispositionFields) -> Result<Self, EntryError> {
        if fields.entry_id.is_empty() {
            let parts = [
                &*fields.edge_event_id,
                &fields.snapshot_id,
                &fields.disposition,
            ];
            fields.entry_id = keyed_id("disposition", &parts)?;
        }
        text(&fields.entry_id, ENTRY_ID_MAX, "entry_id")?;
        text(&fields.edge_event_id, EDGE_EVENT_ID_MAX, "edge_event_id")?;
        text(&fields.snapshot_id, SNAPSHOT_ID_MAX, "snapshot_id")?;
        text(&fields.disposition, DISPOSITION_MAX, "disposition")?;
        text(&fields.reason, DISPOSITION_REASON_MAX, "reason")?;
        Ok(Self { fields })
    }
}

/// Python `_validate_text` for an `entry_id` given to a queue operation.
pub(super) fn validate_entry_id(entry_id: &str) -> Result<(), EntryError> {
    text(entry_id, ENTRY_ID_MAX, "entry_id")
}

/// Python `_validate_text`: non-empty printable ASCII within `maximum`
/// characters; an `entry_id` must also carry no path separator.
fn text(value: &str, maximum: usize, field: &'static str) -> Result<(), EntryError> {
    let printable = value.bytes().all(|byte| (b' '..=b'~').contains(&byte));
    refuse_if(
        value.is_empty() || value.len() > maximum || !printable,
        field,
    )?;
    refuse_if(field == "entry_id" && value.contains(['/', '\\']), field)
}

/// Python `_validate_bytes`.
fn within(length: usize, maximum: usize, field: &'static str) -> Result<(), EntryError> {
    refuse_if(length > maximum, field)
}

pub(super) const fn refuse_if(refused: bool, field: &'static str) -> Result<(), EntryError> {
    if refused {
        Err(EntryError { field })
    } else {
        Ok(())
    }
}
