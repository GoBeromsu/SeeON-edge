//! Python entry fields, `DeliveryEntry`, `_serialize` bytes and `_keyed_id`.

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;

use sha2::{Digest, Sha256};

use super::entry::refuse_if;
use super::{
    ClipEntry, EntryError, EntryKind, EventEntry, QueueError, SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
};
use crate::b64;
use crate::json::{Json, Serialiser};

/// Python `EventEntry` fields; an empty `entry_id` takes the Python default.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventFields {
    pub edge_event_id: String,
    pub event_type: String,
    pub detected_at: String,
    pub camera_id: String,
    pub facility_id: String,
    pub decision_trace: Vec<u8>,
    pub values: Vec<u8>,
    pub shed_detail_keys: Vec<String>,
    pub entry_id: String,
}

/// Python `SnapshotAttachmentEntry` fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotAttachmentFields {
    pub edge_event_id: String,
    pub snapshot_id: String,
    pub sha256: String,
    pub media_reference: String,
    pub size_bytes: i64,
    pub mime_type: String,
    pub entry_id: String,
}

/// Python `ClipEntry` fields; `None` serialises as `null`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClipFields {
    pub clip_id: String,
    pub event_ids: Vec<String>,
    pub camera_id: String,
    pub facility_id: String,
    pub local_state: String,
    pub state_version: i64,
    pub media_reference: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<i64>,
    pub mime_type: Option<String>,
    pub codec: Option<String>,
    pub duration_ms: Option<i64>,
    pub clip_start_at: Option<String>,
    pub clip_end_at: Option<String>,
    pub finalized_at: Option<String>,
    pub unavailable_reason: Option<String>,
    pub entry_id: String,
}

/// Python `SnapshotDispositionEntry` fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotDispositionFields {
    pub edge_event_id: String,
    pub snapshot_id: String,
    pub disposition: String,
    pub reason: String,
    pub entry_id: String,
}

/// One accepted entry of any kind (Python `DeliveryEntry`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeliveryEntry {
    Event(EventEntry),
    Clip(ClipEntry),
    SnapshotAttachment(SnapshotAttachmentEntry),
    SnapshotDisposition(SnapshotDispositionEntry),
}

impl DeliveryEntry {
    pub fn entry_id(&self) -> &str {
        match self {
            Self::Event(entry) => entry.entry_id(),
            Self::Clip(entry) => entry.entry_id(),
            Self::SnapshotAttachment(entry) => entry.entry_id(),
            Self::SnapshotDisposition(entry) => entry.entry_id(),
        }
    }

    pub fn kind(&self) -> EntryKind {
        match self {
            Self::Event(_) => EntryKind::Event,
            Self::Clip(_) => EntryKind::Clip,
            Self::SnapshotAttachment(_) => EntryKind::SnapshotAttachment,
            Self::SnapshotDisposition(_) => EntryKind::SnapshotDisposition,
        }
    }

    /// Python `_serialize`: `asdict` plus `kind`, the event byte fields as
    /// standard base64, sorted keys, compact separators, ASCII escapes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, QueueError> {
        let mut members = match self {
            Self::Event(entry) => event_members(entry.fields()),
            Self::Clip(entry) => clip_members(entry.fields()),
            Self::SnapshotAttachment(entry) => attachment_members(entry.fields()),
            Self::SnapshotDisposition(entry) => disposition_members(entry.fields()),
        };
        members.push(("kind", string(self.kind().as_str())));
        let members = members
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value));
        let text = Serialiser::ModelSelection.canonical(&Json::Object(members.collect()))?;
        Ok(text.into_bytes())
    }
}

/// Python `_keyed_id`: `<prefix>-<sha256 of the parts joined by NUL>`.
/// A non-ASCII part is refused on `entry_id`, where Python raises
/// `UnicodeEncodeError` from `str.encode("ascii")`.
pub fn keyed_id(prefix: &str, parts: &[&str]) -> Result<String, EntryError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    refuse_if(!parts.iter().all(|part| part.is_ascii()), "entry_id")?;
    let digest = Sha256::digest(parts.join("\0").as_bytes());
    let mut id = String::with_capacity(prefix.len() + 65);
    id.push_str(prefix);
    id.push('-');
    for byte in digest {
        id.push(char::from(HEX[usize::from(byte >> 4)]));
        id.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    Ok(id)
}

type Members = Vec<(&'static str, Json)>;

fn event_members(fields: &EventFields) -> Members {
    let keys = fields.shed_detail_keys.iter().map(|key| string(key));
    vec![
        ("edge_event_id", string(&fields.edge_event_id)),
        ("event_type", string(&fields.event_type)),
        ("detected_at", string(&fields.detected_at)),
        ("camera_id", string(&fields.camera_id)),
        ("facility_id", string(&fields.facility_id)),
        (
            "decision_trace_b64",
            Json::Str(b64::encode(&fields.decision_trace)),
        ),
        ("values_b64", Json::Str(b64::encode(&fields.values))),
        ("shed_detail_keys", Json::Array(keys.collect())),
        ("entry_id", string(&fields.entry_id)),
    ]
}

fn attachment_members(fields: &SnapshotAttachmentFields) -> Members {
    vec![
        ("edge_event_id", string(&fields.edge_event_id)),
        ("snapshot_id", string(&fields.snapshot_id)),
        ("sha256", string(&fields.sha256)),
        ("media_reference", string(&fields.media_reference)),
        ("size_bytes", Json::Int(i128::from(fields.size_bytes))),
        ("mime_type", string(&fields.mime_type)),
        ("entry_id", string(&fields.entry_id)),
    ]
}

fn clip_members(fields: &ClipFields) -> Members {
    let event_ids = fields.event_ids.iter().map(|event_id| string(event_id));
    vec![
        ("clip_id", string(&fields.clip_id)),
        ("event_ids", Json::Array(event_ids.collect())),
        ("camera_id", string(&fields.camera_id)),
        ("facility_id", string(&fields.facility_id)),
        ("local_state", string(&fields.local_state)),
        ("state_version", Json::Int(i128::from(fields.state_version))),
        (
            "media_reference",
            optional_text(fields.media_reference.as_deref()),
        ),
        ("sha256", optional_text(fields.sha256.as_deref())),
        ("size_bytes", optional_int(fields.size_bytes)),
        ("mime_type", optional_text(fields.mime_type.as_deref())),
        ("codec", optional_text(fields.codec.as_deref())),
        ("duration_ms", optional_int(fields.duration_ms)),
        (
            "clip_start_at",
            optional_text(fields.clip_start_at.as_deref()),
        ),
        ("clip_end_at", optional_text(fields.clip_end_at.as_deref())),
        (
            "finalized_at",
            optional_text(fields.finalized_at.as_deref()),
        ),
        (
            "unavailable_reason",
            optional_text(fields.unavailable_reason.as_deref()),
        ),
        ("entry_id", string(&fields.entry_id)),
    ]
}

fn disposition_members(fields: &SnapshotDispositionFields) -> Members {
    vec![
        ("edge_event_id", string(&fields.edge_event_id)),
        ("snapshot_id", string(&fields.snapshot_id)),
        ("disposition", string(&fields.disposition)),
        ("reason", string(&fields.reason)),
        ("entry_id", string(&fields.entry_id)),
    ]
}

fn string(value: &str) -> Json {
    Json::Str(value.to_owned())
}

fn optional_text(value: Option<&str>) -> Json {
    value.map_or(Json::Null, string)
}

fn optional_int(value: Option<i64>) -> Json {
    value.map_or(Json::Null, |value| Json::Int(i128::from(value)))
}

/// `<status>.<ordinal>.<original>` to `<original>`, which must end in `.json`.
pub(super) fn recoverable_name(name: &OsStr) -> Option<&OsStr> {
    let name = name.as_bytes();
    let remainder = &name[name.iter().position(|byte| *byte == b'.')? + 1..];
    let original = &remainder[remainder.iter().position(|byte| *byte == b'.')? + 1..];
    original
        .ends_with(b".json")
        .then(|| OsStr::from_bytes(original))
}

/// The `kind` of one published entry file.
pub(super) fn entry_kind(bytes: &[u8]) -> Result<EntryKind, QueueError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| QueueError::CorruptEntry)?;
    value
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .and_then(EntryKind::from_wire)
        .ok_or(QueueError::CorruptEntry)
}
