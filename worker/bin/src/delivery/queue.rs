//! Python `shared/events/delivery_queue.py`: one JSON file per accepted
//! entry, byte-identical to the Python producer, guarded by one `flock` on
//! `.delivery-queue.lock` so both languages may share a directory.

use std::collections::BTreeMap;
use std::fmt;
use std::io;

use crate::json::JsonError;

mod durable;
mod entry;
mod serial;
mod store;

pub use entry::{ClipEntry, EventEntry, SnapshotAttachmentEntry, SnapshotDispositionEntry};
pub use serial::{
    ClipFields, DeliveryEntry, EventFields, SnapshotAttachmentFields, SnapshotDispositionFields,
    keyed_id,
};
pub use store::DeliveryQueue;

pub const MAX_ACCEPTED_ENTRIES: usize = 4096;
pub const MAX_ACCEPTED_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_DEAD_LETTERED_ENTRIES: usize = MAX_ACCEPTED_ENTRIES;
pub const MAX_DEAD_LETTERED_BYTES: u64 = MAX_ACCEPTED_BYTES;
pub const LOCK_FILE_NAME: &str = ".delivery-queue.lock";

/// Python `EntryKind`; the wire value is the `kind` member of every file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntryKind {
    Event,
    Clip,
    SnapshotAttachment,
    SnapshotDisposition,
}

impl EntryKind {
    pub const ALL: [EntryKind; 4] = [
        EntryKind::Event,
        EntryKind::Clip,
        EntryKind::SnapshotAttachment,
        EntryKind::SnapshotDisposition,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Event => "EVENT",
            Self::Clip => "CLIP",
            Self::SnapshotAttachment => "SNAPSHOT_ATTACHMENT",
            Self::SnapshotDisposition => "SNAPSHOT_DISPOSITION",
        }
    }

    pub fn from_wire(value: &str) -> Option<EntryKind> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == value)
    }
}

/// Python `AdmissionFault`: why an admission was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionFault {
    EntryCapacity,
    ByteCapacity,
    Conflict,
    LockUnavailable,
}

impl AdmissionFault {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EntryCapacity => "entry_capacity",
            Self::ByteCapacity => "byte_capacity",
            Self::Conflict => "conflict",
            Self::LockUnavailable => "lock_unavailable",
        }
    }
}

/// Python `AdmissionResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionResult {
    pub accepted: bool,
    pub fault: Option<AdmissionFault>,
    pub already_admitted: bool,
}

impl AdmissionResult {
    pub(crate) const fn admitted(already_admitted: bool) -> Self {
        Self {
            accepted: true,
            fault: None,
            already_admitted,
        }
    }

    pub(crate) const fn refused(fault: AdmissionFault) -> Self {
        Self {
            accepted: false,
            fault: Some(fault),
            already_admitted: false,
        }
    }
}

/// Python `DeliveryQueueCapacitySnapshot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacitySnapshot {
    pub accepted_count: usize,
    pub accepted_bytes: u64,
    pub max_accepted_entries: usize,
    pub max_accepted_bytes: u64,
    /// All four kinds are present; an absent kind counts zero.
    pub by_kind: BTreeMap<EntryKind, usize>,
    pub dead_lettered_count: usize,
    pub dead_lettered_bytes: u64,
    /// `YYYY-MM-DDTHH:MM:SS.mmmZ` from the oldest event file's mtime.
    pub oldest_event_accepted_at: Option<String>,
}

/// A constructor refusal naming the first field that failed validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryError {
    pub field: &'static str,
}

impl fmt::Display for EntryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "delivery entry field {} is outside its envelope",
            self.field
        )
    }
}
impl std::error::Error for EntryError {}

/// Every way a queue operation can fail; refusals by policy are values of
/// [`AdmissionResult`] or `false`, never errors.
#[derive(Debug)]
pub enum QueueError {
    Io(io::Error),
    InvalidEntryId(EntryError),
    Json(JsonError),
    /// A published file is not a JSON object with a known `kind`.
    CorruptEntry,
    /// A retained dead-letter file name is not `<status>.<ordinal>.<file>.json`.
    UnrecoverableName,
}

impl fmt::Display for QueueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "delivery queue I/O failed: {error}"),
            Self::InvalidEntryId(error) => error.fmt(formatter),
            Self::Json(error) => write!(formatter, "delivery entry is not encodable: {error}"),
            Self::CorruptEntry => formatter.write_str("delivery queue holds a corrupt entry"),
            Self::UnrecoverableName => {
                formatter.write_str("dead-lettered entry name cannot be requeued")
            }
        }
    }
}

impl std::error::Error for QueueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidEntryId(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::CorruptEntry | Self::UnrecoverableName => None,
        }
    }
}

impl From<io::Error> for QueueError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<JsonError> for QueueError {
    fn from(error: JsonError) -> Self {
        Self::Json(error)
    }
}

/// The refusal for adding `length` bytes to a queue of `count` entries and
/// `bytes` bytes; nothing is ever evicted to make room.
fn capacity_fault(count: usize, bytes: u64, length: usize) -> Option<AdmissionFault> {
    let length = u64::try_from(length).unwrap_or(u64::MAX);
    if count + 1 > MAX_ACCEPTED_ENTRIES {
        Some(AdmissionFault::EntryCapacity)
    } else if bytes.saturating_add(length) > MAX_ACCEPTED_BYTES {
        Some(AdmissionFault::ByteCapacity)
    } else {
        None
    }
}
