//! Durable `edge_event_id` per source key for one camera (G20), ported from
//! `worker/pipeline/decision/event_identity.py`. A restart resolves the same
//! key to the same id, so the backend never sees a re-minted duplicate.

mod journal;
mod line;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};

use crate::seam::{Clock, IdSource};
use line::Record;

/// Python `RETENTION_SEC`: 90 days.
pub const RETENTION_SEC: f64 = 7_776_000.0;
/// Python `MAX_JOURNAL_BYTES`: 16 MiB of encoded lines.
pub const MAX_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub retention_sec: f64,
    pub max_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            retention_sec: RETENTION_SEC,
            max_bytes: MAX_JOURNAL_BYTES,
        }
    }
}

/// Python `EventIdentityStoreError`; line numbers count from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityError {
    Malformed {
        line: usize,
    },
    ConflictingSourceKey {
        line: usize,
    },
    /// Reading, writing or syncing the journal, invalid UTF-8 in it
    /// (`InvalidData`), or an id source failure.
    Io(io::ErrorKind),
}

impl fmt::Display for IdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed { line } => {
                write!(formatter, "event identity journal: malformed line {line}")
            }
            Self::ConflictingSourceKey { line } => {
                write!(
                    formatter,
                    "event identity journal: conflicting source key at line {line}"
                )
            }
            Self::Io(kind) => write!(formatter, "event identity journal: {kind}"),
        }
    }
}

impl std::error::Error for IdentityError {}

impl From<io::Error> for IdentityError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

/// `<state_dir>/event-identities/<sha256(camera_id) hex>.jsonl`.
pub fn event_identity_path(camera_id: &str, state_dir: &Path) -> PathBuf {
    let digest: String = Sha256::digest(camera_id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    state_dir
        .join("event-identities")
        .join(format!("{digest}.jsonl"))
}

pub struct EventIdentityStore {
    path: Option<PathBuf>,
    limits: Limits,
    clock: Arc<dyn Clock>,
    ids: Arc<dyn IdSource>,
    records: HashMap<String, Record>,
}

impl EventIdentityStore {
    /// Loads the journal, keeps what retention and the byte budget allow, and
    /// rewrites the file when it held any record. `None` keeps ids in memory.
    pub fn open(
        path: Option<PathBuf>,
        limits: Limits,
        clock: Arc<dyn Clock>,
        ids: Arc<dyn IdSource>,
    ) -> Result<Self, IdentityError> {
        let now = wall_seconds(clock.as_ref());
        let loaded = journal::load(path.as_deref(), now)?;
        let records = select_retained(loaded.values().cloned(), now, limits);
        let store = Self {
            path,
            limits,
            clock,
            ids,
            records,
        };
        if !loaded.is_empty() {
            store.rewrite(&store.records)?;
        }
        Ok(store)
    }

    /// The id for `source_key`: a known one, one another writer journaled
    /// since, or a newly minted one that is journaled before it is returned.
    pub fn resolve(&mut self, source_key: &str) -> Result<String, IdentityError> {
        if let Some(record) = self.records.get(source_key) {
            return Ok(record.edge_event_id.clone());
        }
        self.refresh()?;
        if let Some(record) = self.records.get(source_key) {
            return Ok(record.edge_event_id.clone());
        }
        let minted = self.ids.uuid4()?;
        let edge_event_id =
            line::normalize_uuid4(&minted).ok_or(IdentityError::Io(io::ErrorKind::InvalidData))?;
        let record = Record {
            source_key: source_key.to_owned(),
            edge_event_id,
            recorded_at: self.now(),
        };
        let mut updated = self.records.clone();
        updated.insert(source_key.to_owned(), record.clone());
        let retained = select_retained(updated.into_values(), self.now(), self.limits);
        self.rewrite(&retained)?;
        let resolved = retained
            .get(source_key)
            .unwrap_or(&record)
            .edge_event_id
            .clone();
        self.records = retained;
        Ok(resolved)
    }

    /// Adds journaled keys this store does not know yet, unfiltered.
    fn refresh(&mut self) -> Result<(), IdentityError> {
        let loaded = journal::load(self.path.as_deref(), self.now())?;
        for (source_key, record) in loaded {
            self.records.entry(source_key).or_insert(record);
        }
        Ok(())
    }

    /// Oldest first, ties by source key; nothing without a path.
    fn rewrite(&self, records: &HashMap<String, Record>) -> Result<(), IdentityError> {
        let Some(path) = self.path.as_deref() else {
            return Ok(());
        };
        let mut ordered: Vec<&Record> = records.values().collect();
        ordered.sort_by(|left, right| chronological(left, right));
        let payload: String = ordered.iter().map(|record| record.encode()).collect();
        journal::replace(path, &payload)
    }

    fn now(&self) -> f64 {
        wall_seconds(self.clock.as_ref())
    }
}

/// Records dated within `[now - retention, now]`, newest first until the
/// next encoded line would exceed the byte budget.
fn select_retained(
    records: impl Iterator<Item = Record>,
    now: f64,
    limits: Limits,
) -> HashMap<String, Record> {
    let cutoff = now - limits.retention_sec;
    let mut eligible: Vec<Record> = records
        .filter(|record| cutoff <= record.recorded_at && record.recorded_at <= now)
        .collect();
    eligible.sort_by(|left, right| chronological(right, left));
    let mut retained = HashMap::new();
    let mut size = 0usize;
    for record in eligible {
        let length = record.encode().len();
        if size + length > limits.max_bytes {
            break;
        }
        size += length;
        retained.insert(record.source_key.clone(), record);
    }
    retained
}

/// Python's `(recorded_at, source_key)` tuple order.
fn chronological(left: &Record, right: &Record) -> Ordering {
    left.recorded_at
        .partial_cmp(&right.recorded_at)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.source_key.cmp(&right.source_key))
}

/// Python `time.time()`: seconds since the epoch, negative before it.
fn wall_seconds(clock: &dyn Clock) -> f64 {
    match clock.wall().duration_since(UNIX_EPOCH) {
        Ok(since) => since.as_secs_f64(),
        Err(before) => -before.duration().as_secs_f64(),
    }
}
