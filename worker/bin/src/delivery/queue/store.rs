//! Python `DeliveryQueue`: the published entry files are the only durable
//! state, and every operation re-reads them under the queue's `flock`.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use super::durable::{self, ENTRY_SUFFIX, QueueLock};
use super::entry::validate_entry_id;
use super::serial::{entry_kind, recoverable_name};
use super::{
    AdmissionFault, AdmissionResult, CapacitySnapshot, DeliveryEntry, EntryKind, LOCK_FILE_NAME,
    MAX_ACCEPTED_BYTES, MAX_ACCEPTED_ENTRIES, MAX_DEAD_LETTERED_BYTES, MAX_DEAD_LETTERED_ENTRIES,
    QueueError, capacity_fault,
};

/// Backend statuses meaning the backend already holds the entry.
const ACKNOWLEDGED_STATUSES: [u16; 5] = [200, 201, 202, 204, 409];

/// A bounded on-disk delivery queue, interchangeable with the Python one.
#[derive(Debug)]
pub struct DeliveryQueue {
    directory: PathBuf,
    lock_path: PathBuf,
}

impl DeliveryQueue {
    /// Python `DeliveryQueue(directory, recover=...)`. With `recover` the
    /// orphan temps are removed and the totals rescanned under the lock.
    pub fn open(directory: &Path, recover: bool) -> Result<DeliveryQueue, QueueError> {
        durable::create_private_directory(directory)?;
        let queue = DeliveryQueue {
            directory: directory.to_path_buf(),
            lock_path: directory.join(LOCK_FILE_NAME),
        };
        durable::touch(&queue.lock_path)?;
        if recover {
            let _lock = queue.lock()?;
            durable::remove_orphan_temps(directory)?;
            durable::scan_totals(directory)?;
        }
        Ok(queue)
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Admit one entry, waiting for the queue lock.
    pub fn try_admit(&self, entry: &DeliveryEntry) -> Result<AdmissionResult, QueueError> {
        let (payload, target) = self.prepare(entry)?;
        let _lock = self.lock()?;
        self.admit_locked(&payload, &target)
    }

    /// Admit one entry without waiting; a held lock is `LockUnavailable`.
    pub fn try_admit_nonblocking(
        &self,
        entry: &DeliveryEntry,
    ) -> Result<AdmissionResult, QueueError> {
        let (payload, target) = self.prepare(entry)?;
        match QueueLock::try_exclusive(&self.lock_path)? {
            Some(_lock) => self.admit_locked(&payload, &target),
            None => Ok(AdmissionResult::refused(AdmissionFault::LockUnavailable)),
        }
    }

    /// Delete exactly one committed entry; `false` when it is not there.
    pub fn acknowledge(&self, entry_id: &str) -> Result<bool, QueueError> {
        validate_entry_id(entry_id).map_err(QueueError::InvalidEntryId)?;
        let target = self.entry_path(entry_id);
        let _lock = self.lock()?;
        if !target.try_exists()? {
            return Ok(false);
        }
        fs::remove_file(&target)?;
        durable::fsync_directory(&self.directory)?;
        durable::scan_totals(&self.directory)?;
        Ok(true)
    }

    /// Delete an entry the backend holds; any other status leaves it alone.
    pub fn acknowledge_backend(&self, entry_id: &str, status: u16) -> Result<bool, QueueError> {
        if !ACKNOWLEDGED_STATUSES.contains(&status) {
            return Ok(false);
        }
        self.acknowledge(entry_id)
    }

    /// Move a refused entry to `<status>.<ordinal>.<file>` in the dead-letter
    /// directory; `false` when it is missing or the retention area is full.
    pub fn dead_letter(&self, entry_id: &str, status: u16) -> Result<bool, QueueError> {
        validate_entry_id(entry_id).map_err(QueueError::InvalidEntryId)?;
        let source = self.entry_path(entry_id);
        let _lock = self.lock()?;
        if !source.try_exists()? {
            return Ok(false);
        }
        let retention = self.dead_letter_directory();
        let created = !retention.try_exists()?;
        fs::create_dir_all(&retention)?;
        if created {
            durable::fsync_directory(durable::parent_of(&retention))?;
        }
        let (count, bytes) = durable::retained_totals(&retention)?;
        let size = fs::metadata(&source)?.len();
        if count >= MAX_DEAD_LETTERED_ENTRIES
            || bytes.saturating_add(size) > MAX_DEAD_LETTERED_BYTES
        {
            return Ok(false);
        }
        let file_name = format!("{entry_id}{ENTRY_SUFFIX}");
        let destination_at =
            |ordinal: u64| retention.join(format!("{status}.{ordinal}.{file_name}"));
        let mut ordinal = 0_u64;
        while destination_at(ordinal).try_exists()? {
            ordinal += 1;
        }
        fs::hard_link(&source, destination_at(ordinal))?;
        durable::fsync_directory(&retention)?;
        fs::remove_file(&source)?;
        durable::fsync_directory(&self.directory)?;
        durable::scan_totals(&self.directory)?;
        Ok(true)
    }

    /// Python `dead_letter_directory`: `<parent>/<name>-dead-letter`.
    pub fn dead_letter_directory(&self) -> PathBuf {
        durable::dead_letter_directory(&self.directory)
    }

    /// Return one retained entry to the live queue under the queue lock;
    /// `false` leaves it retained (different bytes live, or the queue is full).
    pub fn requeue_dead_lettered(&self, retained: &Path) -> Result<bool, QueueError> {
        let payload = fs::read(retained)?;
        let original = retained
            .file_name()
            .and_then(recoverable_name)
            .ok_or(QueueError::UnrecoverableName)?;
        let target = self.directory.join(original);
        let retention = durable::parent_of(retained);
        let _lock = self.lock()?;
        let (count, bytes) = durable::scan_totals(&self.directory)?;
        if let Some(existing) = durable::read_if_exists(&target)? {
            if existing != payload {
                return Ok(false);
            }
        } else if capacity_fault(count, bytes, payload.len()).is_some() {
            return Ok(false);
        } else {
            durable::publish(&self.directory, &target, &payload)?;
        }
        fs::remove_file(retained)?;
        durable::fsync_directory(retention)?;
        Ok(true)
    }

    /// Every published entry as JSON, in file-name order.
    pub fn entries(&self) -> Result<Vec<serde_json::Value>, QueueError> {
        let paths = {
            let _lock = self.lock()?;
            durable::published_paths(&self.directory)?
        };
        let mut entries = Vec::with_capacity(paths.len());
        for path in &paths {
            let value = serde_json::from_slice(&fs::read(path)?);
            entries.push(value.map_err(|_| QueueError::CorruptEntry)?);
        }
        Ok(entries)
    }

    /// One locked, file-system-derived view of queue capacity.
    pub fn capacity_snapshot(&self) -> Result<CapacitySnapshot, QueueError> {
        let _lock = self.lock()?;
        let paths = durable::published_paths(&self.directory)?;
        let mut by_kind: BTreeMap<EntryKind, usize> =
            EntryKind::ALL.iter().map(|kind| (*kind, 0)).collect();
        let mut accepted_bytes = 0_u64;
        let mut oldest_event: Option<(i64, i64)> = None;
        for path in &paths {
            let metadata = fs::metadata(path)?;
            accepted_bytes = accepted_bytes.saturating_add(metadata.len());
            let kind = entry_kind(&fs::read(path)?)?;
            *by_kind.entry(kind).or_default() += 1;
            let modified = (metadata.mtime(), metadata.mtime_nsec());
            if kind == EntryKind::Event && oldest_event.is_none_or(|oldest| modified < oldest) {
                oldest_event = Some(modified);
            }
        }
        let (dead_lettered_count, dead_lettered_bytes) =
            durable::retained_totals(&self.dead_letter_directory())?;
        Ok(CapacitySnapshot {
            accepted_count: paths.len(),
            accepted_bytes,
            max_accepted_entries: MAX_ACCEPTED_ENTRIES,
            max_accepted_bytes: MAX_ACCEPTED_BYTES,
            by_kind,
            dead_lettered_count,
            dead_lettered_bytes,
            oldest_event_accepted_at: oldest_event
                .map(|(seconds, nanoseconds)| durable::accepted_at(seconds, nanoseconds)),
        })
    }

    /// Python `accepted_count`: rescanned under the lock.
    pub fn accepted_count(&self) -> Result<usize, QueueError> {
        let _lock = self.lock()?;
        Ok(durable::scan_totals(&self.directory)?.0)
    }

    /// Python `accepted_bytes`: rescanned under the lock.
    pub fn accepted_bytes(&self) -> Result<u64, QueueError> {
        let _lock = self.lock()?;
        Ok(durable::scan_totals(&self.directory)?.1)
    }

    fn lock(&self) -> Result<QueueLock, QueueError> {
        Ok(QueueLock::exclusive(&self.lock_path)?)
    }

    fn entry_path(&self, entry_id: &str) -> PathBuf {
        self.directory.join(format!("{entry_id}{ENTRY_SUFFIX}"))
    }

    fn prepare(&self, entry: &DeliveryEntry) -> Result<(Vec<u8>, PathBuf), QueueError> {
        Ok((entry.to_bytes()?, self.entry_path(entry.entry_id())))
    }

    /// Python `_admit_unlocked`; the caller holds the queue lock.
    fn admit_locked(&self, payload: &[u8], target: &Path) -> Result<AdmissionResult, QueueError> {
        durable::remove_orphan_temps(&self.directory)?;
        let (count, bytes) = durable::scan_totals(&self.directory)?;
        if let Some(existing) = durable::read_if_exists(target)? {
            return Ok(if existing == payload {
                // publish fsyncs the file before rename, but a previous attempt
                // may have failed its directory sync after the entry appeared.
                durable::fsync_directory(&self.directory)?;
                AdmissionResult::admitted(true)
            } else {
                AdmissionResult::refused(AdmissionFault::Conflict)
            });
        }
        if let Some(fault) = capacity_fault(count, bytes, payload.len()) {
            return Ok(AdmissionResult::refused(fault));
        }
        durable::publish(&self.directory, target, payload)?;
        Ok(AdmissionResult::admitted(false))
    }
}
