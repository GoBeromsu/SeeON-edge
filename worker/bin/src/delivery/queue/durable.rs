//! File-system steps of the Python queue: the `flock` guard, the durable
//! write (`_write_durable` plus `_fsync_directory`), the published and orphan
//! scans, and the UTC timestamp of `capacity_snapshot`.

use std::fs::{self, DirBuilder, File, FileTimes};
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rustix::fs::{FlockOperation, Mode, OFlags};
use rustix::io::Errno;

use crate::seam::{IdSource, RandomIds};

/// Every published entry file ends with this suffix (Python `_ENTRY_SUFFIX`).
pub(super) const ENTRY_SUFFIX: &str = ".json";

/// An exclusive `flock` on the queue's lock file, released on drop. Each
/// guard opens its own descriptor, so two threads of one process exclude
/// each other exactly as two processes do.
pub(super) struct QueueLock {
    file: File,
}

impl QueueLock {
    /// Python `_locked`: waits for `LOCK_EX`.
    pub(super) fn exclusive(path: &Path) -> io::Result<QueueLock> {
        let file = open_lock(path)?;
        loop {
            match rustix::fs::flock(&file, FlockOperation::LockExclusive) {
                Err(Errno::INTR) => continue,
                outcome => break outcome?,
            }
        }
        Ok(QueueLock { file })
    }

    /// Python `_try_locked`: `LOCK_EX | LOCK_NB`; `None` when another holder has it.
    pub(super) fn try_exclusive(path: &Path) -> io::Result<Option<QueueLock>> {
        let file = open_lock(path)?;
        match rustix::fs::flock(&file, FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => Ok(Some(QueueLock { file })),
            Err(Errno::AGAIN) => Ok(None),
            Err(errno) => Err(errno.into()),
        }
    }
}

impl Drop for QueueLock {
    fn drop(&mut self) {
        // Closing the descriptor releases the lock even if this unlock fails.
        let _ = rustix::fs::flock(&self.file, FlockOperation::Unlock);
    }
}

fn open_lock(path: &Path) -> io::Result<File> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::RDWR;
    Ok(File::from(rustix::fs::open(
        path,
        flags,
        Mode::RUSR | Mode::WUSR,
    )?))
}

/// Python `Path.touch(mode=0o600, exist_ok=True)` for the lock file: create
/// it when missing and set both of its times to now.
pub(super) fn touch(path: &Path) -> io::Result<()> {
    let now = SystemTime::now();
    open_lock(path)?.set_times(FileTimes::new().set_accessed(now).set_modified(now))
}

/// Python `mkdir(mode=0o700, parents=True, exist_ok=True)`, then a parent
/// fsync when this call created the queue directory.
pub(super) fn create_private_directory(directory: &Path) -> io::Result<()> {
    let created = !directory.try_exists()?;
    let parent = parent_of(directory);
    fs::create_dir_all(parent)?;
    match DirBuilder::new().mode(0o700).create(directory) {
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists && directory.is_dir() => {}
        outcome => outcome?,
    }
    if created {
        fsync_directory(parent)?;
    }
    Ok(())
}

/// Python `Path.parent`: `.` for a bare name.
pub(super) fn parent_of(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if parent.as_os_str().is_empty() => Path::new("."),
        Some(parent) => parent,
        None => path,
    }
}

/// Python `dead_letter_directory`: `<parent>/<name>-dead-letter`.
pub(super) fn dead_letter_directory(directory: &Path) -> PathBuf {
    let mut name = directory.file_name().unwrap_or_default().to_os_string();
    name.push("-dead-letter");
    directory.with_file_name(name)
}

/// Python `_fsync_directory`.
pub(super) fn fsync_directory(directory: &Path) -> io::Result<()> {
    let flags = OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::RDONLY;
    let descriptor = rustix::fs::open(directory, flags, Mode::empty())?;
    Ok(rustix::fs::fsync(&descriptor)?)
}

/// Python `_write_durable`: exclusive create 0o600, write, fsync.
fn write_durable(path: &Path, payload: &[u8]) -> io::Result<()> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY;
    let mut file = File::from(rustix::fs::open(path, flags, Mode::RUSR | Mode::WUSR)?);
    file.write_all(payload)?;
    Ok(rustix::fs::fsync(&file)?)
}

/// The admission publish step: dot-prefixed temp, durable write, rename onto
/// `target`, directory fsync; a temp left by a failed step is removed.
pub(super) fn publish(directory: &Path, target: &Path, payload: &[u8]) -> io::Result<()> {
    let token = RandomIds.uuid4()?.replace('-', "");
    let temporary = directory.join(format!(".{token}.tmp"));
    let published = write_durable(&temporary, payload)
        .and_then(|()| fs::rename(&temporary, target))
        .and_then(|()| fsync_directory(directory));
    if published.is_err() && temporary.try_exists()? {
        fs::remove_file(&temporary)?;
    }
    published
}

/// The bytes at `path`, or `None` when nothing is there.
pub(super) fn read_if_exists(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// Python `_published_paths` (`glob("*.json")`, dotfiles included), sorted.
pub(super) fn published_paths(directory: &Path) -> io::Result<Vec<PathBuf>> {
    names_where(directory, |name| name.ends_with(ENTRY_SUFFIX.as_bytes()))
}

/// Python `_scan_totals`: published count and summed sizes.
pub(super) fn scan_totals(directory: &Path) -> io::Result<(usize, u64)> {
    let paths = published_paths(directory)?;
    let mut bytes = 0_u64;
    for path in &paths {
        bytes = bytes.saturating_add(fs::metadata(path)?.len());
    }
    Ok((paths.len(), bytes))
}

/// Python `_remove_orphan_temps`: unlink every `.*.tmp`, then fsync once.
pub(super) fn remove_orphan_temps(directory: &Path) -> io::Result<()> {
    let orphans = names_where(directory, is_orphan_temp)?;
    for path in &orphans {
        fs::remove_file(path)?;
    }
    if !orphans.is_empty() {
        fsync_directory(directory)?;
    }
    Ok(())
}

/// The glob `.*.tmp`: a leading dot, anything, then `.tmp`.
fn is_orphan_temp(name: &[u8]) -> bool {
    name.len() >= 5 && name.starts_with(b".") && name.ends_with(b".tmp")
}

fn names_where(directory: &Path, keep: impl Fn(&[u8]) -> bool) -> io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if keep(entry.file_name().as_bytes()) {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

/// Count and bytes of the regular files in `directory` (symlinks followed,
/// dangling ones skipped), or zero when it is not a directory.
pub(super) fn retained_totals(directory: &Path) -> io::Result<(usize, u64)> {
    match fs::metadata(directory) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Ok((0, 0)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok((0, 0)),
        Err(error) => return Err(error),
    }
    let mut totals = (0_usize, 0_u64);
    for entry in fs::read_dir(directory)? {
        match fs::metadata(entry?.path()) {
            Ok(metadata) if metadata.is_file() => {
                totals = (totals.0 + 1, totals.1.saturating_add(metadata.len()));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(totals)
}

/// Python `datetime.fromtimestamp(mtime, UTC).isoformat(timespec="milliseconds")`
/// with `+00:00` written as `Z`; the float timestamp rounds to microseconds first.
pub(super) fn accepted_at(seconds: i64, nanoseconds: i64) -> String {
    let micros = (nanoseconds + 500) / 1000;
    let (seconds, micros) = if micros >= 1_000_000 {
        (seconds + 1, 0)
    } else {
        (seconds, micros)
    };
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let clock = seconds.rem_euclid(86_400);
    let (hour, minute, second) = (clock / 3600, clock % 3600 / 60, clock % 60);
    let millis = micros / 1000;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Days since 1970-01-01 to the proleptic Gregorian date (H. Hinnant).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}
