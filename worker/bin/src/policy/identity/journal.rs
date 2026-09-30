//! Journal file access: `_load` over universal newlines, and `_atomic_replace`
//! with `_fsync_directory` (`event_identity.py` L119-L208).

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use super::IdentityError;
use super::line::{self, Record};

/// Name attempts before a create refusal, like `mkstemp`'s bounded retries.
const TEMPORARY_ATTEMPTS: u32 = 64;

static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Records by source key; a later line for the same key and id wins.
pub(super) fn load(
    path: Option<&Path>,
    now: f64,
) -> Result<HashMap<String, Record>, IdentityError> {
    let mut loaded = HashMap::new();
    let Some(path) = path else {
        return Ok(loaded);
    };
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if is_absent(&error) => return Ok(loaded),
        Err(error) => return Err(error.into()),
    };
    // Python text mode reads "\r\n", "\r" and "\n" as one line end each.
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    for (index, raw) in text.split('\n').enumerate() {
        let number = index + 1;
        if line::strip(raw).is_empty() {
            continue;
        }
        let record = Record::parse(raw, now).ok_or(IdentityError::Malformed { line: number })?;
        let conflicting = loaded
            .get(&record.source_key)
            .is_some_and(|previous: &Record| previous.edge_event_id != record.edge_event_id);
        if conflicting {
            return Err(IdentityError::ConflictingSourceKey { line: number });
        }
        loaded.insert(record.source_key.clone(), record);
    }
    Ok(loaded)
}

/// `Path.exists()` is false for these; anything else is a refusal.
fn is_absent(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
    )
}

/// Writes `payload` to a 0600 sibling temporary, syncs it, renames it over
/// `path` and syncs the directory. The temporary never outlives the call.
pub(super) fn replace(path: &Path, payload: &str) -> Result<(), IdentityError> {
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    fs::create_dir_all(parent)?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let (temporary, mut file) = create_temporary(parent, &name)?;
    let written = write_and_rename(&mut file, payload, &temporary, path, parent);
    drop(file);
    if written.is_err() && temporary.exists() {
        // The write error is the refusal; a failed cleanup cannot improve it.
        let _ = fs::remove_file(&temporary);
    }
    written.map_err(IdentityError::from)
}

fn create_temporary(parent: &Path, name: &str) -> io::Result<(PathBuf, File)> {
    let mut last = io::Error::from(io::ErrorKind::AlreadyExists);
    for _ in 0..TEMPORARY_ATTEMPTS {
        let serial = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{name}.{}-{serial}.tmp", process::id()));
        let opened = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&candidate);
        match opened {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => last = error,
            Err(error) => return Err(error),
        }
    }
    Err(last)
}

fn write_and_rename(
    file: &mut File,
    payload: &str,
    temporary: &Path,
    path: &Path,
    parent: &Path,
) -> io::Result<()> {
    file.write_all(payload.as_bytes())?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    File::open(parent)?.sync_all()
}
