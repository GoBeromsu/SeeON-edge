//! `worker/pipeline/trace/replay_trace_writer.py`: one opt-in JSONL file per
//! camera under the replay-trace root, `<sha256(camera_id)[:16]>.jsonl`,
//! header line first, bounded to `max_bytes` with `rotation_count` rotated
//! files kept as `.jsonl.1` (newest) to `.jsonl.N` (oldest). A row that
//! cannot fit a fresh file is refused (`Ok(false)`) and counted; every path
//! is resolved and must stay inside the root.

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::config::env::{Env, replay_trace_dir};
use crate::records::id::sha256_hex;
use crate::trace_out::row::{HEADER_LINE, ReplayRow, RowError};

/// `ReplayTraceWriter.max_bytes` default: 50 MiB.
pub const DEFAULT_MAX_BYTES: u64 = 50 * 1024 * 1024;
/// `ReplayTraceWriter.rotation_count` default.
pub const DEFAULT_ROTATION_COUNT: u32 = 3;

/// Static refusals; I/O failures keep only their kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceError {
    /// `max_bytes` must be positive.
    Bounds,
    /// A trace path resolves outside the root.
    Escape,
    Row(RowError),
    Io(io::ErrorKind),
}

impl fmt::Display for TraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Bounds => "replay trace bounds must be positive",
            Self::Escape => "replay trace path escaped its root",
            Self::Row(_) => "replay trace row is invalid",
            Self::Io(_) => "replay trace file operation failed",
        })
    }
}
impl std::error::Error for TraceError {}

impl From<io::Error> for TraceError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

impl From<RowError> for TraceError {
    fn from(error: RowError) -> Self {
        Self::Row(error)
    }
}

#[derive(Debug)]
pub struct ReplayTraceWriter {
    root: PathBuf,
    path: PathBuf,
    max_bytes: u64,
    rotation_count: u32,
    written_rows_total: u64,
    dropped_rows_total: u64,
}

impl ReplayTraceWriter {
    /// Creates the root (resolved once, as Python's `directory.resolve()`)
    /// and checks the camera file path stays inside it.
    pub fn new(
        directory: &Path,
        camera_id: &str,
        max_bytes: u64,
        rotation_count: u32,
    ) -> Result<Self, TraceError> {
        if max_bytes == 0 {
            return Err(TraceError::Bounds);
        }
        fs::create_dir_all(directory)?;
        let root = fs::canonicalize(directory)?;
        let name = format!("{}.jsonl", &sha256_hex(camera_id.as_bytes())[..16]);
        let path = within_root(&root, &root.join(name))?;
        Ok(Self {
            root,
            path,
            max_bytes,
            rotation_count,
            written_rows_total: 0,
            dropped_rows_total: 0,
        })
    }

    /// The opt-in constructor: a blank or unset `WORKER_REPLAY_TRACE_DIR`
    /// disables the trace (`Ok(None)`); otherwise the default bounds apply.
    pub fn from_env(env: &Env, camera_id: &str) -> Result<Option<Self>, TraceError> {
        replay_trace_dir(env)
            .map(|dir| Self::new(&dir, camera_id, DEFAULT_MAX_BYTES, DEFAULT_ROTATION_COUNT))
            .transpose()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn written_rows_total(&self) -> u64 {
        self.written_rows_total
    }

    pub fn dropped_rows_total(&self) -> u64 {
        self.dropped_rows_total
    }

    /// `append`: `Ok(true)` when the row line was written, `Ok(false)` when
    /// it cannot fit even a fresh file. Rotates first when the current file
    /// would exceed `max_bytes`.
    pub fn append(&mut self, row: &ReplayRow) -> Result<bool, TraceError> {
        let addition = row.encode_line()?;
        fs::create_dir_all(&self.root)?;
        let path = within_root(&self.root, &self.path)?;
        let mut existing = match fs::metadata(&path) {
            Ok(metadata) => metadata.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        let header = HEADER_LINE.len() as u64;
        let addition_len = addition.len() as u64;
        let base = if existing == 0 { header } else { existing };
        if base + addition_len > self.max_bytes {
            self.rotate()?;
            if header + addition_len > self.max_bytes {
                self.dropped_rows_total += 1;
                return Ok(false);
            }
            existing = 0;
        }
        let mut buffer = String::with_capacity(HEADER_LINE.len() + addition.len());
        if existing == 0 {
            buffer.push_str(HEADER_LINE);
        }
        buffer.push_str(&addition);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(buffer.as_bytes())?;
        self.written_rows_total += 1;
        Ok(true)
    }

    /// `_rotate`: with no rotated files the current file is removed;
    /// otherwise the oldest is dropped and each file shifts one index up.
    fn rotate(&self) -> Result<(), TraceError> {
        if self.rotation_count == 0 {
            return remove_if_present(&within_root(&self.root, &self.path)?);
        }
        remove_if_present(&self.rotated(self.rotation_count))?;
        for index in (1..self.rotation_count).rev() {
            let source = within_root(&self.root, &self.rotated(index))?;
            if source.exists() {
                let target = within_root(&self.root, &self.rotated(index + 1))?;
                fs::rename(source, target)?;
            }
        }
        let current = within_root(&self.root, &self.path)?;
        if current.exists() {
            fs::rename(current, within_root(&self.root, &self.rotated(1))?)?;
        }
        Ok(())
    }

    /// `<name>.jsonl.<index>` next to the current file.
    fn rotated(&self, index: u32) -> PathBuf {
        let mut name = self.path.as_os_str().to_owned();
        name.push(format!(".{index}"));
        PathBuf::from(name)
    }
}

fn remove_if_present(path: &Path) -> Result<(), TraceError> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error.into()),
        _ => Ok(()),
    }
}

/// `_within_root`: the fully resolved path must stay under the root. A
/// missing file resolves through its parent; a dangling symlink is refused.
fn within_root(root: &Path, path: &Path) -> Result<PathBuf, TraceError> {
    let resolved = match fs::canonicalize(path) {
        Ok(resolved) => resolved,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if fs::symlink_metadata(path).is_ok() {
                return Err(TraceError::Escape);
            }
            let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
                return Err(TraceError::Escape);
            };
            fs::canonicalize(parent)?.join(name)
        }
        Err(error) => return Err(error.into()),
    };
    if resolved.starts_with(root) {
        Ok(resolved)
    } else {
        Err(TraceError::Escape)
    }
}
