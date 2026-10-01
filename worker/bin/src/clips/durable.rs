//! Crash-safe file operations for the clip store and the sealed sidecars:
//! dot-temp, fsync, rename, parent-directory fsync.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, DirBuilder, File};
use std::io::{self, Read, Write};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;
use sha2::{Digest, Sha256};

/// `0644`: clip-store files the backend serves.
pub const PUBLIC_FILE: Mode = Mode::RUSR
    .union(Mode::WUSR)
    .union(Mode::RGRP)
    .union(Mode::ROTH);
/// `0600`: worker-private files.
pub const PRIVATE_FILE: Mode = Mode::RUSR.union(Mode::WUSR);

/// What a bounded read found at a path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Existing {
    Missing,
    Bytes(Vec<u8>),
    /// Not a regular file, a symlink, empty, or over the bound.
    Unreadable,
}

/// The directory holding `path`; `.` for a bare name.
pub fn parent_of(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

pub fn fsync_dir(directory: &Path) -> io::Result<()> {
    let flags = OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::RDONLY;
    let descriptor = rustix::fs::open(directory, flags, Mode::empty())?;
    rustix::fs::fsync(&descriptor)?;
    Ok(())
}

/// `.<name>.tmp` beside `target`.
pub fn temp_path(target: &Path) -> io::Result<PathBuf> {
    let name = target
        .file_name()
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    let mut temp = OsString::from(".");
    temp.push(name);
    temp.push(".tmp");
    Ok(parent_of(target).join(temp))
}

/// Publishes `payload` at `target`: a stale dot-temp is removed, the new
/// one is created exclusively, written, fsynced, renamed, and the parent
/// directory fsynced. A failure leaves no dot-temp behind.
pub fn write_durable(target: &Path, payload: &[u8], mode: Mode) -> io::Result<()> {
    let temp = temp_path(target)?;
    remove_if_present(&temp)?;
    let result = write_new(&temp, payload, mode)
        .and_then(|()| fs::rename(&temp, target))
        .and_then(|()| fsync_dir(parent_of(target)));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn write_new(path: &Path, payload: &[u8], mode: Mode) -> io::Result<()> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::NOFOLLOW;
    let mut file = File::from(rustix::fs::open(path, flags, mode)?);
    file.write_all(payload)?;
    file.sync_all()
}

/// Moves `source` to `target` durably. Across filesystems the bytes are
/// copied through a dot-temp first; both parents are fsynced.
pub fn move_durable(source: &Path, target: &Path, mode: Mode) -> io::Result<()> {
    match fs::rename(source, target) {
        Ok(()) => {}
        Err(error) if error.raw_os_error() == Some(Errno::XDEV.raw_os_error()) => {
            let temp = temp_path(target)?;
            remove_if_present(&temp)?;
            let copied = copy_new(source, &temp, mode).and_then(|()| fs::rename(&temp, target));
            if copied.is_err() {
                let _ = fs::remove_file(&temp);
            }
            copied?;
            fsync_dir(parent_of(target))?;
            fs::remove_file(source)?;
        }
        Err(error) => return Err(error),
    }
    fsync_dir(parent_of(target))?;
    fsync_dir(parent_of(source))
}

fn copy_new(source: &Path, temp: &Path, mode: Mode) -> io::Result<()> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::WRONLY | OFlags::NOFOLLOW;
    let mut output = File::from(rustix::fs::open(temp, flags, mode)?);
    io::copy(&mut File::open(source)?, &mut output)?;
    output.sync_all()
}

/// Creates one directory level with `mode`; returns whether it was created.
/// A new directory's parent is fsynced.
pub fn create_dir(path: &Path, mode: u32) -> io::Result<bool> {
    match DirBuilder::new().mode(mode).create(path) {
        Ok(()) => {
            fsync_dir(parent_of(path))?;
            Ok(true)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists && path.is_dir() => Ok(false),
        Err(error) => Err(error),
    }
}

/// Removes a directory tree if present and fsyncs its parent.
pub fn remove_tree(path: &Path) -> io::Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => fsync_dir(parent_of(path)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Removes a file if present and fsyncs its parent.
pub fn remove_durable(path: &Path) -> io::Result<()> {
    if remove_if_present(path)? {
        fsync_dir(parent_of(path))?;
    }
    Ok(())
}

fn remove_if_present(path: &Path) -> io::Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

/// Reads a regular, non-empty file of at most `max` bytes without
/// following a final symlink.
pub fn read_bounded(path: &Path, max: u64) -> io::Result<Existing> {
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let descriptor = match rustix::fs::open(path, flags, Mode::empty()) {
        Ok(descriptor) => descriptor,
        Err(errno) if errno == Errno::NOENT => return Ok(Existing::Missing),
        Err(errno) if errno == Errno::LOOP => return Ok(Existing::Unreadable),
        Err(errno) => return Err(errno.into()),
    };
    let file = File::from(descriptor);
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > max {
        return Ok(Existing::Unreadable);
    }
    let mut bytes = Vec::new();
    file.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_or(true, |length| length > max) {
        return Ok(Existing::Unreadable);
    }
    Ok(Existing::Bytes(bytes))
}

/// Streams a file into its lowercase SHA-256 hex digest and byte size.
pub fn sha256_file(path: &Path) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut size = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        size += u64::try_from(read).unwrap_or(u64::MAX);
    }
    Ok((hex(&hasher.finalize()), size))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}
