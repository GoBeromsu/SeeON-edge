use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;

use crate::clips::durable;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(super) enum PublishError {
    Conflict,
    Unreadable,
    Io(io::Error),
}

impl From<io::Error> for PublishError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Atomically publishes a file without replacing an existing directory entry.
pub(super) fn publish_once(
    target: &Path,
    payload: &[u8],
    mode: Mode,
    max_bytes: u64,
) -> Result<(), PublishError> {
    let parent = durable::parent_of(target);
    if reconcile(target, parent, payload, max_bytes)? {
        return Ok(());
    }

    for _ in 0..128 {
        let serial = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".sealed-sidecar-{}-{serial}.tmp",
            std::process::id()
        ));
        let flags =
            OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::WRONLY;
        let descriptor = match rustix::fs::open(&temporary, flags, mode) {
            Ok(descriptor) => descriptor,
            Err(Errno::EXIST) => continue,
            Err(error) => return Err(PublishError::Io(error.into())),
        };
        let mut file = File::from(descriptor);
        if let Err(error) = file.write_all(payload).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(PublishError::Io(error));
        }
        drop(file);

        match fs::hard_link(&temporary, target) {
            Ok(()) => {
                durable::fsync_dir(parent)?;
                fs::remove_file(&temporary)?;
                durable::fsync_dir(parent)?;
                return Ok(());
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                fs::remove_file(&temporary)?;
                durable::fsync_dir(parent)?;
                if reconcile(target, parent, payload, max_bytes)? {
                    return Ok(());
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(PublishError::Io(error));
            }
        }
    }

    Err(PublishError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique sealed-sidecar staging file",
    )))
}

fn reconcile(
    target: &Path,
    parent: &Path,
    expected: &[u8],
    max_bytes: u64,
) -> Result<bool, PublishError> {
    let Some((file, bytes)) = read_existing(target, max_bytes)? else {
        return Ok(false);
    };
    if bytes != expected {
        return Err(PublishError::Conflict);
    }
    file.sync_all()?;
    durable::fsync_dir(parent)?;
    Ok(true)
}

fn read_existing(target: &Path, max_bytes: u64) -> Result<Option<(File, Vec<u8>)>, PublishError> {
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let descriptor = match rustix::fs::open(target, flags, Mode::empty()) {
        Ok(descriptor) => descriptor,
        Err(Errno::NOENT) => return Ok(None),
        Err(Errno::LOOP) => return Err(PublishError::Unreadable),
        Err(error) => return Err(PublishError::Io(error.into())),
    };
    let mut file = File::from(descriptor);
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > max_bytes {
        return Err(PublishError::Unreadable);
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| PublishError::Unreadable)?;
    let mut bytes = Vec::with_capacity(capacity);
    let mut bounded = (&mut file).take(max_bytes.saturating_add(1));
    bounded.read_to_end(&mut bytes)?;
    if bytes.is_empty() || u64::try_from(bytes.len()).map_or(true, |length| length > max_bytes) {
        return Err(PublishError::Unreadable);
    }
    Ok(Some((file, bytes)))
}
