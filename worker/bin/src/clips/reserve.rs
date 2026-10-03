//! Saving sealed recordings, and the space held back so a full clip store
//! can still record why a clip is missing (B13 option 2).
//!
//! `<root>/.reserve` holds preallocated slot files. When a save runs out of
//! space one slot is released and the clip is published UNAVAILABLE with
//! `FINALIZE_FAILED` into the freed space. Media is retained until terminal
//! publication completes. Its `ClipEntry` waits in the delivery queue on the
//! state volume.

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

use rustix::fs::{FallocateFlags, OFlags};
use rustix::io::Errno;
use seeon_deepstream_native::MediaResult;

use super::durable::{self, PRIVATE_FILE, PUBLIC_FILE};
use super::manifest::{ClipMetadata, MediaFacts};
use super::publish::{MANIFEST_FILE, PublishError, Published, Publisher};
use super::recorder::ClipSealed;
use super::store::{ClipStore, DIRECTORY_MODE as STORE_MODE, Reservation, exists};

pub const RESERVE_DIR: &str = ".reserve";
/// Covers a manifest at `MAX_MANIFEST_BYTES`, its temp, the marker and the
/// directory pages, with room to spare.
pub const SLOT_BYTES: u64 = 256 * 1024;
pub const FINALIZE_FAILED: &str = "FINALIZE_FAILED";
pub const ENCODER_FAILED: &str = "ENCODER_FAILED";
pub const NO_FRAMES: &str = "NO_FRAMES";
const RESERVE_MODE: u32 = 0o700;

/// A save that returned a publication or recorded why it could not.
#[derive(Debug)]
pub enum SaveOutcome {
    /// READY, or UNAVAILABLE as the media plane reported.
    Saved(Published),
    /// The store ran out of space for the clip. `Some` holds the completed
    /// UNAVAILABLE `FINALIZE_FAILED` publication. `None` has no completed
    /// result; a partial terminal commit may exist and must be reconciled.
    FinalizeFailed(Option<Published>),
}

/// Out of space, as the store reports it (`ENOSPC` or `EDQUOT`).
pub fn no_space(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::StorageFull
        || error.raw_os_error() == Some(Errno::NOSPC.raw_os_error())
        || error.raw_os_error() == Some(Errno::DQUOT.raw_os_error())
}

pub struct ReservePool {
    directory: PathBuf,
    target: usize,
    slots: Vec<PathBuf>,
}

impl ReservePool {
    /// Two slots per camera and one spare.
    pub fn slots_for(cameras: usize) -> usize {
        cameras.saturating_mul(2).saturating_add(1)
    }

    /// Creates the reserve for `cameras` cameras; a store already too full
    /// leaves the pool short rather than failing.
    pub fn arm(store: &ClipStore, cameras: usize) -> io::Result<Self> {
        durable::create_dir(store.root(), STORE_MODE)?;
        let directory = store.root().join(RESERVE_DIR);
        durable::create_dir(&directory, RESERVE_MODE)?;
        let mut pool = Self {
            directory,
            target: Self::slots_for(cameras),
            slots: Vec::new(),
        };
        pool.rearm()?;
        Ok(pool)
    }

    pub fn target(&self) -> usize {
        self.target
    }

    pub fn available(&self) -> usize {
        self.slots.len()
    }

    /// Restores released slots once space is free again; returns how many
    /// slots are held.
    pub fn rearm(&mut self) -> io::Result<usize> {
        self.slots.clear();
        for index in 0..self.target {
            let path = self.directory.join(format!("slot-{index:03}"));
            match fill_slot(&path) {
                Ok(()) => self.slots.push(path),
                Err(error) if no_space(&error) => {
                    durable::remove_durable(&path)?;
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(self.slots.len())
    }

    /// Releases one slot's space; false when none is left.
    fn release(&mut self) -> io::Result<bool> {
        let Some(path) = self.slots.pop() else {
            return Ok(false);
        };
        durable::remove_durable(&path)?;
        Ok(true)
    }

    /// Saves a sealed recording. Media the plane reported as failed or empty
    /// is published UNAVAILABLE. Running out of space while writing the clip
    /// releases a slot and publishes `FINALIZE_FAILED` in its place; any other
    /// I/O failure is returned unchanged.
    pub fn save(
        &mut self,
        store: &ClipStore,
        publisher: &Publisher<'_>,
        meta: &ClipMetadata,
        sealed: &ClipSealed,
        codec: &str,
    ) -> Result<SaveOutcome, PublishError> {
        let reservation = match self.reserve(store, meta) {
            Err(PublishError::Io(error)) if no_space(&error) => {
                return Ok(SaveOutcome::FinalizeFailed(None));
            }
            other => other?,
        };
        let attempt = if sealed.result != MediaResult::Ok {
            publisher.publish_unavailable(&reservation, meta, ENCODER_FAILED, None)
        } else if !sealed.contains_video {
            publisher.publish_unavailable(&reservation, meta, NO_FRAMES, None)
        } else {
            place_and_publish(publisher, &reservation, meta, sealed, codec)
        };
        match attempt {
            Ok(published) => Ok(SaveOutcome::Saved(published)),
            Err(PublishError::Io(error)) if no_space(&error) => {
                self.finalize_failed(publisher, &reservation, meta, error)
            }
            Err(other) => Err(other),
        }
    }

    /// Reserves the clip directories, releasing one slot if they do not fit.
    fn reserve(
        &mut self,
        store: &ClipStore,
        meta: &ClipMetadata,
    ) -> Result<Reservation, PublishError> {
        match store.reserve(&meta.camera_id, &meta.clip_id) {
            Err(PublishError::Io(error)) if no_space(&error) && self.release()? => {
                store.reserve(&meta.camera_id, &meta.clip_id)
            }
            other => other,
        }
    }

    fn finalize_failed(
        &mut self,
        publisher: &Publisher<'_>,
        reservation: &Reservation,
        meta: &ClipMetadata,
        error: io::Error,
    ) -> Result<SaveOutcome, PublishError> {
        if exists(&reservation.final_dir.join(MANIFEST_FILE)) {
            // A manifest may belong to an incomplete publication. Preserve it
            // and its media for an identical resume, not a different outcome.
            return Err(PublishError::Io(error));
        }
        self.release()?;
        match publisher.publish_unavailable(reservation, meta, FINALIZE_FAILED, None) {
            Ok(published) => Ok(SaveOutcome::FinalizeFailed(Some(published))),
            Err(PublishError::Io(again)) if no_space(&again) => {
                Ok(SaveOutcome::FinalizeFailed(None))
            }
            Err(other) => Err(other),
        }
    }
}

fn place_and_publish(
    publisher: &Publisher<'_>,
    reservation: &Reservation,
    meta: &ClipMetadata,
    sealed: &ClipSealed,
    codec: &str,
) -> Result<Published, PublishError> {
    let artifact = reservation.artifact_path();
    durable::move_durable(&sealed.path, &artifact, PUBLIC_FILE)?;
    let (sha256, size) = durable::sha256_file(&artifact)?;
    let facts = MediaFacts {
        sha256,
        size_bytes: i64::try_from(size).map_err(|_| PublishError::MissingMedia)?,
        codec: codec.to_owned(),
        duration_ms: i64::try_from(sealed.duration_ms).unwrap_or(i64::MAX),
    };
    publisher.publish_ready(reservation, meta, &facts)
}

/// Creates or keeps one slot file holding `SLOT_BYTES` of allocated space.
fn fill_slot(path: &std::path::Path) -> io::Result<()> {
    let flags = OFlags::CLOEXEC | OFlags::CREATE | OFlags::WRONLY | OFlags::NOFOLLOW;
    let file = File::from(rustix::fs::open(path, flags, PRIVATE_FILE)?);
    match rustix::fs::fallocate(&file, FallocateFlags::empty(), 0, SLOT_BYTES) {
        Ok(()) => {}
        Err(errno) if errno == Errno::OPNOTSUPP => write_zeros(&file)?,
        Err(errno) => return Err(errno.into()),
    }
    file.sync_all()
}

fn write_zeros(mut file: &File) -> io::Result<()> {
    let block = [0_u8; 64 * 1024];
    let mut written = 0_u64;
    while written < SLOT_BYTES {
        file.write_all(&block)?;
        written += block.len() as u64;
    }
    Ok(())
}
