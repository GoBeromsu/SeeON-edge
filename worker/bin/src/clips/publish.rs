//! Terminal clip publication into the clip store: media first, then the
//! manifest, then the delivery entry, then the terminal marker. A missing
//! marker re-admits the stable entry, even if it was acknowledged meanwhile:
//! publication is at least once across the ACK-before-marker window.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;

use crate::delivery::{
    AdmissionFault, ClipEntry, DeliveryEntry, DeliveryQueue, EntryError, QueueError,
};
use crate::json::{Json, Serialiser};

use super::durable::{self, PUBLIC_FILE};
use super::entry::clip_entry;
use super::manifest::{ClipMetadata, ManifestError, MediaFacts, Terminal, manifest_bytes};
use super::store::{DIRECTORY_MODE, Reservation, exists};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const MEDIA_FILE: &str = "clip.mp4";
pub const TERMINAL_MARKER: &str = "terminal-outcome.json";
pub const CORRUPT: &str = "CORRUPT";

#[derive(Debug)]
pub enum PublishError {
    Io(io::Error),
    /// The reservation names an unsafe clip id, a blank camera, or another clip.
    Reservation(&'static str),
    /// A different or unreadable manifest or terminal marker occupies the clip.
    Conflict,
    /// A READY publication found neither the staged artifact nor the media.
    MissingMedia,
    Manifest(ManifestError),
    Entry(EntryError),
    Queue(QueueError),
    Refused(Option<AdmissionFault>),
}

impl From<io::Error> for PublishError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "clip publication failed: {self:?}")
    }
}

impl std::error::Error for PublishError {}

/// What one publication call produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Published {
    pub clip_id: String,
    pub manifest_path: PathBuf,
    pub manifest_bytes: Vec<u8>,
    pub video_path: Option<PathBuf>,
    pub entry: ClipEntry,
    /// The identical manifest was already published.
    pub resumed: bool,
    /// This call admitted the entry to the delivery queue.
    pub admitted: bool,
}

fn marker_bytes(entry: &ClipEntry, manifest: &[u8]) -> Result<Vec<u8>, PublishError> {
    let state = entry.fields().local_state.clone();
    let model = Json::Object(vec![
        (
            "clip_id".to_owned(),
            Json::Str(entry.fields().clip_id.clone()),
        ),
        (
            "entry_id".to_owned(),
            Json::Str(entry.entry_id().to_owned()),
        ),
        (
            "manifest_sha256".to_owned(),
            Json::Str(durable::sha256_hex(manifest)),
        ),
        ("local_state".to_owned(), Json::Str(state)),
    ]);
    let text = Serialiser::ModelSelection
        .canonical(&model)
        .map_err(|error| PublishError::Manifest(ManifestError::Json(error)))?;
    Ok(format!("{text}\n").into_bytes())
}

/// Publishes terminal clips into one store and announces them on one queue.
pub struct Publisher<'q> {
    queue: &'q DeliveryQueue,
}

impl<'q> Publisher<'q> {
    pub fn new(queue: &'q DeliveryQueue) -> Self {
        Self { queue }
    }

    pub fn publish_ready(
        &self,
        reservation: &Reservation,
        meta: &ClipMetadata,
        media: &MediaFacts,
    ) -> Result<Published, PublishError> {
        self.publish(reservation, meta, &Terminal::Ready(media.clone()))
    }

    pub fn publish_unavailable(
        &self,
        reservation: &Reservation,
        meta: &ClipMetadata,
        reason_code: &str,
        source_error_reason: Option<&str>,
    ) -> Result<Published, PublishError> {
        let terminal = Terminal::Unavailable {
            reason_code: reason_code.to_owned(),
            source_error_reason: source_error_reason.map(str::to_owned),
        };
        self.publish(reservation, meta, &terminal)
    }

    /// The recorder reported corruption and no media survives.
    pub fn publish_corrupt(
        &self,
        reservation: &Reservation,
        meta: &ClipMetadata,
    ) -> Result<Published, PublishError> {
        self.publish_unavailable(reservation, meta, CORRUPT, None)
    }

    /// The recorder reported corruption but the media exists: it is published
    /// READY with the measured facts; without media it is CORRUPT.
    pub fn publish_existing_corrupt(
        &self,
        reservation: &Reservation,
        meta: &ClipMetadata,
        media: &MediaFacts,
    ) -> Result<Published, PublishError> {
        if exists(&reservation.artifact_path()) || exists(&reservation.final_dir.join(MEDIA_FILE)) {
            self.publish_ready(reservation, meta, media)
        } else {
            self.publish_corrupt(reservation, meta)
        }
    }

    fn publish(
        &self,
        reservation: &Reservation,
        meta: &ClipMetadata,
        terminal: &Terminal,
    ) -> Result<Published, PublishError> {
        reservation.check(meta)?;
        let bytes = manifest_bytes(meta, terminal).map_err(PublishError::Manifest)?;
        let entry = clip_entry(meta, terminal).map_err(PublishError::Entry)?;
        let manifest_path = reservation.final_dir.join(MANIFEST_FILE);
        let resumed = confirm_existing(&manifest_path, &bytes)?;
        let marker = reservation.final_dir.join(TERMINAL_MARKER);
        let marker_payload = marker_bytes(&entry, &bytes)?;
        let marked = confirm_existing(&marker, &marker_payload)?;
        let ready = matches!(terminal, Terminal::Ready(_));
        let video_path = reservation.final_dir.join(MEDIA_FILE);
        if !resumed {
            durable::create_dir(&reservation.final_dir, DIRECTORY_MODE)?;
            if ready {
                place_media(&reservation.artifact_path(), &video_path)?;
            }
            durable::write_durable(&manifest_path, &bytes, PUBLIC_FILE)?;
        }
        let admitted = !(resumed && marked);
        if admitted {
            let result = self
                .queue
                .try_admit(&DeliveryEntry::from(entry.clone()))
                .map_err(PublishError::Queue)?;
            if !result.accepted {
                return Err(PublishError::Refused(result.fault));
            }
            if !marked {
                durable::write_durable(&marker, &marker_payload, PUBLIC_FILE)?;
            }
        }
        if !ready {
            durable::remove_durable(&video_path)?;
        }
        durable::remove_tree(&reservation.staging_dir)?;
        Ok(Published {
            clip_id: reservation.clip_id.clone(),
            manifest_path,
            manifest_bytes: bytes,
            video_path: ready.then_some(video_path),
            entry,
            resumed,
            admitted,
        })
    }
}

/// Validates bounded bytes and fsyncs the same open file, then its parent.
/// Existing contradictory files are never replaced.
fn confirm_existing(path: &Path, expected: &[u8]) -> Result<bool, PublishError> {
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let descriptor = match rustix::fs::open(path, flags, Mode::empty()) {
        Ok(descriptor) => descriptor,
        Err(Errno::NOENT) => return Ok(false),
        Err(Errno::LOOP) => return Err(PublishError::Conflict),
        Err(errno) => return Err(PublishError::Io(errno.into())),
    };
    let mut file = File::from(descriptor);
    let metadata = file.metadata()?;
    let bound = expected.len() as u64;
    if !metadata.is_file() || metadata.len() != bound {
        return Err(PublishError::Conflict);
    }
    let mut existing = Vec::new();
    file.by_ref()
        .take(bound.saturating_add(1))
        .read_to_end(&mut existing)?;
    if existing != expected {
        return Err(PublishError::Conflict);
    }
    file.sync_all()?;
    durable::fsync_dir(durable::parent_of(path))?;
    Ok(true)
}

/// Moves the staged artifact to the final media path; a rerun after the move
/// finds the media already in place.
fn place_media(artifact: &Path, video: &Path) -> Result<(), PublishError> {
    if exists(artifact) {
        durable::move_durable(artifact, video, PUBLIC_FILE)?;
        Ok(())
    } else if exists(video) {
        Ok(())
    } else {
        Err(PublishError::MissingMedia)
    }
}
