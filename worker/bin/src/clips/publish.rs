//! Terminal clip publication into the clip store: media first, then the
//! manifest, then the delivery entry, then the terminal marker. Every write
//! is durable (dot-temp, fsync, rename, parent fsync) and a rerun of the
//! same publication is a resume, never a second announcement.

use std::io;
use std::path::{Path, PathBuf};

use crate::delivery::{
    AdmissionFault, ClipEntry, DeliveryEntry, DeliveryQueue, EntryError, QueueError,
};
use crate::json::{Json, Serialiser};

use super::durable::{self, Existing, PUBLIC_FILE};
use super::entry::clip_entry;
use super::manifest::{
    ClipMetadata, MAX_MANIFEST_BYTES, ManifestError, MediaFacts, Terminal, manifest_bytes,
};
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
    /// A different or unreadable manifest already occupies the clip.
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
        let resumed = match durable::read_bounded(&manifest_path, MAX_MANIFEST_BYTES as u64)? {
            Existing::Missing => false,
            Existing::Bytes(existing) if existing == bytes => true,
            Existing::Bytes(_) | Existing::Unreadable => return Err(PublishError::Conflict),
        };
        let ready = matches!(terminal, Terminal::Ready(_));
        let video_path = reservation.final_dir.join(MEDIA_FILE);
        if !resumed {
            durable::create_dir(&reservation.final_dir, DIRECTORY_MODE)?;
            if ready {
                place_media(&reservation.artifact_path(), &video_path)?;
            }
            durable::write_durable(&manifest_path, &bytes, PUBLIC_FILE)?;
        }
        let marker = reservation.final_dir.join(TERMINAL_MARKER);
        let admitted = !(resumed && exists(&marker));
        if admitted {
            let result = self
                .queue
                .try_admit(&DeliveryEntry::from(entry.clone()))
                .map_err(PublishError::Queue)?;
            if !result.accepted {
                return Err(PublishError::Refused(result.fault));
            }
            durable::write_durable(&marker, &marker_bytes(&entry, &bytes)?, PUBLIC_FILE)?;
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
