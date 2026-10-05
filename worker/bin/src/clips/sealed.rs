//! Flow sealed-clip sidecars (`flow_sealed_sidecar.py`): a sealed Flow clip
//! is written to `flow-sealed/` before it is published and retired after
//! publication succeeds; ready clips may also retire after confirmed absence.

mod immutable;
mod observation;
mod payload;
mod read;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::durable::{self, PUBLIC_FILE};
use crate::json::Serialiser;

pub use observation::{SealedObservation, SealedUnavailable};

const MAX_SIDECAR_BYTES: u64 = 1 << 20;

/// The sidecar directory name under the worker state directory.
pub const SIDECAR_DIR: &str = "flow-sealed";
/// The sidecar directory is private to the worker.
pub const SIDECAR_DIR_MODE: u32 = 0o700;

/// One contributor of a sealed clip; `detected_at` is kept as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedContributor {
    pub event_ref: String,
    pub detected_at: String,
}

/// `RecordingSealed`: the closed Flow recording awaiting publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedClip {
    pub clip_id: String,
    pub path: String,
    pub duration_ms: i64,
    pub boundary: String,
    pub contributors: Vec<SealedContributor>,
}

/// The event fields a sidecar carries (`_event_payload`).
#[derive(Clone, Debug, PartialEq)]
pub struct SealedEvent {
    pub domain: String,
    pub event_type: String,
    pub identity: String,
    pub camera_id: String,
    pub facility_id: String,
    pub time_sec: f64,
    /// Absent model probability is JSON null; never a fabricated number.
    pub probability: Option<f64>,
}

/// A persisted sidecar read back for replay.
#[derive(Clone, Debug, PartialEq)]
pub struct Recovery {
    pub sealed: SealedObservation,
    /// Keyed by event identity, which is the contributor's `event_ref`.
    pub events: BTreeMap<String, SealedEvent>,
    pub camera_id: String,
    pub sidecar_path: PathBuf,
}

/// Sidecars for one camera in file-name order, plus every sidecar that
/// could not be read. Malformed sidecars stay on disk (B12).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pending {
    pub recoveries: Vec<Recovery>,
    pub malformed: Vec<PathBuf>,
}

#[derive(Debug)]
pub enum SealedError {
    /// A contributor has no event; nothing is written.
    UnknownRef,
    /// The clip id cannot name a file inside the sidecar directory.
    InvalidClipId,
    /// A probability or time is NaN or infinite.
    NonFinite,
    /// Event references, identities, or camera attribution contradict.
    InvalidAttribution,
    /// An unavailable observation is not a valid native-media result.
    InvalidObservation,
    /// The canonical observation exceeds the bounded sidecar size.
    PayloadTooLarge,
    /// A sidecar already exists with different or malformed bytes.
    ImmutableConflict,
    /// The existing sidecar is not a bounded regular non-symlink file.
    ImmutableUnreadable,
    /// Missing-media retirement cannot discard unavailable evidence.
    NegativeMissingMedia,
    /// A sidecar is not a readable sealed record.
    Malformed,
    Io(io::Error),
}

impl From<io::Error> for SealedError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl std::fmt::Display for SealedError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "flow sealed sidecar refused: {self:?}")
    }
}

impl std::error::Error for SealedError {}

/// What replaying one sidecar did.
#[derive(Debug, PartialEq)]
pub enum ReplayOutcome<T, E> {
    /// Publication succeeded (including a resumed terminal); retire the sidecar.
    Published(T),
    /// The caller confirmed absence after checking the terminal and all owned
    /// media locations. Unavailable observations reject this outcome and stay
    /// persisted; a failed or malformed terminal must be `Failed`.
    MissingMedia,
    /// Publication failed; the sidecar is kept for the next replay.
    Failed(E),
}

/// Counts for one camera's replay pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayReport {
    pub published: usize,
    pub missing_media: usize,
    pub failed: usize,
    pub malformed: usize,
}

/// `FlowSealedSidecars` over one directory.
#[derive(Clone, Debug)]
pub struct SealedSidecars {
    directory: PathBuf,
}

impl SealedSidecars {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Publishes `<clip_id>.json` once. Identical observations are idempotent;
    /// contradictory existing evidence is preserved and refused.
    pub fn persist(
        &self,
        observation: &SealedObservation,
        events: &BTreeMap<String, SealedEvent>,
    ) -> Result<PathBuf, SealedError> {
        let target = self.sidecar_path(observation.clip_id())?;
        let mut ordered: Vec<&SealedContributor> = observation.contributors().iter().collect();
        ordered.sort_by(|left, right| left.detected_at.cmp(&right.detected_at));
        let written = ordered
            .iter()
            .map(|contributor| events.get(&contributor.event_ref))
            .collect::<Option<Vec<&SealedEvent>>>()
            .ok_or(SealedError::UnknownRef)?;
        let camera_id = validate_attribution(&ordered, &written)?;
        let payload = match observation {
            SealedObservation::Ready(sealed) => payload::sidecar_json(sealed, &ordered, &written),
            SealedObservation::Unavailable(unavailable) => {
                if unavailable
                    .path
                    .as_ref()
                    .is_some_and(|path| path.is_empty())
                    || (unavailable.native_result == 0 && unavailable.contains_video)
                    || unavailable.contributors.is_empty()
                    || camera_id.trim().is_empty()
                {
                    return Err(SealedError::InvalidObservation);
                }
                payload::unavailable_sidecar_json(unavailable, camera_id, &ordered, &written)
            }
        };
        let text = Serialiser::ModelSelection
            .canonical(&payload)
            .map_err(|_| SealedError::NonFinite)?;
        if u64::try_from(text.len()).map_or(true, |length| length > MAX_SIDECAR_BYTES) {
            return Err(SealedError::PayloadTooLarge);
        }
        fs::create_dir_all(durable::parent_of(&self.directory))?;
        durable::create_dir(&self.directory, SIDECAR_DIR_MODE)?;
        immutable::publish_once(&target, text.as_bytes(), PUBLIC_FILE, MAX_SIDECAR_BYTES).map_err(
            |error| match error {
                immutable::PublishError::Conflict => SealedError::ImmutableConflict,
                immutable::PublishError::Unreadable => SealedError::ImmutableUnreadable,
                immutable::PublishError::Io(error) => SealedError::Io(error),
            },
        )?;
        Ok(target)
    }

    /// `pending_for_camera`: this camera's sidecars in file-name order.
    pub fn pending(&self, camera_id: &str) -> Result<Pending, SealedError> {
        read::pending(&self.directory, camera_id)
    }

    /// Always calls the publication owner, which checks terminal state and all
    /// owned media locations. `Published` retires either observation;
    /// `MissingMedia` retires only a ready observation.
    pub fn replay_one<T, E>(
        &self,
        recovery: &Recovery,
        publish: impl FnOnce(&Recovery) -> ReplayOutcome<T, E>,
    ) -> Result<ReplayOutcome<T, E>, SealedError> {
        match publish(recovery) {
            ReplayOutcome::Published(published) => {
                self.remove(&recovery.sidecar_path)?;
                Ok(ReplayOutcome::Published(published))
            }
            ReplayOutcome::MissingMedia => {
                if matches!(&recovery.sealed, SealedObservation::Unavailable(_)) {
                    return Err(SealedError::NegativeMissingMedia);
                }
                self.remove(&recovery.sidecar_path)?;
                Ok(ReplayOutcome::MissingMedia)
            }
            ReplayOutcome::Failed(error) => Ok(ReplayOutcome::Failed(error)),
        }
    }

    /// Replays every pending sidecar of `camera_id`; a failure continues
    /// with the next one.
    pub fn replay<T, E>(
        &self,
        camera_id: &str,
        mut publish: impl FnMut(&Recovery) -> ReplayOutcome<T, E>,
    ) -> Result<ReplayReport, SealedError> {
        let pending = self.pending(camera_id)?;
        let mut report = ReplayReport {
            malformed: pending.malformed.len(),
            ..ReplayReport::default()
        };
        for recovery in &pending.recoveries {
            match self.replay_one(recovery, &mut publish)? {
                ReplayOutcome::Published(_) => report.published += 1,
                ReplayOutcome::MissingMedia => report.missing_media += 1,
                ReplayOutcome::Failed(_) => report.failed += 1,
            }
        }
        Ok(report)
    }

    /// Removes a sidecar if present and fsyncs the directory.
    pub fn remove(&self, sidecar_path: &Path) -> Result<(), SealedError> {
        durable::remove_durable(sidecar_path)?;
        Ok(())
    }

    fn sidecar_path(&self, clip_id: &str) -> Result<PathBuf, SealedError> {
        if !observation::valid_clip_id(clip_id) {
            return Err(SealedError::InvalidClipId);
        }
        Ok(self.directory.join(format!("{clip_id}.json")))
    }
}

fn validate_attribution<'a>(
    ordered: &[&'a SealedContributor],
    events: &[&'a SealedEvent],
) -> Result<&'a str, SealedError> {
    let mut identities = BTreeSet::new();
    let mut camera_id = None;
    for (contributor, event) in ordered.iter().zip(events) {
        if contributor.event_ref.trim().is_empty()
            || event.identity.trim().is_empty()
            || event.identity != contributor.event_ref
            || !identities.insert(event.identity.as_str())
            || event.camera_id.trim().is_empty()
        {
            return Err(SealedError::InvalidAttribution);
        }
        match camera_id {
            Some(camera) if camera != event.camera_id => {
                return Err(SealedError::InvalidAttribution);
            }
            Some(_) => {}
            None => camera_id = Some(event.camera_id.as_str()),
        }
    }
    Ok(camera_id.unwrap_or(""))
}
