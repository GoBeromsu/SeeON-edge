//! Flow sealed-clip sidecars (`flow_sealed_sidecar.py`): a sealed Flow clip
//! is written to `flow-sealed/` before it is published and retired only
//! after publication succeeds or its owner confirms missing media.

mod payload;
mod read;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::durable::{self, PUBLIC_FILE};
use crate::json::Serialiser;

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
    pub sealed: SealedClip,
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
    /// media locations; retire the sidecar. A failed or malformed terminal
    /// must be `Failed`, not `MissingMedia`.
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

    /// Writes `<clip_id>.json` durably. Contributors are stably sorted by
    /// the `detected_at` text; an unknown contributor refuses before the
    /// directory is created.
    pub fn persist(
        &self,
        sealed: &SealedClip,
        events: &BTreeMap<String, SealedEvent>,
    ) -> Result<PathBuf, SealedError> {
        let target = self.sidecar_path(&sealed.clip_id)?;
        let mut ordered: Vec<&SealedContributor> = sealed.contributors.iter().collect();
        ordered.sort_by(|left, right| left.detected_at.cmp(&right.detected_at));
        let written = ordered
            .iter()
            .map(|contributor| events.get(&contributor.event_ref))
            .collect::<Option<Vec<&SealedEvent>>>()
            .ok_or(SealedError::UnknownRef)?;
        let payload = payload::sidecar_json(sealed, &ordered, &written);
        let text = Serialiser::ModelSelection
            .canonical(&payload)
            .map_err(|_| SealedError::NonFinite)?;
        fs::create_dir_all(durable::parent_of(&self.directory))?;
        durable::create_dir(&self.directory, SIDECAR_DIR_MODE)?;
        durable::write_durable(&target, text.as_bytes(), PUBLIC_FILE)?;
        Ok(target)
    }

    /// `pending_for_camera`: this camera's sidecars in file-name order.
    pub fn pending(&self, camera_id: &str) -> Result<Pending, SealedError> {
        read::pending(&self.directory, camera_id)
    }

    /// Always calls the publication owner, which checks terminal state and all
    /// owned media locations. `Published` and explicit `MissingMedia` retire
    /// the sidecar; `Failed` keeps it unchanged.
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
        let unsafe_id =
            clip_id.is_empty() || clip_id.starts_with('.') || clip_id.contains(['/', '\0']);
        if unsafe_id {
            return Err(SealedError::InvalidClipId);
        }
        Ok(self.directory.join(format!("{clip_id}.json")))
    }
}
