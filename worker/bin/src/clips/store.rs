//! The clip store layout: `clips/<id>/` for published clips and
//! `clips/.staging/<id>/` for clips still being produced.

use std::path::{Path, PathBuf};

use super::durable;
use super::manifest::ClipMetadata;
use super::publish::PublishError;

/// The product clip store root.
pub const PRODUCT_ROOT: &str = "/var/lib/clip-store";
pub const ARTIFACT_FILE: &str = "artifact.mp4";
const STAGING: &str = ".staging";
pub(crate) const DIRECTORY_MODE: u32 = 0o755;

/// A clip store rooted at one directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipStore {
    root: PathBuf,
}

/// The directories one clip owns while it is being produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub camera_id: String,
    pub clip_id: String,
    pub final_dir: PathBuf,
    pub staging_dir: PathBuf,
}

fn safe_component(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('.')
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

impl ClipStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn clip_dir(&self, clip_id: &str) -> PathBuf {
        self.root.join("clips").join(clip_id)
    }

    pub fn staging_dir(&self, clip_id: &str) -> PathBuf {
        self.root.join("clips").join(STAGING).join(clip_id)
    }

    /// Validates the ids and creates the store and staging directories.
    pub fn reserve(&self, camera_id: &str, clip_id: &str) -> Result<Reservation, PublishError> {
        if !safe_component(clip_id) {
            return Err(PublishError::Reservation("clip_id"));
        }
        if camera_id.trim().is_empty() {
            return Err(PublishError::Reservation("camera_id"));
        }
        let clips = self.root.join("clips");
        durable::create_dir(&self.root, DIRECTORY_MODE)?;
        durable::create_dir(&clips, DIRECTORY_MODE)?;
        durable::create_dir(&clips.join(STAGING), DIRECTORY_MODE)?;
        let staging_dir = self.staging_dir(clip_id);
        durable::create_dir(&staging_dir, DIRECTORY_MODE)?;
        Ok(Reservation {
            camera_id: camera_id.to_owned(),
            clip_id: clip_id.to_owned(),
            final_dir: self.clip_dir(clip_id),
            staging_dir,
        })
    }
}

impl Reservation {
    pub fn artifact_path(&self) -> PathBuf {
        self.staging_dir.join(ARTIFACT_FILE)
    }

    pub(crate) fn check(&self, meta: &ClipMetadata) -> Result<(), PublishError> {
        if !safe_component(&self.clip_id) || meta.clip_id != self.clip_id {
            return Err(PublishError::Reservation("clip_id"));
        }
        if meta.camera_id != self.camera_id {
            return Err(PublishError::Reservation("camera_id"));
        }
        Ok(())
    }
}

pub(crate) fn exists(path: &Path) -> bool {
    path.symlink_metadata().is_ok_and(|meta| meta.is_file())
}
