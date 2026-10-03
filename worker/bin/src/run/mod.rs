//! Stage 4 boot steps 1–4 and their admission contracts.
//! The caller owns the run loop, process exit and later media/recorder startup.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use seeon_worker_runtime::evidence::EngineDigest;
use sha2::{Digest, Sha256};

use crate::config::Checked;
use crate::config::model_bundle::AdmissionKind;
use crate::config::model_bundle::flow_boot::FlowBootKind;
use crate::config::model_bundle::identity::IdentityKind;

pub mod boot;
pub mod calibration;
pub mod cameras;
pub mod clip_output;
pub mod config_digest;
pub mod decision;
mod delivery;
mod event_payload;
pub mod events;
pub mod execution;
pub mod exporter;
pub mod fall_evidence;
pub mod media_config;
pub(crate) mod model_sources;
pub mod models;
pub mod policy;
pub mod pump;
pub mod records;
mod replay;
pub mod runtime_manifest;
pub mod settings;
pub mod status;

#[cfg(test)]
mod model_tests;

pub use boot::{BootFailure, Booted, boot};
pub use settings::{BootPolicy, Settings, SettingsError};
pub use status::{BootStatusContext, ReportIdentity};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelRole {
    Fall,
    Bed,
    StoredPose,
}

pub struct EngineFiles {
    pub fall: PathBuf,
    pub bed: PathBuf,
    pub stored_pose: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineFileKind {
    Io(io::ErrorKind),
    NotRegular,
    Empty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineFileError {
    pub model: ModelRole,
    pub kind: EngineFileKind,
}

/// Actual engine bytes, not a placeholder identity. Image provenance remains
/// the deployment authority; these hashes bind the GPU call evidence to files.
pub struct EngineArtifact {
    pub path: PathBuf,
    pub digest: EngineDigest,
}

pub struct ModelEngines {
    pub fall: EngineArtifact,
    pub bed: EngineArtifact,
    pub stored_pose: EngineArtifact,
}

impl ModelEngines {
    pub(crate) fn admit(files: &EngineFiles) -> Result<Self, EngineFileError> {
        Ok(Self {
            fall: digest(&files.fall, ModelRole::Fall)?,
            bed: digest(&files.bed, ModelRole::Bed)?,
            stored_pose: digest(&files.stored_pose, ModelRole::StoredPose)?,
        })
    }
}

fn digest(path: &Path, model: ModelRole) -> Result<EngineArtifact, EngineFileError> {
    let io_error = |error: io::Error| EngineFileError {
        model,
        kind: EngineFileKind::Io(error.kind()),
    };
    // Reject non-regular paths before open, so a configured FIFO cannot block boot.
    if !fs::metadata(path).map_err(io_error)?.is_file() {
        return Err(EngineFileError {
            model,
            kind: EngineFileKind::NotRegular,
        });
    }
    let mut file = File::open(path).map_err(io_error)?;
    let metadata = file.metadata().map_err(io_error)?;
    if !metadata.is_file() {
        return Err(EngineFileError {
            model,
            kind: EngineFileKind::NotRegular,
        });
    }
    if metadata.len() == 0 {
        return Err(EngineFileError {
            model,
            kind: EngineFileKind::Empty,
        });
    }
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(EngineArtifact {
        path: path.to_path_buf(),
        digest: EngineDigest::new(hash.finalize().into()),
    })
}

/// All Flow boot wiring, retained for the later media configuration.
pub struct FlowSettings {
    pub engine: PathBuf,
    pub identity_path: PathBuf,
    pub infer_config: PathBuf,
    pub tracker_config: PathBuf,
    pub tracker_library: PathBuf,
    pub onnx: PathBuf,
    pub parser_library: PathBuf,
    pub record_dir: PathBuf,
    pub record_cache_seconds: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub batch_size: u32,
    pub rtsp_reconnect_interval_sec: u32,
    pub identity: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IdentityError {
    Environment,
    Selection,
    Bundle(AdmissionKind),
    Packaged(AdmissionKind),
    Calibration(calibration::CalibrationError),
    Conformance(crate::config::model_bundle::conformance::ConformanceError),
    OutputClassCount,
    SelectionThreshold,
    Engine(IdentityKind),
    Flow(FlowBootKind),
    FlowValue(&'static str),
    Model(EngineFileError),
}

/// Selected bundle proof, actual engine hashes and the verified Flow identity.
pub struct Admitted {
    /// Preserves the established optional packaged selection and LKG read result.
    pub checked: Checked,
    pub flow: FlowSettings,
    pub engines: ModelEngines,
    pub fall: fall_evidence::FallEvidence,
}
