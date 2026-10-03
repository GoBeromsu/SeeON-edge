//! Language-neutral applied provenance. Schema 2 names Rust and actual loaded
//! runtime versions; it never invents Python environment facts or engine/ONNX linkage.
mod components;
mod content;
mod store;

use std::fmt;
use std::path::{Path, PathBuf};

use super::{Booted, ModelRole, media_config::MediaAssembly};
use crate::config::pull::PulledConfig;
use crate::inference::Runtime;
use crate::json::{Json, JsonError, Serialiser};
use crate::records::id::sha256_hex;

pub const SCHEMA_VERSION: i128 = 2;
pub const MAX_MANIFEST_BYTES: usize = 128 * 1024;
pub const MAX_MANIFESTS: usize = 512;

fn object<const N: usize>(fields: [(&str, Json); N]) -> Json {
    Json::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn text(value: impl Into<String>) -> Json {
    Json::Str(value.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    Missing(&'static str),
    Invalid(&'static str),
    Contradictory(&'static str),
    RuntimeFacts,
    Canonical(JsonError),
    Capacity,
    Conflict,
    Io(std::io::ErrorKind),
}
impl From<std::io::Error> for ManifestError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.kind())
    }
}
impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(field) => write!(f, "applied manifest identity missing: {field}"),
            Self::Invalid(field) => write!(f, "applied manifest identity invalid: {field}"),
            Self::Contradictory(field) => {
                write!(f, "applied manifest identity contradictory: {field}")
            }
            Self::RuntimeFacts => f.write_str("linked runtime version query failed"),
            Self::Canonical(_) => {
                f.write_str("applied manifest content is not safe canonical JSON")
            }
            Self::Capacity => f.write_str("applied manifest storage bound reached"),
            Self::Conflict => {
                f.write_str("immutable applied manifest conflicts with stored content")
            }
            Self::Io(_) => f.write_str("applied manifest persistence failed"),
        }
    }
}
impl std::error::Error for ManifestError {}

#[derive(Clone, Copy)]
struct ModelRuntimeFacts<'a> {
    fall: &'a Runtime,
    bed: &'a Runtime,
    stored_pose: &'a Runtime,
}
impl<'a> ModelRuntimeFacts<'a> {
    fn new(
        fall: Option<&'a Runtime>,
        bed: Option<&'a Runtime>,
        stored_pose: Option<&'a Runtime>,
    ) -> Result<Self, ManifestError> {
        Ok(Self {
            fall: fall.ok_or(ManifestError::Missing("fall_runtime"))?,
            bed: bed.ok_or(ManifestError::Missing("bed_runtime"))?,
            stored_pose: stored_pose.ok_or(ManifestError::Missing("stored_pose_runtime"))?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    canonical: String,
    sha256: String,
}
impl Manifest {
    fn freeze(value: &Json) -> Result<Self, ManifestError> {
        let canonical = Serialiser::Provenance
            .canonical(value)
            .map_err(ManifestError::Canonical)?;
        if canonical.len() > MAX_MANIFEST_BYTES {
            return Err(ManifestError::Capacity);
        }
        let sha256 = sha256_hex(canonical.as_bytes());
        Ok(Self { canonical, sha256 })
    }
    pub fn canonical(&self) -> &str {
        &self.canonical
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Requires the process GPU lease. Identical content reuses its immutable
    /// file. Capacity pressure refuses new content; referenced history is never pruned.
    pub fn persist(&self, state_dir: &Path) -> Result<PathBuf, ManifestError> {
        store::persist(self, state_dir)
    }
}

/// Called only after model boot and media/policy assembly admission. Auxiliary
/// facts come from retained owners; live media still requires verified GPU facts.
pub fn build(
    booted: &Booted,
    config: &PulledConfig,
    media: &MediaAssembly,
) -> Result<Manifest, ManifestError> {
    let runtimes = ModelRuntimeFacts::new(
        booted.models.runtime(ModelRole::Fall),
        booted.models.runtime(ModelRole::Bed),
        booted.models.runtime(ModelRole::StoredPose),
    )?;
    let versions =
        seeon_deepstream_native::runtime_versions().map_err(|_| ManifestError::RuntimeFacts)?;
    Manifest::freeze(&content::build(
        &booted.settings,
        &booted.admitted,
        &booted.gpu,
        config,
        media,
        versions,
        runtimes,
    )?)
}

#[cfg(test)]
#[path = "runtime_manifest_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "runtime_manifest_input_tests.rs"]
mod input_tests;
