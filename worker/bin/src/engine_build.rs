//! Offline FP32 engine build and its receipt. A receipt exists only after the
//! native exclusive build succeeds and the created regular file hashes.
//! `image_digest` is the caller-declared deployment authority, not a measured
//! container identity. A later failure returns no receipt and leaves the
//! artifact for explicit owner cleanup; this module never deletes or overwrites.
//! Live pose builds live in [`live`] and return this same private receipt.
//! That path does not replace this factory or its FP32 profile.
//! The offline command selects either four TensorRT receipts or one live GPU
//! receipt plus immutable original ONNX bytes for auxiliary ORT CPU models.
//! CPU model bytes are identity inputs, never engine or execution receipts.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use rustix::fs::{Mode, OFlags};
use seeon_deepstream_native::{StateError, build_engine};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::config::model_bundle::identity::deployment_image_digest;
use crate::config::model_bundle::identity::native_profile as profile;
use crate::records::id::sha256_hex;
use crate::run::ModelRole;

mod command;
mod identity;
mod live;
pub use command::execute_engine_build;
pub use identity::{
    BuiltEngine, CpuModelBytes, EngineSet, FlowArtifacts, IdentityError, IdentityRequest,
    publish_identity,
};

pub use live::{LiveBuildError, LiveBuildRequest, build_live_pose};

fn engine_basename(path: &Path) -> Result<&str, BuildError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && !name.contains('/') && !name.contains('\0'))
        .ok_or(BuildError::Native(StateError::InvalidPath))
}

const MAX_ENGINE_BYTES: u64 = 512 * 1024 * 1024;

/// Borrowed inputs for one exclusive FP32 build.
#[derive(Clone, Copy)]
pub struct BuildRequest<'a> {
    pub role: ModelRole,
    pub onnx: &'a [u8],
    pub expected_onnx_sha256: &'a str,
    pub engine: &'a Path,
    /// Caller-declared deployment image authority, not a measured container id.
    pub image_digest: &'a str,
    pub device: i32,
}

/// Why a receipt was not produced. An exclusive file may still exist.
#[derive(Debug)]
pub enum BuildError {
    SourceDigest,
    ImageDigest,
    Native(StateError),
    Output,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::SourceDigest => "captured ONNX digest does not match the expected sha256",
            Self::ImageDigest => "deployment image digest is invalid",
            Self::Native(_) => "FP32 engine build failed",
            Self::Output => "built engine could not be hashed as a bounded regular file",
        })
    }
}

impl std::error::Error for BuildError {}

/// Measured receipt. Fields are private and there is no public constructor or
/// `Deserialize`: only verified [`build_fp32`] or [`build_live_pose`] builds produce one.
#[derive(Debug, PartialEq, Eq)]
pub struct EngineReceipt {
    document: Value,
}

impl EngineReceipt {
    /// Immutable measured entry for the aggregate engine identity document.
    pub fn document(&self) -> &Value {
        &self.document
    }
}

/// Validates the captured bytes and declared image, then builds exclusively.
/// Hardware fields come only from the native result; TF32 must be off.
pub fn build_fp32(request: BuildRequest<'_>) -> Result<EngineReceipt, BuildError> {
    if !crate::config::is_hex(request.expected_onnx_sha256, 64) {
        return Err(BuildError::SourceDigest);
    }
    let actual = sha256_hex(request.onnx);
    if request.expected_onnx_sha256 != actual {
        return Err(BuildError::SourceDigest);
    }
    let image_digest = deployment_image_digest(request.image_digest)
        .map(str::to_owned)
        .ok_or(BuildError::ImageDigest)?;
    let engine = engine_basename(request.engine)?.to_owned();
    let (input, dimensions) = profile(request.role);
    let native = build_engine(
        request.onnx,
        request.engine,
        request.device,
        input,
        dimensions,
    )
    .map_err(BuildError::Native)?;
    if native.tf32_enabled
        || native.trt_version <= 0
        || native.compute_major <= 0
        || native.compute_minor < 0
    {
        return Err(BuildError::Output);
    }
    let input = input.to_str().map_err(|_| BuildError::Output)?;
    let engine_sha256 = output_digest(request.engine).map_err(|_| BuildError::Output)?;
    Ok(EngineReceipt {
        document: json!({
            "engine": engine,
            "onnx_sha256": actual,
            "engine_sha256": engine_sha256,
            "precision": "fp32",
            "tf32_enabled": native.tf32_enabled,
            "trt_version": native.trt_version,
            "device_name": native.device_name,
            "compute_capability": format!("{}.{}", native.compute_major, native.compute_minor),
            "device": request.device,
            "input": input,
            "dimensions": dimensions,
            "image_digest": image_digest,
        }),
    })
}

/// Nofollow, nonblocking regular-file capture. Streams through EOF, at most one
/// byte past 512 MiB, and requires that length to equal the descriptor size.
fn output_digest(path: &Path) -> io::Result<String> {
    let flags = OFlags::CLOEXEC | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK;
    let file = File::from(rustix::fs::open(path, flags, Mode::empty())?);
    let info = file.metadata()?;
    let declared = info.len();
    if !info.is_file() || declared == 0 || declared > MAX_ENGINE_BYTES {
        return Err(io::Error::other(
            "engine output is not a bounded regular file",
        ));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut actual = 0_u64;
    let mut reader = file.take(declared + 1);
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let count_u64 = u64::try_from(count).map_err(io::Error::other)?;
        actual = actual
            .checked_add(count_u64)
            .ok_or_else(|| io::Error::other("length"))?;
        if actual > declared {
            return Err(io::Error::other("engine exceeds descriptor size"));
        }
        hasher.update(&buffer[..count]);
    }
    if actual != declared || actual > MAX_ENGINE_BYTES {
        return Err(io::Error::other(
            "hashed length differs from descriptor length",
        ));
    }
    let mut encoded = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(encoded, "{byte:02x}").map_err(io::Error::other)?;
    }
    Ok(encoded)
}
