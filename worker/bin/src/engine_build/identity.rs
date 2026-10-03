//! Aggregate inference-identity publisher. One immutable document records the
//! selected engine receipts, CPU source hashes and current Flow fingerprints. Those
//! fingerprints describe the files supplied here; they are not proof a child
//! consumed those bytes. A late publication failure may leave the destination
//! present. This module never overwrites or deletes an existing destination
//! or engine, and it does not create a destination parent.

mod store;
#[cfg(test)]
mod tests;

use std::path::Path;

use serde_json::Value;

use super::{EngineReceipt, output_digest};
use crate::config::is_hex;
use crate::config::model_bundle::identity::{
    SchemaError, deployment_image_digest, validate_entries, validate_hybrid_entries,
};

pub use store::publish_identity;

/// One receipt and the engine file it claims to describe.
#[derive(Clone, Copy)]
pub struct BuiltEngine<'a> {
    pub receipt: &'a EngineReceipt,
    pub path: &'a Path,
}

/// Captured CPU ONNX bytes, not engines or accelerator execution receipts.
#[derive(Clone, Copy)]
pub struct CpuModelBytes<'a> {
    pub stored_pose: &'a [u8],
    pub bed: &'a [u8],
    pub fall: &'a [u8],
}

/// Explicit artifact variants; auxiliary ONNX never occupies an engine receipt.
#[derive(Clone, Copy)]
pub enum EngineSet<'a> {
    TensorRt {
        live_pose: BuiltEngine<'a>,
        stored_pose: BuiltEngine<'a>,
        bed: BuiltEngine<'a>,
        fall: BuiltEngine<'a>,
    },
    OnnxRuntimeCpu {
        live_pose: BuiltEngine<'a>,
        models: CpuModelBytes<'a>,
    },
}

/// Current Flow artifacts. Ordinary symlinks are followed, as the Flow
/// contract follows them; selected-bundle strictness does not apply.
#[derive(Clone, Copy)]
pub struct FlowArtifacts<'a> {
    pub parser_lib: &'a Path,
    pub infer_config: &'a Path,
    pub tracker_config: &'a Path,
    pub tracker_library: &'a Path,
}

/// Borrowed inputs for one exclusive identity publication.
#[derive(Clone, Copy)]
pub struct IdentityRequest<'a> {
    pub engines: EngineSet<'a>,
    pub flow: FlowArtifacts<'a>,
    pub image_digest: &'a str,
    pub batch_size: u32,
    pub destination: &'a Path,
}

/// Why an identity document was not published. A destination may already exist
/// after a late publication or fsync failure.
#[derive(Debug)]
pub enum IdentityError {
    Receipt,
    Image,
    Batch,
    Output,
    Model,
    Flow,
    Io,
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Receipt => "engine receipt does not match the approved profile",
            Self::Image => "deployment image digest is invalid or does not match",
            Self::Batch => "batch size is outside 1..=16",
            Self::Output => "engine file does not match its receipt",
            Self::Model => "CPU ONNX source is empty or exceeds the capture bound",
            Self::Flow => "flow artifact is not a bounded readable regular file",
            Self::Io => "identity publication failed; destination may already exist",
        })
    }
}

impl std::error::Error for IdentityError {}

pub(super) fn document(request: IdentityRequest<'_>) -> Result<Vec<u8>, IdentityError> {
    if !(1..=16).contains(&request.batch_size) {
        return Err(IdentityError::Batch);
    }
    let image = deployment_image_digest(request.image_digest).ok_or(IdentityError::Image)?;
    let mut document = match request.engines {
        EngineSet::TensorRt {
            live_pose,
            stored_pose,
            bed,
            fall,
        } => {
            let engines = checked_engines(
                [
                    ("live_pose", live_pose),
                    ("stored_pose", stored_pose),
                    ("bed", bed),
                    ("fall", fall),
                ],
                image,
                request.batch_size,
            )?;
            serde_json::json!({"schema_version": 1, "engines": engines})
        }
        EngineSet::OnnxRuntimeCpu { live_pose, models } => {
            let engines = serde_json::json!({"live_pose": live_pose.receipt.document()});
            let auxiliary = serde_json::json!({
                "runtime": "onnxruntime",
                "provider": "cpu",
                "models": {
                    "stored_pose": {"onnx_sha256": store::onnx_fingerprint(models.stored_pose)?},
                    "bed": {"onnx_sha256": store::onnx_fingerprint(models.bed)?},
                    "fall": {"onnx_sha256": store::onnx_fingerprint(models.fall)?},
                },
            });
            validate_hybrid_entries(&engines, &auxiliary, image, request.batch_size)
                .map_err(schema_error)?;
            bound_output(live_pose.receipt.document(), live_pose.path)?;
            serde_json::json!({"schema_version": 2, "engines": engines, "auxiliary": auxiliary})
        }
    };
    document["batch_size"] = request.batch_size.into();
    document["flow"] = store::flow_fingerprints(request.flow)?;
    let mut bytes = serde_json::to_vec(&document).map_err(|_| IdentityError::Receipt)?;
    bytes.push(b'\n');
    if bytes.len() > 64 * 1024 {
        return Err(IdentityError::Receipt);
    }
    Ok(bytes)
}

fn checked_engines(
    engines: [(&str, BuiltEngine<'_>); 4],
    image: &str,
    batch: u32,
) -> Result<Value, IdentityError> {
    let assembled = Value::Object(
        engines
            .iter()
            .map(|(key, built)| ((*key).to_owned(), built.receipt.document().clone()))
            .collect(),
    );
    validate_entries(&assembled, image, batch).map_err(schema_error)?;
    for (key, built) in engines {
        let document = assembled.get(key).ok_or(IdentityError::Receipt)?;
        bound_output(document, built.path)?;
    }
    Ok(assembled)
}

fn schema_error(error: SchemaError) -> IdentityError {
    match error {
        SchemaError::Image => IdentityError::Image,
        SchemaError::Receipt => IdentityError::Receipt,
    }
}

fn bound_output(document: &Value, path: &Path) -> Result<(), IdentityError> {
    let engine = text(document, "engine")?;
    let expected = text(document, "engine_sha256")?;
    if !is_hex(expected, 64) || basename(path)? != engine {
        return Err(IdentityError::Output);
    }
    if output_digest(path).map_err(|_| IdentityError::Output)? != expected {
        return Err(IdentityError::Output);
    }
    Ok(())
}

fn text<'a>(document: &'a Value, key: &str) -> Result<&'a str, IdentityError> {
    document
        .get(key)
        .and_then(Value::as_str)
        .ok_or(IdentityError::Receipt)
}

fn basename(path: &Path) -> Result<&str, IdentityError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && !name.contains('/') && !name.contains('\0'))
        .ok_or(IdentityError::Output)
}
