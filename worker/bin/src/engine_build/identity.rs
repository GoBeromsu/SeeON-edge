//! Aggregate engine-identity publisher. One immutable document records four
//! measured receipts and the current Flow artifact fingerprints. Those
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
    SchemaError, deployment_image_digest, validate_entries,
};

pub use store::publish_identity;

/// One receipt and the engine file it claims to describe.
#[derive(Clone, Copy)]
pub struct BuiltEngine<'a> {
    pub receipt: &'a EngineReceipt,
    pub path: &'a Path,
}

/// The four engines of one offline identity. Roles are not interchangeable.
#[derive(Clone, Copy)]
pub struct EngineSet<'a> {
    pub live_pose: BuiltEngine<'a>,
    pub stored_pose: BuiltEngine<'a>,
    pub bed: BuiltEngine<'a>,
    pub fall: BuiltEngine<'a>,
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
    let engines = checked_engines(&request.engines, image, request.batch_size)?;
    let mut bytes = serde_json::to_vec(&serde_json::json!({
        "schema_version": 1,
        "batch_size": request.batch_size,
        "engines": engines,
        "flow": store::flow_fingerprints(request.flow)?,
    }))
    .map_err(|_| IdentityError::Receipt)?;
    bytes.push(b'\n');
    if bytes.len() > 64 * 1024 {
        return Err(IdentityError::Receipt);
    }
    Ok(bytes)
}

fn checked_engines(
    engines: &EngineSet<'_>,
    image: &str,
    batch: u32,
) -> Result<Value, IdentityError> {
    let assembled = serde_json::json!({
        "live_pose": engines.live_pose.receipt.document().clone(),
        "stored_pose": engines.stored_pose.receipt.document().clone(),
        "bed": engines.bed.receipt.document().clone(),
        "fall": engines.fall.receipt.document().clone(),
    });
    validate_entries(&assembled, image, batch).map_err(schema_error)?;
    for (built, key) in [
        (engines.live_pose, "live_pose"),
        (engines.stored_pose, "stored_pose"),
        (engines.bed, "bed"),
        (engines.fall, "fall"),
    ] {
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
