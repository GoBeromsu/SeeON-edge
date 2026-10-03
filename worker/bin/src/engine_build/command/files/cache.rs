//! Cache validity and ownership are different questions. A prior coherent
//! identity may own an engine even when the requested model or image changes.

use super::super::{AuxiliaryLayout, Captured, Layout};
use crate::config::model_bundle::identity::{
    fingerprint, validate_entries, validate_hybrid_entries, verify_aggregate,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum CommandError {
    Source,
    Image,
    Batch,
    Config,
    Parser,
    Path,
    Collision,
    Target,
    Hardware,
    Live(crate::engine_build::LiveBuildError),
    Native(crate::engine_build::BuildError),
    Identity(crate::engine_build::IdentityError),
    Io,
}
impl std::fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Source => "engine source capture failed",
            Self::Image => "deployment image digest is invalid",
            Self::Batch => "batch size is outside 1..=16",
            Self::Config => "infer config must define the served keys once",
            Self::Parser => "custom-lib-path does not identify the parser library",
            Self::Path => "engine-build path is not a replaceable regular file",
            Self::Collision => "engine-build input and output paths collide",
            Self::Target => "existing target is not recorded by the selected provider identity",
            Self::Hardware => "current GPU hardware could not be read",
            Self::Live(_) => "live pose engine build failed",
            Self::Native(_) => "native FP32 engine build failed",
            Self::Identity(_) => "engine identity publication failed",
            Self::Io => "engine-build filesystem commit failed",
        })
    }
}
impl std::error::Error for CommandError {}

pub(in crate::engine_build::command) fn cache_candidate(
    layout: &Layout,
    captured: &Captured,
    batch: u32,
) -> Option<BTreeMap<String, String>> {
    if fingerprint(&layout.final_served, "infer_config_sha256")
        .ok()
        .as_deref()
        != Some(crate::records::id::sha256_hex(captured.served.as_bytes()).as_str())
    {
        return None;
    }
    let flow = flow_files(layout);
    verify_aggregate(
        &layout.identity,
        super::identity_inputs(layout, captured, batch, &flow),
    )
    .ok()
}

pub(in crate::engine_build::command) fn authorize(
    layout: &Layout,
    force: bool,
    captured: &Captured,
) -> Result<(), CommandError> {
    let targets: Vec<_> = super::finals(layout, captured.served_output).collect();
    let mut existing = Vec::new();
    for path in targets {
        if exists(path)? {
            existing.push(path);
        }
    }
    if existing.is_empty() || force {
        return Ok(());
    }
    let document = prior_document(layout).ok_or(CommandError::Target)?;
    for (role, _, path) in layout.engines() {
        if !exists(path)? {
            continue;
        }
        let entry = &document["engines"][role];
        if entry["engine"].as_str() != path.file_name().and_then(|name| name.to_str())
            || entry["engine_sha256"].as_str() != Some(current_digest(path)?.as_str())
        {
            return Err(CommandError::Target);
        }
    }
    if captured.served_output
        && exists(&layout.final_served)?
        && document["flow"]["infer_config_sha256"].as_str()
            != Some(current_digest(&layout.final_served)?.as_str())
    {
        return Err(CommandError::Target);
    }
    Ok(())
}

fn prior_document(layout: &Layout) -> Option<Value> {
    let bytes = super::paths::read_bounded(&layout.identity, 64 * 1024).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    let object = value.as_object()?;
    let (version, fields) = match &layout.auxiliary {
        AuxiliaryLayout::TensorRt { .. } => (1, 4),
        AuxiliaryLayout::OnnxRuntimeCpu => (2, 5),
    };
    if object.len() != fields || value["schema_version"].as_u64() != Some(version) {
        return None;
    }
    let batch = u32::try_from(value["batch_size"].as_u64()?).ok()?;
    let image = value["engines"]["live_pose"]["image_digest"].as_str()?;
    match &layout.auxiliary {
        AuxiliaryLayout::TensorRt { .. } => {
            validate_entries(&value["engines"], image, batch).ok()?;
        }
        AuxiliaryLayout::OnnxRuntimeCpu => {
            validate_hybrid_entries(&value["engines"], &value["auxiliary"], image, batch).ok()?;
        }
    }
    let flow = value["flow"].as_object()?;
    if flow.len() != 4
        || ![
            "infer_config_sha256",
            "parser_lib_sha256",
            "tracker_config_sha256",
            "tracker_library_sha256",
        ]
        .iter()
        .all(|key| {
            flow.get(*key)
                .and_then(Value::as_str)
                .is_some_and(|sha| crate::config::is_hex(sha, 64))
        })
    {
        return None;
    }
    Some(value)
}

fn flow_files(layout: &Layout) -> Vec<(&'static str, PathBuf)> {
    vec![
        ("parser_lib_sha256", layout.parser_lib.clone()),
        ("infer_config_sha256", layout.final_served.clone()),
        ("tracker_config_sha256", layout.tracker_config.clone()),
        ("tracker_library_sha256", layout.tracker_library.clone()),
    ]
}
fn exists(path: &Path) -> Result<bool, CommandError> {
    match rustix::fs::lstat(path) {
        Err(rustix::io::Errno::NOENT) => Ok(false),
        Ok(info) if rustix::fs::FileType::from_raw_mode(info.st_mode).is_file() => Ok(true),
        Ok(_) => Err(CommandError::Path),
        Err(_) => Err(CommandError::Io),
    }
}
fn current_digest(path: &Path) -> Result<String, CommandError> {
    fingerprint(path, "artifact").map_err(|_| CommandError::Io)
}
