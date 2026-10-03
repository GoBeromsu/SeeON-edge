//! Bounded captures, exclusive staging, and same-filesystem commit.
//! Existing engines are never unlinked directly. Staging stays on failure.

mod cache;
mod paths;
mod publication;

use std::path::{Path, PathBuf};

use rustix::fs::{self, Mode};

use super::{AuxiliaryLayout, Captured, Layout, Planned};
use crate::cli::{AuxiliaryBuild, EngineBuildFlags};
use crate::config::model_bundle::identity::{
    EnginePaths, IdentityInputs, OBSERVER_LIBRARY, fingerprint,
};
use crate::seam::{IdSource, RandomIds};

pub use cache::CommandError;
pub(super) use cache::{authorize, cache_candidate};
pub(super) use paths::{absolute_text, read_text};
pub(super) use publication::{commit, identity_staging, write_exclusive};

const PRIVATE_DIR: Mode = Mode::RWXU;

pub(super) fn prepare(
    flags: &EngineBuildFlags,
    captured: &Captured,
) -> Result<Planned, CommandError> {
    let served = served_output(flags);
    let mut targets = vec![flags.engine.as_path()];
    match &flags.auxiliary {
        AuxiliaryBuild::TensorRt {
            stored_pose_engine,
            bed_engine,
            fall_engine,
        } => {
            targets.extend([
                stored_pose_engine.as_path(),
                bed_engine.as_path(),
                fall_engine.as_path(),
            ]);
        }
        AuxiliaryBuild::OnnxRuntimeCpu => {}
    }
    let served_index = targets.len();
    targets.extend([served, flags.identity.as_path()]);
    let inputs = [
        flags.onnx.as_path(),
        flags.bed_onnx.as_path(),
        captured.fall.source_path.as_path(),
        flags.parser_lib.as_path(),
        flags.infer_config.as_path(),
        flags.tracker_config.as_path(),
        flags.tracker_library.as_path(),
    ];
    paths::reject_collisions(&targets, &inputs, served_index)?;
    let token = RandomIds.uuid4().map_err(|_| CommandError::Io)?;
    Ok(Planned {
        layout: layout_of(flags, served, &token)?,
        token,
    })
}

fn layout_of(flags: &EngineBuildFlags, served: &Path, token: &str) -> Result<Layout, CommandError> {
    let staged = |role: &str, path: &Path| -> Result<PathBuf, CommandError> {
        Ok(paths::parent_of(path)?
            .join(format!(".engine-stage-{role}-{token}"))
            .join(paths::basename(path)?))
    };
    let auxiliary = match &flags.auxiliary {
        AuxiliaryBuild::TensorRt {
            stored_pose_engine,
            bed_engine,
            fall_engine,
        } => AuxiliaryLayout::TensorRt {
            stored: staged("stored", stored_pose_engine)?,
            bed: staged("bed", bed_engine)?,
            fall: staged("fall", fall_engine)?,
            final_stored: stored_pose_engine.clone(),
            final_bed: bed_engine.clone(),
            final_fall: fall_engine.clone(),
        },
        AuxiliaryBuild::OnnxRuntimeCpu => AuxiliaryLayout::OnnxRuntimeCpu,
    };
    Ok(Layout {
        live: staged("live", &flags.engine)?,
        auxiliary,
        served: staged("served", served)?,
        identity: flags.identity.clone(),
        final_live: flags.engine.clone(),
        final_served: served.to_path_buf(),
        parser_lib: flags.parser_lib.clone(),
        tracker_config: flags.tracker_config.clone(),
        tracker_library: flags.tracker_library.clone(),
        observer: PathBuf::from(OBSERVER_LIBRARY),
    })
}

pub(super) fn materialize(planned: Planned) -> Result<Layout, CommandError> {
    for path in planned.layout.staging() {
        let parent = path.parent().ok_or(CommandError::Path)?;
        let name = parent
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if !name.starts_with(".engine-stage-") || !name.ends_with(&planned.token) {
            return Err(CommandError::Path);
        }
        fs::mkdir(parent, PRIVATE_DIR).map_err(|_| CommandError::Io)?;
    }
    Ok(planned.layout)
}

pub(super) fn served_text(
    flags: &crate::cli::EngineBuildFlags,
) -> Result<(String, bool), CommandError> {
    let template = read_text(&flags.infer_config)?;
    let Some(_) = flags.served_infer_config.as_deref() else {
        return Ok((template, false));
    };
    Ok((
        super::render_served(&template, &absolute_text(&flags.engine)?, flags.batch_size)?,
        true,
    ))
}

pub(super) fn verify_existing_served(
    text: &str,
    engine: &Path,
    batch: u32,
) -> Result<(), CommandError> {
    if !crate::config::model_bundle::identity::engine_only_config(text) {
        return Err(CommandError::Config);
    }
    let engine_line = format!("model-engine-file={}", absolute_text(engine)?);
    let batch_line = format!("batch-size={batch}");
    let engine_count = text
        .lines()
        .filter(|line| line.starts_with("model-engine-file="))
        .count();
    let batch_count = text
        .lines()
        .filter(|line| line.starts_with("batch-size="))
        .count();
    if engine_count != 1
        || batch_count != 1
        || !text.lines().any(|line| line == engine_line)
        || !text.lines().any(|line| line == batch_line)
    {
        return Err(CommandError::Config);
    }
    Ok(())
}

pub(super) fn fingerprint_of(path: &Path) -> Result<String, CommandError> {
    fingerprint(path, "artifact").map_err(|_| CommandError::Io)
}

pub(super) fn capture_inputs(
    flags: &crate::cli::EngineBuildFlags,
) -> Result<Vec<(PathBuf, String)>, CommandError> {
    [
        &flags.parser_lib,
        &flags.tracker_config,
        &flags.tracker_library,
    ]
    .into_iter()
    .map(|path| Ok((path.clone(), fingerprint_of(path)?)))
    .collect()
}

pub(super) fn require_replaceable(layout: &Layout, write_served: bool) -> Result<(), CommandError> {
    for path in finals(layout, write_served) {
        paths::require_replaceable(path)?;
    }
    Ok(())
}

pub(super) fn recheck_inputs(layout: &Layout, captured: &Captured) -> Result<(), CommandError> {
    if fingerprint_of(&layout.observer)? != captured.observer {
        return Err(CommandError::Io);
    }
    for (path, expected) in &captured.input_digests {
        if fingerprint_of(path)? != *expected {
            return Err(CommandError::Io);
        }
    }
    if !captured.served_output
        && fingerprint_of(&layout.final_served)?
            != crate::records::id::sha256_hex(captured.served.as_bytes())
    {
        return Err(CommandError::Io);
    }
    Ok(())
}

pub(super) fn identity_inputs<'a>(
    layout: &'a Layout,
    captured: &'a Captured,
    batch: u32,
    flow: &'a [(&'a str, PathBuf)],
) -> IdentityInputs<'a> {
    IdentityInputs {
        engines: match &layout.auxiliary {
            AuxiliaryLayout::TensorRt {
                final_stored,
                final_bed,
                final_fall,
                ..
            } => EnginePaths::TensorRt {
                live_pose: &layout.final_live,
                stored_pose: final_stored,
                bed: final_bed,
                fall: final_fall,
            },
            AuxiliaryLayout::OnnxRuntimeCpu => EnginePaths::OnnxRuntimeCpu {
                live_pose: &layout.final_live,
            },
        },
        pose_onnx_sha256: &captured.pose.sha256,
        bed_onnx_sha256: &captured.bed.sha256,
        fall_onnx_sha256: &captured.fall.sha256,
        flow,
        observer_library: &layout.observer,
        image_digest: &captured.image,
        configured_batch: Some(batch),
        deployed_batch: None,
    }
}

fn served_output(flags: &crate::cli::EngineBuildFlags) -> &Path {
    flags
        .served_infer_config
        .as_deref()
        .unwrap_or(flags.infer_config.as_path())
}

fn finals(layout: &Layout, write_served: bool) -> impl Iterator<Item = &Path> {
    layout
        .engines()
        .map(|(_, _, target)| target)
        .chain(write_served.then_some(layout.final_served.as_path()))
        .chain(Some(layout.identity.as_path()))
}
