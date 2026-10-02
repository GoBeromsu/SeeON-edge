//! Offline aggregate engine-build command. Capture and policy run before any
//! native call. Engines land in owned staging, then rename onto requested
//! targets; identity is published last. A failure may leave mixed files. Old
//! or missing identity must refuse boot. This is not multi-file atomic, and
//! it does not roll back, cancel, or claim no side effects on error.
//!
//! `--served-infer-config` absent keeps the infer template read-only and
//! verifies it. Present renders that exact output, which may be the template.
//! Cache reuse requires `verify_aggregate` plus current hardware. `--force`
//! rebuilds. Unknown existing identity requires `--force`.

mod files;
mod infer_config;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use super::identity::publish_identity;
use super::sources::{self, CapturedOnnx};
use super::{
    BuildRequest, BuiltEngine, EngineReceipt, EngineSet, FlowArtifacts, IdentityRequest,
    LiveBuildRequest, build_fp32, build_live_pose,
};
use crate::config::env::Env;
use crate::config::model_bundle::identity::{
    OBSERVER_LIBRARY, deployment_image_digest, hardware_matches,
};
use crate::run::ModelRole;
use seeon_deepstream_native::hardware_identity;

pub use files::CommandError;
use infer_config::{render_served, verify_parser};

struct Captured {
    pose: CapturedOnnx,
    bed: CapturedOnnx,
    fall: CapturedOnnx,
    served: String,
    served_output: bool,
    image: String,
    observer: String,
    input_digests: Vec<(PathBuf, String)>,
}

struct Layout {
    live: PathBuf,
    stored: PathBuf,
    bed: PathBuf,
    fall: PathBuf,
    served: PathBuf,
    identity: PathBuf,
    final_live: PathBuf,
    final_stored: PathBuf,
    final_bed: PathBuf,
    final_fall: PathBuf,
    final_served: PathBuf,
    parser_lib: PathBuf,
    tracker_config: PathBuf,
    tracker_library: PathBuf,
    observer: PathBuf,
}

struct Planned {
    layout: Layout,
    token: String,
}

/// Builds or reuses the four requested engines and publishes identity last.
pub fn execute_engine_build(
    env: &Env,
    flags: crate::cli::EngineBuildFlags,
) -> Result<(), CommandError> {
    let captured = capture(env, &flags)?;
    let planned = files::prepare(&flags, &captured)?;
    files::require_replaceable(&planned.layout, captured.served_output)?;
    if !flags.force
        && let Some(media) = files::cache_candidate(&planned.layout, &captured, flags.batch_size)
        && hardware_matches(&media, 0, &hardware()?)
    {
        return Ok(());
    }
    files::authorize(&planned.layout, flags.force, &captured)?;
    let layout = files::materialize(planned)?;
    let built = build_all(&captured, &layout, flags.batch_size)?;
    files::require_replaceable(&layout, captured.served_output)?;
    files::recheck_inputs(&layout, &captured)?;
    files::commit(
        &layout,
        &publish_staged(&built, &layout, &flags, &captured)?,
        captured.served_output,
    )
}

fn hardware() -> Result<seeon_deepstream_native::GpuHardwareIdentity, CommandError> {
    hardware_identity(0).map_err(|_| CommandError::Hardware)
}

fn capture(env: &Env, flags: &crate::cli::EngineBuildFlags) -> Result<Captured, CommandError> {
    let pose = sources::capture_image_onnx(&flags.onnx).map_err(|_| CommandError::Source)?;
    let bed = sources::capture_image_onnx(&flags.bed_onnx).map_err(|_| CommandError::Source)?;
    let fall = sources::capture_fall(env).map_err(|_| CommandError::Source)?;
    let image = deployment_image_digest(&flags.image_digest)
        .map(str::to_owned)
        .ok_or(CommandError::Image)?;
    if !(1..=16).contains(&flags.batch_size) {
        return Err(CommandError::Batch);
    }
    let (served, served_output) = files::served_text(flags)?;
    verify_parser(&served, &flags.parser_lib)?;
    if !served_output {
        files::verify_existing_served(&served, &flags.engine, flags.batch_size)?;
    }
    let observer = files::fingerprint_of(Path::new(OBSERVER_LIBRARY))?;
    let input_digests = files::capture_inputs(flags)?;
    Ok(Captured {
        pose,
        bed,
        fall,
        served,
        served_output,
        image,
        observer,
        input_digests,
    })
}
fn build_all(
    captured: &Captured,
    layout: &Layout,
    batch: u32,
) -> Result<[EngineReceipt; 4], CommandError> {
    let live = build_live_pose(LiveBuildRequest {
        onnx: &captured.pose.bytes,
        expected_onnx_sha256: &captured.pose.sha256,
        engine: &layout.live,
        image_digest: &captured.image,
        infer_config: &captured.served,
        batch_size: batch,
    })
    .map_err(CommandError::Live)?;
    Ok([
        live,
        native(
            ModelRole::StoredPose,
            &captured.pose,
            &layout.stored,
            &captured.image,
        )?,
        native(ModelRole::Bed, &captured.bed, &layout.bed, &captured.image)?,
        native(
            ModelRole::Fall,
            &captured.fall,
            &layout.fall,
            &captured.image,
        )?,
    ])
}

fn native(
    role: ModelRole,
    source: &CapturedOnnx,
    engine: &Path,
    image: &str,
) -> Result<EngineReceipt, CommandError> {
    build_fp32(BuildRequest {
        role,
        onnx: &source.bytes,
        expected_onnx_sha256: &source.sha256,
        engine,
        image_digest: image,
        device: 0,
    })
    .map_err(CommandError::Native)
}

fn publish_staged(
    built: &[EngineReceipt; 4],
    layout: &Layout,
    flags: &crate::cli::EngineBuildFlags,
    captured: &Captured,
) -> Result<PathBuf, CommandError> {
    if captured.served_output {
        files::write_exclusive(&layout.served, captured.served.as_bytes())?;
    }
    let destination = files::identity_staging(&flags.identity)?;
    let engines = EngineSet {
        live_pose: built_of(&built[0], &layout.live),
        stored_pose: built_of(&built[1], &layout.stored),
        bed: built_of(&built[2], &layout.bed),
        fall: built_of(&built[3], &layout.fall),
    };
    let infer = if captured.served_output {
        &layout.served
    } else {
        &layout.final_served
    };
    publish_identity(IdentityRequest {
        engines,
        flow: FlowArtifacts {
            parser_lib: &flags.parser_lib,
            infer_config: infer,
            tracker_config: &flags.tracker_config,
            tracker_library: &flags.tracker_library,
        },
        image_digest: &captured.image,
        batch_size: flags.batch_size,
        destination: &destination,
    })
    .map_err(CommandError::Identity)?;
    Ok(destination)
}

fn built_of<'a>(receipt: &'a EngineReceipt, path: &'a Path) -> BuiltEngine<'a> {
    BuiltEngine { receipt, path }
}
