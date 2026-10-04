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
use super::{
    BuildRequest, BuiltEngine, CpuModelBytes, EngineReceipt, EngineSet, FlowArtifacts,
    IdentityRequest, LiveBuildRequest, build_fp32, build_live_pose,
};
use crate::config::env::Env;
use crate::config::model_bundle::identity::{
    OBSERVER_LIBRARY, deployment_image_digest, hardware_matches,
};
use crate::run::ModelRole;
use crate::run::model_sources::{self, CapturedOnnx};
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
    observer_path: PathBuf,
    input_digests: Vec<(PathBuf, String)>,
}

enum AuxiliaryLayout {
    TensorRt {
        stored: PathBuf,
        bed: PathBuf,
        fall: PathBuf,
        final_stored: PathBuf,
        final_bed: PathBuf,
        final_fall: PathBuf,
    },
    OnnxRuntimeCpu,
}

struct Layout {
    live: PathBuf,
    auxiliary: AuxiliaryLayout,
    served: PathBuf,
    identity: PathBuf,
    final_live: PathBuf,
    final_served: PathBuf,
    parser_lib: PathBuf,
    tracker_config: PathBuf,
    tracker_library: PathBuf,
    observer: PathBuf,
}

impl Layout {
    /// GPU role, staging path and final path; CPU auxiliary models never enter
    /// this iterator. Publication order remains live, stored pose, bed, fall.
    fn engines(&self) -> impl Iterator<Item = (&'static str, &Path, &Path)> {
        let auxiliary = match &self.auxiliary {
            AuxiliaryLayout::TensorRt {
                stored,
                bed,
                fall,
                final_stored,
                final_bed,
                final_fall,
            } => Some([
                ("stored_pose", stored.as_path(), final_stored.as_path()),
                ("bed", bed.as_path(), final_bed.as_path()),
                ("fall", fall.as_path(), final_fall.as_path()),
            ]),
            AuxiliaryLayout::OnnxRuntimeCpu => None,
        };
        std::iter::once(("live_pose", self.live.as_path(), self.final_live.as_path()))
            .chain(auxiliary.into_iter().flatten())
    }

    fn staging(&self) -> impl Iterator<Item = &Path> {
        self.engines()
            .map(|(_, staged, _)| staged)
            .chain(std::iter::once(self.served.as_path()))
    }
}

struct Planned {
    layout: Layout,
    token: String,
}

enum BuiltEngines {
    TensorRt([EngineReceipt; 4]),
    OnnxRuntimeCpu(EngineReceipt),
}

/// Builds or reuses the selected GPU engines and publishes identity last.
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
    let pose = model_sources::capture_image_onnx(&flags.onnx).map_err(|_| CommandError::Source)?;
    let bed =
        model_sources::capture_image_onnx(&flags.bed_onnx).map_err(|_| CommandError::Source)?;
    let fall = model_sources::capture_fall(env).map_err(|_| CommandError::Source)?;
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
    let observer_path = PathBuf::from(OBSERVER_LIBRARY);
    let observer = files::fingerprint_of(&observer_path)?;
    let input_digests = files::capture_inputs(flags)?;
    Ok(Captured {
        pose,
        bed,
        fall,
        served,
        served_output,
        image,
        observer,
        observer_path,
        input_digests,
    })
}
fn build_all(
    captured: &Captured,
    layout: &Layout,
    batch: u32,
) -> Result<BuiltEngines, CommandError> {
    let live = build_live_pose(LiveBuildRequest {
        onnx: &captured.pose.bytes,
        expected_onnx_sha256: &captured.pose.sha256,
        engine: &layout.live,
        image_digest: &captured.image,
        infer_config: &captured.served,
        batch_size: batch,
    })
    .map_err(CommandError::Live)?;
    match &layout.auxiliary {
        AuxiliaryLayout::TensorRt {
            stored, bed, fall, ..
        } => Ok(BuiltEngines::TensorRt([
            live,
            native(
                ModelRole::StoredPose,
                &captured.pose,
                stored,
                &captured.image,
            )?,
            native(ModelRole::Bed, &captured.bed, bed, &captured.image)?,
            native(ModelRole::Fall, &captured.fall, fall, &captured.image)?,
        ])),
        AuxiliaryLayout::OnnxRuntimeCpu => Ok(BuiltEngines::OnnxRuntimeCpu(live)),
    }
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
    built: &BuiltEngines,
    layout: &Layout,
    flags: &crate::cli::EngineBuildFlags,
    captured: &Captured,
) -> Result<PathBuf, CommandError> {
    if captured.served_output {
        files::write_exclusive(&layout.served, captured.served.as_bytes())?;
    }
    let destination = files::identity_staging(&flags.identity)?;
    let engines = engines_of(built, layout, captured)?;
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

fn engines_of<'a>(
    built: &'a BuiltEngines,
    layout: &'a Layout,
    captured: &'a Captured,
) -> Result<EngineSet<'a>, CommandError> {
    match (built, &layout.auxiliary) {
        (
            BuiltEngines::TensorRt(receipts),
            AuxiliaryLayout::TensorRt {
                stored, bed, fall, ..
            },
        ) => Ok(EngineSet::TensorRt {
            live_pose: built_of(&receipts[0], &layout.live),
            stored_pose: built_of(&receipts[1], stored),
            bed: built_of(&receipts[2], bed),
            fall: built_of(&receipts[3], fall),
        }),
        (BuiltEngines::OnnxRuntimeCpu(live), AuxiliaryLayout::OnnxRuntimeCpu) => {
            Ok(EngineSet::OnnxRuntimeCpu {
                live_pose: built_of(live, &layout.live),
                models: CpuModelBytes {
                    stored_pose: &captured.pose.bytes,
                    bed: &captured.bed.bytes,
                    fall: &captured.fall.bytes,
                },
            })
        }
        _ => Err(CommandError::Config),
    }
}

fn built_of<'a>(receipt: &'a EngineReceipt, path: &'a Path) -> BuiltEngine<'a> {
    BuiltEngine { receipt, path }
}
