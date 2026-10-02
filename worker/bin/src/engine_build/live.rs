//! Offline live-pose engine build. A receipt exists only after exclusive
//! publication of one unambiguous regular engine, an exit-zero profile, and
//! one complete V3 observation. Failure keeps the owned scratch and never
//! deletes or overwrites an existing target.

use std::path::Path;

use serde_json::{Value, json};

use super::{EngineReceipt, engine_basename, output_digest};
use crate::config::model_bundle::identity::deployment_image_digest;
use crate::config::model_bundle::onnx_shape::{Dim, input_dims_bytes};
use crate::records::id::sha256_hex;
const MAX_ONNX_BYTES: usize = 512 * 1024 * 1024;

mod observation;
mod process;
use observation::{Observation, Profile, library_identity, observation_line, profile};
use process::{Limits, ProcessOutput, Scratch, Staged, copy_exclusive, stage_files as stage};

/// Borrowed inputs for one exclusive live-pose build. `image_digest` is the
/// caller-declared deployment authority, not a measured container identity.
#[derive(Clone, Copy)]
pub struct LiveBuildRequest<'a> {
    pub onnx: &'a [u8],
    pub expected_onnx_sha256: &'a str,
    pub engine: &'a Path,
    pub image_digest: &'a str,
    pub infer_config: &'a str,
    pub batch_size: u32,
}

/// Why a live receipt was not produced. Owned scratch and logs may remain.
#[derive(Debug)]
pub enum LiveBuildError {
    SourceDigest,
    ImageDigest,
    BatchSize,
    Path,
    Config,
    OnnxShape,
    TargetExists,
    Observer,
    Process,
    Profile,
    Output,
}

impl std::fmt::Display for LiveBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::SourceDigest => "captured ONNX digest does not match the expected sha256",
            Self::ImageDigest => "deployment image digest is invalid",
            Self::BatchSize => "live pose batch size must be 1..=16",
            Self::Path => "live pose engine path is not one private basename",
            Self::Config => "infer config requires one property section with one engine and batch",
            Self::OnnxShape => "captured ONNX input shape is not a pose profile",
            Self::TargetExists => "live pose refuses an existing engine target",
            Self::Observer => "image-owned nvdsinfer observer does not match the SDK library",
            Self::Process => "owned nvinfer build process failed",
            Self::Profile => "nvinfer profile or observation does not match the live pose build",
            Self::Output => "built engine could not be published as a bounded regular file",
        })
    }
}

impl std::error::Error for LiveBuildError {}

struct Prepared<'a> {
    request: LiveBuildRequest<'a>,
    onnx_sha256: String,
    image_digest: String,
    engine_name: String,
    library: observation::LibraryIdentity,
}

/// Validates captured bytes, image, config, and shape before any effect.
pub fn build_live_pose(request: LiveBuildRequest<'_>) -> Result<EngineReceipt, LiveBuildError> {
    build_prepared(&prepare(request)?, Limits::production())
}

fn checked(request: LiveBuildRequest<'_>) -> Result<(String, String, String), LiveBuildError> {
    if !valid_onnx_length(request.onnx.len()) {
        return Err(LiveBuildError::OnnxShape);
    }
    if !crate::config::is_hex(request.expected_onnx_sha256, 64) {
        return Err(LiveBuildError::SourceDigest);
    }
    let onnx_sha256 = sha256_hex(request.onnx);
    if request.expected_onnx_sha256 != onnx_sha256 {
        return Err(LiveBuildError::SourceDigest);
    }
    let image_digest = deployment_image_digest(request.image_digest)
        .map(str::to_owned)
        .ok_or(LiveBuildError::ImageDigest)?;
    if !(1..=16).contains(&request.batch_size) {
        return Err(LiveBuildError::BatchSize);
    }
    let engine_name = engine_basename(request.engine)
        .map_err(|_| LiveBuildError::Path)?
        .to_owned();
    if request
        .engine
        .to_str()
        .is_none_or(|text| text.bytes().any(|byte| byte.is_ascii_control()))
    {
        return Err(LiveBuildError::Path);
    }
    process::render_infer_config(
        request.infer_config,
        "model.onnx",
        "child.engine",
        request.batch_size,
    )?;
    validate_pose_shape(request)?;
    require_absent_target(request.engine)?;
    Ok((onnx_sha256, image_digest, engine_name))
}
fn prepare(request: LiveBuildRequest<'_>) -> Result<Prepared<'_>, LiveBuildError> {
    let (onnx_sha256, image_digest, engine_name) = checked(request)?;
    Ok(Prepared {
        request,
        onnx_sha256,
        image_digest,
        engine_name,
        library: library_identity()?,
    })
}

fn validate_pose_shape(request: LiveBuildRequest<'_>) -> Result<(), LiveBuildError> {
    let dims =
        input_dims_bytes(request.onnx, request.engine).map_err(|_| LiveBuildError::OnnxShape)?;
    let [batch, channels, height, width] = dims.as_slice() else {
        return Err(LiveBuildError::OnnxShape);
    };
    for (dim, expected) in [(channels, 3), (height, 640), (width, 640)] {
        if matches!(dim, Dim::Value(value) if *value != expected) {
            return Err(LiveBuildError::OnnxShape);
        }
    }
    let fixed = matches!(batch, Dim::Value(value) if *value != i64::from(request.batch_size));
    if fixed || (request.batch_size > 1 && matches!(batch, Dim::Value(_))) {
        return Err(LiveBuildError::OnnxShape);
    }
    Ok(())
}

fn valid_onnx_length(length: usize) -> bool {
    (1..=MAX_ONNX_BYTES).contains(&length)
}

fn require_absent_target(path: &Path) -> Result<(), LiveBuildError> {
    match rustix::fs::lstat(path) {
        Ok(_) => Err(LiveBuildError::TargetExists),
        Err(rustix::io::Errno::NOENT) => Ok(()),
        Err(_) => Err(LiveBuildError::Path),
    }
}

fn build_prepared(
    prepared: &Prepared<'_>,
    limits: Limits,
) -> Result<EngineReceipt, LiveBuildError> {
    let parent = prepared
        .request
        .engine
        .parent()
        .ok_or(LiveBuildError::Path)?;
    let scratch = Scratch::create(parent)?;
    let built = stage_and_finish(&scratch, prepared, limits);
    if let Err(error) = &built {
        let _ = scratch.record(error.to_string().as_bytes());
    }
    built
}

fn stage_and_finish(
    scratch: &Scratch,
    prepared: &Prepared<'_>,
    limits: Limits,
) -> Result<EngineReceipt, LiveBuildError> {
    let staged = stage(scratch, prepared)?;
    let output = process::run_bounded(&staged.argv, scratch, limits, &prepared.library)?;
    finish(prepared, &staged, output)
}

fn finish(
    prepared: &Prepared<'_>,
    staged: &Staged,
    output: ProcessOutput,
) -> Result<EngineReceipt, LiveBuildError> {
    if output.status != 0 {
        return Err(LiveBuildError::Process);
    }
    let parsed = profile(&output.merged, prepared.request.batch_size)?;
    let observed = observation_line(&output.merged)?;
    if !observed.fp16 {
        return Err(LiveBuildError::Profile);
    }
    let source = adopted_source(staged)?;
    let engine_sha256 = publish(prepared.request.engine, source)?;
    Ok(EngineReceipt {
        document: receipt(prepared, &parsed, &observed, engine_sha256),
    })
}

fn adopted_source(staged: &Staged) -> Result<&Path, LiveBuildError> {
    let candidates = [staged.child_engine.as_path(), staged.generated.as_path()];
    let mut source = None;
    for candidate in candidates {
        match rustix::fs::lstat(candidate) {
            Err(rustix::io::Errno::NOENT) => {}
            Err(_) => return Err(LiveBuildError::Output),
            Ok(_) if !process::regular_engine(candidate) => return Err(LiveBuildError::Output),
            Ok(_) if source.replace(candidate).is_some() => return Err(LiveBuildError::Output),
            Ok(_) => {}
        }
    }
    source.ok_or(LiveBuildError::Output)
}

fn publish(target: &Path, source: &Path) -> Result<String, LiveBuildError> {
    copy_exclusive(source, target).map_err(|_| LiveBuildError::Output)?;
    output_digest(target).map_err(|_| LiveBuildError::Output)
}

fn receipt(
    prepared: &Prepared<'_>,
    parsed: &Profile,
    observed: &Observation,
    engine_sha256: String,
) -> Value {
    json!({
        "engine": prepared.engine_name,
        "onnx_sha256": prepared.onnx_sha256,
        "engine_sha256": engine_sha256,
        "precision": "fp16",
        "tf32_enabled": observed.tf32,
        "trt_version": observed.trt,
        "device_name": observed.device_name,
        "compute_capability": format!("{}.{}", observed.sm_major, observed.sm_minor),
        "device": observed.device,
        "input": "images",
        "image_digest": prepared.image_digest,
        "min_dimensions": parsed.min,
        "opt_dimensions": parsed.opt,
        "max_dimensions": parsed.max,
        "observer_library_sha256": prepared.library.sha256,
    })
}
#[cfg(test)]
mod cpu_tests {
    use std::fs;
    use std::path::Path;

    use super::observation::{observation_line, profile};
    use super::process::{gst_argv, render_infer_config};
    use super::{LiveBuildError, LiveBuildRequest, checked};
    use crate::records::id::sha256_hex;
    use crate::seam::{IdSource, RandomIds};

    const IMAGE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const TEMPLATE: &str = "[property]\nonnx-file=old.onnx\nmodel-engine-file=old.engine\nbatch-size=9\nnetwork-mode=2\n";
    // Minimal first-input shape, not an executable ONNX graph:
    // Model.graph -> Graph.input -> ValueInfo.type -> tensor_type -> shape.
    const POSE: &[u8] = &[
        0x3a, 27, 0x5a, 25, 0x12, 23, 0x0a, 21, 0x12, 19, 0x0a, 3, 0x12, 1, b'N', 0x0a, 2, 0x08, 3,
        0x0a, 3, 0x08, 0x80, 0x05, 0x0a, 3, 0x08, 0x80, 0x05,
    ];

    fn scratch() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            ".live-pose-test-{}",
            RandomIds.uuid4().expect("id")
        ));
        fs::create_dir(&path).expect("exclusive scratch");
        path
    }

    fn request<'a>(
        engine: &'a Path,
        onnx: &'a [u8],
        expected: &'a str,
        batch: u32,
    ) -> LiveBuildRequest<'a> {
        LiveBuildRequest {
            onnx,
            expected_onnx_sha256: expected,
            engine,
            image_digest: IMAGE,
            infer_config: TEMPLATE,
            batch_size: batch,
        }
    }

    #[test]
    fn refusals_stop_before_effects() {
        let dir = scratch();
        let engine = dir.join("pose.engine");
        let expected = sha256_hex(POSE);
        assert!(checked(request(&engine, POSE, &expected, 1)).is_ok());
        assert!(checked(request(&engine, POSE, &expected, 16)).is_ok());
        assert!(matches!(
            checked(request(&engine, POSE, "abcd", 1)),
            Err(LiveBuildError::SourceDigest)
        ));
        assert!(matches!(
            super::build_live_pose(request(&engine, POSE, &"0".repeat(64), 1)),
            Err(LiveBuildError::SourceDigest)
        ));
        let mut bad_image = request(&engine, POSE, &expected, 1);
        bad_image.image_digest = "sha256:abcd";
        assert!(matches!(
            checked(bad_image),
            Err(LiveBuildError::ImageDigest)
        ));
        assert!(matches!(
            checked(request(&engine, POSE, &expected, 0)),
            Err(LiveBuildError::BatchSize)
        ));
        assert!(matches!(
            checked(request(&engine, POSE, &expected, 17)),
            Err(LiveBuildError::BatchSize)
        ));
        let mut bad_config = request(&engine, POSE, &expected, 1);
        bad_config.infer_config =
            "[property]\nonnx-file=a\nonnx-file=b\nmodel-engine-file=c\nbatch-size=1\n";
        assert!(matches!(checked(bad_config), Err(LiveBuildError::Config)));
        fs::write(&engine, b"already").unwrap();
        assert!(matches!(
            checked(request(&engine, POSE, &expected, 1)),
            Err(LiveBuildError::TargetExists)
        ));
        assert_eq!(fs::read(&engine).unwrap(), b"already");
        assert!(!dir.read_dir().unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".live-pose-")
        }));
        fs::remove_file(&engine).unwrap();
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn captured_onnx_length_is_inclusive_without_large_allocations() {
        assert!(!super::valid_onnx_length(0));
        assert!(super::valid_onnx_length(1));
        assert!(super::valid_onnx_length(super::MAX_ONNX_BYTES));
        assert!(!super::valid_onnx_length(super::MAX_ONNX_BYTES + 1));
    }

    #[test]
    fn dynamic_spatial_axes_reach_the_real_profile_gate() {
        let dir = scratch();
        let engine = dir.join("pose.engine");
        let mut dynamic = POSE.to_vec();
        for symbol in [b'H', b'W'] {
            let index = dynamic
                .windows(5)
                .position(|bytes| bytes == [0x0a, 3, 0x08, 0x80, 0x05])
                .unwrap();
            dynamic[index + 2..index + 5].copy_from_slice(&[0x12, 1, symbol]);
        }
        let expected = sha256_hex(&dynamic);
        assert!(checked(request(&engine, &dynamic, &expected, 2)).is_ok());
        let mut wrong = POSE.to_vec();
        let index = wrong
            .windows(5)
            .position(|bytes| bytes == [0x0a, 3, 0x08, 0x80, 0x05])
            .unwrap();
        wrong[index + 3] = 0x81; // A concrete 641 contradicts the served 640.
        let expected = sha256_hex(&wrong);
        assert!(matches!(
            checked(request(&engine, &wrong, &expected, 2)),
            Err(LiveBuildError::OnnxShape)
        ));
        fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn argv_is_exact_and_profile_names_are_required() {
        let one = gst_argv("/tmp/private.txt", 1);
        assert_eq!(
            one,
            [
                "gst-launch-1.0",
                "-e",
                "nvstreammux",
                "name=mux",
                "batch-size=1",
                "width=640",
                "height=640",
                "live-source=0",
                "batched-push-timeout=40000",
                "!",
                "nvinfer",
                "config-file-path=/tmp/private.txt",
                "!",
                "fakesink",
                "sync=false",
                "videotestsrc",
                "num-buffers=1",
                "pattern=black",
                "!",
                "video/x-raw,format=I420,width=640,height=640,framerate=30/1",
                "!",
                "nvvideoconvert",
                "!",
                "video/x-raw(memory:NVMM),format=NV12",
                "!",
                "mux.sink_0",
            ]
        );
        let two = gst_argv("/tmp/private.txt", 2);
        assert_eq!(two.len(), one.len() + 11);
        assert!(two.ends_with(&["mux.sink_1".to_owned()]));
        let rendered = render_infer_config(TEMPLATE, "model.onnx", "child.engine", 2).unwrap();
        assert_eq!(rendered.matches("onnx-file=").count(), 1);
        assert!(rendered.contains("network-mode=2\n"));
        let v3 = "INFO: SEEON_BUILD_OBSERVATION_V3 flags=65 fp16=1 tf32=1 trt=101600 device=0 sm_major=12 sm_minor=0 device_name=NVIDIA GeForce RTX 5070 Ti";
        let good = "INFO: SEEON_BUILD_PROFILE_V1 profiles=1 input_float=1 output_float=1 input_dims=-1x3x640x640 min=1x3x640x640 opt=1x3x640x640 max=1x3x640x640 output_dims=-1x300x57";
        let log = format!("{v3}\n{good}");
        assert!(observation_line(&log).unwrap().fp16);
        assert!(profile(&log, 1).is_ok());
        assert!(
            observation_line(&format!(
                "{v3}\nINFO: SEEON_BUILD_OBSERVATION_V2 flags=1\n{good}"
            ))
            .is_err()
        );
        assert!(
            observation_line(&format!(
                "INFO: SEEON_BUILD_OBSERVATION_V3 unavailable\n{v3}\n{good}"
            ))
            .is_err()
        );
        assert!(profile(&log.replace("input_float=1", "input_float=0"), 1).is_err());
        assert!(
            profile(
                &log.replace("output_dims=-1x300x57", "output_dims=-1x301x57"),
                1
            )
            .is_err()
        );
    }
}
