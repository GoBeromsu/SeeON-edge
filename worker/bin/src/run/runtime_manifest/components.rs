use seeon_worker_runtime::cpu::Threads;

use super::{ManifestError, ModelRuntimeFacts, object, text};
use crate::config::model_bundle::identity::{AuxiliaryRuntime, deployment_image_digest};
use crate::inference::Runtime;
use crate::json::Json;
use crate::records::id::sha256_hex;
use crate::run::{Admitted, AdmittedModels, ModelRole, media_config::MediaAssembly};

pub(super) fn image_digest(reference: &str) -> Result<&str, ManifestError> {
    deployment_image_digest(reference).ok_or(ManifestError::Invalid("worker_image_digest"))
}

pub(super) fn models(
    admitted: &Admitted,
    selected: AuxiliaryRuntime,
    runtimes: ModelRuntimeFacts<'_>,
    fall_enabled: bool,
) -> Result<Json, ManifestError> {
    let fall_use = if fall_enabled {
        "live_score"
    } else {
        "warm_owner"
    };
    let mut components = match (&admitted.models, selected) {
        (AdmittedModels::TensorRt(engines), AuxiliaryRuntime::TensorRt) => {
            if admitted.checked.cpu_model_hashes.is_some() {
                return Err(ManifestError::Contradictory("cpu_model_hashes"));
            }
            tensor_rt(runtimes.fall, "fall_runtime")?;
            tensor_rt(runtimes.bed, "bed_runtime")?;
            tensor_rt(runtimes.stored_pose, "stored_pose_runtime")?;
            [
                ("fall", &engines.fall, fall_use),
                ("bed", &engines.bed, "warm_owner"),
                ("stored_pose", &engines.stored_pose, "warm_owner"),
            ]
            .into_iter()
            .map(|(role, engine, use_kind)| {
                object([
                    ("role", text(role)),
                    ("admitted_engine_sha256", text(engine.digest.to_string())),
                    ("use", text(use_kind)),
                    ("runtime", text("tensorrt")),
                ])
            })
            .collect::<Vec<_>>()
        }
        (AdmittedModels::OnnxRuntimeCpu(models), AuxiliaryRuntime::OnnxRuntimeCpu) => {
            let hashes = admitted
                .checked
                .cpu_model_hashes
                .as_ref()
                .ok_or(ManifestError::Missing("cpu_model_hashes"))?;
            let components = vec![
                cpu_component(
                    ModelRole::Fall,
                    &hashes.fall,
                    &models.fall,
                    runtimes.fall,
                    fall_use,
                )?,
                cpu_component(
                    ModelRole::Bed,
                    &hashes.bed,
                    &models.bed,
                    runtimes.bed,
                    "warm_owner",
                )?,
                cpu_component(
                    ModelRole::StoredPose,
                    &hashes.stored_pose,
                    &models.stored_pose,
                    runtimes.stored_pose,
                    "warm_owner",
                )?,
            ];
            if hashes.fall != admitted.fall.model_version {
                return Err(ManifestError::Contradictory("fall_onnx_sha256"));
            }
            components
        }
        _ => return Err(ManifestError::Contradictory("auxiliary_runtime")),
    };
    let pose = admitted
        .flow
        .identity
        .get("engine_sha256")
        .ok_or(ManifestError::Missing("pose_engine_sha256"))?;
    if !crate::config::is_hex(pose, 64) {
        return Err(ManifestError::Invalid("pose_engine_sha256"));
    }
    components.push(object([
        ("role", text("pose")),
        ("admitted_engine_sha256", text(pose)),
        ("use", text("media_plane_infer")),
        ("runtime", text("tensorrt")),
    ]));
    Ok(Json::Array(components))
}

fn tensor_rt(runtime: &Runtime, field: &'static str) -> Result<(), ManifestError> {
    match runtime {
        Runtime::TensorRt => Ok(()),
        Runtime::OnnxRuntimeCpu(_) => Err(ManifestError::Contradictory(field)),
    }
}

fn cpu_component(
    role: ModelRole,
    hash: &str,
    bytes: &[u8],
    runtime: &Runtime,
    use_kind: &str,
) -> Result<Json, ManifestError> {
    let (role, hash_field, runtime_field) = match role {
        ModelRole::Fall => ("fall", "fall_onnx_sha256", "fall_runtime"),
        ModelRole::Bed => ("bed", "bed_onnx_sha256", "bed_runtime"),
        ModelRole::StoredPose => (
            "stored_pose",
            "stored_pose_onnx_sha256",
            "stored_pose_runtime",
        ),
    };
    if !crate::config::is_hex(hash, 64) || bytes.is_empty() {
        return Err(ManifestError::Invalid(hash_field));
    }
    if sha256_hex(bytes) != hash {
        return Err(ManifestError::Contradictory(hash_field));
    }
    let Runtime::OnnxRuntimeCpu(info) = runtime else {
        return Err(ManifestError::Contradictory(runtime_field));
    };
    if info.abi_version != 1
        || info.input_count != 1
        || !(1..=2).contains(&info.output_count)
        || info.runtime_version.is_empty()
        || info.runtime_version.len() >= 64
        || info
            .runtime_version
            .chars()
            .any(|ch| ch.is_whitespace() || ch.is_control())
    {
        return Err(ManifestError::Invalid(runtime_field));
    }
    // This is the native session's accepted configuration policy, not a
    // measurement of OS workers. Default deliberately carries no thread count.
    let policy = match info.threads {
        Threads::Default => "default",
        Threads::Single => "single",
    };
    Ok(object([
        ("role", text(role)),
        ("admitted_onnx_sha256", text(hash)),
        ("use", text(use_kind)),
        ("runtime", text("onnxruntime")),
        ("provider", text("cpu")),
        ("runtime_version", text(&info.runtime_version)),
        ("thread_policy", text(policy)),
    ]))
}

pub(super) fn media_plan(media: &MediaAssembly) -> Json {
    match media {
        MediaAssembly::Idle => object([("state", text("idle"))]),
        MediaAssembly::Configured(media) => object([
            ("state", text("configured")),
            ("decode_backend", text("nvdec")),
            ("mux_width", Json::Int(i128::from(media.mux_width))),
            ("mux_height", Json::Int(i128::from(media.mux_height))),
            (
                "mux_batch_timeout_us",
                Json::Int(i128::from(media.mux_batch_timeout_us)),
            ),
            ("mux_live_source", Json::Bool(media.mux_live_source)),
            ("tracker_width", Json::Int(i128::from(media.tracker_width))),
            (
                "tracker_height",
                Json::Int(i128::from(media.tracker_height)),
            ),
            (
                "record_cache_seconds",
                Json::Int(i128::from(media.record_cache_seconds)),
            ),
            (
                "rtsp_reconnect_interval_sec",
                Json::Int(i128::from(media.rtsp_reconnect_interval_sec)),
            ),
        ]),
    }
}
