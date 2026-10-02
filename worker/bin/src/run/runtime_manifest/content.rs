use seeon_deepstream_native::RuntimeVersions;

use super::{ManifestError, SCHEMA_VERSION, components, object, text};
use crate::config::build_revision::BuildRevisionError;
use crate::config::pull::PulledConfig;
use crate::json::Json;
use crate::relay::cameras::WorkerConfigPayload;
use crate::run::cameras::{
    CameraPolicyError, PolicyNumberSource, admitted_fall, resolve_fall_policy,
};
use crate::run::media_config::MediaAssembly;
use crate::run::{Admitted, Settings};
use crate::telemetry::gpu::GpuStatus;

fn required<'a>(value: Option<&'a str>, field: &'static str) -> Result<&'a str, ManifestError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or(ManifestError::Missing(field))
}
fn count(value: usize) -> Result<Json, ManifestError> {
    i128::try_from(value)
        .map(Json::Int)
        .map_err(|_| ManifestError::Invalid("integer"))
}
fn fall_policy(
    config: &PulledConfig,
    camera_id: &str,
    admitted: &Admitted,
    enabled: bool,
) -> Result<(Json, Json), ManifestError> {
    if !enabled {
        return Ok((Json::Null, Json::Null));
    }
    let selected = admitted_fall(&config.policies, camera_id).map_err(|error| match error {
        CameraPolicyError::PolicyMissing => ManifestError::Missing("fall_policy"),
        _ => ManifestError::Invalid("fall_policy"),
    })?;
    let applied = resolve_fall_policy(&config.policies, camera_id, &admitted.fall.calibration)
        .map_err(|_| ManifestError::Invalid("fall_policy"))?;
    Ok((
        selected.as_json(),
        object([
            (
                "transition_threshold",
                Json::Float(applied.transition_threshold),
            ),
            ("threshold_source", number_source(applied.threshold_source)),
            ("transition_votes", count(applied.transition_votes)?),
            ("transition_window", count(applied.transition_window)?),
            (
                "confirmation_rule_source",
                number_source(applied.confirmation_rule_source),
            ),
            ("temperature", Json::Float(applied.temperature)),
        ]),
    ))
}

fn effective_fall(config: &WorkerConfigPayload) -> Result<bool, ManifestError> {
    config
        .domain_selection()
        .resolve()
        .map(|domains| domains.fall)
        .map_err(|_| ManifestError::Invalid("domain_selection"))
}
fn number_source(value: PolicyNumberSource) -> Json {
    text(match value {
        PolicyNumberSource::Receipt => "receipt",
        PolicyNumberSource::Default => "default",
    })
}

pub(super) fn build(
    settings: &Settings,
    admitted: &Admitted,
    gpu: &GpuStatus,
    config: &PulledConfig,
    media: &MediaAssembly,
    versions: RuntimeVersions,
) -> Result<Json, ManifestError> {
    let revision = settings
        .build_revision()
        .map_err(|error| match error {
            BuildRevisionError::Invalid => ManifestError::Invalid("worker_build_revision"),
            BuildRevisionError::Mismatch => ManifestError::Contradictory("worker_build_revision"),
        })?
        .ok_or(ManifestError::Missing("worker_build_revision"))?;
    let image = required(
        admitted
            .flow
            .identity
            .get("image_digest")
            .map(String::as_str),
        "worker_image_digest",
    )?;
    let image = components::image_digest(image)?;
    let driver = required(gpu.driver_version.as_deref(), "driver_version")?;
    let device = required(gpu.device_name.as_deref(), "device_name")?;
    if !gpu.cuda_context_ok || versions.trt_version <= 0 || versions.cuda_runtime_version <= 0 {
        return Err(ManifestError::Invalid("accelerator_runtime"));
    }
    let directive = config.directive;
    if directive.version < 0 || directive.generation < 0 || directive.registry < 0 {
        return Err(ManifestError::Invalid("configuration_revision"));
    }
    let sources = match media {
        MediaAssembly::Idle if config.cameras.is_empty() => &[][..],
        MediaAssembly::Configured(media) if media.sources.len() == config.cameras.len() => {
            &media.sources[..]
        }
        _ => return Err(ManifestError::Contradictory("media_roster")),
    };
    let fall_enabled = effective_fall(&config.config)?;
    let mut cameras = Vec::new();
    for (index, camera) in config.cameras.iter().enumerate() {
        let source = &sources[index];
        if usize::try_from(source.source_id).ok() != Some(index) {
            return Err(ManifestError::Contradictory("media_source"));
        }
        let (declared_policy, applied_policy) =
            fall_policy(config, &camera.camera_id, admitted, fall_enabled)?;
        let content = object([
            ("camera_id", text(&camera.camera_id)),
            ("facility_id", text(&camera.facility_id)),
            ("source_id", Json::Int(i128::from(source.source_id))),
            (
                "source_generation",
                Json::Int(i128::from(source.binding.generation)),
            ),
            ("stream_epoch", Json::Int(i128::from(source.binding.epoch))),
            ("module_qualified_id", text("fall.v2")),
            ("enabled", Json::Bool(fall_enabled)),
            ("declared_policy", declared_policy),
            ("applied_policy", applied_policy),
        ]);
        cameras.push((&camera.camera_id, content));
    }
    cameras.sort_by(|left, right| left.0.cmp(right.0));
    let components = components::engines(admitted, fall_enabled)?;
    let fall = &admitted.fall;
    let bundle = match &admitted.checked.selection {
        Some((_, proof)) => object([
            ("authority", text("selected")),
            ("bundle_sha256", text(&proof.bundle_sha256)),
            (
                "verified_identities",
                Json::Object(
                    proof
                        .identities
                        .iter()
                        .map(|(key, value)| (key.clone(), text(value)))
                        .collect(),
                ),
            ),
        ]),
        None => object([("authority", text("packaged"))]),
    };
    let media_plan = components::media_plan(media);
    Ok(object([
        ("manifest_schema_version", Json::Int(SCHEMA_VERSION)),
        (
            "build",
            object([
                ("worker_build_revision", text(revision)),
                ("worker_image_digest", text(image)),
                (
                    "edge_database_schema_version",
                    Json::Int(crate::config::pull::EDGE_DATABASE_SCHEMA_VERSION),
                ),
                ("implementation_language", text("rust")),
                ("package_version", text(env!("CARGO_PKG_VERSION"))),
                ("os_name", text(std::env::consts::OS)),
                ("architecture", text(std::env::consts::ARCH)),
                ("inference_runtime", text("tensorrt")),
                (
                    "inference_runtime_version_encoded",
                    Json::Int(i128::from(versions.trt_version)),
                ),
                (
                    "cuda_runtime_version_encoded",
                    Json::Int(i128::from(versions.cuda_runtime_version)),
                ),
                ("driver_version", text(driver)),
                ("device_name", text(device)),
            ]),
        ),
        (
            "configuration",
            object([
                ("config_version", Json::Int(directive.version)),
                ("restart_generation", Json::Int(directive.generation)),
                ("registry_version", Json::Int(directive.registry)),
                (
                    "pulled_config_sha256",
                    text(
                        crate::run::config_digest::config_digest(&config.payload)
                            .map_err(|_| ManifestError::Invalid("config_digest"))?,
                    ),
                ),
            ]),
        ),
        ("profile", text("flow")),
        ("media_plan", media_plan),
        ("components", components),
        (
            "cameras",
            Json::Array(cameras.into_iter().map(|(_, value)| value).collect()),
        ),
        ("bundle", bundle),
        (
            "fall_evidence",
            object([
                ("published_onnx_sha256", text(&fall.model_version)),
                (
                    "published_weights_sha256",
                    text(&fall.published_weights_digest),
                ),
                ("calibration_sha256", text(&fall.calibration_digest)),
                ("preprocessing_identity", text(&fall.preprocessing_identity)),
            ]),
        ),
    ]))
}
