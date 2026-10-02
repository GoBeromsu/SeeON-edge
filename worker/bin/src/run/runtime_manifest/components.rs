use super::{ManifestError, object, text};
use crate::config::model_bundle::identity::deployment_image_digest;
use crate::json::Json;
use crate::run::{Admitted, media_config::MediaAssembly};

pub(super) fn image_digest(reference: &str) -> Result<&str, ManifestError> {
    deployment_image_digest(reference).ok_or(ManifestError::Invalid("worker_image_digest"))
}

pub(super) fn engines(admitted: &Admitted, fall_enabled: bool) -> Result<Json, ManifestError> {
    let engines = &admitted.engines;
    let fall_use = if fall_enabled {
        "live_score"
    } else {
        "warm_owner"
    };
    let mut components: Vec<_> = [
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
    .collect();
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
