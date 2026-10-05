//! The sidecar payload (`persist`, `_event_payload`).

use super::{SealedClip, SealedContributor, SealedEvent, SealedUnavailable};
use crate::json::Json;

pub(super) fn sidecar_json(
    sealed: &SealedClip,
    ordered: &[&SealedContributor],
    events: &[&SealedEvent],
) -> Json {
    let text = |value: &str| Json::Str(value.to_owned());
    let contributors = ordered
        .iter()
        .map(|contributor| {
            Json::Object(vec![
                ("detected_at".to_owned(), text(&contributor.detected_at)),
                ("event_ref".to_owned(), text(&contributor.event_ref)),
            ])
        })
        .collect();
    let event_values = events
        .iter()
        .map(|event| {
            Json::Object(vec![
                ("domain".to_owned(), text(&event.domain)),
                ("event_type".to_owned(), text(&event.event_type)),
                ("identity".to_owned(), text(&event.identity)),
                ("camera_id".to_owned(), text(&event.camera_id)),
                ("facility_id".to_owned(), text(&event.facility_id)),
                ("time_sec".to_owned(), Json::Float(event.time_sec)),
                (
                    "probability".to_owned(),
                    match event.probability {
                        None => Json::Null,
                        Some(probability) => Json::Float(probability),
                    },
                ),
            ])
        })
        .collect();
    let camera_id = events.first().map_or("", |event| event.camera_id.as_str());
    Json::Object(vec![
        ("clip_id".to_owned(), text(&sealed.clip_id)),
        ("path".to_owned(), text(&sealed.path)),
        (
            "duration_ms".to_owned(),
            Json::Int(i128::from(sealed.duration_ms)),
        ),
        ("camera_id".to_owned(), text(camera_id)),
        ("boundary".to_owned(), text(&sealed.boundary)),
        ("contributors".to_owned(), Json::Array(contributors)),
        ("events".to_owned(), Json::Array(event_values)),
    ])
}

pub(super) fn unavailable_sidecar_json(
    unavailable: &SealedUnavailable,
    camera_id: &str,
    ordered: &[&SealedContributor],
    events: &[&SealedEvent],
) -> Json {
    let text = |value: &str| Json::Str(value.to_owned());
    let contributors = ordered
        .iter()
        .map(|contributor| {
            Json::Object(vec![
                ("detected_at".to_owned(), text(&contributor.detected_at)),
                ("event_ref".to_owned(), text(&contributor.event_ref)),
            ])
        })
        .collect();
    let event_values = events
        .iter()
        .map(|event| {
            Json::Object(vec![
                ("domain".to_owned(), text(&event.domain)),
                ("event_type".to_owned(), text(&event.event_type)),
                ("identity".to_owned(), text(&event.identity)),
                ("camera_id".to_owned(), text(&event.camera_id)),
                ("facility_id".to_owned(), text(&event.facility_id)),
                ("time_sec".to_owned(), Json::Float(event.time_sec)),
                (
                    "probability".to_owned(),
                    match event.probability {
                        None => Json::Null,
                        Some(probability) => Json::Float(probability),
                    },
                ),
            ])
        })
        .collect();
    let mut unavailable_fields = vec![
        ("clip_id".to_owned(), text(&unavailable.clip_id)),
        ("camera_id".to_owned(), text(camera_id)),
        (
            "duration_ms".to_owned(),
            Json::Int(i128::from(unavailable.duration_ms)),
        ),
        ("boundary".to_owned(), text(&unavailable.boundary)),
        ("contributors".to_owned(), Json::Array(contributors)),
        ("events".to_owned(), Json::Array(event_values)),
        (
            "native_result".to_owned(),
            Json::Int(i128::from(unavailable.native_result)),
        ),
        (
            "contains_video".to_owned(),
            Json::Bool(unavailable.contains_video),
        ),
    ];
    if let Some(path) = &unavailable.path {
        unavailable_fields.push(("path".to_owned(), text(path)));
    }
    Json::Object(vec![(
        "unavailable".to_owned(),
        Json::Object(unavailable_fields),
    )])
}
