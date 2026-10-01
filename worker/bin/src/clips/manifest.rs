//! Clip manifest (schema version 2) and its delivery entry: one model, two
//! renderings. The bytes are the sorted, compact, ASCII-escaped serialisation
//! the Python `evidence_manifest` writer produces, plus one newline.

use crate::json::{Json, JsonError, Serialiser};

use super::time::Utc;

pub const MANIFEST_SCHEMA_VERSION: i128 = 2;
/// Terminal manifests and entries are the second state of a clip.
pub const TERMINAL_STATE_VERSION: i64 = 2;
pub const MAX_MANIFEST_BYTES: usize = 65_536;
pub const MIME_TYPE: &str = "video/mp4";

/// One event folded into a flow clip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contributor {
    pub event_ref: String,
    pub detected_at: Utc,
}

/// A flow clip's extension block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extension {
    pub boundary: String,
    pub contributors: Vec<Contributor>,
    pub duration_ms: i64,
}

/// Everything a manifest says about a clip that is not its media.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipMetadata {
    pub clip_id: String,
    pub camera_id: String,
    pub facility_id: String,
    pub domain: String,
    pub event_type: String,
    pub event_refs: Vec<String>,
    pub detected_at: Utc,
    pub started_at: Utc,
    pub clip_start_at: Utc,
    pub clip_end_at: Utc,
    pub finalized_at: Utc,
    pub duration_ms: i64,
    pub encoder: String,
    pub truncation_reasons: Vec<String>,
    pub extension: Option<Extension>,
}

/// Facts measured from the published media file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaFacts {
    pub sha256: String,
    pub size_bytes: i64,
    pub codec: String,
    pub duration_ms: i64,
}

/// How a clip ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Terminal {
    Ready(MediaFacts),
    Unavailable {
        reason_code: String,
        source_error_reason: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestError {
    Blank(&'static str),
    TooLarge,
    Json(JsonError),
    MissingEvent,
    SpansCameras,
    SpansFacilities,
    NonPositiveDuration,
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blank(field) => write!(formatter, "manifest field {field} is blank"),
            Self::TooLarge => write!(formatter, "manifest exceeds {MAX_MANIFEST_BYTES} bytes"),
            Self::Json(error) => write!(formatter, "manifest serialisation: {error:?}"),
            Self::MissingEvent => write!(formatter, "flow contributor has no event"),
            Self::SpansCameras => write!(formatter, "flow clip spans cameras"),
            Self::SpansFacilities => write!(formatter, "flow clip spans facilities"),
            Self::NonPositiveDuration => write!(formatter, "flow clip duration must be positive"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// The clip-store-relative media path a READY manifest names.
pub fn media_reference(clip_id: &str) -> String {
    format!("clips/{clip_id}/clip.mp4")
}

fn text(value: &str) -> Json {
    Json::Str(value.to_owned())
}

fn seconds(millis: i64) -> Json {
    Json::Float(millis as f64 / 1000.0)
}

fn texts(values: &[String]) -> Json {
    Json::Array(values.iter().map(|value| text(value)).collect())
}

fn extension_json(extension: &Extension) -> Json {
    let contributors = extension
        .contributors
        .iter()
        .map(|c| {
            Json::Object(vec![
                ("detected_at".to_owned(), text(&c.detected_at.iso_micros())),
                ("event_ref".to_owned(), text(&c.event_ref)),
            ])
        })
        .collect();
    Json::Object(vec![
        ("boundary".to_owned(), text(&extension.boundary)),
        ("contributors".to_owned(), Json::Array(contributors)),
        ("duration_s".to_owned(), seconds(extension.duration_ms)),
    ])
}

fn validate(meta: &ClipMetadata) -> Result<(), ManifestError> {
    let required = [
        ("clip_id", meta.clip_id.as_str()),
        ("camera_id", meta.camera_id.as_str()),
        ("facility_id", meta.facility_id.as_str()),
        ("domain", meta.domain.as_str()),
        ("event_type", meta.event_type.as_str()),
        ("encoder", meta.encoder.as_str()),
    ];
    for (field, value) in required {
        if value.trim().is_empty() {
            return Err(ManifestError::Blank(field));
        }
    }
    match meta.event_refs.first() {
        Some(first) if !first.trim().is_empty() => Ok(()),
        _ => Err(ManifestError::Blank("event_refs")),
    }
}

/// The manifest as a JSON model; key order is irrelevant (the serialiser sorts).
pub fn manifest_json(meta: &ClipMetadata, terminal: &Terminal) -> Result<Json, ManifestError> {
    validate(meta)?;
    let mut fields = vec![
        ("camera_id", text(&meta.camera_id)),
        ("clip_end_at", text(&meta.clip_end_at.iso_millis())),
        ("clip_id", text(&meta.clip_id)),
        ("clip_start_at", text(&meta.clip_start_at.iso_millis())),
        ("detected_at", text(&meta.detected_at.iso_micros())),
        ("domain", text(&meta.domain)),
        ("duration_s", seconds(meta.duration_ms)),
        ("encoder", text(&meta.encoder)),
        ("event_ref", text(&meta.event_refs[0])),
        ("event_refs", texts(&meta.event_refs)),
        ("event_type", text(&meta.event_type)),
        ("finalized", Json::Bool(true)),
        ("finalized_at", text(&meta.finalized_at.iso_millis())),
        (
            "manifest_schema_version",
            Json::Int(MANIFEST_SCHEMA_VERSION),
        ),
        ("started_at", text(&meta.started_at.iso_micros())),
        (
            "state_version",
            Json::Int(i128::from(TERMINAL_STATE_VERSION)),
        ),
        ("truncation_reasons", texts(&meta.truncation_reasons)),
    ];
    if let Some(extension) = &meta.extension {
        fields.push(("extension", extension_json(extension)));
    }
    match terminal {
        Terminal::Ready(media) => fields.extend([
            ("codec", text(&media.codec)),
            ("duration_ms", Json::Int(i128::from(media.duration_ms))),
            ("mime_type", text(MIME_TYPE)),
            ("path", text(&media_reference(&meta.clip_id))),
            ("recovery_state", text("MEDIA_VERIFIED")),
            ("sha256", text(&media.sha256)),
            ("size_bytes", Json::Int(i128::from(media.size_bytes))),
            ("state", text("READY")),
            ("video_available", Json::Bool(true)),
        ]),
        Terminal::Unavailable {
            reason_code,
            source_error_reason,
        } => {
            if reason_code.trim().is_empty() {
                return Err(ManifestError::Blank("reason_code"));
            }
            fields.extend([
                ("path", Json::Null),
                ("reason_code", text(reason_code)),
                ("recovery_state", text("UNAVAILABLE")),
                ("state", text("UNAVAILABLE")),
                ("video_available", Json::Bool(false)),
            ]);
            if let Some(reason) = source_error_reason {
                fields.push(("source_error_reason", text(reason)));
            }
        }
    }
    Ok(Json::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    ))
}

/// The exact bytes written to `manifest.json`.
pub fn manifest_bytes(meta: &ClipMetadata, terminal: &Terminal) -> Result<Vec<u8>, ManifestError> {
    let model = manifest_json(meta, terminal)?;
    let mut bytes = Serialiser::ModelSelection
        .canonical(&model)
        .map_err(ManifestError::Json)?
        .into_bytes();
    bytes.push(b'\n');
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(ManifestError::TooLarge);
    }
    Ok(bytes)
}
