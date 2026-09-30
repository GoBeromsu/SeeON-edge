//! `shared/events/replay_wire.py`: the replay trace body a GET `/replay`
//! answer carries, `{camera_id, frames, truncation}` encoded by
//! `canonical_json` (sorted keys, `,`/`:` separators, `allow_nan=False`).
//! Frames are the trace store's frame dicts, carried as JSON objects; the
//! truncation mirrors `worker/pipeline/trace/models.py` `TraceTruncation`.

use std::fmt;

use crate::json::{Json, JsonError, Serialiser};

/// `canonical_json`: `json.dumps(sort_keys=True, separators=(",", ":"),
/// allow_nan=False)`, ASCII escapes included.
const CANONICAL: Serialiser = Serialiser::ModelSelection;

/// `FrameKey`: `(worker_boot_id, camera_id, stream_epoch, seq)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameKey {
    pub worker_boot_id: String,
    pub camera_id: String,
    pub stream_epoch: u64,
    pub seq: u64,
}

impl FrameKey {
    fn to_json(&self) -> Json {
        Json::Array(vec![
            Json::Str(self.worker_boot_id.clone()),
            Json::Str(self.camera_id.clone()),
            uint(self.stream_epoch),
            uint(self.seq),
        ])
    }
}

/// `TraceTruncation`: the nine fields the wire requires.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Truncation {
    pub handoff_dropped_frames: u64,
    pub pruned_frames: u64,
    pub oldest_retained_seq: Option<u64>,
    pub newest_retained_seq: Option<u64>,
    pub persistence_failed_frames: u64,
    pub retention_blocked_frames: u64,
    pub oldest_retained_key: Option<FrameKey>,
    pub newest_retained_key: Option<FrameKey>,
    pub detail_unavailable_reason: Option<String>,
}

impl Truncation {
    fn to_json(&self) -> Json {
        let seq = |value: Option<u64>| value.map_or(Json::Null, uint);
        let key = |value: &Option<FrameKey>| value.as_ref().map_or(Json::Null, FrameKey::to_json);
        let reason = self
            .detail_unavailable_reason
            .as_ref()
            .map_or(Json::Null, |text| Json::Str(text.clone()));
        Json::Object(vec![
            member("handoff_dropped_frames", uint(self.handoff_dropped_frames)),
            member("pruned_frames", uint(self.pruned_frames)),
            member("oldest_retained_seq", seq(self.oldest_retained_seq)),
            member("newest_retained_seq", seq(self.newest_retained_seq)),
            member(
                "persistence_failed_frames",
                uint(self.persistence_failed_frames),
            ),
            member(
                "retention_blocked_frames",
                uint(self.retention_blocked_frames),
            ),
            member("oldest_retained_key", key(&self.oldest_retained_key)),
            member("newest_retained_key", key(&self.newest_retained_key)),
            member("detail_unavailable_reason", reason),
        ])
    }
}

/// `ReplayTrace` on the wire.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayWire {
    pub camera_id: String,
    pub frames: Vec<Json>,
    pub truncation: Truncation,
}

/// `ReplayWireError` causes; no value text is carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WireError {
    EmptyCamera,
    NoFrames,
    /// A frame is not a JSON object.
    FrameNotObject,
    /// Non-finite, duplicate-key or otherwise unencodable content.
    Json(JsonError),
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyCamera => "replay camera_id is required",
            Self::NoFrames => "replay frames are required",
            Self::FrameNotObject => "replay frames must be objects",
            Self::Json(_) => "replay trace does not encode",
        })
    }
}
impl std::error::Error for WireError {}

impl ReplayWire {
    /// `ReplayTrace.__post_init__` plus the frame shape `decode_replay_trace`
    /// requires.
    pub fn validate(&self) -> Result<(), WireError> {
        if self.camera_id.is_empty() {
            return Err(WireError::EmptyCamera);
        }
        if self.frames.is_empty() {
            return Err(WireError::NoFrames);
        }
        if self
            .frames
            .iter()
            .any(|frame| !matches!(frame, Json::Object(_)))
        {
            return Err(WireError::FrameNotObject);
        }
        Ok(())
    }

    /// `ReplayTrace.as_dict()`.
    pub fn to_json(&self) -> Json {
        Json::Object(vec![
            member("camera_id", Json::Str(self.camera_id.clone())),
            member("frames", Json::Array(self.frames.clone())),
            member("truncation", self.truncation.to_json()),
        ])
    }

    /// `ReplayTrace.canonical_json()`; a NaN or infinity anywhere is refused.
    pub fn canonical_json(&self) -> Result<String, WireError> {
        self.validate()?;
        CANONICAL
            .canonical(&self.to_json())
            .map_err(WireError::Json)
    }
}

fn member(key: &str, value: Json) -> (String, Json) {
    (key.to_owned(), value)
}

fn uint(value: u64) -> Json {
    Json::Int(i128::from(value))
}
