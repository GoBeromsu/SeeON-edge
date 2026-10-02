//! `contracts/replay_trace.py`: one replay-trace-v2 row, its validation and
//! the `encode_jsonl` line (insertion order, `,`/`:` separators, ASCII
//! escapes). Each Python `ValueError`/`TypeError` is a typed [`RowError`];
//! the int, float and bool type checks and the bbox5, COCO-17 and xy lengths
//! are carried by the Rust types instead.

use std::collections::BTreeSet;
use std::fmt;

use crate::json::{Json, JsonError};
use crate::trace_out::jsonl::encode_ordered;
use crate::trace_out::vocab::{Lifecycle, Source, SourceEvent};

pub const REPLAY_TRACE_VERSION: &str = "replay-trace-v2";
/// `encode_jsonl(ReplayTraceHeader(), [])`.
pub const HEADER_LINE: &str = "{\"version\":\"replay-trace-v2\"}\n";
/// COCO-17 keypoints per track.
pub const KEYPOINTS: usize = 17;

/// `ReplayTrack`: bbox is `(x1, y1, x2, y2, confidence)`, keypoints are
/// `(x, y, confidence)`; every value is a unit float.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayTrack {
    pub track_id: i128,
    pub lifecycle: Lifecycle,
    pub bbox: [f64; 5],
    pub keypoints: [[f64; 3]; KEYPOINTS],
}

/// The three `bed_polygon*` fields, present together or not at all.
#[derive(Clone, Debug, PartialEq)]
pub struct BedPolygon {
    pub id: String,
    pub polygon: Vec<[f64; 2]>,
    pub image_size: [u64; 2],
}

/// `ReplayRow`, fields in the Python declaration order.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayRow {
    pub camera_id: String,
    pub seq: u64,
    pub pts_ns: u64,
    pub epoch: u64,
    pub source_event: SourceEvent,
    pub source: Source,
    pub tracks: Vec<ReplayTrack>,
    pub bed: Option<BedPolygon>,
    pub night_window_active: bool,
    pub frame_width: u64,
    pub frame_height: u64,
}

/// Static refusals; no value text is carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowError {
    /// A coordinate or confidence is non-finite or outside `0.0..=1.0`.
    UnitValue,
    /// `x1 > x2` or `y1 > y2`.
    BboxOrder,
    FrameSize,
    CameraId,
    BedPolygonId,
    /// Fewer than three polygon points.
    BedPolygonPoints,
    BedImageSize,
    /// A non-`frame` row carries tracks.
    ControlRowTracks,
    DuplicateTrackId,
    Encode(JsonError),
}

impl fmt::Display for RowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnitValue => "coordinates and confidence must be finite unit floats",
            Self::BboxOrder => "bbox corners must be ordered",
            Self::FrameSize => "frame_width and frame_height must be positive integers",
            Self::CameraId => "camera_id is required",
            Self::BedPolygonId => "invalid bed_polygon_id",
            Self::BedPolygonPoints => "bed_polygon must contain at least three xy points",
            Self::BedImageSize => "bed_polygon_image_size must contain positive integers",
            Self::ControlRowTracks => "control rows must not contain tracks",
            Self::DuplicateTrackId => "track_id values must be unique per row",
            Self::Encode(_) => "replay row does not encode",
        })
    }
}
impl std::error::Error for RowError {}

fn unit(values: &[f64]) -> Result<(), RowError> {
    let inside = |value: &f64| value.is_finite() && (0.0..=1.0).contains(value);
    if values.iter().all(inside) {
        Ok(())
    } else {
        Err(RowError::UnitValue)
    }
}

impl ReplayTrack {
    /// `ReplayTrack.__post_init__`: `_unit_box`, then each keypoint.
    pub fn validate(&self) -> Result<(), RowError> {
        unit(&self.bbox)?;
        if self.bbox[0] > self.bbox[2] || self.bbox[1] > self.bbox[3] {
            return Err(RowError::BboxOrder);
        }
        self.keypoints.iter().try_for_each(|point| unit(point))
    }

    fn to_json(&self) -> Json {
        let points = self.keypoints.iter().map(|point| floats(point)).collect();
        Json::Object(vec![
            member("track_id", Json::Int(self.track_id)),
            member("lifecycle", Json::Str(self.lifecycle.as_str().to_owned())),
            member("bbox", floats(&self.bbox)),
            member("keypoints", Json::Array(points)),
        ])
    }
}

impl ReplayRow {
    /// The tracks first (Python builds them before the row), then
    /// `ReplayRow.__post_init__` in its order.
    pub fn validate(&self) -> Result<(), RowError> {
        self.tracks.iter().try_for_each(ReplayTrack::validate)?;
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err(RowError::FrameSize);
        }
        if self.camera_id.is_empty() {
            return Err(RowError::CameraId);
        }
        if let Some(bed) = &self.bed {
            if bed.id.is_empty() {
                return Err(RowError::BedPolygonId);
            }
            if bed.polygon.len() < 3 {
                return Err(RowError::BedPolygonPoints);
            }
            bed.polygon.iter().try_for_each(|point| unit(point))?;
            if bed.image_size.contains(&0) {
                return Err(RowError::BedImageSize);
            }
        }
        if self.source_event != SourceEvent::Frame && !self.tracks.is_empty() {
            return Err(RowError::ControlRowTracks);
        }
        let mut seen = BTreeSet::new();
        if !self.tracks.iter().all(|track| seen.insert(track.track_id)) {
            return Err(RowError::DuplicateTrackId);
        }
        Ok(())
    }

    /// `dataclasses.asdict(row)`: absent bed fields are `null`.
    pub fn to_json(&self) -> Json {
        let (id, polygon, size) = match &self.bed {
            Some(bed) => (
                Json::Str(bed.id.clone()),
                Json::Array(bed.polygon.iter().map(|point| floats(point)).collect()),
                Json::Array(bed.image_size.iter().map(|&side| uint(side)).collect()),
            ),
            None => (Json::Null, Json::Null, Json::Null),
        };
        Json::Object(vec![
            member("camera_id", Json::Str(self.camera_id.clone())),
            member("seq", uint(self.seq)),
            member("pts_ns", uint(self.pts_ns)),
            member("epoch", uint(self.epoch)),
            member(
                "source_event",
                Json::Str(self.source_event.as_str().to_owned()),
            ),
            member("source", Json::Str(self.source.as_str().to_owned())),
            member(
                "tracks",
                Json::Array(self.tracks.iter().map(ReplayTrack::to_json).collect()),
            ),
            member("bed_polygon_id", id),
            member("bed_polygon", polygon),
            member("bed_polygon_image_size", size),
            member("night_window_active", Json::Bool(self.night_window_active)),
            member("frame_width", uint(self.frame_width)),
            member("frame_height", uint(self.frame_height)),
        ])
    }

    /// One validated `encode_jsonl` row line, trailing newline included.
    pub fn encode_line(&self) -> Result<String, RowError> {
        self.validate()?;
        let mut line = encode_ordered(&self.to_json()).map_err(RowError::Encode)?;
        line.push('\n');
        Ok(line)
    }
}

fn member(key: &str, value: Json) -> (String, Json) {
    (key.to_owned(), value)
}

fn uint(value: u64) -> Json {
    Json::Int(i128::from(value))
}

fn floats(values: &[f64]) -> Json {
    Json::Array(values.iter().map(|&value| Json::Float(value)).collect())
}
