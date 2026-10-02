//! Source-geometry and bed-window normalisation for one replay row.
//!
//! Division follows `_unit_bbox` and `_persisted_polygon`. A zero persisted
//! image side falls back to the accepted source size, as Python `or` does.
//! The first stored polygon is always offered to row validation. A typed
//! window or clock problem is returned so the frame can be dropped.

use std::time::SystemTime;

use seeon_worker::detection_window::{AwareDateTime, DetectionWindow, DetectionWindowError};

use crate::policy::ingest::ObservedPose;
use crate::trace_out::{BedPolygon, KEYPOINTS, ReplayRow, ReplayTrack, Source, SourceEvent};

pub(super) enum GeometryError {
    Window(DetectionWindowError),
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Window(error) => write!(formatter, "{error}"),
        }
    }
}

/// Python `_unit_bbox` plus the per-point division. No clipping or repair.
pub(super) fn unit_observation(
    pose: &ObservedPose,
    width: u64,
    height: u64,
) -> Result<([f64; 5], [[f64; 3]; KEYPOINTS]), ()> {
    if pose.keypoints.len() != KEYPOINTS || width == 0 || height == 0 {
        return Err(());
    }
    let (width, height) = (width as f64, height as f64);
    let bbox = [
        integer_zero(pose.bbox[0]) / width,
        integer_zero(pose.bbox[1]) / height,
        integer_zero(pose.bbox[2]) / width,
        integer_zero(pose.bbox[3]) / height,
        pose.confidence,
    ];
    let mut keypoints = [[0.0; 3]; KEYPOINTS];
    for (point, source) in keypoints.iter_mut().zip(&pose.keypoints) {
        *point = [source[0] / width, source[1] / height, source[2]];
    }
    Ok((bbox, keypoints))
}

// Python PersonBox corners pass through int(), unlike confidence and scores.
fn integer_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// Build one track and validate it before the caller touches prior history.
pub(super) fn validated_track(
    pose: &ObservedPose,
    known: bool,
    width: u64,
    height: u64,
) -> Result<ReplayTrack, ()> {
    let (bbox, keypoints) = unit_observation(pose, width, height)?;
    let track = ReplayTrack {
        track_id: i128::from(pose.track_id),
        lifecycle: if known {
            crate::trace_out::Lifecycle::Tracked
        } else {
            crate::trace_out::Lifecycle::New
        },
        bbox,
        keypoints,
    };
    track.validate().map_err(|_| ())?;
    Ok(track)
}

/// First persisted polygon, even when its points will fail row validation.
/// A zero persisted side uses the accepted source size.
pub(super) fn persisted_polygon(
    points: &[(i128, i128)],
    image_width: Option<u64>,
    image_height: Option<u64>,
    fallback_width: u64,
    fallback_height: u64,
) -> BedPolygon {
    let width = nonzero(image_width).unwrap_or(fallback_width);
    let height = nonzero(image_height).unwrap_or(fallback_height);
    let (width_f, height_f) = (width as f64, height as f64);
    let polygon = points
        .iter()
        .map(|&(x, y)| [x as f64 / width_f, y as f64 / height_f])
        .collect();
    BedPolygon {
        id: "persisted".to_owned(),
        polygon,
        image_size: [width, height],
    }
}

fn nonzero(value: Option<u64>) -> Option<u64> {
    value.filter(|side| *side != 0)
}
pub(super) struct RowMeta {
    pub pts: u64,
    pub epoch: u64,
    pub event: SourceEvent,
    pub width: u64,
    pub height: u64,
}

pub(super) fn base_row(camera_id: &str, seq: u64, meta: &RowMeta) -> ReplayRow {
    ReplayRow {
        camera_id: camera_id.to_owned(),
        seq,
        pts_ns: meta.pts,
        epoch: meta.epoch,
        source_event: meta.event,
        // The sole Flow producer's association_pass declares NVDCF_STRATEGY.
        source: Source::Nvdcf,
        tracks: Vec::new(),
        bed: None,
        night_window_active: false,
        frame_width: meta.width,
        frame_height: meta.height,
    }
}

/// Enabled bed-exit's admitted window, or `false` when that window is absent.
pub(super) fn night_window_active(
    enabled: bool,
    window: Option<&DetectionWindow>,
    wall: SystemTime,
) -> Result<bool, GeometryError> {
    if !enabled {
        return Ok(false);
    }
    let Some(window) = window else {
        return Ok(false);
    };
    let now = AwareDateTime::from_utc_system_time(wall).map_err(GeometryError::Window)?;
    window.contains(now).map_err(GeometryError::Window)
}
