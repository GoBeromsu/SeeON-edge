//! File-free publisher conformance admission before model warmup.
//! Mirrors `ort_pose_bbox56._parse_conformance/_validate_runner_conformance`
//! and `worker._validate_fall_bundle_conformance`, not a second feature extractor.

use std::fmt;

use seeon_worker::pose_bbox56::{
    COCO17_KEYPOINT_ORDER, FALL_WINDOW_FRAMES, POSE_BBOX56_CONFIDENCE_GATE, POSE_BBOX56_DIM,
    POSE_BBOX56_PREPROCESSING_IDENTITY,
};
use seeon_worker::temporal::CURRENT_TEMPORAL_PROFILE;

use crate::config::lookup;
use crate::json::Json;
use crate::policy::fall::window::FALL_STRIDE_FRAMES;

type Object = [(String, Json)];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConformanceError {
    Shape(&'static str),
    Mismatch(&'static str),
}

impl fmt::Display for ConformanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(field) => write!(formatter, "invalid conformance field shape: {field}"),
            Self::Mismatch(field) => write!(formatter, "conformance differs from runner: {field}"),
        }
    }
}
impl std::error::Error for ConformanceError {}

const NORMALIZATION: &str = "clip finite raw coordinates to inclusive raw bounds, then divide x by frame_width and y by frame_height";

/// Admit only the feature contract this binary implements. Extra publisher
/// metadata is allowed; the tail map and normalization denominators are exact.
/// Python admits numeric-equal floats for length/window, but not stride/tail.
pub fn validate(document: &Json) -> Result<(), ConformanceError> {
    let Json::Object(document) = document else {
        return Err(ConformanceError::Shape("document"));
    };
    let vector = object(document, "vector")?;
    let tail = object(vector, "tail_indices")?;
    let confidence = object(document, "confidence")?;
    let temporal = object(document, "temporal")?;
    let coordinates = object(document, "coordinate_system")?;
    text(
        document,
        "preprocessing_identity",
        POSE_BBOX56_PREPROCESSING_IDENTITY,
    )?;
    numeric(vector, "length", POSE_BBOX56_DIM as f64)?;
    numeric(temporal, "window_frames", FALL_WINDOW_FRAMES as f64)?;
    numeric(confidence, "gate", POSE_BBOX56_CONFIDENCE_GATE)?;
    integer(temporal, "stride_frames", i128::from(FALL_STRIDE_FRAMES))?;
    numeric(temporal, "fps", CURRENT_TEMPORAL_PROFILE.pose_fps())?;
    let Some(Json::Array(order)) = lookup(document, "keypoint_order") else {
        return Err(ConformanceError::Shape("keypoint_order"));
    };
    if order.len() != COCO17_KEYPOINT_ORDER.len()
        || !order
            .iter()
            .zip(COCO17_KEYPOINT_ORDER)
            .all(|(value, expected)| matches!(value, Json::Str(actual) if actual == expected))
    {
        return Err(ConformanceError::Mismatch("keypoint_order"));
    }
    if tail.len() != 5 {
        return Err(ConformanceError::Mismatch("tail_indices"));
    }
    for (name, index) in [
        ("x1", 51),
        ("y1", 52),
        ("x2", 53),
        ("y2", 54),
        ("valid", 55),
    ] {
        integer(tail, name, index)?;
    }
    text(coordinates, "origin", "top_left")?;
    text(coordinates, "xy_normalization_rule", NORMALIZATION)?;
    let denominators = object(coordinates, "xy_normalization_denominators")?;
    if denominators.len() != 2 {
        return Err(ConformanceError::Mismatch("xy_normalization_denominators"));
    }
    text(denominators, "x", "frame_width")?;
    text(denominators, "y", "frame_height")
}

fn object<'a>(parent: &'a Object, field: &'static str) -> Result<&'a Object, ConformanceError> {
    match lookup(parent, field) {
        Some(Json::Object(value)) => Ok(value),
        _ => Err(ConformanceError::Shape(field)),
    }
}

fn text(parent: &Object, field: &'static str, expected: &str) -> Result<(), ConformanceError> {
    match lookup(parent, field) {
        Some(Json::Str(value)) if value == expected => Ok(()),
        Some(Json::Str(_)) => Err(ConformanceError::Mismatch(field)),
        _ => Err(ConformanceError::Shape(field)),
    }
}

fn numeric(parent: &Object, field: &'static str, expected: f64) -> Result<(), ConformanceError> {
    let value = match lookup(parent, field) {
        Some(Json::Int(value)) => *value as f64,
        Some(Json::Float(value)) => *value,
        _ => return Err(ConformanceError::Shape(field)),
    };
    if value == expected {
        Ok(())
    } else {
        Err(ConformanceError::Mismatch(field))
    }
}

fn integer(parent: &Object, field: &'static str, expected: i128) -> Result<(), ConformanceError> {
    match lookup(parent, field) {
        Some(Json::Int(value)) if *value == expected => Ok(()),
        Some(Json::Int(_)) => Err(ConformanceError::Mismatch(field)),
        _ => Err(ConformanceError::Shape(field)),
    }
}
