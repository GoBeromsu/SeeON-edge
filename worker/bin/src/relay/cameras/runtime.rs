//! Python `CameraRuntimeConfig` and `BedZoneRegionConfig`
//! (`worker/runtime/config/camera_models.py`) as `_runtime_camera`
//! (`worker/runtime/config/pull_models.py`) builds them.

use std::collections::BTreeSet;

use super::{CameraConfigError, CameraPayload};
use crate::config::lookup;
use crate::json::Json;

/// `_runtime_camera`'s fps when the relay omits one
/// (`CURRENT_TEMPORAL_PROFILE.target_fps`, `worker/types/temporal_profile.py`).
pub const DEFAULT_CAMERA_FPS: f64 = 30.0;
const DECODE_BACKENDS: [&str; 4] = ["auto", "nvdec", "opencv", "cpu"];

/// Python `BedZoneRegionConfig`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BedZoneRegion {
    pub id: String,
    pub polygon: Vec<(i128, i128)>,
    pub origin: String,
}

/// Python `CameraRuntimeConfig` as `_runtime_camera` builds it.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeCamera {
    pub camera_id: String,
    pub facility_id: String,
    pub rtsp_url: String,
    pub fps: f64,
    pub frame_stride: u64,
    pub decode_backend: Option<String>,
    pub label: Option<String>,
    pub bed_zone_regions: Vec<BedZoneRegion>,
    pub bed_zone_image_width: Option<u64>,
    pub bed_zone_image_height: Option<u64>,
}

impl RuntimeCamera {
    /// `_runtime_camera` plus the `CameraRuntimeConfig` validators; a
    /// refusal names the camera and the field, and fails the whole config.
    pub(super) fn from_payload(payload: &CameraPayload) -> Result<Self, CameraConfigError> {
        let refuse = |field: &'static str| CameraConfigError::RuntimeCamera {
            camera_id: payload.camera_id.clone(),
            field,
        };
        let camera_id = &payload.camera_id;
        if camera_id.trim().is_empty() || camera_id.contains(['\0', '\n', '\r']) {
            return Err(refuse("camera_id"));
        }
        let facility_id = payload.resolved_facility_id().trim();
        if facility_id.is_empty() {
            return Err(refuse("facility_id"));
        }
        let rtsp_url = payload.rtsp_url.as_deref().unwrap_or("").trim();
        if !rtsp_url
            .get(..7)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("rtsp://"))
        {
            return Err(refuse("rtsp_url"));
        }
        let decode_backend = match &payload.decode_backend {
            None => None,
            Some(value) => {
                let normalised = value.trim().to_lowercase();
                if !DECODE_BACKENDS.contains(&normalised.as_str()) {
                    return Err(refuse("decode_backend"));
                }
                Some(normalised)
            }
        };
        if !payload.bed_zone_regions.is_empty() {
            let (Some(width), Some(height)) =
                (payload.bed_zone_image_width, payload.bed_zone_image_height)
            else {
                return Err(refuse("bed_zone_image_size"));
            };
            let ids: BTreeSet<&str> = payload
                .bed_zone_regions
                .iter()
                .map(|r| r.id.as_str())
                .collect();
            if ids.len() != payload.bed_zone_regions.len() {
                return Err(refuse("bed_zone_regions"));
            }
            let (width, height) = (i128::from(width), i128::from(height));
            let inside =
                |&(x, y): &(i128, i128)| (0..width).contains(&x) && (0..height).contains(&y);
            if !payload
                .bed_zone_regions
                .iter()
                .all(|r| r.polygon.iter().all(inside))
            {
                return Err(refuse("bed_zone_regions"));
            }
        }
        Ok(Self {
            camera_id: camera_id.clone(),
            facility_id: facility_id.to_owned(),
            rtsp_url: rtsp_url.to_owned(),
            fps: payload.fps.unwrap_or(DEFAULT_CAMERA_FPS),
            frame_stride: payload.frame_stride.unwrap_or(1),
            decode_backend,
            label: payload.label.clone(),
            bed_zone_regions: payload.bed_zone_regions.clone(),
            bed_zone_image_width: payload.bed_zone_image_width,
            bed_zone_image_height: payload.bed_zone_image_height,
        })
    }
}

impl BedZoneRegion {
    /// `BedZoneRegionConfig`: exactly `id` (1-64 characters, not blank),
    /// `polygon` (3-16 integer pairs) and `origin` (`manual` or `model`).
    pub(super) fn parse(value: &Json) -> Option<Self> {
        let Json::Object(members) = value else {
            return None;
        };
        if members
            .iter()
            .any(|(key, _)| !matches!(key.as_str(), "id" | "polygon" | "origin"))
        {
            return None;
        }
        let id = match lookup(members, "id") {
            Some(Json::Str(id))
                if (1..=64).contains(&id.chars().count()) && !id.trim().is_empty() =>
            {
                id.clone()
            }
            _ => return None,
        };
        let polygon = match lookup(members, "polygon") {
            Some(Json::Array(points)) if (3..=16).contains(&points.len()) => points
                .iter()
                .map(|point| match point {
                    Json::Array(pair) => match pair.as_slice() {
                        [Json::Int(x), Json::Int(y)] => Some((*x, *y)),
                        _ => None,
                    },
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?,
            _ => return None,
        };
        let origin = match lookup(members, "origin") {
            Some(Json::Str(origin)) if matches!(origin.as_str(), "manual" | "model") => {
                origin.clone()
            }
            _ => return None,
        };
        Some(Self {
            id,
            polygon,
            origin,
        })
    }
}
