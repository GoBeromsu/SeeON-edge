//! Python `_CameraPayload` (`worker/runtime/config/pull_models.py`): one
//! roster entry of the pulled worker config and the row the restart check
//! reads.

use super::runtime::BedZoneRegion;
use crate::config::lookup;
use crate::json::Json;

const MAX_BED_ZONE_REGIONS: usize = 8;

/// One roster entry that passed `_CameraPayload.model_validate`.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPayload {
    pub camera_id: String,
    pub facility_id: Option<String>,
    pub space_id: Option<String>,
    pub label: Option<String>,
    pub rtsp_url: Option<String>,
    pub online: bool,
    pub space_name: Option<String>,
    pub floor_name: Option<String>,
    pub created_at: Option<String>,
    pub fps: Option<f64>,
    pub frame_stride: Option<u64>,
    pub decode_backend: Option<String>,
    /// `None` (key omitted) and `Some(vec![])` (explicit opt-out) differ.
    pub domains: Option<Vec<String>>,
    pub bed_zone_regions: Vec<BedZoneRegion>,
    pub bed_zone_image_width: Option<u64>,
    pub bed_zone_image_height: Option<u64>,
}

/// Python `PulledCameraConfig`: the roster row the restart check reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PulledCamera {
    pub camera_id: String,
    pub space_id: String,
    pub label: String,
    pub rtsp_url: Option<String>,
    pub online: bool,
    pub space_name: Option<String>,
    pub floor_name: Option<String>,
    pub created_at: Option<String>,
}

impl CameraPayload {
    /// `_CameraPayload.model_validate`; `None` drops the entry, as
    /// `resolved_cameras` does. Unknown keys are ignored.
    pub(super) fn parse(entry: &Json) -> Option<Self> {
        let Json::Object(members) = entry else {
            return None;
        };
        let bed_zone_regions = match lookup(members, "bed_zone_regions") {
            None => Vec::new(),
            Some(Json::Array(items)) if items.len() <= MAX_BED_ZONE_REGIONS => items
                .iter()
                .map(BedZoneRegion::parse)
                .collect::<Option<Vec<_>>>()?,
            Some(_) => return None,
        };
        Some(Self {
            camera_id: match lookup(members, "camera_id") {
                Some(Json::Str(text)) if !text.is_empty() => text.clone(),
                _ => return None,
            },
            facility_id: text(members, "facility_id", 1)?,
            space_id: text(members, "space_id", 1)?,
            label: text(members, "label", 1)?,
            rtsp_url: text(members, "rtsp_url", 0)?,
            online: match lookup(members, "online") {
                None => true,
                Some(Json::Bool(online)) => *online,
                Some(_) => return None,
            },
            space_name: text(members, "space_name", 0)?,
            floor_name: text(members, "floor_name", 0)?,
            created_at: text(members, "created_at", 0)?,
            fps: match lookup(members, "fps") {
                None | Some(Json::Null) => None,
                Some(Json::Int(value)) if *value > 0 => Some(*value as f64),
                Some(Json::Float(value)) if *value > 0.0 => Some(*value),
                Some(_) => return None,
            },
            frame_stride: positive(members, "frame_stride")?,
            decode_backend: text(members, "decode_backend", 0)?,
            domains: match lookup(members, "domains") {
                None | Some(Json::Null) => None,
                Some(Json::Array(items)) => Some(
                    items
                        .iter()
                        .map(|item| match item {
                            Json::Str(name) => Some(name.clone()),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?,
                ),
                Some(_) => return None,
            },
            bed_zone_regions,
            bed_zone_image_width: positive(members, "bed_zone_image_width")?,
            bed_zone_image_height: positive(members, "bed_zone_image_height")?,
        })
    }

    /// `resolved_facility_id`: the facility, else the placeholder `local`.
    pub fn resolved_facility_id(&self) -> &str {
        self.facility_id.as_deref().unwrap_or("local")
    }

    /// `resolved_space_id`: the space, else the facility, else empty.
    pub fn resolved_space_id(&self) -> &str {
        self.space_id
            .as_deref()
            .or(self.facility_id.as_deref())
            .unwrap_or("")
    }

    /// The `PulledCameraConfig` row `to_pulled_config` builds.
    pub fn pulled(&self) -> PulledCamera {
        PulledCamera {
            camera_id: self.camera_id.clone(),
            space_id: self.resolved_space_id().to_owned(),
            label: self.label.clone().unwrap_or_else(|| self.camera_id.clone()),
            rtsp_url: self.rtsp_url.clone(),
            online: self.online,
            space_name: self.space_name.clone(),
            floor_name: self.floor_name.clone(),
            created_at: self.created_at.clone(),
        }
    }
}

/// An optional string of at least `min_chars` characters: `Some(None)` when
/// absent or null, `None` when the entry must be dropped.
fn text(members: &[(String, Json)], key: &str, min_chars: usize) -> Option<Option<String>> {
    match lookup(members, key) {
        None | Some(Json::Null) => Some(None),
        Some(Json::Str(text)) if text.chars().count() >= min_chars => Some(Some(text.clone())),
        Some(_) => None,
    }
}

/// An optional integer greater than zero, with the same three outcomes.
fn positive(members: &[(String, Json)], key: &str) -> Option<Option<u64>> {
    match lookup(members, key) {
        None | Some(Json::Null) => Some(None),
        Some(Json::Int(value)) if *value > 0 => u64::try_from(*value).ok().map(Some),
        Some(_) => None,
    }
}
