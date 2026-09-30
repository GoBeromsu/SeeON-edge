//! Python `BackendWorkerConfigPayload` (`worker/runtime/config/pull_models.py`):
//! the fleet fields of the worker config the relay serves, parsed with the
//! same refusals (`_reject_unimplemented_policy_payload`, `_require_version`)
//! and the same fail-open resolvers (windows, domains, clip subdir, cameras).
//! Integer and boolean fields accept only JSON integers and booleans.

mod camera;
mod domains;
mod runtime;
mod window;

pub use camera::{CameraPayload, PulledCamera};
pub use domains::{DomainSelection, KNOWN_DOMAINS};
pub use runtime::{BedZoneRegion, DEFAULT_CAMERA_FPS, RuntimeCamera};
pub use window::DetectionWindow;

use crate::config::lkg::Directive;
use crate::config::lookup;
use crate::json::Json;

/// Why a pulled worker config was refused as a whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CameraConfigError {
    /// The payload is not a JSON object.
    NotObject,
    /// `clip` or `models` is present (`_reject_unimplemented_policy_payload`).
    Unimplemented(&'static str),
    /// A top-level field is missing or has the wrong type or range.
    Field(&'static str),
    /// Neither `registry_version` nor `config_version` (`_require_version`).
    MissingVersion,
    /// `cameras` declared entries and none of them parsed.
    NoCamerasParsed,
    /// A parsed camera fails the `CameraRuntimeConfig` checks.
    RuntimeCamera {
        camera_id: String,
        field: &'static str,
    },
}

/// A validated `BackendWorkerConfigPayload`.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerConfigPayload {
    registry_version: Option<i128>,
    config_version: Option<i128>,
    restart_epoch: Option<i128>,
    night_window: Option<DetectionWindow>,
    detection_windows: Option<Vec<(String, Json)>>,
    cameras: Vec<Json>,
    domains: Option<Vec<(String, Json)>>,
    clip_store_subdir: Json,
    detection_policies: Json,
    clip_export_enabled: bool,
    clip_export_version: i128,
}

impl WorkerConfigPayload {
    /// `BackendWorkerConfigPayload.model_validate`. Unknown keys are ignored.
    pub fn parse(value: &Json) -> Result<Self, CameraConfigError> {
        let Json::Object(members) = value else {
            return Err(CameraConfigError::NotObject);
        };
        for key in ["clip", "models"] {
            if lookup(members, key).is_some() {
                return Err(CameraConfigError::Unimplemented(key));
            }
        }
        let payload = Self {
            registry_version: version(members, "registry_version")?,
            config_version: version(members, "config_version")?,
            restart_epoch: version(members, "restart_epoch")?,
            night_window: match lookup(members, "night_window") {
                None | Some(Json::Null) => None,
                Some(window) => Some(
                    DetectionWindow::shape(window)
                        .ok_or(CameraConfigError::Field("night_window"))?,
                ),
            },
            detection_windows: object_or_null(members, "detection_windows")?,
            cameras: match lookup(members, "cameras") {
                Some(Json::Array(items)) => items.clone(),
                _ => return Err(CameraConfigError::Field("cameras")),
            },
            domains: object_or_null(members, "domains")?,
            clip_store_subdir: lookup(members, "clip_store_subdir")
                .cloned()
                .unwrap_or(Json::Null),
            detection_policies: lookup(members, "detection_policies")
                .cloned()
                .unwrap_or(Json::Null),
            clip_export_enabled: match lookup(members, "clip_export_enabled") {
                None => false,
                Some(Json::Bool(enabled)) => *enabled,
                Some(_) => return Err(CameraConfigError::Field("clip_export_enabled")),
            },
            clip_export_version: match lookup(members, "clip_export_version") {
                None => 0,
                Some(Json::Int(value)) if *value >= 0 => *value,
                Some(_) => return Err(CameraConfigError::Field("clip_export_version")),
            },
        };
        if payload.registry_version.is_none() && payload.config_version.is_none() {
            return Err(CameraConfigError::MissingVersion);
        }
        Ok(payload)
    }

    /// `directive`: (`restart_epoch` or 0, `config_version` else the
    /// registry version, `registry_version` or 0).
    pub fn directive(&self) -> Directive {
        let registry = self.registry_version.unwrap_or(0);
        Directive {
            generation: self.restart_epoch.unwrap_or(0),
            version: self.config_version.unwrap_or(registry),
            registry,
        }
    }

    /// `resolved_clip_store_subdir`: a non-blank relative path without `..`.
    pub fn clip_store_subdir(&self) -> Option<&str> {
        let Json::Str(value) = &self.clip_store_subdir else {
            return None;
        };
        let refused = value.trim().is_empty()
            || value.starts_with('/')
            || value.split('/').any(|part| part == "..");
        (!refused).then_some(value.as_str())
    }

    /// `resolved_cameras`: the entries that parse, in roster order.
    pub fn cameras(&self) -> Vec<CameraPayload> {
        self.cameras
            .iter()
            .filter_map(CameraPayload::parse)
            .collect()
    }

    /// The `to_pulled_config` roster.
    pub fn pulled_cameras(&self) -> Vec<PulledCamera> {
        self.cameras().iter().map(CameraPayload::pulled).collect()
    }

    /// The `to_worker_config` cameras: every parsed camera with an RTSP URL.
    /// Declared cameras of which none parsed, or a camera the runtime
    /// checks refuse, refuse the whole config.
    pub fn runtime_cameras(&self) -> Result<Vec<RuntimeCamera>, CameraConfigError> {
        let cameras = self.cameras();
        let runtime = cameras
            .iter()
            .filter(|camera| camera.rtsp_url.is_some())
            .map(RuntimeCamera::from_payload)
            .collect::<Result<Vec<_>, _>>()?;
        if !self.cameras.is_empty() && cameras.is_empty() {
            return Err(CameraConfigError::NoCamerasParsed);
        }
        Ok(runtime)
    }

    pub fn clip_export_enabled(&self) -> bool {
        self.clip_export_enabled
    }

    pub fn clip_export_version(&self) -> i128 {
        self.clip_export_version
    }

    /// The raw `detection_policies` value (`Json::Null` when absent); the
    /// policy bundle is parsed by its own owner.
    pub fn detection_policies(&self) -> &Json {
        &self.detection_policies
    }
}

/// An optional version: absent, null or an integer of at least zero.
fn version(
    members: &[(String, Json)],
    key: &'static str,
) -> Result<Option<i128>, CameraConfigError> {
    match lookup(members, key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Int(value)) if *value >= 0 => Ok(Some(*value)),
        Some(_) => Err(CameraConfigError::Field(key)),
    }
}

/// An optional object: absent, null or a JSON object.
fn object_or_null(
    members: &[(String, Json)],
    key: &'static str,
) -> Result<Option<Vec<(String, Json)>>, CameraConfigError> {
    match lookup(members, key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Object(fields)) => Ok(Some(fields.clone())),
        Some(_) => Err(CameraConfigError::Field(key)),
    }
}
