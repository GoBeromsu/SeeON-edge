//! The runtime-status body (Python `telemetry/wire.py:facility_payload`)
//! and its publisher (Python `RuntimeStatusSender._post` over
//! `RelayRuntimeStatusTransport.send`). `seq` rises per facility on every
//! attempt, failed ones included, so the next accepted status carries the
//! number of attempts the relay missed.

use crate::delivery::CapacitySnapshot;
use crate::json::{Json, Serialiser};

use super::PayloadError;
use super::gpu::{GpuStatus, text};

/// Python `RUNTIME_STATUS_PATH`, relative to the relay base URL.
pub const RUNTIME_STATUS_PATH: &str = "api/v1/relay/runtime-status";

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeStatus {
    pub requested: String,
    pub selected: Option<String>,
    pub fallback_count: u64,
    pub last_reason: Option<String>,
    pub updated_at_sec: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetectionStatus {
    pub expected: bool,
    pub inference_admitted: u64,
    pub inference_succeeded: u64,
    pub inference_overwritten: u64,
    pub decision_completed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraStatus {
    pub camera_id: String,
    pub decode: DecodeStatus,
    pub measured_fps: Option<f64>,
    pub detection: Option<DetectionStatus>,
}

/// Python `ClipRecorderStatus`; every counter is `None` while unavailable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClipRecorderStatus {
    pub available: bool,
    pub dropped_frames: Option<u64>,
    pub dropped_events: Option<u64>,
    pub failed_writes: Option<u64>,
    pub finalized_clips: Option<u64>,
    pub video_unavailable_clips: Option<u64>,
    pub active_clips: Option<u64>,
    pub encoder: Option<String>,
}

impl ClipRecorderStatus {
    /// No recorder: `available` false and every other field `None`.
    pub fn unavailable() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipExportStatus {
    pub enabled: bool,
    pub version: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkerStatus {
    pub alive: bool,
    pub pid: Option<u32>,
    pub started_at_sec: Option<f64>,
    pub profile_boot_error: Option<String>,
}

impl WorkerStatus {
    /// This process, started at `started_at_sec`.
    pub fn current(alive: bool, started_at_sec: f64, profile_boot_error: Option<String>) -> Self {
        Self {
            alive,
            pid: Some(std::process::id()),
            started_at_sec: Some(started_at_sec),
            profile_boot_error,
        }
    }
}

/// One facility's status before `seq` and `generation` are stamped.
#[derive(Clone, Debug, PartialEq)]
pub struct FacilityStatus {
    pub facility_id: String,
    pub cameras: Vec<CameraStatus>,
    pub clip_recorder: ClipRecorderStatus,
    pub clip_export: ClipExportStatus,
    pub gpu: Option<GpuStatus>,
    pub worker: Option<WorkerStatus>,
    pub delivery_queue: Option<CapacitySnapshot>,
}

fn camera(status: &CameraStatus) -> Result<Json, PayloadError> {
    if status.camera_id.is_empty() {
        return Err(PayloadError::BlankId("camera_id"));
    }
    let decode = &status.decode;
    let mut members = vec![
        ("camera_id".into(), Json::Str(status.camera_id.clone())),
        (
            "decode".into(),
            Json::Object(vec![
                ("requested".into(), Json::Str(decode.requested.clone())),
                ("selected".into(), text(decode.selected.as_deref())),
                ("fallback_count".into(), count(decode.fallback_count)),
                ("last_reason".into(), text(decode.last_reason.as_deref())),
                ("updated_at_sec".into(), Json::Float(decode.updated_at_sec)),
            ]),
        ),
    ];
    if let Some(fps) = status.measured_fps {
        if !(fps.is_finite() && fps >= 0.0) {
            return Err(PayloadError::Range("measured_fps"));
        }
        members.push(("measured_fps".into(), Json::Float(fps)));
    }
    if let Some(detection) = status.detection {
        members.push((
            "detection".into(),
            Json::Object(vec![
                ("expected".into(), Json::Bool(detection.expected)),
                (
                    "inference_admitted".into(),
                    count(detection.inference_admitted),
                ),
                (
                    "inference_succeeded".into(),
                    count(detection.inference_succeeded),
                ),
                (
                    "inference_overwritten".into(),
                    count(detection.inference_overwritten),
                ),
                (
                    "decision_completed".into(),
                    count(detection.decision_completed),
                ),
            ]),
        ));
    }
    Ok(Json::Object(members))
}

impl FacilityStatus {
    /// The wire object stamped with `seq` and `generation`: cameras sorted
    /// by id; `gpu`, `worker` and `delivery_queue` only when present.
    pub fn payload(&self, seq: u64, generation: Option<u64>) -> Result<Json, PayloadError> {
        if self.facility_id.is_empty() {
            return Err(PayloadError::BlankId("facility_id"));
        }
        let mut cameras: Vec<&CameraStatus> = self.cameras.iter().collect();
        cameras.sort_by(|left, right| left.camera_id.cmp(&right.camera_id));
        let cameras = cameras.into_iter().map(camera).collect::<Result<_, _>>()?;
        let export = self.clip_export;
        let mut members = vec![
            ("facility_id".into(), Json::Str(self.facility_id.clone())),
            ("generation".into(), generation.map_or(Json::Null, count)),
            ("seq".into(), count(seq)),
            ("cameras".into(), Json::Array(cameras)),
            (
                "clip_export".into(),
                Json::Object(vec![
                    ("enabled".into(), Json::Bool(export.enabled)),
                    ("version".into(), Json::Int(i128::from(export.version))),
                ]),
            ),
            ("clip_recorder".into(), recorder(&self.clip_recorder)),
        ];
        if let Some(gpu) = &self.gpu {
            members.push(("gpu".into(), gpu.payload()));
        }
        if let Some(status) = &self.worker {
            members.push(("worker".into(), worker(status)));
        }
        if let Some(snapshot) = &self.delivery_queue {
            members.push(("delivery_queue".into(), delivery_queue(snapshot)));
        }
        Ok(Json::Object(members))
    }

    /// The request body: Python `encode_json(payload)`.
    pub fn body(&self, seq: u64, generation: Option<u64>) -> Result<Vec<u8>, PayloadError> {
        Ok(Serialiser::ModelSelection
            .canonical(&self.payload(seq, generation)?)?
            .into_bytes())
    }
}

mod parts;
mod sender;

pub use parts::delivery_queue;
use parts::{count, recorder, worker};
pub use sender::{StatusSender, parse_acceptance};
