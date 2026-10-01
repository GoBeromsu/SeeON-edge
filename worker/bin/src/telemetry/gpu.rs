//! The runtime-status `gpu` block (Python `wire.RelayGpuPayload`, filled by
//! `worker.py:_gpu_payload`) from the native `device_report`. The native
//! `NvmlStatus::reason` texts are not the wire texts; [`nvml_error`] maps
//! each status to the text the Python probe reports.

use std::time::{SystemTime, UNIX_EPOCH};

use seeon_deepstream_native::{GpuDeviceReport, NvmlStatus, StateError, device_report};

use crate::json::Json;
use crate::seam::Clock;

/// `nvml_error` when the native report itself failed (unknown status,
/// malformed text or a negative ordinal).
pub const REPORT_FAILED: &str = "NVML device report failed";

/// One `gpu` block of a runtime-status body.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuStatus {
    pub nvml_available: bool,
    pub cuda_context_ok: bool,
    pub driver_version: Option<String>,
    pub device_name: Option<String>,
    pub captured_at_sec: f64,
    pub nvml_error: Option<String>,
}

/// The wire `nvml_error` of `status`; `None` when NVML is usable.
pub fn nvml_error(status: NvmlStatus) -> Option<&'static str> {
    match status {
        NvmlStatus::Ok => None,
        NvmlStatus::LibraryMissing => Some("NVML Shared Library Not Found"),
        NvmlStatus::SymbolMissing => Some("Function Not Found"),
        NvmlStatus::InitFailed => Some("nvmlInit failed"),
        NvmlStatus::DeviceCountFailed => Some("nvmlDeviceGetCount failed"),
        NvmlStatus::NoDevice => Some("NVML initialized but no GPU devices are visible"),
    }
}

impl GpuStatus {
    /// The block for `report` captured at `captured_at_sec`; a failed report
    /// is an unavailable GPU, never an error.
    pub fn from_report(report: Result<GpuDeviceReport, StateError>, captured_at_sec: f64) -> Self {
        match report {
            Ok(report) => Self {
                nvml_available: report.nvml_available(),
                cuda_context_ok: report.cuda_context_ok,
                nvml_error: nvml_error(report.nvml).map(str::to_owned),
                driver_version: report.driver_version,
                device_name: report.device_name,
                captured_at_sec,
            },
            Err(_) => Self {
                nvml_available: false,
                cuda_context_ok: false,
                driver_version: None,
                device_name: None,
                captured_at_sec,
                nvml_error: Some(REPORT_FAILED.to_owned()),
            },
        }
    }

    /// Probes CUDA ordinal `device` now, stamped with `clock`'s wall time.
    pub fn probe(device: i32, clock: &dyn Clock) -> Self {
        Self::from_report(device_report(device), unix_seconds(clock.wall()))
    }

    /// The wire object.
    pub fn payload(&self) -> Json {
        Json::Object(vec![
            ("nvml_available".into(), Json::Bool(self.nvml_available)),
            ("cuda_context_ok".into(), Json::Bool(self.cuda_context_ok)),
            (
                "driver_version".into(),
                text(self.driver_version.as_deref()),
            ),
            ("device_name".into(), text(self.device_name.as_deref())),
            ("captured_at_sec".into(), Json::Float(self.captured_at_sec)),
            ("nvml_error".into(), text(self.nvml_error.as_deref())),
        ])
    }
}

/// Seconds since the Unix epoch as Python `time.time()` reports them; a
/// time before the epoch is 0.
pub fn unix_seconds(at: SystemTime) -> f64 {
    at.duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64())
}

pub(crate) fn text(value: Option<&str>) -> Json {
    value.map_or(Json::Null, |value| Json::Str(value.to_owned()))
}
