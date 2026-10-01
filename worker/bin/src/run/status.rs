//! One consuming runtime-status attempt with the existing wire and a 2s deadline.

use std::time::Duration;

use crate::exit::Exit;
use crate::relay::RelayClient;
use crate::telemetry::gpu::GpuStatus;
use crate::telemetry::status::{
    ClipExportStatus, ClipRecorderStatus, FacilityStatus, RUNTIME_STATUS_PATH, WorkerStatus,
    parse_acceptance,
};

pub const BOOT_REPORT_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BootReason {
    GpuLease,
    EngineIdentity,
    CudaUnavailable,
    EngineOpen,
}

impl BootReason {
    pub const fn wire(self) -> &'static str {
        match self {
            Self::GpuLease => "gpu_lease",
            Self::EngineIdentity => "engine_identity",
            Self::CudaUnavailable => "cuda_unavailable",
            Self::EngineOpen => "engine_open",
        }
    }

    pub const fn exit(self) -> Exit {
        match self {
            Self::GpuLease | Self::EngineIdentity => Exit::RefuseToStart,
            Self::CudaUnavailable | Self::EngineOpen => Exit::FatalAccelerator,
        }
    }
}

/// Caller-owned facility authority, sequence and effective export policy.
/// Boot does not pull config or invent a facility when no context is available.
pub struct ReportIdentity {
    pub facility_id: String,
    pub seq: u64,
    pub generation: Option<u64>,
    pub clip_export: ClipExportStatus,
    pub started_at_sec: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportContextError {
    Facility,
    StartedAt,
    ExportVersion,
    Relay,
}

impl ReportContextError {
    pub const fn exit(self) -> Exit {
        Exit::Config
    }
}

/// Path-free, token-free outcome; failure never changes the boot exit reason.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportOutcome {
    Accepted { generation: u64 },
    Transport,
    Status(u16),
    Malformed,
    Payload,
}

/// Construct before acquiring the lease. No Debug/Clone and no exposed client:
/// consuming `send` permits exactly one attempt, with no periodic sender/retry.
pub struct BootStatusContext {
    client: RelayClient,
    identity: ReportIdentity,
}

impl BootStatusContext {
    pub fn new(
        base_url: &str,
        token: &str,
        identity: ReportIdentity,
    ) -> Result<Self, ReportContextError> {
        if identity.facility_id.trim().is_empty() {
            return Err(ReportContextError::Facility);
        }
        if !identity.started_at_sec.is_finite() || identity.started_at_sec < 0.0 {
            return Err(ReportContextError::StartedAt);
        }
        if identity.clip_export.version < 0 {
            return Err(ReportContextError::ExportVersion);
        }
        let client = RelayClient::new(base_url, token, BOOT_REPORT_TIMEOUT)
            .map_err(|_| ReportContextError::Relay)?;
        Ok(Self { client, identity })
    }

    pub fn send(self, reason: BootReason, gpu: &GpuStatus) -> ReportOutcome {
        let status = FacilityStatus {
            facility_id: self.identity.facility_id,
            cameras: Vec::new(),
            clip_recorder: ClipRecorderStatus::unavailable(),
            clip_export: self.identity.clip_export,
            gpu: Some(gpu.clone()),
            worker: Some(WorkerStatus::current(
                false,
                self.identity.started_at_sec,
                Some(reason.wire().to_owned()),
            )),
            delivery_queue: None,
        };
        let body = match status.body(self.identity.seq, self.identity.generation) {
            Ok(body) => body,
            Err(_) => return ReportOutcome::Payload,
        };
        let response = match self.client.post_json(RUNTIME_STATUS_PATH, &body) {
            Ok(response) => response,
            Err(_) => return ReportOutcome::Transport,
        };
        if !(200..300).contains(&response.status) {
            return ReportOutcome::Status(response.status);
        }
        match parse_acceptance(&response.body) {
            Ok(generation) => ReportOutcome::Accepted { generation },
            Err(_) => ReportOutcome::Malformed,
        }
    }
}
