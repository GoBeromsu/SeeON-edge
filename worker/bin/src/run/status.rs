//! One consuming boot-report outcome; assigned facilities use a 2s status POST.

use std::time::Duration;

use crate::config::pull::PulledConfig;
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
    Accepted {
        generation: u64,
    },
    Transport,
    Status(u16),
    Malformed,
    Payload,
    /// Neither camera context nor enrollment supplies a facility for a wire body.
    NoFacility,
}

/// Construct before acquiring the lease. No Debug/Clone and no exposed client:
/// consuming `send` permits one attempt, or `NoFacility` for an unenrolled empty
/// roster whose valid wire body cannot be constructed. No periodic sender/retry.
pub struct BootStatusContext {
    client: RelayClient,
    identity: Option<ReportIdentity>,
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
        Ok(Self {
            client,
            identity: Some(identity),
        })
    }

    /// Fresh/LKG config is resolved before the lease, as in the Python CLI.
    /// A single failure report uses the first facility in canonical sorted
    /// order, or genuine enrollment when no camera supplies one. An unenrolled
    /// empty roster still runs boot gates without inventing a facility.
    pub fn for_config(
        base_url: &str,
        token: &str,
        config: &PulledConfig,
        started_at_sec: f64,
    ) -> Result<Self, ReportContextError> {
        if !started_at_sec.is_finite() || started_at_sec < 0.0 {
            return Err(ReportContextError::StartedAt);
        }
        let clip_export = ClipExportStatus {
            enabled: config.config.clip_export_enabled(),
            version: i64::try_from(config.config.clip_export_version())
                .map_err(|_| ReportContextError::ExportVersion)?,
        };
        match config
            .cameras
            .iter()
            .map(|camera| camera.facility_id.as_str())
            .min()
            .or_else(|| config.config.enrolled_facility_id())
        {
            Some(facility_id) => Self::new(
                base_url,
                token,
                ReportIdentity {
                    facility_id: facility_id.to_owned(),
                    seq: 1,
                    generation: None,
                    clip_export,
                    started_at_sec,
                },
            ),
            None => Ok(Self {
                client: RelayClient::new(base_url, token, BOOT_REPORT_TIMEOUT)
                    .map_err(|_| ReportContextError::Relay)?,
                identity: None,
            }),
        }
    }

    pub fn send(self, reason: BootReason, gpu: &GpuStatus) -> ReportOutcome {
        let Some(identity) = self.identity else {
            return ReportOutcome::NoFacility;
        };
        let status = FacilityStatus {
            facility_id: identity.facility_id,
            cameras: Vec::new(),
            clip_recorder: ClipRecorderStatus::unavailable(),
            clip_export: identity.clip_export,
            gpu: Some(gpu.clone()),
            worker: Some(WorkerStatus::current(
                false,
                identity.started_at_sec,
                Some(reason.wire().to_owned()),
            )),
            delivery_queue: None,
        };
        let body = match status.body(identity.seq, identity.generation) {
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
