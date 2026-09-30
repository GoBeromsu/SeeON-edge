//! Wire shapes of execution records (`shared/events/execution_records.py`
//! `WireProvenance`, `WireRecord`, `WireGap`, `WireBatch`). Every value is
//! validated at construction; ids are derived, never supplied.

use crate::json::Json;
use crate::records::id::{
    ContractError, content_id, identity, int, optional, sha256_hex_field, text,
};

/// `PROCESS_SCOPE`: generation and epoch of process-scoped kinds.
pub const PROCESS_SCOPE: u64 = 0;

/// `RECORD_KINDS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RecordKind {
    SdkFrame,
    PolicyConsume,
    ModelScore,
    PolicyDecision,
    EventDelivery,
    BackendAcceptance,
}

impl RecordKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SdkFrame => "sdk.frame",
            Self::PolicyConsume => "policy.consume",
            Self::ModelScore => "model.score",
            Self::PolicyDecision => "policy.decision",
            Self::EventDelivery => "event.delivery",
            Self::BackendAcceptance => "backend.acceptance",
        }
    }

    /// `PROCESS_SCOPED_KINDS`.
    pub fn is_process_scoped(self) -> bool {
        self == Self::BackendAcceptance
    }
}

/// `TIME_QUALITIES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeQuality {
    Monotonic,
    Wall,
    Pts,
    Unknown,
}

impl TimeQuality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Monotonic => "monotonic",
            Self::Wall => "wall",
            Self::Pts => "pts",
            Self::Unknown => "unknown",
        }
    }
}

/// `WireProvenance`: the execution identity every record of a batch points at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub worker_build_revision: String,
    pub worker_image_digest: String,
    pub model_digest: String,
    pub calibration_digest: String,
    pub preprocessing_identity: String,
    pub config_digest: String,
    pub policy_identity: String,
}

impl Provenance {
    fn members(&self) -> [(&'static str, &str); 7] {
        [
            ("worker_build_revision", &self.worker_build_revision),
            ("worker_image_digest", &self.worker_image_digest),
            ("model_digest", &self.model_digest),
            ("calibration_digest", &self.calibration_digest),
            ("preprocessing_identity", &self.preprocessing_identity),
            ("config_digest", &self.config_digest),
            ("policy_identity", &self.policy_identity),
        ]
    }

    /// Every member must be an identity (`WireProvenance.__post_init__`).
    pub fn validate(&self) -> Result<(), ContractError> {
        self.members()
            .into_iter()
            .try_for_each(|(name, value)| identity(value, name))
    }

    pub fn to_json(&self) -> Json {
        let members = self.members().into_iter().map(|(k, v)| text(k, v));
        Json::Object(members.collect())
    }
}

/// The fields of one record before its id is derived (`WireRecord` minus
/// `record_id`). Non-negative integers are `u64`; `source_pts_ns` may be
/// negative.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordBody {
    pub record_kind: RecordKind,
    pub camera_id: String,
    pub worker_boot_id: String,
    pub source_generation: u64,
    pub stream_epoch: u64,
    pub producer: String,
    pub producer_sequence: u64,
    pub observed_at_ns: u64,
    pub time_quality: TimeQuality,
    pub causal_unit_id: String,
    pub outcome: String,
    pub payload: Vec<(String, Json)>,
    pub frame_seq: Option<u64>,
    pub source_pts_ns: Option<i64>,
    pub parent_record_id: Option<String>,
    pub reason: Option<String>,
}

impl RecordBody {
    fn to_json(&self) -> Json {
        let kind = self.record_kind.as_str();
        Json::Object(vec![
            text("record_kind", kind),
            text("camera_id", &self.camera_id),
            text("worker_boot_id", &self.worker_boot_id),
            int("source_generation", self.source_generation),
            int("stream_epoch", self.stream_epoch),
            text("producer", &self.producer),
            int("producer_sequence", self.producer_sequence),
            int("observed_at_ns", self.observed_at_ns),
            text("time_quality", self.time_quality.as_str()),
            text("causal_unit_id", &self.causal_unit_id),
            text("outcome", &self.outcome),
            ("payload".to_owned(), Json::Object(self.payload.clone())),
            optional("frame_seq", self.frame_seq.map(|v| Json::Int(v.into()))),
            optional(
                "source_pts_ns",
                self.source_pts_ns.map(|v| Json::Int(v.into())),
            ),
            optional(
                "parent_record_id",
                self.parent_record_id.clone().map(Json::Str),
            ),
            optional("reason", self.reason.clone().map(Json::Str)),
        ])
    }
}

/// `WireRecord`: one immutable producer observation with its content id.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    body: RecordBody,
    record_id: String,
}

impl Record {
    /// Validates `body` as `WireRecord.__post_init__` and derives `record_id`.
    pub fn new(body: RecordBody) -> Result<Self, ContractError> {
        if body.record_kind.is_process_scoped()
            && (body.source_generation != PROCESS_SCOPE || body.stream_epoch != PROCESS_SCOPE)
        {
            return Err(ContractError::ProcessScope);
        }
        identity(&body.camera_id, "camera_id")?;
        identity(&body.worker_boot_id, "worker_boot_id")?;
        identity(&body.producer, "producer")?;
        identity(&body.causal_unit_id, "causal_unit_id")?;
        identity(&body.outcome, "outcome")?;
        if let Some(parent) = &body.parent_record_id {
            sha256_hex_field(parent, "parent_record_id")?;
        }
        if let Some(reason) = &body.reason {
            identity(reason, "reason")?;
        }
        let record_id = content_id(&body.to_json())?;
        Ok(Self { body, record_id })
    }

    pub fn body(&self) -> &RecordBody {
        &self.body
    }

    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    /// The body plus `record_id`, as `WireRecord.to_json`.
    pub fn to_json(&self) -> Json {
        let mut json = self.body.to_json();
        if let Json::Object(members) = &mut json {
            members.push(text("record_id", &self.record_id));
        }
        json
    }
}

/// `WireGap`: a drop the worker itself observed. `scope` is the
/// `(source_generation, stream_epoch)` pair, both or neither.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gap {
    pub producer: String,
    pub from_sequence: u64,
    pub to_sequence: u64,
    pub from_ns: u64,
    pub to_ns: u64,
    pub record_count: u64,
    pub cause: String,
    pub scope: Option<(u64, u64)>,
}

impl Gap {
    pub fn validate(&self) -> Result<(), ContractError> {
        identity(&self.producer, "producer")?;
        identity(&self.cause, "cause")?;
        if self.to_sequence < self.from_sequence || self.to_ns < self.from_ns {
            return Err(ContractError::GapRange);
        }
        Ok(())
    }

    /// Unscoped gaps omit both scope members (legacy batch identities).
    pub fn to_json(&self) -> Json {
        let mut members = vec![
            text("producer", &self.producer),
            int("from_sequence", self.from_sequence),
            int("to_sequence", self.to_sequence),
            int("from_ns", self.from_ns),
            int("to_ns", self.to_ns),
            int("record_count", self.record_count),
            text("cause", &self.cause),
        ];
        if let Some((generation, epoch)) = self.scope {
            members.push(int("source_generation", generation));
            members.push(int("stream_epoch", epoch));
        }
        Json::Object(members)
    }
}
