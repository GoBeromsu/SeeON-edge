//! Record construction shared by the producers
//! (`worker/pipeline/diagnostics/record_builder.py`): the causal-unit id
//! helpers and `make_record`. Builders are pure: the caller supplies the wall
//! clock stamp, and a contract violation is a typed `Err` the caller drops.

mod decision;
mod delivery;
mod score;

pub use decision::{
    AuthorityRole, Decision, DecisionSource, Numeric, policy_coast_record, policy_decision_record,
};
pub use delivery::{
    Acceptance, Admission, Attempt, AttemptOutcome, backend_acceptance_record,
    delivery_attempt_record, event_delivery_record,
};
pub use score::{
    Counters, FallScore, ModelEvidence, SdkEvidence, model_score_record, policy_consume_record,
    sdk_frame_record,
};

use crate::json::Json;
use crate::records::id::{ContractError, optional};
use crate::records::wire::{Record, RecordBody, RecordKind, TimeQuality};

pub const PRODUCER_SDK: &str = "sdk";
pub const PRODUCER_MODEL: &str = "model";
pub const PRODUCER_POLICY: &str = "policy";
pub const PRODUCER_EVENT: &str = "event";
pub const PRODUCER_BACKEND: &str = "backend";

/// `worker.domains.fall.classifier.FALL_WINDOW_FRAMES`.
pub const FALL_WINDOW_FRAMES: u64 = 30;
/// `worker.domains.registry.FALL_MODULE_QUALIFIED_ID`.
pub const FALL_MODULE_QUALIFIED_ID: &str = "fall.v2";
pub const NO_TRACK: &str = "no-track";
pub const NO_GENERATION: &str = "no-generation";
pub const NO_MODULE: &str = "no-module";

/// Identity of the stream a record is observed against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    pub camera_id: String,
    pub worker_boot_id: String,
    pub source_generation: u64,
    pub stream_epoch: u64,
}

/// The frame a stream-scoped record belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    pub frame_seq: u64,
    pub source_pts_ns: Option<i64>,
}

/// The kind-specific part of a record before `make_record` stamps it.
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub record_kind: RecordKind,
    pub producer: &'static str,
    pub causal_unit_id: String,
    pub outcome: String,
    pub payload: Vec<(String, Json)>,
}

/// Pre-Gate-R placeholder: frames bucketed by the deployed 30-frame window.
pub fn frame_causal_unit_id(stream: &Stream, seq: u64) -> String {
    let Stream {
        camera_id,
        worker_boot_id,
        stream_epoch,
        ..
    } = stream;
    let bucket = seq / FALL_WINDOW_FRAMES;
    format!("{camera_id}:{worker_boot_id}:{stream_epoch}:frame:{bucket}")
}

/// One fall decision: camera, boot, epoch, track, generation. Absence is an
/// explicit token, never 0, because 0 is a real track id and generation.
pub fn fall_causal_unit_id(
    stream: &Stream,
    track_id: Option<u64>,
    generation: Option<u64>,
) -> String {
    let track = track_id.map_or_else(|| NO_TRACK.to_owned(), |id| id.to_string());
    let generation = generation.map_or_else(|| NO_GENERATION.to_owned(), |g| g.to_string());
    let Stream {
        camera_id,
        worker_boot_id,
        stream_epoch,
        ..
    } = stream;
    format!("{camera_id}:{worker_boot_id}:{stream_epoch}:{track}:{generation}")
}

/// A non-fall or unattributed decision: module plus frame bucket.
pub fn module_causal_unit_id(
    stream: &Stream,
    module_qualified_id: Option<&str>,
    frame_seq: u64,
) -> String {
    let module = module_qualified_id.unwrap_or(NO_MODULE);
    let Stream {
        camera_id,
        worker_boot_id,
        stream_epoch,
        ..
    } = stream;
    let bucket = frame_seq / FALL_WINDOW_FRAMES;
    format!("{camera_id}:{worker_boot_id}:{stream_epoch}:{module}:{bucket}")
}

/// `make_record`: wall time quality and producer sequence 0 (the lane
/// assigns the real sequence).
pub fn make_record(
    stream: &Stream,
    observed_at_ns: u64,
    frame: Option<Frame>,
    draft: Draft,
) -> Result<Record, ContractError> {
    Record::new(RecordBody {
        record_kind: draft.record_kind,
        camera_id: stream.camera_id.clone(),
        worker_boot_id: stream.worker_boot_id.clone(),
        source_generation: stream.source_generation,
        stream_epoch: stream.stream_epoch,
        producer: draft.producer.to_owned(),
        producer_sequence: 0,
        observed_at_ns,
        time_quality: TimeQuality::Wall,
        causal_unit_id: draft.causal_unit_id,
        outcome: draft.outcome,
        payload: draft.payload,
        frame_seq: frame.map(|frame| frame.frame_seq),
        source_pts_ns: frame.and_then(|frame| frame.source_pts_ns),
        parent_record_id: None,
        reason: None,
    })
}

fn optional_int(key: &str, value: Option<u64>) -> (String, Json) {
    optional(key, value.map(|number| Json::Int(i128::from(number))))
}

fn optional_text(key: &str, value: Option<&str>) -> (String, Json) {
    optional(key, value.map(|text| Json::Str(text.to_owned())))
}

fn float(key: &str, value: f64) -> (String, Json) {
    (key.to_owned(), Json::Float(value))
}
