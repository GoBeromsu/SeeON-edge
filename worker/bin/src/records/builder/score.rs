//! `sdk.frame`, `policy.consume` and `model.score`
//! (`worker/pipeline/diagnostics/emit_policy.py` L28-135).

use seeon_worker_runtime::evidence::AcceleratorEvidence;

use super::{
    Draft, FALL_WINDOW_FRAMES, Frame, PRODUCER_MODEL, PRODUCER_POLICY, PRODUCER_SDK, Stream,
    fall_causal_unit_id, float, frame_causal_unit_id, make_record, optional_int,
};
use crate::gpu::evidence::to_json as accelerator_json;
use crate::json::Json;
use crate::records::id::{ContractError, int};
use crate::records::wire::{Record, RecordKind};

/// `NativeObservationEvidence` as the SDK probe reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdkEvidence {
    pub sdk_frame_number: Option<u64>,
    pub source_id: Option<u64>,
    pub inference_tensor_present: bool,
    pub raw_output_row_count: u64,
    pub eligible_row_count: u64,
    pub matched_row_count: u64,
}

/// `MetadataCounters` read before and after one consume pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub accepted: u64,
    pub overwritten: u64,
    pub late: u64,
}

/// `ModelEvidence`: the float32 logit and temperature behind a calibrated
/// score, with the class origins of the deployed head.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelEvidence {
    pub raw_logit: f64,
    pub applied_temperature: f64,
    pub class_origins: Vec<String>,
}

/// One calibrated fall probability for a track.
#[derive(Clone, Debug, PartialEq)]
pub struct FallScore {
    pub track_id: u64,
    pub generation: Option<u64>,
    pub fall_transition: f64,
    pub background: f64,
    pub fallen: f64,
    pub evidence: Option<ModelEvidence>,
}

/// `sdk_frame_record`: probe evidence (when present), then the frame counters.
pub fn sdk_frame_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    evidence: Option<&SdkEvidence>,
    native_publish_sequence: u64,
) -> Result<Record, ContractError> {
    let mut payload = evidence.map_or_else(Vec::new, |evidence| {
        vec![
            optional_int("sdk_frame_number", evidence.sdk_frame_number),
            optional_int("source_id", evidence.source_id),
            (
                "inference_tensor_present".to_owned(),
                Json::Bool(evidence.inference_tensor_present),
            ),
            int("raw_output_row_count", evidence.raw_output_row_count),
            int("eligible_row_count", evidence.eligible_row_count),
            int("matched_row_count", evidence.matched_row_count),
        ]
    });
    payload.push(int("seq", frame.frame_seq));
    payload.push(int("source_generation", stream.source_generation));
    payload.push(int("native_publish_sequence", native_publish_sequence));
    let draft = Draft {
        record_kind: RecordKind::SdkFrame,
        producer: PRODUCER_SDK,
        causal_unit_id: frame_causal_unit_id(stream, frame.frame_seq),
        outcome: "accepted".to_owned(),
        payload,
    };
    make_record(stream, observed_at_ns, Some(frame), draft)
}

/// `policy_consume_record`: counter deltas over one consume pass.
pub fn policy_consume_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    before: Counters,
    after: Counters,
    processed_count: u64,
) -> Result<Record, ContractError> {
    let delta = |after: u64, before: u64| i128::from(after) - i128::from(before);
    let draft = Draft {
        record_kind: RecordKind::PolicyConsume,
        producer: PRODUCER_POLICY,
        causal_unit_id: frame_causal_unit_id(stream, frame.frame_seq),
        outcome: "consumed".to_owned(),
        payload: vec![
            int("accepted_delta", delta(after.accepted, before.accepted)),
            int(
                "overwritten_delta",
                delta(after.overwritten, before.overwritten),
            ),
            int("late_delta", delta(after.late, before.late)),
            int("processed_count", processed_count),
        ],
    };
    make_record(stream, observed_at_ns, Some(frame), draft)
}

/// `model_score_record`. The accelerator receipt of the inference call, when
/// there is one, lands in `payload.accelerator` before the id is hashed.
pub fn model_score_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    score: &FallScore,
    accelerator: Option<&AcceleratorEvidence>,
) -> Result<Record, ContractError> {
    let mut payload = vec![
        int("track_id", score.track_id),
        optional_int("generation", score.generation),
        float("fall_transition", score.fall_transition),
        float("background", score.background),
        float("fallen", score.fallen),
        int("window_frames", FALL_WINDOW_FRAMES),
    ];
    if let Some(evidence) = &score.evidence {
        payload.push(float("raw_logit", evidence.raw_logit));
        payload.push(float("applied_temperature", evidence.applied_temperature));
        let origins = evidence
            .class_origins
            .iter()
            .map(|origin| Json::Str(origin.clone()));
        payload.push(("class_origins".to_owned(), Json::Array(origins.collect())));
    }
    if let Some(accelerator) = accelerator {
        payload.push(("accelerator".to_owned(), accelerator_json(accelerator)));
    }
    let draft = Draft {
        record_kind: RecordKind::ModelScore,
        producer: PRODUCER_MODEL,
        causal_unit_id: fall_causal_unit_id(stream, Some(score.track_id), score.generation),
        outcome: "scored".to_owned(),
        payload,
    };
    make_record(stream, observed_at_ns, Some(frame), draft)
}
