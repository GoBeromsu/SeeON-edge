//! Stage3B policy values into Stage3A2 records. Stream/frame identities are
//! supplied by the owner; batch provenance belongs to the exporter, not payloads.

use std::fmt;

use seeon_worker::trace::DecisionTraceSnapshot;
use seeon_worker_runtime::evidence::AcceleratorEvidence;

use super::decision::{DecisionError, adapt_decision};
use crate::json::Json;
use crate::policy::emit::ModelScore;
use crate::records::Record;
use crate::records::builder::{self, DecisionSource, FallScore, Frame, ModelEvidence, Stream};
use crate::records::id::ContractError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordError {
    Decision(DecisionError),
    NegativeIdentity(&'static str),
    MissingScore(&'static str),
    ModelEvidenceShape,
    Contract(ContractError),
}

impl fmt::Display for RecordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("runtime value cannot be represented by an execution record")
    }
}

impl std::error::Error for RecordError {}

pub fn decision_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    snapshot: &DecisionTraceSnapshot,
    source: &DecisionSource,
) -> Result<Record, RecordError> {
    let decision = adapt_decision(snapshot).map_err(RecordError::Decision)?;
    builder::policy_decision_record(stream, frame, observed_at_ns, &decision, source)
        .map_err(RecordError::Contract)
}

/// A scored record requires all three probabilities. An accelerator receipt
/// is included only for an actual accelerator call; CPU execution supplies none.
/// Missing scores are not invented or zero-filled.
/// `ModelEvidence` is calibration evidence, NOT an accelerator receipt.
pub fn model_score_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    score: &ModelScore,
    accelerator: Option<&AcceleratorEvidence>,
) -> Result<Record, RecordError> {
    let identity =
        |value, name| u64::try_from(value).map_err(|_| RecordError::NegativeIdentity(name));
    let probability = |value: Option<f64>, name| value.ok_or(RecordError::MissingScore(name));
    let evidence = match score.evidence {
        None => None,
        Some(evidence) => Some(ModelEvidence {
            raw_logit: evidence.raw_logit,
            applied_temperature: evidence.applied_temperature,
            class_origins: class_origins(score)?,
        }),
    };
    let score = FallScore {
        track_id: identity(score.track_id, "track_id")?,
        generation: score
            .generation
            .map(|g| identity(g, "generation"))
            .transpose()?,
        fall_transition: probability(score.fall_transition, "fall_transition")?,
        background: probability(score.background, "background")?,
        fallen: probability(score.fallen, "fallen")?,
        evidence,
    };
    builder::model_score_record(stream, frame, observed_at_ns, &score, accelerator)
        .map_err(RecordError::Contract)
}

// Stage3B owns the head's class origins. Its public canonical payload method
// is the only frozen API exposing them; do not duplicate its private constants.
fn class_origins(score: &ModelScore) -> Result<Vec<String>, RecordError> {
    let Json::Object(payload) = score.payload() else {
        return Err(RecordError::ModelEvidenceShape);
    };
    let Some((_, Json::Array(origins))) =
        payload.into_iter().find(|(key, _)| key == "class_origins")
    else {
        return Err(RecordError::ModelEvidenceShape);
    };
    origins
        .into_iter()
        .map(|origin| match origin {
            Json::Str(origin) => Ok(origin),
            _ => Err(RecordError::ModelEvidenceShape),
        })
        .collect()
}
