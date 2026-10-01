//! The plain model-score value behind `emit_policy.model_score_record`.

use crate::json::Json;

use super::FALL_WINDOW_FRAMES;

const CLASS_ORIGINS: [&str; 3] = ["derived_complement", "temperature_sigmoid", "constant_zero"];

/// The observed binary score behind one result (`BinaryFallScoreEvidence`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelEvidence {
    pub raw_logit: f64,
    pub applied_temperature: f64,
}

/// The plain model-score value one scored track yields per window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelScore {
    pub track_id: i64,
    pub generation: Option<i64>,
    pub fall_transition: Option<f64>,
    pub background: Option<f64>,
    pub fallen: Option<f64>,
    pub evidence: Option<ModelEvidence>,
}

impl ModelScore {
    /// `model_score_record` payload, in its insertion order.
    pub fn payload(&self) -> Json {
        let float = |value: Option<f64>| value.map_or(Json::Null, Json::Float);
        let generation = self
            .generation
            .map_or(Json::Null, |g| Json::Int(i128::from(g)));
        let mut members: Vec<(String, Json)> = vec![
            ("track_id".into(), Json::Int(i128::from(self.track_id))),
            ("generation".into(), generation),
            ("fall_transition".into(), float(self.fall_transition)),
            ("background".into(), float(self.background)),
            ("fallen".into(), float(self.fallen)),
            ("window_frames".into(), Json::Int(FALL_WINDOW_FRAMES)),
        ];
        if let Some(evidence) = self.evidence {
            let origins = CLASS_ORIGINS.iter().map(|o| Json::Str((*o).to_owned()));
            members.push(("raw_logit".into(), Json::Float(evidence.raw_logit)));
            let temperature = Json::Float(evidence.applied_temperature);
            members.push(("applied_temperature".into(), temperature));
            members.push(("class_origins".into(), Json::Array(origins.collect())));
        }
        Json::Object(members)
    }
}
