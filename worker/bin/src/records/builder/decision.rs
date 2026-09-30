//! `policy.decision`, evaluated or coasted
//! (`worker/pipeline/diagnostics/emit_policy.py` L137-273).

use super::{
    Draft, FALL_MODULE_QUALIFIED_ID, Frame, PRODUCER_POLICY, Stream, fall_causal_unit_id, float,
    make_record, module_causal_unit_id, optional_int, optional_text,
};
use crate::json::Json;
use crate::records::id::{ContractError, int, optional, text};
use crate::records::wire::{Record, RecordKind};

/// `NumericTraceValue`: a decision input is an int or a float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Numeric {
    Int(i64),
    Float(f64),
}

/// `DecisionTraceSnapshot`.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub reason: String,
    pub previous_state: String,
    pub current_state: String,
    pub triggered: bool,
    pub track_id: Option<u64>,
    pub bed_id: Option<u64>,
    pub values: Vec<(String, Numeric)>,
    pub missing_values: Vec<(String, String)>,
}

/// Whether a decision is a cause or a shadow evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityRole {
    Authoritative,
    Shadow,
}

impl AuthorityRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authoritative => "authoritative",
            Self::Shadow => "shadow",
        }
    }
}

/// Which compiled module produced a decision and how it is attributed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionSource {
    pub generation: Option<u64>,
    pub module_qualified_id: Option<String>,
    pub authority_role: AuthorityRole,
    pub decision_trace_id: Option<String>,
}

/// `policy_decision_record`: only the fall module joins the fall unit; any
/// other or unattributed module gets a module-scoped frame unit.
pub fn policy_decision_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    decision: &Decision,
    source: &DecisionSource,
) -> Result<Record, ContractError> {
    let module = source.module_qualified_id.as_deref();
    let causal_unit_id = if module == Some(FALL_MODULE_QUALIFIED_ID) {
        fall_causal_unit_id(stream, decision.track_id, source.generation)
    } else {
        module_causal_unit_id(stream, module, frame.frame_seq)
    };
    let values = decision.values.iter().map(|(key, value)| match value {
        Numeric::Int(number) => int(key, *number),
        Numeric::Float(number) => float(key, *number),
    });
    let missing = decision
        .missing_values
        .iter()
        .map(|(key, value)| text(key, value));
    let draft = Draft {
        record_kind: RecordKind::PolicyDecision,
        producer: PRODUCER_POLICY,
        causal_unit_id,
        outcome: if decision.triggered {
            "triggered".to_owned()
        } else {
            decision.current_state.clone()
        },
        payload: vec![
            text("reason", &decision.reason),
            text("previous_state", &decision.previous_state),
            text("current_state", &decision.current_state),
            ("triggered".to_owned(), Json::Bool(decision.triggered)),
            optional_int("track_id", decision.track_id),
            optional_int("bed_id", decision.bed_id),
            ("values".to_owned(), Json::Object(values.collect())),
            ("missing_values".to_owned(), Json::Object(missing.collect())),
            optional_text("module_qualified_id", module),
            text("authority_role", source.authority_role.as_str()),
            optional_text("decision_trace_id", source.decision_trace_id.as_deref()),
        ],
    };
    make_record(stream, observed_at_ns, Some(frame), draft)
}

/// `policy_coast_record`: a truthful decision for a frame the module did not
/// evaluate (the resampler yielded no row); no track, no score.
pub fn policy_coast_record(
    stream: &Stream,
    frame: Frame,
    observed_at_ns: u64,
    module_qualified_id: Option<&str>,
) -> Result<Record, ContractError> {
    let draft = Draft {
        record_kind: RecordKind::PolicyDecision,
        producer: PRODUCER_POLICY,
        causal_unit_id: module_causal_unit_id(stream, module_qualified_id, frame.frame_seq),
        outcome: "coasted".to_owned(),
        payload: vec![
            text("reason", "score-missing"),
            text("previous_state", "not-evaluated"),
            text("current_state", "not-evaluated"),
            ("triggered".to_owned(), Json::Bool(false)),
            optional("track_id", None),
            optional("bed_id", None),
            ("values".to_owned(), Json::Object(Vec::new())),
            (
                "missing_values".to_owned(),
                Json::Object(vec![text("decision_state", "resample-gap")]),
            ),
            optional_text("module_qualified_id", module_qualified_id),
            text("authority_role", AuthorityRole::Authoritative.as_str()),
            optional("decision_trace_id", None),
        ],
    };
    make_record(stream, observed_at_ns, Some(frame), draft)
}
