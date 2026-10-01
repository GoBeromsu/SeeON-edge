//! The runtime snapshot is the sole source of production `Decision` values.
//! Content identity is `worker/types/trace.py:decision_trace_id`: one SHA-256
//! over canonical snapshot JSON, shared by alert audit and `policy.decision`.

use std::fmt;

use seeon_worker::trace::{DecisionTraceSnapshot, DecisionTraceValueName, NumericTraceValue};

use crate::json::{Json, JsonError, Serialiser};
use crate::records::builder::{Decision, Numeric};
use crate::records::id::sha256_hex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionError {
    IntegerOutOfRange(DecisionTraceValueName),
    Canonical(JsonError),
}

impl fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntegerOutOfRange(_) => {
                formatter.write_str("decision integer exceeds the record numeric range")
            }
            Self::Canonical(_) => formatter.write_str("decision trace canonical form was refused"),
        }
    }
}

impl std::error::Error for DecisionError {}

impl From<JsonError> for DecisionError {
    fn from(error: JsonError) -> Self {
        Self::Canonical(error)
    }
}

/// Preserve domain wire tokens and numeric kinds; never truncate a `usize`.
pub fn adapt_decision(snapshot: &DecisionTraceSnapshot) -> Result<Decision, DecisionError> {
    let values = snapshot
        .values()
        .iter()
        .map(|(name, value)| {
            let value = match value {
                NumericTraceValue::Integer(value) => Numeric::Int(
                    i64::try_from(*value).map_err(|_| DecisionError::IntegerOutOfRange(*name))?,
                ),
                NumericTraceValue::Float(value) => Numeric::Float(value.get()),
            };
            Ok((name.as_str().to_owned(), value))
        })
        .collect::<Result<_, DecisionError>>()?;
    Ok(Decision {
        reason: snapshot.reason.as_str().to_owned(),
        previous_state: snapshot.previous_state.as_str().to_owned(),
        current_state: snapshot.current_state.as_str().to_owned(),
        triggered: snapshot.triggered,
        track_id: snapshot.track_id,
        bed_id: snapshot.bed_id,
        values,
        missing_values: snapshot
            .missing_values()
            .iter()
            .map(|(name, reason)| (name.as_str().to_owned(), reason.as_str().to_owned()))
            .collect(),
    })
}

/// `worker/types/trace.py:decision_trace_id`.
///
/// The body is the snapshot plus module and effective-policy identity only.
/// `None` stays JSON null and is distinct from numeric zero. Snapshot integers
/// keep their full `usize` range; the execution-record `i64` limit belongs to
/// `adapt_decision` and is not applied here. No clock, process, frame pixel,
/// or random input is read. The caller supplies the producing module identity.
pub fn trace_id(
    snapshot: &DecisionTraceSnapshot,
    module_qualified_id: &str,
    effective_policy_id: &str,
) -> Result<String, DecisionError> {
    let body = trace_body(snapshot, module_qualified_id, effective_policy_id)?;
    // `ensure_ascii=False`, compact separators, sorted keys. Not the
    // model-selection ASCII encoder and not the policy-record payload.
    let encoded = Serialiser::ExecutionRecords.canonical(&body)?;
    Ok(sha256_hex(encoded.as_bytes()))
}

fn trace_body(
    snapshot: &DecisionTraceSnapshot,
    module: &str,
    policy: &str,
) -> Result<Json, DecisionError> {
    let text = |value: &str| Json::Str(value.to_owned());
    let values = snapshot
        .values()
        .iter()
        .map(|(name, value)| {
            let number = match value {
                NumericTraceValue::Integer(value) => Json::Int(
                    i128::try_from(*value).map_err(|_| DecisionError::IntegerOutOfRange(*name))?,
                ),
                NumericTraceValue::Float(value) => Json::Float(value.get()),
            };
            Ok((name.as_str().to_owned(), number))
        })
        .collect::<Result<_, DecisionError>>()?;
    let optional =
        |value: Option<u64>| value.map_or(Json::Null, |value| Json::Int(i128::from(value)));
    Ok(Json::Object(vec![
        ("module".to_owned(), text(module)),
        ("effective_policy_id".to_owned(), text(policy)),
        ("reason".to_owned(), text(snapshot.reason.as_str())),
        (
            "previous_state".to_owned(),
            text(snapshot.previous_state.as_str()),
        ),
        (
            "current_state".to_owned(),
            text(snapshot.current_state.as_str()),
        ),
        ("triggered".to_owned(), Json::Bool(snapshot.triggered)),
        ("track_id".to_owned(), optional(snapshot.track_id)),
        ("bed_id".to_owned(), optional(snapshot.bed_id)),
        ("values".to_owned(), Json::Object(values)),
        (
            "missing_values".to_owned(),
            Json::Object(
                snapshot
                    .missing_values()
                    .iter()
                    .map(|(name, reason)| (name.as_str().to_owned(), text(reason.as_str())))
                    .collect(),
            ),
        ),
    ]))
}
