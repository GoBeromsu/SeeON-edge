//! The runtime snapshot is the sole source of production `Decision` values.

use std::fmt;

use seeon_worker::trace::{DecisionTraceSnapshot, DecisionTraceValueName, NumericTraceValue};

use crate::records::builder::{Decision, Numeric};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionError {
    IntegerOutOfRange(DecisionTraceValueName),
}

impl fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("decision integer exceeds the record numeric range")
    }
}

impl std::error::Error for DecisionError {}

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
