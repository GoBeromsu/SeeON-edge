//! Python `dict` and `str` semantics over an insertion-ordered payload.

use crate::json::{Json, Serialiser};

use super::{EmitError, Payload};

/// `_required_text`: `"" if value is None else str(value).strip()`.
pub(super) fn required_text(
    event: &[(String, Json)],
    key: &'static str,
) -> Result<String, EmitError> {
    let text = match lookup(event, key) {
        None | Some(Json::Null) => String::new(),
        Some(Json::Str(text)) => text.clone(),
        Some(Json::Int(number)) => number.to_string(),
        Some(Json::Bool(flag)) => (if *flag { "True" } else { "False" }).to_owned(),
        Some(Json::Float(number)) if number.is_nan() => "nan".to_owned(),
        Some(Json::Float(number)) if number.is_infinite() => {
            (if *number > 0.0 { "inf" } else { "-inf" }).to_owned()
        }
        Some(float @ Json::Float(_)) => Serialiser::ExecutionRecords.canonical(float)?,
        Some(Json::Array(_) | Json::Object(_)) => return Err(EmitError::FieldType(key)),
    };
    let text = text.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c));
    if text.is_empty() {
        return Err(EmitError::BlankField(key));
    }
    Ok(text.to_owned())
}

/// Python `dict(pairs)`: a repeated key keeps its first position, last value.
pub(super) fn dict(pairs: &[(String, Json)]) -> Payload {
    let mut out = Payload::with_capacity(pairs.len());
    for (key, value) in pairs {
        upsert(&mut out, key, value.clone());
    }
    out
}

pub(super) fn lookup<'a>(payload: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    payload
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

pub(super) fn upsert(payload: &mut Payload, key: &str, value: Json) {
    match payload.iter_mut().find(|(name, _)| name == key) {
        Some(slot) => slot.1 = value,
        None => payload.push((key.to_owned(), value)),
    }
}

pub(super) fn remove(payload: &mut Payload, key: &str) -> Option<Json> {
    let index = payload.iter().position(|(name, _)| name == key)?;
    Some(payload.remove(index).1)
}
