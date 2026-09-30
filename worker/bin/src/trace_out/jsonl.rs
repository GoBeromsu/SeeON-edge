//! `json.dumps(value, separators=(",", ":"))`, the `encode_jsonl` line body.

use crate::json::{Json, JsonError, Serialiser};

/// Leaves and keys encode as `json.dumps` does by default: ASCII escapes;
/// rows are validated finite before encoding, so NaN is never reached.
const LEAF: Serialiser = Serialiser::ModelSelection;

/// `json.dumps(value, separators=(",", ":"))` without `sort_keys`: object
/// members keep their insertion order.
pub fn encode_ordered(value: &Json) -> Result<String, JsonError> {
    let mut text = String::new();
    write_ordered(value, &mut text)?;
    Ok(text)
}

fn write_ordered(value: &Json, text: &mut String) -> Result<(), JsonError> {
    match value {
        Json::Array(items) => {
            text.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    text.push(',');
                }
                write_ordered(item, text)?;
            }
            text.push(']');
        }
        Json::Object(members) => {
            text.push('{');
            for (index, (key, item)) in members.iter().enumerate() {
                if members[..index].iter().any(|(seen, _)| seen == key) {
                    return Err(JsonError::DuplicateKey);
                }
                if index > 0 {
                    text.push(',');
                }
                text.push_str(&LEAF.canonical(&Json::Str(key.clone()))?);
                text.push(':');
                write_ordered(item, text)?;
            }
            text.push('}');
        }
        leaf => text.push_str(&LEAF.canonical(leaf)?),
    }
    Ok(())
}
