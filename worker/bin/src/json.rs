//! Python-canonical JSON text (B3): the bytes `json.dumps(value,
//! sort_keys=True, separators=(",", ":"))` writes in each producer's
//! `ensure_ascii` and `allow_nan` mode. Floats use Python `repr`, keys sort
//! by code point, and every refusal is a typed `Err`.

use std::fmt;

use sha2::{Digest, Sha256};

/// A JSON value as the Python producers see it. `Float` holds NaN and ±inf,
/// which `serde_json::Value` cannot. `Object` keeps insertion order; the
/// writer sorts it.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl From<&serde_json::Value> for Json {
    fn from(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(flag) => Self::Bool(*flag),
            serde_json::Value::Number(number) => match (number.as_i64(), number.as_u64()) {
                (Some(signed), _) => Self::Int(i128::from(signed)),
                (None, Some(unsigned)) => Self::Int(i128::from(unsigned)),
                (None, None) => Self::Float(number.as_f64().unwrap_or(f64::NAN)),
            },
            serde_json::Value::String(text) => Self::Str(text.clone()),
            serde_json::Value::Array(items) => Self::Array(items.iter().map(Self::from).collect()),
            serde_json::Value::Object(members) => Self::Object(
                members
                    .iter()
                    .map(|(key, member)| (key.clone(), Self::from(member)))
                    .collect(),
            ),
        }
    }
}

/// Static refusals; no value text is carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JsonError {
    /// NaN or ±inf where the producer passes `allow_nan=False`.
    NonFinite,
    /// A provenance string that is a URL, an absolute path or holds a
    /// NUL, CR or LF.
    UnsafeValue,
    /// Two members of one object share a key; a Python dict cannot.
    DuplicateKey,
}

impl fmt::Display for JsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFinite => "out of range float values are not JSON compliant",
            Self::UnsafeValue => "provenance value must be a path-free, single-line string",
            Self::DuplicateKey => "object holds a duplicate key",
        })
    }
}
impl std::error::Error for JsonError {}

/// The four Python producers of canonical JSON, each with its own mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Serialiser {
    /// `shared/events/execution_records.py:canonical_json`.
    ExecutionRecords,
    /// `worker/tools/fetch_models/manifest.py:canonical_json`.
    FetchModelsManifest,
    /// `contracts/model_selection.py:canonical_json_bytes`.
    ModelSelection,
    /// `worker/runtime/provenance/models.py:canonical_json`.
    Provenance,
}

#[derive(Clone, Copy)]
struct Mode {
    ensure_ascii: bool,
    allow_nan: bool,
}

impl Serialiser {
    const fn mode(self) -> Mode {
        Mode {
            ensure_ascii: matches!(self, Self::ModelSelection),
            allow_nan: matches!(self, Self::ExecutionRecords | Self::Provenance),
        }
    }

    pub fn canonical(self, value: &Json) -> Result<String, JsonError> {
        if self == Self::Provenance {
            reject_unsafe(value)?;
        }
        let mut text = String::new();
        write_value(value, self.mode(), &mut text)?;
        Ok(text)
    }
}

/// `contracts/model_selection.py:canonical_digest`: lowercase hex sha256 of
/// the model-selection canonical bytes.
pub fn model_selection_digest(value: &Json) -> Result<String, JsonError> {
    let text = Serialiser::ModelSelection.canonical(value)?;
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(text.as_bytes()) {
        push_hex(&mut hex, byte);
    }
    Ok(hex)
}

fn reject_unsafe(value: &Json) -> Result<(), JsonError> {
    match value {
        Json::Str(text) => {
            let bytes = text.as_bytes();
            let drive = bytes.len() >= 3
                && bytes[0].is_ascii_alphabetic()
                && bytes[1] == b':'
                && matches!(bytes[2], b'\\' | b'/');
            if text.contains("://")
                || text.starts_with('/')
                || text.starts_with("\\\\")
                || drive
                || text.contains(['\0', '\n', '\r'])
            {
                return Err(JsonError::UnsafeValue);
            }
            Ok(())
        }
        Json::Array(items) => items.iter().try_for_each(reject_unsafe),
        Json::Object(members) => members.iter().try_for_each(|(_, item)| reject_unsafe(item)),
        Json::Null | Json::Bool(_) | Json::Int(_) | Json::Float(_) => Ok(()),
    }
}

fn write_value(value: &Json, mode: Mode, text: &mut String) -> Result<(), JsonError> {
    match value {
        Json::Null => text.push_str("null"),
        Json::Bool(flag) => text.push_str(if *flag { "true" } else { "false" }),
        Json::Int(number) => text.push_str(&number.to_string()),
        Json::Float(number) => write_float(*number, mode.allow_nan, text)?,
        Json::Str(string) => write_str(string, mode.ensure_ascii, text),
        Json::Array(items) => {
            text.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    text.push(',');
                }
                write_value(item, mode, text)?;
            }
            text.push(']');
        }
        Json::Object(members) => {
            let mut sorted: Vec<&(String, Json)> = members.iter().collect();
            sorted.sort_by(|left, right| left.0.cmp(&right.0));
            if sorted.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(JsonError::DuplicateKey);
            }
            text.push('{');
            for (index, (key, member)) in sorted.into_iter().enumerate() {
                if index > 0 {
                    text.push(',');
                }
                write_str(key, mode.ensure_ascii, text);
                text.push(':');
                write_value(member, mode, text)?;
            }
            text.push('}');
        }
    }
    Ok(())
}

/// Python `float.__repr__`: the shortest round-trip digits, positional for
/// decimal exponents -4 through 15, else `d.ddde±XX`.
fn write_float(number: f64, allow_nan: bool, text: &mut String) -> Result<(), JsonError> {
    if !number.is_finite() {
        if !allow_nan {
            return Err(JsonError::NonFinite);
        }
        text.push_str(match (number.is_nan(), number > 0.0) {
            (true, _) => "NaN",
            (false, true) => "Infinity",
            (false, false) => "-Infinity",
        });
        return Ok(());
    }
    if number.is_sign_negative() {
        text.push('-');
    }
    let scientific = format!("{:e}", number.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if !(-4..16).contains(&exponent) {
        let (lead, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        let sign = if exponent < 0 { '-' } else { '+' };
        text.push_str(&format!("{lead}{point}{rest}e{sign}{:02}", exponent.abs()));
    } else if exponent < 0 {
        text.push_str("0.");
        text.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
        text.push_str(&digits);
    } else {
        let point = exponent as usize + 1;
        let (whole, fraction) = digits.split_at(point.min(digits.len()));
        text.push_str(whole);
        text.extend(std::iter::repeat_n('0', point - whole.len()));
        text.push('.');
        text.push_str(if fraction.is_empty() { "0" } else { fraction });
    }
    Ok(())
}

fn write_str(string: &str, ensure_ascii: bool, text: &mut String) {
    text.push('"');
    for character in string.chars() {
        match character {
            '"' => text.push_str("\\\""),
            '\\' => text.push_str("\\\\"),
            '\n' => text.push_str("\\n"),
            '\r' => text.push_str("\\r"),
            '\t' => text.push_str("\\t"),
            '\u{8}' => text.push_str("\\b"),
            '\u{c}' => text.push_str("\\f"),
            _ if character < ' ' || (ensure_ascii && character > '~') => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    text.push_str("\\u");
                    let [high, low] = unit.to_be_bytes();
                    push_hex(text, high);
                    push_hex(text, low);
                }
            }
            _ => text.push(character),
        }
    }
    text.push('"');
}

fn push_hex(text: &mut String, byte: u8) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    text.push(char::from(HEX[usize::from(byte >> 4)]));
    text.push(char::from(HEX[usize::from(byte & 0x0f)]));
}
