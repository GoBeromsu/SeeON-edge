//! One journal line: pydantic `_PersistedIdentity.model_dump_json()` bytes
//! out, `_parse_line` semantics in (`event_identity.py` L148-L171).

use std::cmp::Ordering;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Record {
    pub(super) source_key: String,
    /// Lowercase hyphenated version-4 UUID, as `str(UUID)` writes it.
    pub(super) edge_event_id: String,
    pub(super) recorded_at: f64,
}

impl Record {
    /// Key order, escaping and float text of pydantic's JSON serializer
    /// (serde_json and ryu), plus the newline.
    pub(super) fn encode(&self) -> String {
        format!(
            "{{\"source_key\":{},\"edge_event_id\":\"{}\",\"recorded_at\":{}}}\n",
            Value::String(self.source_key.clone()),
            self.edge_event_id,
            Value::from(self.recorded_at),
        )
    }

    /// A persisted line, or a legacy line without `recorded_at` stamped with
    /// `now`. `None` for anything else and for a record dated after `now`.
    pub(super) fn parse(line: &str, now: f64) -> Option<Self> {
        let text = strip(line);
        let Value::Object(fields) = serde_json::from_str::<Value>(text).ok()? else {
            return None;
        };
        let source_key = fields.get("source_key")?.as_str()?.to_owned();
        let edge_event_id = normalize_uuid4(fields.get("edge_event_id")?.as_str()?)?;
        let recorded_at = match fields.len() {
            2 => now,
            3 if fields.get("recorded_at")?.is_number() => {
                last_member_text(text, "recorded_at")?.parse::<f64>().ok()?
            }
            _ => return None,
        };
        let future = recorded_at.partial_cmp(&now) == Some(Ordering::Greater);
        (!future).then_some(Self {
            source_key,
            edge_event_id,
            recorded_at,
        })
    }
}

/// Python `str.strip()`: Unicode whitespace plus the separators U+001C-U+001F.
pub(super) fn strip(line: &str) -> &str {
    line.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// pydantic `UUID4` input forms (simple, hyphenated, braced, `urn:uuid:`),
/// case-insensitive, version nibble 4; normalised to lowercase hyphenated.
pub(super) fn normalize_uuid4(text: &str) -> Option<String> {
    let hex = match text.len() {
        32 => text.to_owned(),
        36 => dehyphenate(text)?,
        38 => dehyphenate(text.strip_prefix('{')?.strip_suffix('}')?)?,
        45 => dehyphenate(text.strip_prefix("urn:uuid:")?)?,
        _ => return None,
    };
    if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let hex = hex.to_ascii_lowercase();
    if hex.as_bytes()[12] != b'4' {
        return None;
    }
    Some(format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}

fn dehyphenate(text: &str) -> Option<String> {
    let groups: Vec<&str> = text.split('-').collect();
    let lengths = groups.iter().map(|group| group.len());
    lengths.eq([8, 4, 4, 4, 12]).then(|| groups.concat())
}

/// Source text of the last top-level member `name` whose value is a number.
/// serde_json without `float_roundtrip` may round a long decimal one ulp away
/// from Python's `float()`; `str::parse::<f64>` rounds correctly. `text` is an
/// object serde_json already accepted, so strings and nesting are well formed.
fn last_member_text<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let bytes = text.as_bytes();
    let (mut depth, mut index) = (0usize, 0usize);
    let (mut expect_key, mut key, mut value_start) = (false, None::<String>, None);
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let end = string_end(bytes, index)?;
                if depth == 1 && expect_key {
                    key = serde_json::from_str::<String>(&text[index..=end]).ok();
                    expect_key = false;
                }
                index = end;
            }
            b'{' | b'[' => {
                depth += 1;
                expect_key = depth == 1;
            }
            b'}' | b']' => depth = depth.checked_sub(1)?,
            b',' if depth == 1 => expect_key = true,
            b':' if depth == 1 && key.as_deref() == Some(name) => value_start = Some(index + 1),
            _ => {}
        }
        index += 1;
    }
    let rest = text[value_start?..].trim_start_matches([' ', '\t', '\n', '\r']);
    let length = rest
        .bytes()
        .take_while(|byte| matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        .count();
    (length > 0).then(|| &rest[..length])
}

/// Index of the quote that closes the string opened at `start`.
fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => return Some(index),
            _ => index += 1,
        }
    }
    None
}
