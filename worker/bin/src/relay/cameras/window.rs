//! Python `_NightWindowPayload` (`worker/runtime/config/pull_models.py`) and
//! `detection_window_validation_error` (`contracts/worker_config.py`): one
//! pulled detection window with `HH:MM` bounds.

use crate::config::lookup;
use crate::json::Json;

/// One pulled detection window, kept exactly as the relay spelled it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetectionWindow {
    pub start: String,
    pub end: String,
    pub tz: String,
}

impl DetectionWindow {
    /// `_NightWindowPayload.model_validate`: an object with exactly `start`,
    /// `end` and `tz`, each a non-empty string.
    pub(super) fn shape(value: &Json) -> Option<Self> {
        let Json::Object(members) = value else {
            return None;
        };
        if members
            .iter()
            .any(|(key, _)| !matches!(key.as_str(), "start" | "end" | "tz"))
        {
            return None;
        }
        let text = |key: &str| match lookup(members, key) {
            Some(Json::Str(text)) if !text.is_empty() => Some(text.clone()),
            _ => None,
        };
        Some(Self {
            start: text("start")?,
            end: text("end")?,
            tz: text("tz")?,
        })
    }

    /// `detection_window_validation_error(...) is None`, except the IANA zone
    /// is not checked: the crate carries no time-zone database. Both bounds
    /// must parse as `strptime("%H:%M")` and name different minute values.
    /// The original strings are left unchanged; a later root adapter can
    /// normalise a valid pair to ASCII before the ASCII-bounded primitive.
    pub(super) fn is_valid(&self) -> bool {
        self.parsed_minutes().is_some()
    }

    /// Parsed minute-of-day pair, or `None` when either bound is malformed
    /// or the endpoints name the same minute. Crate-private so callers can
    /// normalise without a public test-only accessor.
    pub(crate) fn parsed_minutes(&self) -> Option<(u16, u16)> {
        let start = clock_minutes(&self.start)?;
        let end = clock_minutes(&self.end)?;
        (start != end).then_some((start, end))
    }
}

/// `strptime(text, "%H:%M")` as minutes past midnight.
///
/// CPython 3.12 `_strptime.TimeRE` compiles `%H:%M` to the anchored pattern
/// `(2[0-3]|[0-1]\d|\d):([0-5]\d|\d)`. The bracket classes are ASCII. Only
/// `\d` is a Unicode decimal digit (`Nd`), not `is_numeric` or `isdigit`.
/// `int()` then reads each matched component by its decimal value. A sign,
/// space, or extra character fails the whole-string match.
fn clock_minutes(text: &str) -> Option<u16> {
    let (hour, minute) = text.split_once(':')?;
    Some(hour_field(hour)? * 60 + minute_field(minute)?)
}

/// `2[0-3]|[0-1]Nd|Nd`, anchored to the whole component.
///
/// The ASCII class consumes the following character even when it misses, so
/// `2` plus a non-ASCII character does not fall through to one-digit `Nd`.
fn hour_field(text: &str) -> Option<u16> {
    let mut chars = text.chars();
    let first = chars.next()?;
    let Some(second) = chars.next() else {
        return decimal_value(first).map(u16::from);
    };
    if chars.next().is_some() {
        return None;
    }
    match (first, second) {
        ('0' | '1', units) => {
            Some(u16::from(digit_value(first)) * 10 + u16::from(decimal_value(units)?))
        }
        ('2', units @ '0'..='3') => Some(20 + u16::from(digit_value(units))),
        ('2', _) => None,
        _ => None,
    }
}

/// `[0-5]Nd|Nd`, anchored to the whole component. An ASCII digit outside
/// `0..=5` consumes the following character and fails rather than retrying
/// the one-digit alternative.
fn minute_field(text: &str) -> Option<u16> {
    let mut chars = text.chars();
    let first = chars.next()?;
    let Some(second) = chars.next() else {
        return decimal_value(first).map(u16::from);
    };
    if chars.next().is_some() {
        return None;
    }
    match first {
        '0'..='5' => Some(u16::from(digit_value(first)) * 10 + u16::from(decimal_value(second)?)),
        '6'..='9' => None,
        _ => None,
    }
}

fn digit_value(ch: char) -> u8 {
    u8::try_from(u32::from(ch) - u32::from('0')).expect("ascii digit")
}

/// Decimal value of one Unicode 15.0.0 `Nd` character, or `None`.
fn decimal_value(ch: char) -> Option<u8> {
    let code = u32::from(ch);
    DECIMAL_ZEROES
        .iter()
        .find_map(|zero| code.checked_sub(*zero).filter(|digit| *digit < 10))
        .map(|digit| u8::try_from(digit).expect("digit is 0..=9"))
}

/// Unicode 15.0.0 decimal-digit zeroes, pinned from the deployed image
/// (Python 3.12.3, `unicodedata` 15.0.0). Each entry starts a contiguous
/// run of ten `Nd` characters with values 0..=9. This is a fixed reference
/// table, not a generated file and not `char::is_numeric`: that predicate
/// also accepts numbers (`No`) and other numerics that `strptime` rejects.
const DECIMAL_ZEROES: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];
