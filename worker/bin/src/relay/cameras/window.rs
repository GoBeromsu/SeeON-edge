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

    /// `detection_window_validation_error(...) is None`: both bounds parse
    /// as `strptime("%H:%M")` and name different times. The IANA zone is
    /// not checked: the crate carries no time-zone database.
    pub(super) fn is_valid(&self) -> bool {
        match (clock_minutes(&self.start), clock_minutes(&self.end)) {
            (Some(start), Some(end)) => start != end,
            _ => false,
        }
    }
}

/// `strptime(text, "%H:%M")` as minutes past midnight: the hour is one or
/// two digits up to 23, the minute one or two digits up to 59, and nothing
/// may follow.
fn clock_minutes(text: &str) -> Option<u32> {
    let (hour, minute) = text.split_once(':')?;
    Some(clock_field(hour, 23)? * 60 + clock_field(minute, 59)?)
}

fn clock_field(text: &str, max: u32) -> Option<u32> {
    if text.is_empty() || text.len() > 2 || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let value: u32 = text.parse().ok()?;
    (value <= max).then_some(value)
}
