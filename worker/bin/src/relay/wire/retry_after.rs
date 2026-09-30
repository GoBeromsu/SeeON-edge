//! Python `_retry_after` (`shared/events/evidence_http_transport.py`): a
//! `Retry-After` value becomes seconds to wait, clamped to [0, 900]. A number
//! is taken as delta seconds; otherwise an IMF-fixdate (RFC 9110
//! `Sun, 06 Nov 1994 08:49:37 GMT`) minus the injected wall clock; anything
//! else is no hint at all. The date parser is written by hand on purpose: the
//! crate carries no date dependency.

use std::time::{SystemTime, UNIX_EPOCH};

/// Upper bound on any retry hint, in seconds (Python `min(900.0, ...)`).
pub const MAX_RETRY_AFTER_SECONDS: f64 = 900.0;

const DAY_NAMES: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
const MONTH_NAMES: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];
const SECONDS_PER_DAY: i64 = 86_400;
const MICROS_PER_SECOND: i128 = 1_000_000;

/// Seconds to wait before retrying, or `None` when `value` is absent or is
/// neither a number nor an IMF-fixdate. `now` is the `Clock::wall()` reading.
pub fn retry_after(value: Option<&str>, now: SystemTime) -> Option<f64> {
    let value = value?;
    if let Some(seconds) = python_float(value) {
        return Some(clamp(seconds));
    }
    let target = imf_fixdate_seconds(value)?;
    Some(clamp(seconds_until(target, now)))
}

/// Python `min(900.0, max(0.0, value))`, including its NaN behaviour: a NaN
/// never compares greater, so `max` keeps 0.0.
fn clamp(value: f64) -> f64 {
    let floored = if value > 0.0 { value } else { 0.0 };
    if floored < MAX_RETRY_AFTER_SECONDS {
        floored
    } else {
        MAX_RETRY_AFTER_SECONDS
    }
}

/// Python `float(str)`: surrounding whitespace is ignored and a single `_`
/// may separate two digits. `inf`, `infinity` and `nan` are accepted in any
/// case, as both languages do.
fn python_float(value: &str) -> Option<f64> {
    let trimmed = value.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c));
    let chars: Vec<char> = trimmed.chars().collect();
    let mut cleaned = String::with_capacity(trimmed.len());
    for (index, &c) in chars.iter().enumerate() {
        if c != '_' {
            cleaned.push(c);
            continue;
        }
        let before = index.checked_sub(1).and_then(|i| chars.get(i));
        let after = chars.get(index + 1);
        let between_digits =
            before.is_some_and(char::is_ascii_digit) && after.is_some_and(char::is_ascii_digit);
        if !between_digits {
            return None;
        }
    }
    cleaned.parse::<f64>().ok()
}

/// Seconds since the Unix epoch for an IMF-fixdate, or `None` when `value`
/// is not one. The day name is checked for spelling only, as Python's
/// `parsedate_to_datetime` does not check it against the date either.
fn imf_fixdate_seconds(value: &str) -> Option<i64> {
    let tokens: Vec<&str> = value.split_ascii_whitespace().collect();
    let [day_name, day, month, year, time, zone] = tokens.as_slice() else {
        return None;
    };
    let day_name = day_name.strip_suffix(',')?;
    if !DAY_NAMES
        .iter()
        .any(|name| day_name.eq_ignore_ascii_case(name))
    {
        return None;
    }
    if !zone.eq_ignore_ascii_case("GMT") {
        return None;
    }
    let day = digits(day, 1, 2)?;
    let month = MONTH_NAMES
        .iter()
        .position(|name| month.eq_ignore_ascii_case(name))
        .and_then(|index| i64::try_from(index + 1).ok())?;
    let year = digits(year, 4, 4)?;
    if year == 0 || !(1..=days_in_month(year, month)).contains(&day) {
        return None;
    }
    let mut clock = time.split(':');
    let (Some(hour), Some(minute), Some(second), None) =
        (clock.next(), clock.next(), clock.next(), clock.next())
    else {
        return None;
    };
    let (hour, minute, second) = (
        digits(hour, 2, 2)?,
        digits(minute, 2, 2)?,
        digits(second, 2, 2)?,
    );
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    Some(days_from_civil(year, month, day) * SECONDS_PER_DAY + hour * 3_600 + minute * 60 + second)
}

/// An ASCII decimal of `min..=max` digits.
fn digits(text: &str, min: usize, max: usize) -> Option<i64> {
    let valid = (min..=max).contains(&text.len()) && text.bytes().all(|b| b.is_ascii_digit());
    if !valid {
        return None;
    }
    text.parse().ok()
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// `target - now` in seconds with microsecond resolution, as Python's
/// `timedelta.total_seconds()` reports it.
fn seconds_until(target: i64, now: SystemTime) -> f64 {
    let now_micros = match now.duration_since(UNIX_EPOCH) {
        Ok(since) => i128::try_from(since.as_micros()).unwrap_or(i128::MAX),
        Err(before) => -i128::try_from(before.duration().as_micros()).unwrap_or(i128::MAX),
    };
    let delta = i128::from(target) * MICROS_PER_SECOND - now_micros;
    delta as f64 / MICROS_PER_SECOND as f64
}
