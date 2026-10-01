//! UTC instants at microsecond precision and the two ISO renderings the
//! clip manifest carries: `…SS.ffffffZ` and `…SS.mmmZ` (truncated).

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

const MICROS_PER_SECOND: i64 = 1_000_000;
const SECONDS_PER_DAY: i64 = 86_400;

/// Microseconds since the Unix epoch, UTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Utc(i64);

/// A timestamp outside `YYYY-MM-DDTHH:MM:SS[.f{1,6}](Z|+00:00)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeError;

impl fmt::Display for TimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("timestamp is not a UTC ISO-8601 instant")
    }
}
impl std::error::Error for TimeError {}

impl Utc {
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    pub const fn micros(self) -> i64 {
        self.0
    }

    pub fn from_system(time: SystemTime) -> Self {
        let micros =
            |duration: std::time::Duration| i64::try_from(duration.as_micros()).unwrap_or(i64::MAX);
        match time.duration_since(UNIX_EPOCH) {
            Ok(after) => Self(micros(after)),
            Err(before) => Self(-micros(before.duration())),
        }
    }

    /// Parses the UTC forms the Python producers write.
    pub fn parse(text: &str) -> Result<Self, TimeError> {
        let body = text
            .strip_suffix('Z')
            .or_else(|| text.strip_suffix("+00:00"))
            .ok_or(TimeError)?;
        let (clock, fraction) = match body.split_once('.') {
            Some((clock, fraction)) => (clock, Some(fraction)),
            None => (body, None),
        };
        let bytes = clock.as_bytes();
        let shape = bytes.len() == 19
            && bytes.iter().enumerate().all(|(index, byte)| match index {
                4 | 7 => *byte == b'-',
                10 => *byte == b'T',
                13 | 16 => *byte == b':',
                _ => byte.is_ascii_digit(),
            });
        if !shape {
            return Err(TimeError);
        }
        let field = |range: std::ops::Range<usize>| -> i64 {
            clock[range]
                .bytes()
                .fold(0, |sum, digit| sum * 10 + i64::from(digit - b'0'))
        };
        let (year, month, day) = (field(0..4), field(5..7), field(8..10));
        let (hour, minute, second) = (field(11..13), field(14..16), field(17..19));
        let valid = year >= 1
            && (1..=12).contains(&month)
            && day >= 1
            && day <= days_in_month(year, month)
            && hour < 24
            && minute < 60
            && second < 60;
        if !valid {
            return Err(TimeError);
        }
        let micros = match fraction {
            None => 0,
            Some(digits)
                if (1..=6).contains(&digits.len())
                    && digits.bytes().all(|byte| byte.is_ascii_digit()) =>
            {
                let value = digits
                    .bytes()
                    .fold(0, |sum, digit| sum * 10 + i64::from(digit - b'0'));
                value * 10_i64.pow(6 - u32::try_from(digits.len()).unwrap_or(6))
            }
            Some(_) => return Err(TimeError),
        };
        let seconds = days_from_civil(year, month, day) * SECONDS_PER_DAY
            + hour * 3600
            + minute * 60
            + second;
        Ok(Self(seconds * MICROS_PER_SECOND + micros))
    }

    pub const fn plus_millis(self, millis: i64) -> Self {
        Self(self.0 + millis * 1000)
    }

    /// `YYYY-MM-DDTHH:MM:SS.ffffffZ`.
    pub fn iso_micros(self) -> String {
        format!(
            "{}.{:06}Z",
            self.clock(),
            self.0.rem_euclid(MICROS_PER_SECOND)
        )
    }

    /// `YYYY-MM-DDTHH:MM:SS.mmmZ`, the microseconds truncated.
    pub fn iso_millis(self) -> String {
        format!(
            "{}.{:03}Z",
            self.clock(),
            self.0.rem_euclid(MICROS_PER_SECOND) / 1000
        )
    }

    fn clock(self) -> String {
        let seconds = self.0.div_euclid(MICROS_PER_SECOND);
        let (days, of_day) = (
            seconds.div_euclid(SECONDS_PER_DAY),
            seconds.rem_euclid(SECONDS_PER_DAY),
        );
        let (year, month, day) = civil_from_days(days);
        format!(
            "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
            of_day / 3600,
            of_day / 60 % 60,
            of_day % 60
        )
    }
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Howard Hinnant's `days_from_civil`.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}
