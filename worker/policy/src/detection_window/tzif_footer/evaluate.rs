//! Offset-only port of CPython v3.12.3 Modules/_zoneinfo.c:
//! ymd_to_ord, calendar/dayrule_year_to_timestamp, find_tzrule_ttinfo_fromutc.
//! Fold annotations never change the chosen offset and are not returned here.
use jiff::Timestamp;
use jiff::civil::Time;
use jiff::tz::Offset;

use super::DetectionWindowError;
use super::posix::{FooterRule, RuleDay, TransitionRule};

const EPOCHORDINAL: i64 = 719163;
const NANOS_PER_SECOND: i128 = 1_000_000_000;
const NANOS_PER_DAY: i128 = 86400 * NANOS_PER_SECOND;
const DAYS_BEFORE_MONTH: [i64; 13] = [-1, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const DAYS_IN_MONTH: [i64; 13] = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

pub(super) fn offset_seconds(
    rule: &FooterRule,
    timestamp: Timestamp,
) -> Result<i32, DetectionWindowError> {
    let (std_seconds, dst_seconds, start, end) = match *rule {
        FooterRule::Fixed { std_seconds } => return Ok(std_seconds),
        FooterRule::Alternate {
            std_seconds,
            dst_seconds,
            start,
            end,
        } => (std_seconds, dst_seconds, start, end),
    };
    // zoneinfo_fromutc passes the input UTC datetime's year, not a local year.
    let year = Offset::UTC.to_datetime(timestamp).year();
    let ts = i64::try_from(timestamp.as_nanosecond().div_euclid(NANOS_PER_SECOND))
        .map_err(|_| DetectionWindowError::ClockOutOfRange)?;
    let start = transition_second(&start, year)?
        .checked_sub(i64::from(std_seconds))
        .ok_or(DetectionWindowError::ClockOutOfRange)?;
    let end = transition_second(&end, year)?
        .checked_sub(i64::from(dst_seconds))
        .ok_or(DetectionWindowError::ClockOutOfRange)?;
    let is_dst = if start < end {
        start <= ts && ts < end
    } else {
        ts < end || ts >= start
    };
    Ok(if is_dst { dst_seconds } else { std_seconds })
}

fn is_leap_year(year: i16) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn ymd_to_ord(year: i16, month: u8, day: i64) -> i64 {
    let y = i64::from(year) - 1;
    let mut yearday = DAYS_BEFORE_MONTH[usize::from(month)];
    if month > 2 && is_leap_year(year) {
        yearday += 1;
    }
    y * 365 + y / 4 - y / 100 + y / 400 + yearday + day
}

fn transition_second(rule: &TransitionRule, utc_year: i16) -> Result<i64, DetectionWindowError> {
    if !(1..=9999).contains(&utc_year) {
        return Err(DetectionWindowError::ClockOutOfRange);
    }
    let days = match rule.day {
        RuleDay::MonthWeekDay {
            month,
            week,
            weekday,
        } => {
            let first_day = (ymd_to_ord(utc_year, month, 1) + 6) % 7;
            let mut month_day = (i64::from(weekday) - (first_day + 1)).rem_euclid(7) + 1;
            month_day += (i64::from(week) - 1) * 7;
            let mut days_in_month = DAYS_IN_MONTH[usize::from(month)];
            if month == 2 && is_leap_year(utc_year) {
                days_in_month += 1;
            }
            if month_day > days_in_month {
                month_day -= 7;
            }
            ymd_to_ord(utc_year, month, month_day) - EPOCHORDINAL
        }
        RuleDay::Day { day, julian } => {
            // dayrule_new stores the parsed day unchanged. The pinned C code
            // subtracts one for BOTH day forms and increments at J59 in leaps.
            let base = ymd_to_ord(utc_year, 1, 1) - EPOCHORDINAL - 1;
            let mut stored_day = i64::from(day);
            if julian && day >= 59 && is_leap_year(utc_year) {
                stored_day += 1;
            }
            base + stored_day
        }
    };
    // Typed parser bounds make this safe; signed carries may cross a year.
    Ok(days * 86400 + i64::from(rule.time_seconds))
}

pub(super) fn local_time(
    timestamp: Timestamp,
    offset_seconds: i32,
) -> Result<Time, DetectionWindowError> {
    let delta = i128::from(offset_seconds)
        .checked_mul(NANOS_PER_SECOND)
        .ok_or(DetectionWindowError::ClockOutOfRange)?;
    let shifted = timestamp
        .as_nanosecond()
        .checked_add(delta)
        .ok_or(DetectionWindowError::ClockOutOfRange)?;
    // This is a civil coordinate, not an instant. Do not apply Jiff Timestamp
    // headroom or its Offset bound to the target's carried seconds.
    let day = shifted.div_euclid(NANOS_PER_DAY);
    let first_day = i128::from(1 - EPOCHORDINAL);
    let after_last_day = i128::from(ymd_to_ord(9999, 12, 31) - EPOCHORDINAL + 1);
    if !(first_day..after_last_day).contains(&day) {
        return Err(DetectionWindowError::ClockOutOfRange);
    }
    let day_nanoseconds = shifted.rem_euclid(NANOS_PER_DAY);
    let second = day_nanoseconds / NANOS_PER_SECOND;
    let nanosecond = day_nanoseconds % NANOS_PER_SECOND;
    Time::new(
        (second / 3600) as i8,
        (second / 60 % 60) as i8,
        (second % 60) as i8,
        nanosecond as i32,
    )
    .map_err(|_| DetectionWindowError::ClockOutOfRange)
}

#[cfg(test)]
mod tests {
    use super::super::posix;
    use super::*;

    fn timestamp(text: &str) -> Timestamp {
        text.parse().unwrap()
    }

    fn transition(text: &str, year: i16) -> i64 {
        let FooterRule::Alternate { start, .. } = posix::parse(text.as_bytes()).unwrap() else {
            panic!("expected alternate rule");
        };
        transition_second(&start, year).unwrap()
    }

    #[test]
    fn parsed_day_rules_preserve_the_pinned_c_constructor_and_evaluator_chain() {
        for (text, year, expected) in [
            ("A0B,0/0,365/0", 2024, "2023-12-31T00:00:00Z"),
            ("A0B,1/0,365/0", 2024, "2024-01-01T00:00:00Z"),
            ("A0B,J58/0,J365/0", 2000, "2000-02-27T00:00:00Z"),
            ("A0B,J59/0,J365/0", 1900, "1900-02-28T00:00:00Z"),
            ("A0B,J59/0,J365/0", 2000, "2000-02-29T00:00:00Z"),
            ("A0B,J59/0,J365/0", 2100, "2100-02-28T00:00:00Z"),
            ("A0B,J60/0,J365/0", 2000, "2000-03-01T00:00:00Z"),
            ("A0B,60/0,365/0", 2000, "2000-02-29T00:00:00Z"),
            ("A0B,J1/-167:99:99,J365/0", 2024, "2023-12-24T23:19:21Z"),
        ] {
            assert_eq!(
                i128::from(transition(text, year)),
                timestamp(expected).as_nanosecond() / NANOS_PER_SECOND,
                "{text}: {year}"
            );
        }
    }

    #[test]
    fn month_week_five_chooses_the_last_weekday() {
        for (text, year, expected) in [
            ("A0B,M2.5.0,M11.1.0", 2024, "2024-02-25T02:00:00Z"),
            ("A0B,M2.5.0,M11.1.0", 2025, "2025-02-23T02:00:00Z"),
            ("A0B,M3.5.0,M11.1.0", 2024, "2024-03-31T02:00:00Z"),
        ] {
            assert_eq!(
                i128::from(transition(text, year)),
                timestamp(expected).as_nanosecond() / NANOS_PER_SECOND
            );
        }
    }

    #[test]
    fn fromutc_selection_keeps_boundaries_southern_negative_and_equal_rules() {
        for (text, instant, expected) in [
            ("A5B,M3.2.0,M11.1.0", "2024-03-10T06:59:59.999999Z", -18000),
            ("A5B,M3.2.0,M11.1.0", "2024-03-10T07:00:00Z", -14400),
            ("A5B,M3.2.0,M11.1.0", "2024-11-03T05:59:59.999999Z", -14400),
            ("A5B,M3.2.0,M11.1.0", "2024-11-03T06:00:00Z", -18000),
            ("A-10B,M10.1.0,M4.1.0", "2024-01-01T00:00:00Z", 39600),
            ("A-10B,M10.1.0,M4.1.0", "2024-07-01T00:00:00Z", 36000),
            ("A0B1,M3.2.0,M11.1.0", "2024-07-01T00:00:00Z", -3600),
            ("A0B0,M3.2.0,M11.1.0", "2024-07-01T00:00:00Z", 0),
            ("A0B-1,0/0,0/1", "2024-01-01T00:00:00Z", 3600),
            ("A-24B,J1/0,J2/0", "2023-12-31T12:00:00Z", 86400),
            ("A-24:99:99B,M3.2.0,M11.1.0", "2024-07-01T00:00:00Z", 96039),
        ] {
            assert_eq!(
                offset_seconds(&posix::parse(text.as_bytes()).unwrap(), timestamp(instant)),
                Ok(expected),
                "{text}: {instant}"
            );
        }
    }

    #[test]
    fn civil_conversion_keeps_carried_offsets_microseconds_and_bounds() {
        for (instant, offset, expected) in [
            ("1969-12-31T23:59:59.123456Z", 96039, "02:40:38.123456"),
            ("1970-01-01T00:00:00.000001Z", -92439, "22:19:21.000001"),
        ] {
            assert_eq!(
                local_time(timestamp(instant), offset),
                Ok(expected.parse().unwrap())
            );
        }
        for (instant, offset) in [
            ("0001-01-01T00:00:00Z", -1),
            ("9999-12-30T00:00:00Z", 172800),
        ] {
            assert_eq!(
                local_time(timestamp(instant), offset),
                Err(DetectionWindowError::ClockOutOfRange)
            );
        }
        let upper = Timestamp::from_microsecond(Timestamp::MAX.as_microsecond()).unwrap();
        assert!(
            Timestamp::from_nanosecond(upper.as_nanosecond() + 86400 * NANOS_PER_SECOND).is_err()
        );
        assert_eq!(
            local_time(upper, 86400),
            Ok(Offset::UTC.to_datetime(upper).time())
        );
    }
}
