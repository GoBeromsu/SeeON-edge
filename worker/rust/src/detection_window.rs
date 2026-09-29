//! Port of worker/domains/detection_window.py, not the old NightWindow alias.
//!
//! Construction reads ONLY `zoneinfo_dir / tz`. Supply the same IANA TZif tree
//! selected by Python ZoneInfo on site (normally /usr/share/zoneinfo); record
//! its tzdata release and file hashes in differential/deployment evidence.
//! There is no bundled database, environment/system-zone lookup, or UTC fallback.
//! Reconstruct a window after a tzdata update: loaded rules are immutable.
//!
//! Admission limits are explicit: 256-byte zone names, 4096-byte paths, 1 MiB
//! TZif files, ASCII one/two-digit HH:MM components (Python also accepts some
//! Unicode decimal digits), Python years 1..=9999, microsecond civil precision,
//! and integral-second UTC offsets strictly inside +/-24 hours. External clocks
//! also require Python-range intermediate UTC time and Jiff's checked timestamp
//! range (which reserves headroom at the upper bound). Same-target clocks never
//! enter that instant domain. No clamping of out-of-range dates is accepted.
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use jiff::Timestamp;
pub use jiff::civil::DateTime;
use jiff::civil::Time;
use jiff::tz::{Offset, TimeZone};

pub const MAX_ZONE_NAME_BYTES: usize = 256;
pub const MAX_ZONE_PATH_BYTES: usize = 4096;
pub const MAX_TZIF_BYTES: usize = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionWindowError {
    InvalidTime,
    NaiveClock,
    InvalidClock,
    ClockOutOfRange,
    InvalidZoneName,
    InvalidZoneDirectory,
    ZoneDataUnavailable,
    ZoneDataTooLarge,
    InvalidZoneData,
}
impl std::fmt::Display for DetectionWindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidTime => "window time must use ASCII HH:MM (one or two digits per field)",
            Self::NaiveClock => "now must be timezone-aware",
            Self::InvalidClock => {
                "clock requires Python-range civil time and an integral-second UTC offset"
            }
            Self::ClockOutOfRange => "clock conversion exceeds the supported datetime range",
            Self::InvalidZoneName => {
                "zone name must be a bounded relative IANA key without dot components"
            }
            Self::InvalidZoneDirectory => {
                "zoneinfo directory must be explicit, absolute, and within the path bound"
            }
            Self::ZoneDataUnavailable => {
                "named TZif file is unavailable in the explicit zoneinfo directory"
            }
            Self::ZoneDataTooLarge => "TZif file exceeds the byte bound",
            Self::InvalidZoneData => "named zone data is not valid TZif",
        })
    }
}
impl std::error::Error for DetectionWindowError {}

/// Python tzinfo object identity relative to THIS evaluation's target.
/// Recompute after target replacement, removal or re-enabling; this is not
/// global zone identity. Equal zone names, bytes or offsets do not imply Same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockRelation {
    SameTargetTzinfo,
    DifferentTzinfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidatedClock {
    SameTargetCivil(DateTime),
    ExternalInstant(Timestamp),
}

/// An injected aware datetime, never a machine-local or wall-clock read.
/// Supply Python `now.utcoffset()` and the actual target-relative tzinfo identity.
/// Same-target clocks preserve civil fields even in DST gaps. External clocks
/// use the actual fold-selected offset; no additional fold field is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwareDateTime(ValidatedClock);
impl AwareDateTime {
    pub fn new(
        local: DateTime,
        utc_offset_seconds: Option<i32>,
        relation: ClockRelation,
    ) -> Result<Self, DetectionWindowError> {
        let seconds = utc_offset_seconds.ok_or(DetectionWindowError::NaiveClock)?;
        if !(1..=9999).contains(&local.year())
            || local.subsec_nanosecond() % 1000 != 0
            || !(-86_399..=86_399).contains(&seconds)
        {
            return Err(DetectionWindowError::InvalidClock);
        }
        let clock = match relation {
            ClockRelation::SameTargetTzinfo => ValidatedClock::SameTargetCivil(local),
            ClockRelation::DifferentTzinfo => {
                let offset = Offset::from_seconds(seconds)
                    .map_err(|_| DetectionWindowError::InvalidClock)?;
                let timestamp = offset
                    .to_timestamp(local)
                    .map_err(|_| DetectionWindowError::ClockOutOfRange)?;
                // Python may overflow while subtracting the source offset,
                // even if a subsequent target conversion would be in range.
                if !(1..=9999).contains(&Offset::UTC.to_datetime(timestamp).year()) {
                    return Err(DetectionWindowError::ClockOutOfRange);
                }
                ValidatedClock::ExternalInstant(timestamp)
            }
        };
        Ok(Self(clock))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectionWindow {
    start: Time,
    end: Time,
    zone: TimeZone,
    source_path: PathBuf,
}
impl DetectionWindow {
    pub fn from_zoneinfo_dir(
        start: &str,
        end: &str,
        tz: &str,
        zoneinfo_dir: &Path,
    ) -> Result<Self, DetectionWindowError> {
        let start = parse_hhmm(start)?;
        let end = parse_hhmm(end)?;
        if tz.is_empty()
            || tz.len() > MAX_ZONE_NAME_BYTES
            || tz
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
            || !tz
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"/_-+.".contains(&c))
        {
            return Err(DetectionWindowError::InvalidZoneName);
        }
        let source_path = zoneinfo_dir.join(tz);
        if !zoneinfo_dir.is_absolute() || source_path.as_os_str().len() > MAX_ZONE_PATH_BYTES {
            return Err(DetectionWindowError::InvalidZoneDirectory);
        }
        let file =
            File::open(&source_path).map_err(|_| DetectionWindowError::ZoneDataUnavailable)?;
        let mut bytes = Vec::new();
        file.take(MAX_TZIF_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| DetectionWindowError::ZoneDataUnavailable)?;
        if bytes.len() > MAX_TZIF_BYTES {
            return Err(DetectionWindowError::ZoneDataTooLarge);
        }
        let zone = TimeZone::tzif(tz, &bytes).map_err(|_| DetectionWindowError::InvalidZoneData)?;
        Ok(Self {
            start,
            end,
            zone,
            source_path,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    /// Same-target identity preserves Python's civil-time short circuit.
    /// External historical/POSIX rules are interpreted by Jiff; checking the
    /// round trip detects saturation at Jiff's civil range edges.
    pub fn contains(&self, now: AwareDateTime) -> Result<bool, DetectionWindowError> {
        let time = match now.0 {
            ValidatedClock::SameTargetCivil(local) => local.time(),
            ValidatedClock::ExternalInstant(timestamp) => {
                let offset = self.zone.to_offset(timestamp);
                let local = offset.to_datetime(timestamp);
                if !(1..=9999).contains(&local.year())
                    || offset.to_timestamp(local).ok() != Some(timestamp)
                {
                    return Err(DetectionWindowError::ClockOutOfRange);
                }
                local.time()
            }
        };
        Ok(if self.start <= self.end {
            self.start <= time && time < self.end
        } else {
            time >= self.start || time < self.end
        })
    }
}

fn parse_hhmm(value: &str) -> Result<Time, DetectionWindowError> {
    let (hour, minute) = value
        .split_once(':')
        .ok_or(DetectionWindowError::InvalidTime)?;
    fn component(value: &str) -> Result<i8, DetectionWindowError> {
        if !(1..=2).contains(&value.len()) || !value.bytes().all(|c| c.is_ascii_digit()) {
            return Err(DetectionWindowError::InvalidTime);
        }
        value.parse().map_err(|_| DetectionWindowError::InvalidTime)
    }
    Time::new(component(hour)?, component(minute)?, 0, 0)
        .map_err(|_| DetectionWindowError::InvalidTime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ClockRelation::{DifferentTzinfo, SameTargetTzinfo};

    // Onsite tests must use the SAME tree as their Python reference. Missing
    // system data is a test error, never a skip or an alternate database.
    fn window(start: &str, end: &str, tz: &str) -> DetectionWindow {
        let root = std::env::var_os("SEEON_TEST_ZONEINFO_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/share/zoneinfo"));
        DetectionWindow::from_zoneinfo_dir(start, end, tz, &root).unwrap()
    }
    fn aware(text: &str, offset: i32, relation: ClockRelation) -> AwareDateTime {
        AwareDateTime::new(text.parse().unwrap(), Some(offset), relation).unwrap()
    }

    #[test]
    fn seoul_python_boundary_facts_use_clock_not_pts() {
        let night = window("21:00", "05:00", "Asia/Seoul");
        for (text, expected) in [
            ("2026-07-31T20:59:59.999999", false),
            ("2026-07-31T21:00:00", true),
            ("2026-07-31T22:00:00", true),
            ("2026-08-01T00:00:00", true),
            ("2026-08-01T04:59:59.999999", true),
            ("2026-08-01T05:00:00", false),
            ("2026-08-01T13:00:00", false),
        ] {
            assert_eq!(
                night
                    .contains(aware(text, 9 * 3600, SameTargetTzinfo))
                    .unwrap(),
                expected
            );
        }
        assert!(
            night
                .contains(aware("2026-07-31T12:00:00", 0, DifferentTzinfo))
                .unwrap()
        );
        assert!(night.source_path().ends_with("Asia/Seoul"));
    }

    #[test]
    fn same_day_and_equal_endpoints_are_half_open() {
        let day = window("9:5", "17:00", "UTC");
        let empty = window("09:05", "09:05", "UTC");
        for (text, expected) in [
            ("2026-01-01T09:04:59.999999", false),
            ("2026-01-01T09:05:00", true),
            ("2026-01-01T16:59:59.999999", true),
            ("2026-01-01T17:00:00", false),
        ] {
            let now = aware(text, 0, SameTargetTzinfo);
            assert_eq!(day.contains(now).unwrap(), expected);
            assert!(!empty.contains(now).unwrap());
        }
    }

    #[test]
    fn dst_gap_fold_historical_and_future_rules_are_not_fixed_offsets() {
        let fold = window("01:30", "02:00", "America/New_York");
        for offset in [-4 * 3600, -5 * 3600] {
            for relation in [SameTargetTzinfo, DifferentTzinfo] {
                assert!(
                    fold.contains(aware("2024-11-03T01:30:00", offset, relation))
                        .unwrap()
                );
            }
        }
        assert!(
            !fold
                .contains(aware("2024-11-03T07:00:00", 0, DifferentTzinfo))
                .unwrap()
        );
        let gap = window("02:00", "03:00", "America/New_York");
        for text in ["2024-03-10T06:59:59", "2024-03-10T07:00:00"] {
            assert!(!gap.contains(aware(text, 0, DifferentTzinfo)).unwrap());
        }
        // Seoul used +08:30 in 1960, not today's +09:00.
        let historical = window("08:30", "08:31", "Asia/Seoul");
        assert!(
            historical
                .contains(aware("1960-01-01T00:00:00", 0, DifferentTzinfo))
                .unwrap()
        );
        let summer = window("08:00", "08:01", "America/New_York");
        assert!(
            summer
                .contains(aware("2050-07-01T12:00:00", 0, DifferentTzinfo))
                .unwrap()
        );
        assert!(
            !summer
                .contains(aware("2050-01-01T12:00:00", 0, DifferentTzinfo))
                .unwrap()
        );
    }

    #[test]
    fn same_target_gap_is_civil_but_external_gap_uses_selected_offset() {
        let gap = window("02:00", "03:00", "America/New_York");
        for (offset, start, end) in [(-5 * 3600, "03:00", "04:00"), (-4 * 3600, "01:00", "02:00")] {
            let text = "2024-03-10T02:30:00";
            assert!(gap.contains(aware(text, offset, SameTargetTzinfo)).unwrap());
            let external = aware(text, offset, DifferentTzinfo);
            assert!(!gap.contains(external).unwrap());
            assert!(
                window(start, end, "America/New_York")
                    .contains(external)
                    .unwrap()
            );
        }
    }

    #[test]
    fn same_target_calendar_extremes_do_not_enter_the_instant_domain() {
        let full = window("23:59", "00:01", "UTC");
        for text in ["0001-01-01T00:00:00", "9999-12-31T23:59:59.999999"] {
            assert!(full.contains(aware(text, 0, SameTargetTzinfo)).unwrap());
        }
        for (text, offset) in [
            ("0001-01-01T00:00:00", 9 * 3600),
            ("9999-12-31T23:59:59.999999", 0),
        ] {
            assert_eq!(
                AwareDateTime::new(text.parse().unwrap(), Some(offset), DifferentTzinfo),
                Err(DetectionWindowError::ClockOutOfRange)
            );
        }
    }

    #[test]
    fn malformed_naive_unknown_and_range_inputs_do_not_fall_back() {
        for text in [
            "",
            "24:00",
            "23:60",
            "001:00",
            "1:",
            "1:2:3",
            " 1:00",
            "1:00 ",
            "-1:00",
            "１２:００",
        ] {
            assert_eq!(parse_hhmm(text), Err(DetectionWindowError::InvalidTime));
        }
        let local: DateTime = "2026-07-31T22:00:00".parse().unwrap();
        for relation in [SameTargetTzinfo, DifferentTzinfo] {
            assert_eq!(
                AwareDateTime::new(local, None, relation),
                Err(DetectionWindowError::NaiveClock)
            );
            for offset in [-86400, 86400] {
                assert_eq!(
                    AwareDateTime::new(local, Some(offset), relation),
                    Err(DetectionWindowError::InvalidClock)
                );
            }
            for text in ["0000-01-01T00:00:00", "2026-01-01T00:00:00.000000001"] {
                assert_eq!(
                    AwareDateTime::new(text.parse().unwrap(), Some(0), relation),
                    Err(DetectionWindowError::InvalidClock)
                );
            }
        }
        for name in ["", "../UTC", "/UTC", "Asia//Seoul", "./UTC", "UTC\0"] {
            assert_eq!(
                DetectionWindow::from_zoneinfo_dir(
                    "00:00",
                    "01:00",
                    name,
                    Path::new("/usr/share/zoneinfo")
                ),
                Err(DetectionWindowError::InvalidZoneName)
            );
        }
        assert_eq!(
            DetectionWindow::from_zoneinfo_dir(
                "00:00",
                "01:00",
                "Unknown/NoSuchZone",
                Path::new("/usr/share/zoneinfo")
            ),
            Err(DetectionWindowError::ZoneDataUnavailable)
        );
        assert_eq!(
            DetectionWindow::from_zoneinfo_dir("00:00", "01:00", "UTC", Path::new("zoneinfo")),
            Err(DetectionWindowError::InvalidZoneDirectory)
        );
        let early = aware("0001-01-01T00:00:00", 0, DifferentTzinfo);
        assert_eq!(
            window("00:00", "01:00", "America/New_York").contains(early),
            Err(DetectionWindowError::ClockOutOfRange)
        );
    }
}
