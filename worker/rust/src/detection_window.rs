//! Port of worker/domains/detection_window.py, not the old NightWindow alias.
//!
//! Construction reads ONLY `zoneinfo_dir / tz`. Supply the same IANA TZif tree
//! selected by Python ZoneInfo on site (normally /usr/share/zoneinfo); record
//! its tzdata release and file hashes in differential/deployment evidence.
//! There is no bundled database, environment/system-zone lookup, or UTC fallback.
//! Reconstruct a window after a tzdata update: loaded rules are immutable.
//!
//! `zoneinfo_path` validates the IANA key and explicit root before any open.
//! `from_tzif_bytes` compiles caller-captured bytes for that same checked path
//! and performs no filesystem IO. `source_path` records that caller-supplied
//! name; it is not evidence that a file was read. `from_zoneinfo_dir` still
//! opens and reads that path itself and still collapses every open/read
//! failure to `ZoneDataUnavailable`. Callers that need to separate Python
//! fail-open (missing or non-TZif) from an actual IO or corruption fault
//! classify the open themselves, then compile the captured bytes. There is no
//! environment lookup, global cache, or second timezone database.
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
use std::time::{SystemTime, UNIX_EPOCH};

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
    // Construct only through AwareDateTime::new's canonicalization: equality
    // must compare instants, not noncanonical seconds/fraction pairs.
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
                // An offset crossing the epoch can produce a noncanonical
                // seconds/fraction pair. Store the checked instant, not its
                // representation, so equivalent external clocks compare equal.
                let timestamp = Timestamp::from_nanosecond(timestamp.as_nanosecond())
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
    /// Convert an injected UTC `SystemTime` to Python microsecond civil precision.
    ///
    /// Truncation is Euclidean floor, so a negative sub-microsecond remainder
    /// moves to the previous microsecond. The civil value is built only through
    /// Jiff's checked timestamp and `Offset::UTC`; this method does not read the
    /// wall clock, environment, or system zone.
    /// Like Python `datetime.now(UTC)`, its tzinfo differs from every target
    /// `ZoneInfo`, including `ZoneInfo("UTC")`.
    pub fn from_utc_system_time(time: SystemTime) -> Result<Self, DetectionWindowError> {
        let nanoseconds = match time.duration_since(UNIX_EPOCH) {
            Ok(duration) => i128::try_from(duration.as_nanos())
                .map_err(|_| DetectionWindowError::ClockOutOfRange)?,
            Err(earlier) => {
                let signed = i128::try_from(earlier.duration().as_nanos())
                    .map_err(|_| DetectionWindowError::ClockOutOfRange)?;
                signed
                    .checked_neg()
                    .ok_or(DetectionWindowError::ClockOutOfRange)?
            }
        };
        let microseconds = i64::try_from(nanoseconds.div_euclid(1_000))
            .map_err(|_| DetectionWindowError::ClockOutOfRange)?;
        let timestamp = Timestamp::from_microsecond(microseconds)
            .map_err(|_| DetectionWindowError::ClockOutOfRange)?;
        let local = Offset::UTC.to_datetime(timestamp);
        // Injected instants use ClockOutOfRange; new validates civil inputs separately.
        if !(1..=9999).contains(&local.year()) {
            return Err(DetectionWindowError::ClockOutOfRange);
        }
        Self::new(local, Some(0), ClockRelation::DifferentTzinfo)
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
        let (start, end) = checked_bounds(start, end)?;
        let source_path = zoneinfo_path(tz, zoneinfo_dir)?;
        let file =
            File::open(&source_path).map_err(|_| DetectionWindowError::ZoneDataUnavailable)?;
        let mut bytes = Vec::new();
        // Bound the read. This does not reject an oversized file before open,
        // and a failed read stays ZoneDataUnavailable: this constructor does
        // not classify IO separately.
        file.take(MAX_TZIF_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| DetectionWindowError::ZoneDataUnavailable)?;
        compile(start, end, tz, source_path, &bytes)
    }

    /// Compile exact captured TZif bytes. `zoneinfo_dir` and `tz` are checked
    /// with the same name/root/path rules as `from_zoneinfo_dir`, then used only
    /// as `source_path`. No file is opened or read. Bytes above 1 MiB are
    /// `ZoneDataTooLarge` before decode; other non-TZif input is
    /// `InvalidZoneData`.
    pub fn from_tzif_bytes(
        start: &str,
        end: &str,
        tz: &str,
        zoneinfo_dir: &Path,
        bytes: &[u8],
    ) -> Result<Self, DetectionWindowError> {
        let (start, end) = checked_bounds(start, end)?;
        let source_path = zoneinfo_path(tz, zoneinfo_dir)?;
        compile(start, end, tz, source_path, bytes)
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
                    || offset
                        .to_timestamp(local)
                        .map(|value| value.as_nanosecond())
                        .ok()
                        != Some(timestamp.as_nanosecond())
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

/// Existing name, root, and joined-path checks, before any open.
/// An invalid key is `InvalidZoneName` even when the root is also invalid.
/// A relative root or a joined path over the byte bound is `InvalidZoneDirectory`.
/// This does not check that the path exists.
pub fn zoneinfo_path(tz: &str, zoneinfo_dir: &Path) -> Result<PathBuf, DetectionWindowError> {
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
    Ok(source_path)
}

fn checked_bounds(start: &str, end: &str) -> Result<(Time, Time), DetectionWindowError> {
    Ok((parse_hhmm(start)?, parse_hhmm(end)?))
}

fn compile(
    start: Time,
    end: Time,
    tz: &str,
    source_path: PathBuf,
    bytes: &[u8],
) -> Result<DetectionWindow, DetectionWindowError> {
    if bytes.len() > MAX_TZIF_BYTES {
        return Err(DetectionWindowError::ZoneDataTooLarge);
    }
    let zone = TimeZone::tzif(tz, bytes).map_err(|_| DetectionWindowError::InvalidZoneData)?;
    Ok(DetectionWindow {
        start,
        end,
        zone,
        source_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ClockRelation::{DifferentTzinfo, SameTargetTzinfo};
    use std::time::Duration;

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
    fn injected(offset: Duration, before_epoch: bool) -> SystemTime {
        if before_epoch {
            UNIX_EPOCH.checked_sub(offset).unwrap()
        } else {
            UNIX_EPOCH.checked_add(offset).unwrap()
        }
    }

    #[test]
    fn injected_utc_epoch_and_microsecond_floor_match_python_civil_time() {
        let cases = [
            (Duration::ZERO, false, "1970-01-01T00:00:00"),
            (Duration::from_nanos(999), false, "1970-01-01T00:00:00"),
            (
                Duration::from_nanos(1_000),
                false,
                "1970-01-01T00:00:00.000001",
            ),
            (Duration::from_nanos(1), true, "1969-12-31T23:59:59.999999"),
            (
                Duration::from_nanos(1_000),
                true,
                "1969-12-31T23:59:59.999999",
            ),
            (
                Duration::from_nanos(1_001),
                true,
                "1969-12-31T23:59:59.999998",
            ),
        ];
        let utc_night = window("23:59", "00:01", "UTC");
        let seoul_night = window("23:59", "00:01", "Asia/Seoul");
        let new_york_evening = window("18:59", "19:01", "America/New_York");
        for (offset, before_epoch, civil) in cases {
            let clock =
                AwareDateTime::from_utc_system_time(injected(offset, before_epoch)).unwrap();
            let expected = aware(civil, 0, DifferentTzinfo);
            assert_eq!(clock, expected);
            assert_eq!(
                utc_night.contains(clock),
                Ok(true),
                "UTC membership for {civil}: {clock:?}"
            );
            assert_eq!(
                seoul_night.contains(clock),
                Ok(false),
                "Seoul membership for {civil}: {clock:?}"
            );
            assert_eq!(
                new_york_evening.contains(clock),
                Ok(true),
                "New York membership for {civil}: {clock:?}"
            );
        }
        assert_eq!(
            aware("1970-01-01T08:59:59.999999", 9 * 3600, DifferentTzinfo),
            aware("1969-12-31T23:59:59.999999", 0, DifferentTzinfo)
        );
        assert_eq!(
            aware("1969-12-31T19:00:00.000001", -5 * 3600, DifferentTzinfo),
            aware("1970-01-01T00:00:00.000001", 0, DifferentTzinfo)
        );
    }

    #[test]
    fn injected_fold_instants_agree_with_existing_utc_aware_construction() {
        // America/New_York repeated 01:30 on 2024-11-03. These are the two UTC
        // instants selected by the -04:00 and -05:00 fold offsets.
        let fold = window("01:30", "02:00", "America/New_York");
        for (seconds, offset) in [(1_730_611_800_i64, -4 * 3600), (1_730_615_400, -5 * 3600)] {
            let instant = UNIX_EPOCH + Duration::from_secs(seconds as u64);
            let injected = AwareDateTime::from_utc_system_time(instant).unwrap();
            let constructed = aware("2024-11-03T01:30:00", offset, DifferentTzinfo);
            assert_eq!(injected, constructed);
            assert_eq!(
                fold.contains(injected).unwrap(),
                fold.contains(constructed).unwrap()
            );
            assert!(fold.contains(injected).unwrap());
        }
    }

    #[test]
    fn injected_utc_refuses_python_and_jiff_range_without_clamping() {
        let python_start = injected(Duration::from_secs(62_135_596_800), true);
        let pre_python = python_start - Duration::from_micros(1);
        assert_eq!(
            AwareDateTime::from_utc_system_time(pre_python),
            Err(DetectionWindowError::ClockOutOfRange)
        );
        assert!(AwareDateTime::from_utc_system_time(python_start).is_ok());

        let past_python = UNIX_EPOCH + Duration::from_secs(253_402_300_800);
        assert_eq!(
            AwareDateTime::from_utc_system_time(past_python),
            Err(DetectionWindowError::ClockOutOfRange)
        );
        assert_eq!(
            AwareDateTime::from_utc_system_time(past_python - Duration::from_micros(1)),
            Err(DetectionWindowError::ClockOutOfRange)
        );

        // Jiff's declared limit reserves offset headroom inside Python's range.
        let last_jiff_microsecond = UNIX_EPOCH
            + Duration::from_micros(u64::try_from(Timestamp::MAX.as_microsecond()).unwrap());
        let past_jiff = last_jiff_microsecond + Duration::from_micros(1);
        assert_eq!(
            AwareDateTime::from_utc_system_time(past_jiff),
            Err(DetectionWindowError::ClockOutOfRange)
        );
        assert!(AwareDateTime::from_utc_system_time(last_jiff_microsecond).is_ok());
    }
}
