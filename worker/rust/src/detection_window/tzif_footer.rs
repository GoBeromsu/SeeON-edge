//! Original-byte native serving, or a pinned footer with guarded body history.
use std::sync::Arc;

use jiff::Timestamp;
use jiff::civil::Time;
use jiff::tz::TimeZone;

use super::DetectionWindowError;

mod evaluate;
mod posix;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CompiledZone {
    Native(TimeZone),
    PythonFooter(Arc<FooterZone>),
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct FooterZone {
    history: TimeZone,
    last_utc_second: Option<i64>,
    rule: posix::FooterRule,
    original_footer: Box<[u8]>,
}

struct FooterView<'a> {
    body_end: usize,
    text: &'a [u8],
    last_utc_second: Option<i64>,
}

pub(super) fn compile(tz: &str, bytes: &[u8]) -> Result<CompiledZone, DetectionWindowError> {
    let view = footer_view(bytes);
    let parsed = view.as_ref().and_then(|view| posix::parse(view.text));
    if let Ok(zone) = TimeZone::tzif(tz, bytes)
        && !parsed.is_some_and(posix::FooterRule::requires_python_evaluator)
    {
        return Ok(CompiledZone::Native(zone));
    }
    let view = view.ok_or(DetectionWindowError::InvalidZoneData)?;
    // Only the suffix changes. This object decodes history, never the full zone.
    // Validate it even for a rejected rule to preserve the error distinction.
    let mut body = bytes[..view.body_end].to_vec();
    body.extend_from_slice(b"\n\n");
    let history = TimeZone::tzif(tz, &body).map_err(|_| DetectionWindowError::InvalidZoneData)?;
    let rule = parsed.ok_or(DetectionWindowError::InvalidZoneFooter)?;
    Ok(CompiledZone::PythonFooter(Arc::new(FooterZone {
        history,
        last_utc_second: view.last_utc_second,
        rule,
        original_footer: view.text.into(),
    })))
}

impl CompiledZone {
    pub(super) fn local_time(&self, timestamp: Timestamp) -> Result<Time, DetectionWindowError> {
        match self {
            Self::Native(zone) => native_local_time(zone, timestamp),
            Self::PythonFooter(zone) => {
                // CPython ignores microseconds here: the entire final second
                // remains historical. No cutoff Timestamp or cutoff + 1.
                let second = timestamp.as_nanosecond().div_euclid(1_000_000_000);
                if zone
                    .last_utc_second
                    .is_some_and(|last| second <= i128::from(last))
                {
                    return native_local_time(&zone.history, timestamp);
                }
                let offset = evaluate::offset_seconds(&zone.rule, timestamp)?;
                evaluate::local_time(timestamp, offset)
            }
        }
    }
}

fn native_local_time(zone: &TimeZone, timestamp: Timestamp) -> Result<Time, DetectionWindowError> {
    let offset = zone.to_offset(timestamp);
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
    Ok(local.time())
}

fn footer_view(bytes: &[u8]) -> Option<FooterView<'_>> {
    if bytes.len() > super::MAX_TZIF_BYTES {
        return None;
    }
    let version = *bytes.get(4)?;
    if !matches!(version, b'2' | b'3' | b'4') {
        return None;
    }
    let second = data_end(bytes, 0, [1, 1, 8, 5, 6, 1])?;
    if bytes.get(second.checked_add(4)?) != Some(&version) {
        return None;
    }
    let end = data_end(bytes, second, [1, 1, 12, 9, 6, 1])?;
    let text = bytes.get(end..)?.strip_prefix(b"\n")?.strip_suffix(b"\n")?;
    if text.is_empty() || !text.is_ascii() || text.contains(&b'\n') {
        return None;
    }
    let header = bytes.get(second..second.checked_add(44)?)?;
    let timecnt = usize::try_from(i32::from_be_bytes(header[32..36].try_into().ok()?)).ok()?;
    let last_utc_second = if timecnt == 0 {
        None
    } else {
        let array = second.checked_add(44)?;
        let last = array.checked_add(timecnt.checked_sub(1)?.checked_mul(8)?)?;
        let raw = bytes.get(last..last.checked_add(8)?)?;
        Some(i64::from_be_bytes(raw.try_into().ok()?))
    };
    Some(FooterView {
        body_end: end,
        text,
        last_utc_second,
    })
}

fn data_end(bytes: &[u8], offset: usize, widths: [usize; 6]) -> Option<usize> {
    let mut end = offset.checked_add(44)?;
    let header = bytes.get(offset..end)?;
    if !header.starts_with(b"TZif") {
        return None;
    }
    for (count, width) in header[20..44].chunks_exact(4).zip(widths) {
        let count = usize::try_from(i32::from_be_bytes(count.try_into().ok()?)).ok()?;
        end = end.checked_add(count.checked_mul(width)?)?;
    }
    bytes.get(..end)?;
    Some(end)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use jiff::tz::Offset;

    use super::super::{AwareDateTime, ClockRelation, DetectionWindow};
    use super::*;

    fn data(footer: &[u8]) -> Vec<u8> {
        let original = std::fs::read("/usr/share/zoneinfo/UTC").unwrap();
        let mut bytes = original.strip_suffix(b"\nUTC0\n").unwrap().to_vec();
        bytes.push(b'\n');
        bytes.extend_from_slice(footer);
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn rejects_python_invalid_footer_only_after_validating_body() {
        for text in [b"INVALID!!!".as_slice(), b"UTC25", b"UT!0", b"\0UTC0"] {
            assert_eq!(
                compile("test", &data(text)),
                Err(DetectionWindowError::InvalidZoneFooter)
            );
        }
        for text in [b"INVALID!!!".as_slice(), b"A0"] {
            let mut corrupt = data(text);
            let second = data_end(&corrupt, 0, [1, 1, 8, 5, 6, 1]).unwrap();
            corrupt[second + 44..second + 48].copy_from_slice(&i32::MAX.to_be_bytes());
            assert_eq!(
                compile("test", &corrupt),
                Err(DetectionWindowError::InvalidZoneData)
            );
        }
    }

    #[test]
    fn serves_python_accepted_footers_without_transitionless_body_substitution() {
        for (text, expected) in [
            (b"A0".as_slice(), "00:00:00"),
            (b"AB0".as_slice(), "00:00:00"),
            (b"<A>0".as_slice(), "00:00:00"),
            (b"ABC0:60".as_slice(), "23:00:00"),
            (b"ABC0:00:60".as_slice(), "23:59:00"),
            (b"EST5\0garbage".as_slice(), "19:00:00"),
        ] {
            let bytes = data(text);
            assert!(TimeZone::tzif("test", &bytes).is_err());
            let zone = compile("test", &bytes).unwrap();
            assert!(matches!(zone, CompiledZone::PythonFooter(_)));
            assert_eq!(
                zone.local_time(Timestamp::UNIX_EPOCH),
                Ok(expected.parse().unwrap())
            );
            assert_eq!(zone, zone.clone());
            assert_eq!(zone, compile("test", &bytes).unwrap());
        }
    }

    #[test]
    fn accepted_day_footers_use_c_semantics_through_the_public_compiled_route() {
        for (text, instant, time, start, end, expected) in [
            (
                "AAA0BBB,0/0,0/0",
                "2050-01-15T08:30:00Z",
                "09:30:00",
                "08:00",
                "09:00",
                false,
            ),
            (
                "AAA0BBB,0/-2,200/26",
                "2050-07-20T01:00:00.999999Z",
                "01:00:00.999999",
                "00:30",
                "01:30",
                true,
            ),
            (
                "AAA0BBB,59/2,M11.1.0/2",
                "2050-02-28T02:00:00Z",
                "03:00:00",
                "01:30",
                "02:30",
                false,
            ),
            (
                "AAA0BBB,M3.2.0/2,299/2",
                "2050-10-26T01:00:00Z",
                "01:00:00",
                "01:30",
                "02:30",
                false,
            ),
            (
                "AAA0BBB,J59/2,M11.1.0/2",
                "2052-02-28T02:00:00Z",
                "02:00:00",
                "01:30",
                "02:30",
                true,
            ),
            (
                "AAA0BBB,M3.2.0/2,J59/2",
                "2052-02-28T01:00:00Z",
                "02:00:00",
                "01:30",
                "02:30",
                true,
            ),
        ] {
            let bytes = data(text.as_bytes());
            let native = TimeZone::tzif("test", &bytes).unwrap();
            let timestamp: Timestamp = instant.parse().unwrap();
            let time: Time = time.parse().unwrap();
            assert_ne!(
                native_local_time(&native, timestamp).unwrap(),
                time,
                "{text}"
            );
            let window = DetectionWindow::from_tzif_bytes(
                start,
                end,
                "test",
                Path::new("/usr/share/zoneinfo"),
                &bytes,
            )
            .unwrap();
            assert!(
                matches!(window.zone, CompiledZone::PythonFooter(_)),
                "{text}"
            );
            assert_eq!(window.zone.local_time(timestamp), Ok(time), "{text}");
            let now = AwareDateTime::new(
                Offset::UTC.to_datetime(timestamp),
                Some(0),
                ClockRelation::DifferentTzinfo,
            )
            .unwrap();
            assert_eq!(window.contains(now), Ok(expected), "{text}: {instant}");
        }
    }

    #[test]
    fn accepted_non_day_footers_keep_the_original_native_zone() {
        for text in ["UTC0", "AAA0BBB,M3.2.0/2,M11.1.0/2"] {
            let bytes = data(text.as_bytes());
            let native = TimeZone::tzif("test", &bytes).unwrap();
            assert_eq!(compile("test", &bytes), Ok(CompiledZone::Native(native)));
        }
    }

    #[test]
    fn accepted_day_footer_keeps_history_through_the_last_whole_second() {
        let mut bytes = data(b"UTC0BBB,60/2,300/2");
        // The body ends while Jiff's day-60 rule is still standard time,
        // but CPython's day-60 rule has already entered daylight time.
        let cutoff: Timestamp = "1970-03-01T12:00:00Z".parse().unwrap();
        let cutoff_second = i64::try_from(cutoff.as_nanosecond() / 1_000_000_000).unwrap();
        let second = data_end(&bytes, 0, [1, 1, 8, 5, 6, 1]).unwrap();
        bytes[second + 32..second + 36].copy_from_slice(&1_i32.to_be_bytes());
        let at = second + 44;
        bytes.splice(at..at, cutoff_second.to_be_bytes().into_iter().chain([0]));
        TimeZone::tzif("test", &bytes).expect("historical fixture must be natively accepted");
        let window = DetectionWindow::from_tzif_bytes(
            "11:59",
            "12:01",
            "test",
            Path::new("/usr/share/zoneinfo"),
            &bytes,
        )
        .unwrap();
        assert!(matches!(window.zone, CompiledZone::PythonFooter(_)));
        for (delta, time, expected) in [
            (-1, "11:59:59.999999", true),
            (0, "12:00:00", true),
            (999_999, "12:00:00.999999", true),
            (1_000_000, "13:00:01", false),
        ] {
            let timestamp = Timestamp::from_microsecond(cutoff.as_microsecond() + delta).unwrap();
            assert_eq!(window.zone.local_time(timestamp), Ok(time.parse().unwrap()));
            let now = AwareDateTime::new(
                Offset::UTC.to_datetime(timestamp),
                Some(0),
                ClockRelation::DifferentTzinfo,
            )
            .unwrap();
            assert_eq!(window.contains(now), Ok(expected));
        }
    }

    #[test]
    fn malformed_layout_and_unclassified_text_do_not_enter_footer_serving() {
        let original = data(b"INVALID!!!");
        let second = data_end(&original, 0, [1, 1, 8, 5, 6, 1]).unwrap();
        for length in [0, 4, 44, second, second + 44, original.len() - 1] {
            assert!(footer_view(&original[..length]).is_none());
        }
        for offset in [20, second + 20] {
            for count in [-1_i32, i32::MAX] {
                let mut bytes = original.clone();
                bytes[offset..offset + 4].copy_from_slice(&count.to_be_bytes());
                assert!(footer_view(&bytes).is_none());
            }
        }
        for text in [b"".as_slice(), b"INVALID!!!\nextra", &[0xff]] {
            assert!(footer_view(&data(text)).is_none());
        }
        for text in [b"".as_slice(), b"UTC0"] {
            assert!(matches!(
                compile("test", &data(text)),
                Ok(CompiledZone::Native(_))
            ));
        }
        let mut short = original[..second + 44].to_vec();
        short[second + 40..second + 44].copy_from_slice(&50_i32.to_be_bytes());
        short.extend_from_slice(b"\nINVALID!!!\n");
        assert!(footer_view(&short).is_none());
        let mut version = original;
        version[second + 4] = b'9';
        assert!(footer_view(&version).is_none());
    }

    #[test]
    fn raw_cutoff_extraction_does_not_enter_the_timestamp_domain() {
        for last in [i64::MIN, -1, i64::MAX] {
            let mut bytes = data(b"A0");
            let second = data_end(&bytes, 0, [1, 1, 8, 5, 6, 1]).unwrap();
            bytes[second + 32..second + 36].copy_from_slice(&1_i32.to_be_bytes());
            let at = second + 44;
            bytes.splice(at..at, last.to_be_bytes().into_iter().chain([0]));
            assert_eq!(footer_view(&bytes).unwrap().last_utc_second, Some(last));
        }
    }

    #[test]
    fn complete_last_second_is_historical_and_missing_cutoff_is_not() {
        let make = |last_utc_second| {
            CompiledZone::PythonFooter(Arc::new(FooterZone {
                history: TimeZone::UTC,
                last_utc_second,
                rule: posix::FooterRule::Fixed { std_seconds: 3600 },
                original_footer: Box::from(b"A-1".as_slice()),
            }))
        };
        let zone = make(Some(-1));
        for (microsecond, time) in [
            (-1_000_001, "23:59:58.999999"),
            (-1_000_000, "23:59:59"),
            (-1, "23:59:59.999999"),
            (0, "01:00:00"),
        ] {
            assert_eq!(
                zone.local_time(Timestamp::from_microsecond(microsecond).unwrap()),
                Ok(time.parse().unwrap())
            );
        }
        let epoch = Timestamp::UNIX_EPOCH;
        assert_eq!(
            make(None).local_time(epoch),
            Ok("01:00:00".parse().unwrap())
        );
        assert_eq!(
            make(Some(i64::MIN)).local_time(epoch),
            Ok("01:00:00".parse().unwrap())
        );
        assert_eq!(
            make(Some(i64::MAX)).local_time(epoch),
            Ok("00:00:00".parse().unwrap())
        );
        assert_ne!(zone, make(None));
    }

    #[test]
    fn equality_keeps_ignored_nul_suffix_bytes() {
        let plain = compile("test", &data(b"A0")).unwrap();
        let first = compile("test", &data(b"A0\0first")).unwrap();
        let second = compile("test", &data(b"A0\0second")).unwrap();
        assert_ne!(plain, first);
        assert_ne!(first, second);
    }
}
