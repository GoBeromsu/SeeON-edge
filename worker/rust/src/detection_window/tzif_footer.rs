//! Rejection-only classification, never an alternative timezone constructor.
use jiff::tz::TimeZone;

mod posix;

/// Called only after compilation of the original bytes has failed.
pub(super) fn invalid_footer(tz: &str, bytes: &[u8]) -> bool {
    let Some((body_end, footer)) = footer(bytes) else {
        return false;
    };
    // Jiff rejection is not Python rejection: short names, numeric carries and
    // C-string termination are accepted by CPython. Unsupported serving rules
    // remain typed faults, never an invented ALWAYS window.
    if posix::accepts(footer) {
        return false;
    }
    // Prove the body independently, without making this diagnostic timezone
    // available to any caller. Serving can use only the original captured data.
    let mut body = bytes[..body_end].to_vec();
    body.extend_from_slice(b"\n\n");
    TimeZone::tzif(tz, &body).is_ok()
}

fn footer(bytes: &[u8]) -> Option<(usize, &[u8])> {
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
    Some((end, text))
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
            assert!(invalid_footer("test", &data(text)));
        }
        let mut corrupt = data(b"INVALID!!!");
        let second = data_end(&corrupt, 0, [1, 1, 8, 5, 6, 1]).unwrap();
        corrupt[second + 44..second + 48].copy_from_slice(&i32::MAX.to_be_bytes());
        assert!(!invalid_footer("test", &corrupt));
    }

    #[test]
    fn jiff_unsupported_python_accepted_footers_never_become_always() {
        for text in [
            b"A0".as_slice(),
            b"AB0",
            b"<A>0",
            b"ABC0:60",
            b"ABC0:00:60",
            b"EST5\0garbage",
        ] {
            let bytes = data(text);
            assert!(TimeZone::tzif("test", &bytes).is_err());
            assert!(!invalid_footer("test", &bytes));
        }
    }

    #[test]
    fn malformed_layout_and_unclassified_text_stay_typed() {
        let original = data(b"INVALID!!!");
        let second = data_end(&original, 0, [1, 1, 8, 5, 6, 1]).unwrap();
        for length in [0, 4, 44, second, second + 44, original.len() - 1] {
            assert!(!invalid_footer("test", &original[..length]));
        }
        for offset in [20, second + 20] {
            for count in [-1_i32, i32::MAX] {
                let mut bytes = original.clone();
                bytes[offset..offset + 4].copy_from_slice(&count.to_be_bytes());
                assert!(!invalid_footer("test", &bytes));
            }
        }
        for text in [b"".as_slice(), b"UTC0", b"INVALID!!!\nextra", &[0xff]] {
            assert!(!invalid_footer("test", &data(text)));
        }
        let mut short = original[..second + 44].to_vec();
        short[second + 40..second + 44].copy_from_slice(&50_i32.to_be_bytes());
        short.extend_from_slice(b"\nINVALID!!!\n");
        assert!(!invalid_footer("test", &short));
        let mut version = original;
        version[second + 4] = b'9';
        assert!(!invalid_footer("test", &version));
    }
}
