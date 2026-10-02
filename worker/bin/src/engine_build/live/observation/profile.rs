//! One getter-produced profile line. Implicit and FullDims text is not evidence.

use super::{LiveBuildError, Profile};

const MARKER: &str = "SEEON_BUILD_PROFILE_";
const INFO: &str = "INFO: ";
const PREFIX: &str = "INFO: SEEON_BUILD_PROFILE_V1 ";
const UNAVAILABLE: &str = "INFO: SEEON_BUILD_PROFILE_V1 unavailable";
const FIELDS: [&str; 8] = [
    "profiles=",
    "input_float=",
    "output_float=",
    "input_dims=",
    "min=",
    "opt=",
    "max=",
    "output_dims=",
];

pub(in crate::engine_build::live) fn profile(
    output: &str,
    batch: u32,
) -> Result<Profile, LiveBuildError> {
    if !(1..=16).contains(&batch) {
        return Err(LiveBuildError::Profile);
    }
    let mut accepted = 0_u32;
    let mut rejected = 0_u32;
    let mut body = None;
    for line in output.lines() {
        match classify(line) {
            Class::Absent => {}
            Class::Rejected => rejected += 1,
            Class::Accepted(item) => {
                accepted += 1;
                body = Some(item);
            }
        }
    }
    if accepted != 1 || rejected != 0 {
        return Err(LiveBuildError::Profile);
    }
    parse_body(body.ok_or(LiveBuildError::Profile)?, batch)
}

enum Class<'a> {
    Absent,
    Rejected,
    Accepted(&'a str),
}

fn classify(line: &str) -> Class<'_> {
    if line == UNAVAILABLE || !line.starts_with(INFO) && line.contains(MARKER) {
        return Class::Rejected;
    }
    let Some(rest) = line.strip_prefix(INFO) else {
        return Class::Absent;
    };
    if !rest.starts_with(MARKER) {
        return if rest.contains(MARKER) {
            Class::Rejected
        } else {
            Class::Absent
        };
    }
    match line.strip_prefix(PREFIX) {
        Some(body) if body.starts_with("profiles=") => Class::Accepted(body),
        _ => Class::Rejected,
    }
}

fn parse_body(body: &str, batch: u32) -> Result<Profile, LiveBuildError> {
    let mut parts = body.split(' ');
    flag_one(parts.next(), "profiles=")?;
    flag_one(parts.next(), "input_float=")?;
    flag_one(parts.next(), "output_float=")?;
    let input = axes::<4>(parts.next(), "input_dims=")?;
    let min = axes::<4>(parts.next(), "min=")?;
    let opt = axes::<4>(parts.next(), "opt=")?;
    let max = axes::<4>(parts.next(), "max=")?;
    let output = axes::<3>(parts.next(), "output_dims=")?;
    if parts.next().is_some() {
        return Err(LiveBuildError::Profile);
    }
    let batch = i32::try_from(batch).map_err(|_| LiveBuildError::Profile)?;
    let expected = [batch, 3, 640, 640];
    let input_batch = input[0] == -1 || input[0] == batch;
    let output_batch = output[0] == -1 || output[0] == batch;
    if !input_batch
        || input[1..] != [3, 640, 640]
        || !output_batch
        || output[1..] != [300, 57]
        || min != [1, 3, 640, 640]
        || opt != expected
        || max != expected
    {
        return Err(LiveBuildError::Profile);
    }
    Ok(Profile { min, opt, max })
}

fn flag_one(part: Option<&str>, prefix: &str) -> Result<(), LiveBuildError> {
    match ascii_i32(part, prefix)? {
        1 => Ok(()),
        _ => Err(LiveBuildError::Profile),
    }
}

fn axes<const N: usize>(part: Option<&str>, prefix: &str) -> Result<[i32; N], LiveBuildError> {
    let text = part
        .and_then(|item| item.strip_prefix(prefix))
        .ok_or(LiveBuildError::Profile)?;
    if FIELDS.iter().any(|field| text.contains(field)) {
        return Err(LiveBuildError::Profile);
    }
    let mut values = [0_i32; N];
    let mut parts = text.split('x');
    for value in &mut values {
        *value = ascii_i32(parts.next(), "")?;
    }
    if parts.next().is_some() {
        return Err(LiveBuildError::Profile);
    }
    Ok(values)
}

fn ascii_i32(part: Option<&str>, prefix: &str) -> Result<i32, LiveBuildError> {
    let text = part
        .and_then(|item| item.strip_prefix(prefix))
        .ok_or(LiveBuildError::Profile)?;
    let digits = text.strip_prefix('-').unwrap_or(text);
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return Err(LiveBuildError::Profile);
    }
    text.parse().map_err(|_| LiveBuildError::Profile)
}

#[cfg(test)]
mod tests {
    use super::profile;

    fn marker(batch: i32, input_batch: i32, output_batch: i32) -> String {
        format!(
            "INFO: SEEON_BUILD_PROFILE_V1 profiles=1 input_float=1 output_float=1 input_dims={input_batch}x3x640x640 min=1x3x640x640 opt={batch}x3x640x640 max={batch}x3x640x640 output_dims={output_batch}x300x57"
        )
    }

    #[test]
    fn measured_batches_accept_dynamic_or_requested_leading_dims() {
        for batch in [1, 2, 16] {
            let fixed = marker(batch, batch, batch);
            let dynamic = marker(batch, -1, -1);
            let seen = profile(&fixed, batch as u32).unwrap();
            assert_eq!(seen.min, [1, 3, 640, 640]);
            assert_eq!(seen.opt, [batch, 3, 640, 640]);
            assert_eq!(seen.max, [batch, 3, 640, 640]);
            assert!(profile(&dynamic, batch as u32).is_ok());
            assert!(
                profile(
                    &format!("noise\n{fixed}\nImplicit Engine Info layers 0"),
                    batch as u32
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn only_one_complete_named_getter_line_is_evidence() {
        let good = marker(1, -1, 1);
        let fulldims = "INFO: [FullDims Engine Info]: layers num: 2\n0 INPUT kFLOAT images 3x640x640 min: 1x3x640x640 opt: 1x3x640x640 Max: 1x3x640x640\n";
        assert!(profile(fulldims, 1).is_err());
        assert!(profile(&format!("{fulldims}{good}"), 1).is_ok());
        assert!(profile("INFO: SEEON_BUILD_PROFILE_V1 unavailable", 1).is_err());
        assert!(
            profile(
                &format!("INFO: SEEON_BUILD_PROFILE_V1 unavailable\n{good}"),
                1
            )
            .is_err()
        );
        assert!(profile(&format!("BLAH{good}"), 1).is_err());
        assert!(
            profile(
                &format!("INFO: SEEON_BUILD_PROFILE_V2 profiles=1\n{good}"),
                1
            )
            .is_err()
        );
        assert!(profile(&format!("{good}\n{good}"), 1).is_err());
        assert!(profile(&format!("{good}\n{}", good.replace("640", "641")), 1).is_err());
        assert!(profile(&good.replace("profiles=1", "profiles=1 profiles=1"), 1).is_err());
        assert!(profile(&good.replace(" input_float=1", ""), 1).is_err());
        assert!(profile(&format!("{good} extra=1"), 1).is_err());
        assert!(profile(&good.replace("input_float=1", "input_float=0"), 1).is_err());
        assert!(profile(&good.replace("output_float=1", "output_float=2"), 1).is_err());
        assert!(profile(&good.replace("profiles=1", "profiles=2"), 1).is_err());
        assert!(profile(&good.replace("-1x3x640x640", "-1x3x640"), 1).is_err());
        assert!(profile(&good.replace("1x300x57", "1x300"), 1).is_err());
        assert!(profile(&good.replace("-1x3x640x640", "2x3x640x640"), 1).is_err());
        assert!(profile(&good.replace("1x300x57", "-1x301x57"), 1).is_err());
        assert!(profile(&good.replace("min=1x3x640x640", "min=2x3x640x640"), 1).is_err());
        assert!(profile(&good.replace("opt=1x3x640x640", "opt=2x3x640x640"), 1).is_err());
        assert!(profile(&good.replace("max=1x3x640x640", "max=1x3x641x640"), 1).is_err());
        assert!(profile(&good.replace("-1", "-01"), 1).is_err());
        assert!(profile(&good.replace("640", "999999999999"), 1).is_err());
        assert!(profile(&good, 0).is_err());
        assert!(profile(&good, 17).is_err());
        assert!(profile(&marker(2, 2, 2), 1).is_err());
    }
}
