//! Fixed V3 observation and one getter profile. Hardware facts come only
//! from that one complete line and the mapped SDK library. A filename, an
//! environment guess, or an FP32 receipt is not a device fact.

mod library;
mod profile;

use std::path::Path;

pub(super) use library::{LibraryIdentity, library_identity as capture_library};

use super::LiveBuildError;

pub(super) const OBSERVER_RESULT: &str = "/opt/seeon/nvdsinfer-observer/build-result.json";
use crate::config::model_bundle::identity::OBSERVER_LIBRARY;
const FAMILY: &str = "SEEON_BUILD_OBSERVATION_";
const INFO: &str = "INFO: ";
const NAME_LIMIT: usize = 256;

pub(super) use profile::profile;
pub(super) fn library_identity() -> Result<LibraryIdentity, LiveBuildError> {
    capture_library(Path::new(OBSERVER_RESULT), Path::new(OBSERVER_LIBRARY))
}

#[derive(Debug)]
pub(super) struct Observation {
    pub(super) fp16: bool,
    pub(super) tf32: bool,
    pub(super) trt: i32,
    pub(super) device: i32,
    pub(super) sm_major: i32,
    pub(super) sm_minor: i32,
    pub(super) device_name: String,
}

pub(super) struct Profile {
    pub(super) min: [i32; 4],
    pub(super) opt: [i32; 4],
    pub(super) max: [i32; 4],
}

pub(super) fn observation_line(output: &str) -> Result<Observation, LiveBuildError> {
    let mut accepted = 0_u32;
    let mut rejected = 0_u32;
    let mut parsed = None;
    for line in output.lines() {
        match family_payload(line) {
            Family::Absent => {}
            Family::Rejected => rejected += 1,
            Family::Accepted(payload) => {
                accepted += 1;
                if parsed.is_none() {
                    parsed = Some(parse_v3(payload)?);
                }
            }
        }
    }
    if accepted != 1 || rejected != 0 {
        return Err(LiveBuildError::Profile);
    }
    parsed.ok_or(LiveBuildError::Profile)
}

enum Family<'a> {
    Absent,
    Rejected,
    Accepted(&'a str),
}

fn family_payload(line: &str) -> Family<'_> {
    let Some(rest) = line.strip_prefix(INFO) else {
        return if line.contains(FAMILY) {
            Family::Rejected
        } else {
            Family::Absent
        };
    };
    let Some(payload) = rest.strip_prefix(FAMILY) else {
        return if rest.contains(FAMILY) {
            Family::Rejected
        } else {
            Family::Absent
        };
    };
    match payload.strip_prefix("V3 ") {
        Some(body) if body == "unavailable" || body.starts_with("flags=") => Family::Accepted(body),
        Some(_) => Family::Rejected,
        None => Family::Rejected,
    }
}

fn parse_v3(line: &str) -> Result<Observation, LiveBuildError> {
    if line == "unavailable" {
        return Err(LiveBuildError::Profile);
    }
    let mut parts = line.splitn(8, ' ');
    let _flags = ascii_u32(parts.next(), "flags=")?;
    let fp16 = flag(parts.next(), "fp16=")?;
    let tf32 = flag(parts.next(), "tf32=")?;
    let trt = ascii_i32(parts.next(), "trt=")?;
    let device = ascii_i32(parts.next(), "device=")?;
    let sm_major = ascii_i32(parts.next(), "sm_major=")?;
    let sm_minor = ascii_i32(parts.next(), "sm_minor=")?;
    let name = parts
        .next()
        .and_then(|item| item.strip_prefix("device_name="))
        .ok_or(LiveBuildError::Profile)?;
    if !fp16 || trt <= 0 || device < 0 || sm_major <= 0 || sm_minor < 0 || !valid_name(name) {
        return Err(LiveBuildError::Profile);
    }
    Ok(Observation {
        fp16,
        tf32,
        trt,
        device,
        sm_major,
        sm_minor,
        device_name: name.to_owned(),
    })
}

fn valid_name(name: &str) -> bool {
    let trimmed = name.trim();
    !trimmed.is_empty()
        && trimmed.len() == name.len()
        && name.len() <= NAME_LIMIT
        && name
            .bytes()
            .all(|byte| byte.is_ascii_graphic() || byte == b' ')
}

fn ascii_u32(part: Option<&str>, prefix: &str) -> Result<u32, LiveBuildError> {
    let text = part
        .and_then(|item| item.strip_prefix(prefix))
        .ok_or(LiveBuildError::Profile)?;
    if !text.bytes().all(|byte| byte.is_ascii_digit()) || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(LiveBuildError::Profile);
    }
    text.parse().map_err(|_| LiveBuildError::Profile)
}

fn flag(part: Option<&str>, prefix: &str) -> Result<bool, LiveBuildError> {
    match ascii_u32(part, prefix)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(LiveBuildError::Profile),
    }
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
    use super::{observation_line, profile};

    const V3: &str = "INFO: SEEON_BUILD_OBSERVATION_V3 flags=65 fp16=1 tf32=0 trt=101600 device=0 sm_major=12 sm_minor=0 device_name=NVIDIA GeForce RTX 5070 Ti";
    const PROFILE: &str = "INFO: SEEON_BUILD_PROFILE_V1 profiles=1 input_float=1 output_float=1 input_dims=-1x3x640x640 min=1x3x640x640 opt=1x3x640x640 max=1x3x640x640 output_dims=-1x300x57";

    #[test]
    fn protocol_rejects_hidden_and_wrong_shape() {
        let log = format!("{V3}\n{PROFILE}");
        let seen = observation_line(&log).unwrap();
        assert!(seen.fp16);
        assert!(!seen.tf32);
        assert!(profile(&log, 1).is_ok());
        let hidden = format!(
            "BLAHINFO: SEEON_BUILD_OBSERVATION_V3 flags=1 fp16=1 tf32=1 trt=1 device=0 sm_major=1 sm_minor=0 device_name=GPU\n{V3}"
        );
        assert!(observation_line(&hidden).is_err());
        let blank = V3.replace("device_name=NVIDIA GeForce RTX 5070 Ti", "device_name=   ");
        assert!(observation_line(&blank).is_err());
        let duplicate = format!(
            "{V3}\nINFO: SEEON_BUILD_OBSERVATION_V3 flags=1 fp16=1 tf32=1 trt=1 device=0 sm_major=1 sm_minor=0 device_name=Other GPU"
        );
        assert!(observation_line(&duplicate).is_err());
        assert!(
            observation_line(&format!("INFO: SEEON_BUILD_OBSERVATION_V2 flags=1\n{V3}")).is_err()
        );
        let fulldims = "INFO: [FullDims Engine Info]: layers num: 2\n0 INPUT kFLOAT images 3x640x640 min: 1x3x640x640 opt: 1x3x640x640 Max: 1x3x640x640\n";
        assert!(profile(fulldims, 1).is_err());
        assert!(profile(&format!("{fulldims}{PROFILE}"), 1).is_ok());
        let batch_two = PROFILE
            .replace("opt=1x3x640x640", "opt=2x3x640x640")
            .replace("max=1x3x640x640", "max=2x3x640x640");
        assert!(profile(&batch_two, 2).is_ok());
        assert!(profile(PROFILE, 2).is_err());
        assert!(profile(&format!("{PROFILE}\n{PROFILE}"), 1).is_err());
        assert!(profile("INFO: SEEON_BUILD_PROFILE_V1 unavailable", 1).is_err());
    }
}
