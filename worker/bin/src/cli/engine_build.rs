//! `engine-build` flags, matching `worker/tools/edge_engine_build.py` plus the
//! approved required pose and bed/fall paths. Parsing only: this module never
//! builds an engine.

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

/// Required path flags, in declaration order.
const REQUIRED_PATHS: [&str; 11] = [
    "--onnx",
    "--engine",
    "--identity",
    "--parser-lib",
    "--infer-config",
    "--tracker-config",
    "--tracker-library",
    "--stored-pose-engine",
    "--bed-onnx",
    "--bed-engine",
    "--fall-engine",
];
const REQUIRED_COUNT: usize = REQUIRED_PATHS.len();

const IMAGE_DIGEST: &str = "--image-digest";
const BATCH_SIZE: &str = "--batch-size";
const SERVED_INFER_CONFIG: &str = "--served-infer-config";
const FORCE: &str = "--force";

/// Parsed `engine-build` flags. Paths keep their `OsString`, including
/// non-UTF-8 bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineBuildFlags {
    pub onnx: PathBuf,
    pub engine: PathBuf,
    pub identity: PathBuf,
    pub parser_lib: PathBuf,
    pub infer_config: PathBuf,
    pub tracker_config: PathBuf,
    pub tracker_library: PathBuf,
    pub stored_pose_engine: PathBuf,
    pub bed_onnx: PathBuf,
    pub bed_engine: PathBuf,
    pub fall_engine: PathBuf,
    pub served_infer_config: Option<PathBuf>,
    pub image_digest: String,
    pub batch_size: u32,
    pub force: bool,
}

/// Why `engine-build` was refused. Variants carry no secret or raw value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseError {
    /// A required flag was omitted, or a path/image value was empty.
    MissingRequired,
    /// An argument that is not a known flag, or a positional.
    UnknownFlag,
    /// A value flag whose next argument was absent or itself a flag.
    MissingValue,
    /// `--force=<value>`: the flag takes no value.
    UnexpectedValue,
    /// `--image-digest` or `--batch-size` was not UTF-8.
    NotUtf8,
    /// `--batch-size` was not a canonical positive ASCII decimal in `1..=16`.
    InvalidBatch,
}

/// Split the ASCII separator without decoding a possibly non-UTF-8 path.
fn split_flag(argument: &OsStr) -> (&OsStr, Option<&OsStr>) {
    let bytes = argument.as_bytes();
    match bytes.iter().position(|byte| *byte == b'=') {
        Some(index) => (
            OsStr::from_bytes(&bytes[..index]),
            Some(OsStr::from_bytes(&bytes[index + 1..])),
        ),
        None => (argument, None),
    }
}

fn path_slot<'a>(flags: &'a mut EngineBuildFlags, name: &str) -> Option<&'a mut PathBuf> {
    Some(match name {
        "--onnx" => &mut flags.onnx,
        "--engine" => &mut flags.engine,
        "--identity" => &mut flags.identity,
        "--parser-lib" => &mut flags.parser_lib,
        "--infer-config" => &mut flags.infer_config,
        "--tracker-config" => &mut flags.tracker_config,
        "--tracker-library" => &mut flags.tracker_library,
        "--stored-pose-engine" => &mut flags.stored_pose_engine,
        "--bed-onnx" => &mut flags.bed_onnx,
        "--bed-engine" => &mut flags.bed_engine,
        "--fall-engine" => &mut flags.fall_engine,
        _ => return None,
    })
}

fn take_value<'a>(
    attached: Option<&'a OsStr>,
    remaining: &mut impl Iterator<Item = &'a OsString>,
) -> Result<&'a OsStr, ParseError> {
    match attached {
        Some(value) => Ok(value),
        None => match remaining.next() {
            Some(next) if !next.as_encoded_bytes().starts_with(b"-") => Ok(next),
            _ => Err(ParseError::MissingValue),
        },
    }
}

fn nonempty(value: &OsStr) -> Result<&OsStr, ParseError> {
    if value.is_empty() {
        Err(ParseError::MissingRequired)
    } else {
        Ok(value)
    }
}

/// Canonical positive ASCII decimal in `1..=16`: no sign, leading zero, or
/// surrounding whitespace.
fn parse_batch(value: &OsStr) -> Result<u32, ParseError> {
    let text = value.to_str().ok_or(ParseError::NotUtf8)?;
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ParseError::InvalidBatch);
    }
    if text.len() > 1 && text.starts_with('0') {
        return Err(ParseError::InvalidBatch);
    }
    let batch: u32 = text.parse().map_err(|_| ParseError::InvalidBatch)?;
    if (1..=16).contains(&batch) {
        Ok(batch)
    } else {
        Err(ParseError::InvalidBatch)
    }
}

/// Parses `engine-build` arguments. A value is `--name=value` or the next
/// argument, which must not start with `-`. A repeated flag keeps the last
/// value. `--force` is boolean and rejects `--force=value`.
pub fn parse(arguments: &[OsString]) -> Result<EngineBuildFlags, ParseError> {
    let mut flags = EngineBuildFlags {
        onnx: PathBuf::new(),
        engine: PathBuf::new(),
        identity: PathBuf::new(),
        parser_lib: PathBuf::new(),
        infer_config: PathBuf::new(),
        tracker_config: PathBuf::new(),
        tracker_library: PathBuf::new(),
        stored_pose_engine: PathBuf::new(),
        bed_onnx: PathBuf::new(),
        bed_engine: PathBuf::new(),
        fall_engine: PathBuf::new(),
        served_infer_config: None,
        image_digest: String::new(),
        batch_size: 0,
        force: false,
    };
    let mut saw_image = false;
    let mut saw_path = [false; REQUIRED_COUNT];

    let mut saw_batch = false;
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        let (flag, attached) = split_flag(argument);
        let Some(name) = flag.to_str() else {
            return Err(ParseError::UnknownFlag);
        };
        if let Some(index) = REQUIRED_PATHS.iter().position(|required| *required == name) {
            let value = PathBuf::from(nonempty(take_value(attached, &mut remaining)?)?);
            *path_slot(&mut flags, name).expect("required path") = value;
            saw_path[index] = true;
        } else if name == IMAGE_DIGEST {
            let value = nonempty(take_value(attached, &mut remaining)?)?;
            flags.image_digest = value.to_str().ok_or(ParseError::NotUtf8)?.to_owned();
            saw_image = true;
        } else if name == BATCH_SIZE {
            flags.batch_size = parse_batch(take_value(attached, &mut remaining)?)?;
            saw_batch = true;
        } else if name == SERVED_INFER_CONFIG {
            let value = nonempty(take_value(attached, &mut remaining)?)?;
            flags.served_infer_config = Some(PathBuf::from(value));
        } else if name == FORCE {
            if attached.is_some() {
                return Err(ParseError::UnexpectedValue);
            }
            flags.force = true;
        } else {
            return Err(ParseError::UnknownFlag);
        }
    }
    let paths_complete = saw_path.iter().all(|seen| *seen);
    if !saw_image || !saw_batch || !paths_complete {
        return Err(ParseError::MissingRequired);
    }
    Ok(flags)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    use super::{EngineBuildFlags, ParseError, parse};

    fn arguments(texts: &[&str]) -> Vec<OsString> {
        texts.iter().copied().map(OsString::from).collect()
    }

    fn required_pairs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("--onnx", "/models/pose.onnx"),
            ("--engine", "/models/pose.engine"),
            ("--identity", "/state/identity.json"),
            ("--parser-lib", "/opt/libnvdsinfer_custom_impl_Yolo.so"),
            ("--infer-config", "/opt/infer.txt"),
            ("--tracker-config", "/opt/tracker.txt"),
            ("--tracker-library", "/opt/libnvds_nvmultiobjecttracker.so"),
            ("--stored-pose-engine", "/models/stored-pose.engine"),
            ("--bed-onnx", "/models/bed.onnx"),
            ("--bed-engine", "/models/bed.engine"),
            ("--fall-engine", "/models/fall.engine"),
            ("--image-digest", "sha256:abc"),
            ("--batch-size", "4"),
        ]
    }

    fn expected(served: Option<&str>, force: bool) -> EngineBuildFlags {
        EngineBuildFlags {
            onnx: PathBuf::from("/models/pose.onnx"),
            engine: PathBuf::from("/models/pose.engine"),
            identity: PathBuf::from("/state/identity.json"),
            parser_lib: PathBuf::from("/opt/libnvdsinfer_custom_impl_Yolo.so"),
            infer_config: PathBuf::from("/opt/infer.txt"),
            tracker_config: PathBuf::from("/opt/tracker.txt"),
            tracker_library: PathBuf::from("/opt/libnvds_nvmultiobjecttracker.so"),
            stored_pose_engine: PathBuf::from("/models/stored-pose.engine"),
            bed_onnx: PathBuf::from("/models/bed.onnx"),
            bed_engine: PathBuf::from("/models/bed.engine"),
            fall_engine: PathBuf::from("/models/fall.engine"),
            served_infer_config: served.map(PathBuf::from),
            image_digest: "sha256:abc".to_owned(),
            batch_size: 4,
            force,
        }
    }

    fn full(style: &str) -> Vec<OsString> {
        let mut out = Vec::new();
        for (name, value) in required_pairs() {
            match style {
                "equals" => out.push(OsString::from(format!("{name}={value}"))),
                "split" => {
                    out.push(OsString::from(name));
                    out.push(OsString::from(value));
                }
                _ => unreachable!(),
            }
        }
        out
    }

    #[test]
    fn equals_and_split_preserve_paths() {
        let mut equals = full("equals");
        equals.push(OsString::from("--served-infer-config=/opt/served.txt"));
        equals.push(OsString::from("--force"));
        assert_eq!(parse(&equals), Ok(expected(Some("/opt/served.txt"), true)));

        let mut split = full("split");
        split.push(OsString::from("--served-infer-config"));
        split.push(OsString::from("/opt/served.txt"));
        assert_eq!(parse(&split), Ok(expected(Some("/opt/served.txt"), false)));

        let mut non_utf8 = full("split");
        non_utf8.push(OsString::from("--onnx"));
        non_utf8.push(OsString::from_vec(vec![0xff, b'/', b'a']));
        let parsed = parse(&non_utf8).expect("non-utf8 path");
        assert_eq!(
            parsed.onnx,
            PathBuf::from(OsString::from_vec(vec![0xff, b'/', b'a']))
        );
        assert_eq!(parsed.batch_size, 4);
        let mut attached = full("equals");
        attached.push(OsString::from_vec(b"--onnx=/models/\xff.onnx".to_vec()));
        assert_eq!(
            parse(&attached).expect("attached non-UTF-8 path").onnx,
            PathBuf::from(OsString::from_vec(b"/models/\xff.onnx".to_vec())),
        );
    }

    #[test]
    fn duplicate_last_wins() {
        let mut values = full("equals");
        values.extend(arguments(&[
            "--batch-size",
            "1",
            "--batch-size=16",
            "--image-digest=sha256:first",
            "--image-digest",
            "sha256:abc",
            "--engine=/first.engine",
            "--engine",
            "/models/pose.engine",
            "--force",
            "--force",
        ]));
        let mut expected = expected(None, true);
        expected.batch_size = 16;
        assert_eq!(parse(&values), Ok(expected));
    }

    #[test]
    fn required_omissions_are_missing() {
        let pairs = required_pairs();
        for skip in 0..pairs.len() {
            let kept: Vec<&str> = pairs
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != skip)
                .flat_map(|(_, (name, value))| [*name, *value])
                .collect();
            assert_eq!(
                parse(&arguments(&kept)),
                Err(ParseError::MissingRequired),
                "omitted {}",
                pairs[skip].0
            );
        }
        let mut empty = full("split");
        empty.push(OsString::from("--onnx="));
        assert_eq!(parse(&empty), Err(ParseError::MissingRequired));
        let mut blank_image = full("equals");
        blank_image.push(OsString::from("--image-digest="));
        assert_eq!(parse(&blank_image), Err(ParseError::MissingRequired));
    }

    #[test]
    fn batch_bounds_and_shape_are_refused() {
        for value in [
            "0", "17", "00", "01", "4.0", "+4", "-1", " 4", "4 ", "0x4", "",
        ] {
            let mut arguments = full("equals");
            arguments.push(OsString::from(format!("--batch-size={value}")));
            assert_eq!(
                parse(&arguments),
                Err(ParseError::InvalidBatch),
                "{value:?}"
            );
        }
        let mut non_utf8 = full("split");
        non_utf8.push(OsString::from("--batch-size"));
        non_utf8.push(OsString::from_vec(vec![0xff]));
        assert_eq!(parse(&non_utf8), Err(ParseError::NotUtf8));
        let mut image = full("split");
        image.push(OsString::from("--image-digest"));
        image.push(OsString::from_vec(vec![0xff]));
        assert_eq!(parse(&image), Err(ParseError::NotUtf8));
    }

    #[test]
    fn unknown_force_value_and_missing_value_are_refused() {
        let unknown = [vec!["--unknown"], vec!["positional"]];
        for case in unknown {
            let mut values = full("equals");
            values.extend(arguments(&case));
            assert_eq!(parse(&values), Err(ParseError::UnknownFlag), "{case:?}");
        }
        let mut force = full("split");
        force.push(OsString::from("--force=true"));
        assert_eq!(parse(&force), Err(ParseError::UnexpectedValue));

        for flag in [
            "--onnx",
            "--engine",
            "--served-infer-config",
            "--image-digest",
            "--batch-size",
        ] {
            let mut arguments = full("equals");
            arguments.push(OsString::from(flag));
            assert_eq!(parse(&arguments), Err(ParseError::MissingValue), "{flag}");
            arguments.push(OsString::from("--force"));
            assert_eq!(parse(&arguments), Err(ParseError::MissingValue), "{flag}");
        }
        assert_eq!(
            parse(&arguments(&["--force"])),
            Err(ParseError::MissingRequired)
        );
    }
}
