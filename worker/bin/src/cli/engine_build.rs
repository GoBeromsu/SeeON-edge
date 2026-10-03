//! Offline `engine-build` flags with an explicit auxiliary runtime. Parsing
//! only: this module never builds an engine or opens a model.

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

use crate::config::model_bundle::identity::AuxiliaryRuntime;

/// Common required path flags, in declaration order.
const REQUIRED_PATHS: [&str; 8] = [
    "--onnx",
    "--engine",
    "--identity",
    "--parser-lib",
    "--infer-config",
    "--tracker-config",
    "--tracker-library",
    "--bed-onnx",
];
const REQUIRED_COUNT: usize = REQUIRED_PATHS.len();
const AUXILIARY_PATHS: [&str; 3] = ["--stored-pose-engine", "--bed-engine", "--fall-engine"];

const AUXILIARY_RUNTIME: &str = "--auxiliary-runtime";
const IMAGE_DIGEST: &str = "--image-digest";
const BATCH_SIZE: &str = "--batch-size";
const SERVED_INFER_CONFIG: &str = "--served-infer-config";
const FORCE: &str = "--force";

/// Auxiliary artifacts selected for the offline build. CPU models use the
/// captured original ONNX sources and have no auxiliary GPU output paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuxiliaryBuild {
    TensorRt {
        stored_pose_engine: PathBuf,
        bed_engine: PathBuf,
        fall_engine: PathBuf,
    },
    OnnxRuntimeCpu,
}

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
    pub auxiliary: AuxiliaryBuild,
    pub bed_onnx: PathBuf,
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
    /// A non-path value was not UTF-8.
    NotUtf8,
    /// `--batch-size` was not a canonical positive ASCII decimal in `1..=16`.
    InvalidBatch,
    /// `--auxiliary-runtime` was empty or not one of the two supported modes.
    InvalidAuxiliaryRuntime,
    /// An auxiliary GPU output was supplied with the CPU runtime.
    ForbiddenAuxiliaryOutput,
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

fn parse_auxiliary_runtime(value: &OsStr) -> Result<AuxiliaryRuntime, ParseError> {
    AuxiliaryRuntime::parse(value.to_str().ok_or(ParseError::NotUtf8)?)
        .ok_or(ParseError::InvalidAuxiliaryRuntime)
}

/// Parses `engine-build` arguments. A value is `--name=value` or the next
/// argument, which must not start with `-`. A repeated flag keeps the last
/// value. TensorRT is the default and requires all auxiliary GPU outputs;
/// CPU mode refuses any of them. `--force` rejects `--force=value`.
pub fn parse(arguments: &[OsString]) -> Result<EngineBuildFlags, ParseError> {
    let mut paths: [Option<PathBuf>; REQUIRED_COUNT] = std::array::from_fn(|_| None);
    let mut image_digest = None;
    let mut batch_size = None;
    let mut served_infer_config = None;
    let mut force = false;
    let mut auxiliary_paths: [Option<PathBuf>; 3] = [None, None, None];
    let mut auxiliary_runtime = AuxiliaryRuntime::TensorRt;

    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        let (flag, attached) = split_flag(argument);
        let Some(name) = flag.to_str() else {
            return Err(ParseError::UnknownFlag);
        };
        if let Some(index) = REQUIRED_PATHS.iter().position(|required| *required == name) {
            paths[index] = Some(PathBuf::from(nonempty(take_value(
                attached,
                &mut remaining,
            )?)?));
        } else if let Some(index) = AUXILIARY_PATHS.iter().position(|output| *output == name) {
            auxiliary_paths[index] = Some(PathBuf::from(nonempty(take_value(
                attached,
                &mut remaining,
            )?)?));
        } else if name == AUXILIARY_RUNTIME {
            auxiliary_runtime = parse_auxiliary_runtime(take_value(attached, &mut remaining)?)?;
        } else if name == IMAGE_DIGEST {
            let value = nonempty(take_value(attached, &mut remaining)?)?;
            image_digest = Some(value.to_str().ok_or(ParseError::NotUtf8)?.to_owned());
        } else if name == BATCH_SIZE {
            batch_size = Some(parse_batch(take_value(attached, &mut remaining)?)?);
        } else if name == SERVED_INFER_CONFIG {
            let value = nonempty(take_value(attached, &mut remaining)?)?;
            served_infer_config = Some(PathBuf::from(value));
        } else if name == FORCE {
            if attached.is_some() {
                return Err(ParseError::UnexpectedValue);
            }
            force = true;
        } else {
            return Err(ParseError::UnknownFlag);
        }
    }
    if image_digest.is_none() || batch_size.is_none() || paths.iter().any(Option::is_none) {
        return Err(ParseError::MissingRequired);
    }
    let auxiliary = match auxiliary_runtime {
        AuxiliaryRuntime::TensorRt => {
            let [stored_pose_engine, bed_engine, fall_engine] = auxiliary_paths;
            AuxiliaryBuild::TensorRt {
                stored_pose_engine: stored_pose_engine.ok_or(ParseError::MissingRequired)?,
                bed_engine: bed_engine.ok_or(ParseError::MissingRequired)?,
                fall_engine: fall_engine.ok_or(ParseError::MissingRequired)?,
            }
        }
        AuxiliaryRuntime::OnnxRuntimeCpu => {
            if auxiliary_paths.iter().any(Option::is_some) {
                return Err(ParseError::ForbiddenAuxiliaryOutput);
            }
            AuxiliaryBuild::OnnxRuntimeCpu
        }
    };
    let [
        onnx,
        engine,
        identity,
        parser_lib,
        infer_config,
        tracker_config,
        tracker_library,
        bed_onnx,
    ] = paths;
    Ok(EngineBuildFlags {
        onnx: onnx.ok_or(ParseError::MissingRequired)?,
        engine: engine.ok_or(ParseError::MissingRequired)?,
        identity: identity.ok_or(ParseError::MissingRequired)?,
        parser_lib: parser_lib.ok_or(ParseError::MissingRequired)?,
        infer_config: infer_config.ok_or(ParseError::MissingRequired)?,
        tracker_config: tracker_config.ok_or(ParseError::MissingRequired)?,
        tracker_library: tracker_library.ok_or(ParseError::MissingRequired)?,
        auxiliary,
        bed_onnx: bed_onnx.ok_or(ParseError::MissingRequired)?,
        served_infer_config,
        image_digest: image_digest.ok_or(ParseError::MissingRequired)?,
        batch_size: batch_size.ok_or(ParseError::MissingRequired)?,
        force,
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    use super::{AuxiliaryBuild, EngineBuildFlags, ParseError, parse};

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
            auxiliary: AuxiliaryBuild::TensorRt {
                stored_pose_engine: PathBuf::from("/models/stored-pose.engine"),
                bed_engine: PathBuf::from("/models/bed.engine"),
                fall_engine: PathBuf::from("/models/fall.engine"),
            },
            bed_onnx: PathBuf::from("/models/bed.onnx"),
            served_infer_config: served.map(PathBuf::from),
            image_digest: "sha256:abc".to_owned(),
            batch_size: 4,
            force,
        }
    }

    fn full(style: &str) -> Vec<OsString> {
        path_arguments(style, true)
    }

    fn common(style: &str) -> Vec<OsString> {
        path_arguments(style, false)
    }

    fn path_arguments(style: &str, auxiliary_outputs: bool) -> Vec<OsString> {
        let mut out = Vec::new();
        for (name, value) in required_pairs() {
            if !auxiliary_outputs && super::AUXILIARY_PATHS.contains(&name) {
                continue;
            }
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

    fn value_flag(style: &str, name: &str, value: &str) -> Vec<OsString> {
        match style {
            "equals" => vec![OsString::from(format!("{name}={value}"))],
            "split" => arguments(&[name, value]),
            _ => unreachable!(),
        }
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
            "--stored-pose-engine=/first-stored.engine",
            "--stored-pose-engine",
            "/models/stored-pose.engine",
            "--bed-engine=/first-bed.engine",
            "--bed-engine",
            "/models/bed.engine",
            "--fall-engine=/first-fall.engine",
            "--fall-engine",
            "/models/fall.engine",
            "--force",
            "--force",
        ]));
        let mut expected = expected(None, true);
        expected.batch_size = 16;
        assert_eq!(parse(&values), Ok(expected));
    }

    #[test]
    fn auxiliary_modes_accept_both_syntaxes_before_or_after_paths() {
        for style in ["equals", "split"] {
            for mode_first in [false, true] {
                for mode in ["tensorrt", "onnxruntime-cpu"] {
                    let paths = if mode == "tensorrt" {
                        full(style)
                    } else {
                        common(style)
                    };
                    let runtime = value_flag(style, "--auxiliary-runtime", mode);
                    let values: Vec<_> = if mode_first {
                        runtime.into_iter().chain(paths).collect()
                    } else {
                        paths.into_iter().chain(runtime).collect()
                    };
                    let mut expected = expected(None, false);
                    if mode == "onnxruntime-cpu" {
                        expected.auxiliary = AuxiliaryBuild::OnnxRuntimeCpu;
                    }
                    assert_eq!(parse(&values), Ok(expected), "{style} {mode_first} {mode}");
                }
            }
        }
    }

    #[test]
    fn cpu_refuses_each_auxiliary_gpu_output_in_either_order() {
        for style in ["equals", "split"] {
            for name in super::AUXILIARY_PATHS {
                for output_first in [false, true] {
                    let mut values = common(style);
                    let runtime = value_flag(style, "--auxiliary-runtime", "onnxruntime-cpu");
                    let output = value_flag(style, name, "/unused.engine");
                    if output_first {
                        values.extend(output);
                        values.extend(runtime);
                    } else {
                        values.extend(runtime);
                        values.extend(output);
                    }
                    assert_eq!(parse(&values), Err(ParseError::ForbiddenAuxiliaryOutput));
                }
            }
        }
    }

    #[test]
    fn auxiliary_runtime_duplicates_select_the_last_valid_value() {
        let mut tensor_rt = full("equals");
        tensor_rt.extend(arguments(&[
            "--auxiliary-runtime=onnxruntime-cpu",
            "--auxiliary-runtime",
            "tensorrt",
        ]));
        assert_eq!(parse(&tensor_rt), Ok(expected(None, false)));

        let mut cpu = common("split");
        cpu.extend(arguments(&[
            "--auxiliary-runtime=tensorrt",
            "--auxiliary-runtime",
            "onnxruntime-cpu",
        ]));
        let mut expected_cpu = expected(None, false);
        expected_cpu.auxiliary = AuxiliaryBuild::OnnxRuntimeCpu;
        assert_eq!(parse(&cpu), Ok(expected_cpu));

        tensor_rt.push(OsString::from("--auxiliary-runtime=onnxruntime-cpu"));
        assert_eq!(parse(&tensor_rt), Err(ParseError::ForbiddenAuxiliaryOutput));
        cpu.push(OsString::from("--auxiliary-runtime=tensorrt"));
        assert_eq!(parse(&cpu), Err(ParseError::MissingRequired));
        assert_eq!(parse(&common("equals")), Err(ParseError::MissingRequired));
    }

    #[test]
    fn auxiliary_runtime_unknown_empty_and_non_utf8_values_are_refused() {
        for style in ["equals", "split"] {
            for value in [
                "",
                "cpu",
                "onnxruntime",
                "TensorRt",
                " tensorrt",
                "tensorrt ",
            ] {
                let mut values = common(style);
                values.extend(value_flag(style, "--auxiliary-runtime", value));
                values.push(OsString::from("--auxiliary-runtime=onnxruntime-cpu"));
                assert_eq!(parse(&values), Err(ParseError::InvalidAuxiliaryRuntime));
            }
        }
        let mut split = common("split");
        split.push(OsString::from("--auxiliary-runtime"));
        split.push(OsString::from_vec(vec![0xff]));
        assert_eq!(parse(&split), Err(ParseError::NotUtf8));
        let mut attached = common("equals");
        attached.push(OsString::from_vec(b"--auxiliary-runtime=\xff".to_vec()));
        assert_eq!(parse(&attached), Err(ParseError::NotUtf8));
    }

    #[test]
    fn cpu_still_requires_every_common_input_and_rejects_empty_gpu_paths() {
        for omitted in super::REQUIRED_PATHS
            .into_iter()
            .chain(["--image-digest", "--batch-size"])
        {
            let kept: Vec<_> = required_pairs()
                .into_iter()
                .filter(|(name, _)| !super::AUXILIARY_PATHS.contains(name) && *name != omitted)
                .flat_map(|(name, value)| [OsString::from(name), OsString::from(value)])
                .chain([OsString::from("--auxiliary-runtime=onnxruntime-cpu")])
                .collect();
            assert_eq!(parse(&kept), Err(ParseError::MissingRequired), "{omitted}");
        }
        for name in super::AUXILIARY_PATHS {
            let mut values = common("equals");
            values.push(OsString::from("--auxiliary-runtime=onnxruntime-cpu"));
            values.push(OsString::from(format!("{name}=")));
            assert_eq!(parse(&values), Err(ParseError::MissingRequired));
        }
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
            "--auxiliary-runtime",
            "--stored-pose-engine",
            "--bed-engine",
            "--fall-engine",
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
