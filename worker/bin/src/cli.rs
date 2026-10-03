//! The `ml-worker` command line: `run` (the default), `check-config`, and the
//! shared worker flags plus explicit Rust auxiliary provider selection.
//! Python's `--config` (B11) and `--max-frames-per-camera` (X11 b) are refused,
//! with or without a value. `engine-build` has its own flag set.

mod engine_build;

use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

use crate::config::model_bundle::identity::AuxiliaryRuntime;
use crate::exit::Exit;
pub use engine_build::{AuxiliaryBuild, EngineBuildFlags, ParseError as EngineBuildParseError};

const CHECK_CONFIG: &str = "check-config";
const ENGINE_BUILD: &str = "engine-build";
const RUN: &str = "run";
const HEARTBEAT_ON_START: &str = "--heartbeat-on-start";
const STATE_DIR: &str = "--state-dir";
const AUXILIARY_RUNTIME: &str = "--auxiliary-runtime";
/// Python flags the Rust worker refuses.
const REFUSED: [&str; 2] = ["--config", "--max-frames-per-camera"];

/// The parsed flags.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Flags {
    /// `--heartbeat-on-start`: one relay heartbeat per camera at start.
    pub heartbeat_on_start: bool,
    /// `--state-dir <path>`; `None` means `$HOME/.local/state/ml-worker`.
    pub state_dir: Option<PathBuf>,
    /// `--auxiliary-runtime`: explicit provider for auxiliary models only.
    pub auxiliary_runtime: AuxiliaryRuntime,
}

/// A command the worker runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Runs the admitted worker until shutdown or a restart directive.
    Run(Flags),
    /// Checks the env, model bundle and engine identity, touching no GPU,
    /// camera or network.
    CheckConfig(Flags),
    /// Builds engines from the approved flag set. Parsing only; the parent
    /// wires execution.
    EngineBuild(Box<EngineBuildFlags>),
}

/// Why the command line was refused; every refusal exits 2, as an
/// `argparse` error does.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliError {
    /// A command this worker does not have.
    Command(OsString),
    /// An argument that is not a known flag.
    UnknownFlag(OsString),
    /// `--config` or `--max-frames-per-camera`.
    RefusedFlag(&'static str),
    /// `--state-dir` or `--auxiliary-runtime` without a value.
    MissingValue,
    /// `--heartbeat-on-start=<value>`: the flag takes no value.
    UnexpectedValue,
    /// An unsupported or non-UTF-8 auxiliary runtime; no raw value is retained.
    InvalidAuxiliaryRuntime,
    /// `engine-build` was refused. The inner error carries no secret or raw value.
    EngineBuild(engine_build::ParseError),
}

impl CliError {
    pub const fn exit(&self) -> Exit {
        Exit::Config
    }
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

/// Parses the flags that follow a command. A value is either
/// `--name=value` or the next argument, which must not start with `-`;
/// repeated value flags keep the last value, as `argparse` does.
pub fn parse_flags(arguments: &[OsString]) -> Result<Flags, CliError> {
    let mut flags = Flags::default();
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        let (flag, value) = split_flag(argument);
        if let Some(refused) = REFUSED.into_iter().find(|name| flag == *name) {
            return Err(CliError::RefusedFlag(refused));
        }
        if flag == HEARTBEAT_ON_START {
            if value.is_some() {
                return Err(CliError::UnexpectedValue);
            }
            flags.heartbeat_on_start = true;
        } else if flag == STATE_DIR || flag == AUXILIARY_RUNTIME {
            let value = match value {
                Some(value) => value,
                None => match remaining.next() {
                    Some(next) if !next.as_encoded_bytes().starts_with(b"-") => next,
                    _ => return Err(CliError::MissingValue),
                },
            };
            if flag == STATE_DIR {
                flags.state_dir = Some(PathBuf::from(value));
            } else {
                flags.auxiliary_runtime = value
                    .to_str()
                    .and_then(AuxiliaryRuntime::parse)
                    .ok_or(CliError::InvalidAuxiliaryRuntime)?;
            }
        } else {
            return Err(CliError::UnknownFlag(argument.clone()));
        }
    }
    Ok(flags)
}

/// Parses the arguments after the program name.
pub fn parse(arguments: &[OsString]) -> Result<Command, CliError> {
    match arguments.split_first() {
        None => Ok(Command::Run(Flags::default())),
        Some((command, flags)) if command == RUN => parse_flags(flags).map(Command::Run),
        Some((command, flags)) if command == ENGINE_BUILD => engine_build::parse(flags)
            .map(|flags| Command::EngineBuild(Box::new(flags)))
            .map_err(CliError::EngineBuild),
        Some((command, flags)) if command == CHECK_CONFIG => {
            parse_flags(flags).map(Command::CheckConfig)
        }
        Some((flag, _)) if flag.as_encoded_bytes().starts_with(b"-") => {
            parse_flags(arguments).map(Command::Run)
        }
        Some((command, _)) => Err(CliError::Command(command.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::ffi::OsStringExt;

    fn arguments(texts: &[&str]) -> Vec<OsString> {
        texts.iter().map(OsString::from).collect()
    }

    #[test]
    fn flags_parse_as_argparse_does() {
        let parsed = parse(&arguments(&[
            "check-config",
            "--state-dir",
            "/a",
            "--heartbeat-on-start",
            "--state-dir=/b",
        ]));
        let flags = Flags {
            heartbeat_on_start: true,
            state_dir: Some(PathBuf::from("/b")),
            auxiliary_runtime: AuxiliaryRuntime::TensorRt,
        };
        assert_eq!(parsed, Ok(Command::CheckConfig(flags)));
        let bare = parse(&arguments(&["check-config"]));
        assert_eq!(bare, Ok(Command::CheckConfig(Flags::default())));
    }

    #[test]
    fn refusals_exit_two() {
        let refused = [
            vec!["unknown-command"],
            vec!["--check-config"],
            vec!["run", "--config", "x.yaml"],
            vec!["run", "--max-frames-per-camera=1"],
            vec!["run", "--heartbeat-on-start=1"],
            vec!["run", "--state-dir"],
            vec!["run", "positional"],
            vec!["--config=x.yaml"],
            vec!["check-config", "--config", "x.yaml"],
            vec!["check-config", "--config=x.yaml"],
            vec!["check-config", "--max-frames-per-camera", "1"],
            vec!["check-config", "--max-frames-per-camera=1"],
            vec!["check-config", "--max-frames-per-camera"],
            vec!["check-config", "--state-dir"],
            vec!["check-config", "--state-dir", "--heartbeat-on-start"],
            vec!["check-config", "--heartbeat-on-start=1"],
            vec!["check-config", "--unknown"],
            vec!["check-config", "positional"],
        ];
        for case in refused {
            let error = parse(&arguments(&case)).expect_err("refused");
            assert_eq!(error.exit(), Exit::Config, "{case:?}");
        }
    }

    #[test]
    fn explicit_and_default_run_share_flags_without_selecting_static_check() {
        let flags = Flags {
            heartbeat_on_start: true,
            state_dir: Some(PathBuf::from("/state")),
            auxiliary_runtime: AuxiliaryRuntime::TensorRt,
        };
        for arguments in [
            arguments(&["run", "--heartbeat-on-start", "--state-dir=/state"]),
            arguments(&["--state-dir", "/state", "--heartbeat-on-start"]),
        ] {
            assert_eq!(parse(&arguments), Ok(Command::Run(flags.clone())));
        }
        assert_eq!(parse(&[]), Ok(Command::Run(Flags::default())));
    }

    #[test]
    fn auxiliary_modes_share_both_syntaxes_and_flag_orders_across_runtime_commands() {
        for command in [None, Some(RUN), Some(CHECK_CONFIG)] {
            for (value, auxiliary_runtime) in [
                ("tensorrt", AuxiliaryRuntime::TensorRt),
                ("onnxruntime-cpu", AuxiliaryRuntime::OnnxRuntimeCpu),
            ] {
                for attached in [false, true] {
                    for mode_first in [false, true] {
                        let mut values: Vec<OsString> =
                            command.into_iter().map(OsString::from).collect();
                        let mode = if attached {
                            vec![OsString::from(format!("{AUXILIARY_RUNTIME}={value}"))]
                        } else {
                            arguments(&[AUXILIARY_RUNTIME, value])
                        };
                        let common = arguments(&[HEARTBEAT_ON_START, STATE_DIR, "/state"]);
                        if mode_first {
                            values.extend(mode);
                            values.extend(common);
                        } else {
                            values.extend(common);
                            values.extend(mode);
                        }
                        let flags = Flags {
                            heartbeat_on_start: true,
                            state_dir: Some(PathBuf::from("/state")),
                            auxiliary_runtime,
                        };
                        let expected = if command == Some(CHECK_CONFIG) {
                            Command::CheckConfig(flags)
                        } else {
                            Command::Run(flags)
                        };
                        assert_eq!(parse(&values), Ok(expected), "{values:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_auxiliary_runtime_keeps_the_last_valid_value() {
        for command in [None, Some(RUN), Some(CHECK_CONFIG)] {
            for (first, last, expected) in [
                (
                    "tensorrt",
                    "onnxruntime-cpu",
                    AuxiliaryRuntime::OnnxRuntimeCpu,
                ),
                ("onnxruntime-cpu", "tensorrt", AuxiliaryRuntime::TensorRt),
            ] {
                for attached_last in [false, true] {
                    let mut values: Vec<OsString> =
                        command.into_iter().map(OsString::from).collect();
                    if attached_last {
                        values.extend(arguments(&[AUXILIARY_RUNTIME, first]));
                        values.push(OsString::from(format!("{AUXILIARY_RUNTIME}={last}")));
                    } else {
                        values.push(OsString::from(format!("{AUXILIARY_RUNTIME}={first}")));
                        values.extend(arguments(&[AUXILIARY_RUNTIME, last]));
                    }
                    let parsed = parse(&values).expect("valid duplicate modes");
                    let flags = match parsed {
                        Command::Run(flags) | Command::CheckConfig(flags) => flags,
                        Command::EngineBuild(_) => panic!("runtime dispatch expected"),
                    };
                    assert_eq!(flags.auxiliary_runtime, expected, "{values:?}");
                }
            }
        }
    }

    #[test]
    fn omitted_auxiliary_mode_preserves_tensor_rt_for_every_runtime_dispatch() {
        for values in [
            Vec::new(),
            arguments(&[RUN]),
            arguments(&[CHECK_CONFIG]),
            arguments(&[HEARTBEAT_ON_START]),
        ] {
            let flags = match parse(&values).expect("runtime command") {
                Command::Run(flags) | Command::CheckConfig(flags) => flags,
                Command::EngineBuild(_) => panic!("runtime dispatch expected"),
            };
            assert_eq!(flags.auxiliary_runtime, AuxiliaryRuntime::TensorRt);
        }
    }

    #[test]
    fn malformed_auxiliary_modes_refuse_before_later_valid_duplicates() {
        for command in [None, Some(RUN), Some(CHECK_CONFIG)] {
            for value in [
                "",
                "cpu",
                "onnxruntime",
                "TensorRt",
                "ONNXRUNTIME-CPU",
                " tensorrt",
                "tensorrt ",
                "onnxruntime-cpu\n",
                "tensorrt=onnxruntime-cpu",
                "tensorrt\0",
            ] {
                for attached in [false, true] {
                    let mut values: Vec<OsString> =
                        command.into_iter().map(OsString::from).collect();
                    if attached {
                        values.push(OsString::from(format!("{AUXILIARY_RUNTIME}={value}")));
                    } else {
                        values.extend(arguments(&[AUXILIARY_RUNTIME, value]));
                    }
                    values.push(OsString::from("--auxiliary-runtime=onnxruntime-cpu"));
                    let error = parse(&values).expect_err("invalid mode");
                    assert_eq!(error, CliError::InvalidAuxiliaryRuntime);
                    assert_eq!(error.exit(), Exit::Config);
                }
            }
        }
    }

    #[test]
    fn non_utf8_and_missing_auxiliary_values_refuse_without_retaining_the_value() {
        for command in [None, Some(RUN), Some(CHECK_CONFIG)] {
            for mode in [
                vec![
                    OsString::from(AUXILIARY_RUNTIME),
                    OsString::from_vec(vec![0xff]),
                ],
                vec![OsString::from_vec(b"--auxiliary-runtime=\xff".to_vec())],
            ] {
                let mut values: Vec<OsString> = command.into_iter().map(OsString::from).collect();
                values.extend(mode);
                let error = parse(&values).expect_err("non-UTF-8 mode");
                assert_eq!(error, CliError::InvalidAuxiliaryRuntime);
                assert_eq!(error.exit(), Exit::Config);
            }
            for following in [
                None,
                Some(HEARTBEAT_ON_START),
                Some("--auxiliary-runtime=tensorrt"),
            ] {
                let mut values: Vec<OsString> = command.into_iter().map(OsString::from).collect();
                values.push(OsString::from(AUXILIARY_RUNTIME));
                values.extend(following.into_iter().map(OsString::from));
                let error = parse(&values).expect_err("missing mode");
                assert_eq!(error, CliError::MissingValue);
                assert_eq!(error.exit(), Exit::Config);
            }
        }
    }

    #[test]
    fn engine_build_dispatches_before_default_run_and_exits_two() {
        let parsed = parse(&arguments(&["engine-build", "--force"])).expect_err("incomplete");
        assert_eq!(
            parsed,
            CliError::EngineBuild(EngineBuildParseError::MissingRequired)
        );
        assert_eq!(parsed.exit(), Exit::Config);
        assert_eq!(
            parse(&arguments(&["--engine-build"])),
            Err(CliError::UnknownFlag(OsString::from("--engine-build")))
        );
        assert_eq!(parse(&[]), Ok(Command::Run(Flags::default())));
    }
}
