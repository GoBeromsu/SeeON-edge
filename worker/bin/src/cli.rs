//! The `ml-worker` command line: `check-config [--state-dir <path>]`, and the
//! flag set of `_build_parser` in `worker/__main__.py` that every command
//! shares. Python's `--config` (B11) and `--max-frames-per-camera` (X11 b)
//! are refused, with or without a value.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use crate::exit::Exit;

const CHECK_CONFIG: &str = "check-config";
const HEARTBEAT_ON_START: &str = "--heartbeat-on-start";
const STATE_DIR: &str = "--state-dir";
/// Python flags the Rust worker refuses.
const REFUSED: [&str; 2] = ["--config", "--max-frames-per-camera"];

/// The parsed flags.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Flags {
    /// `--heartbeat-on-start`: one relay heartbeat per camera at start.
    pub heartbeat_on_start: bool,
    /// `--state-dir <path>`; `None` means `$HOME/.local/state/ml-worker`.
    pub state_dir: Option<PathBuf>,
}

/// A command the worker runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    /// Checks the env, model bundle and engine identity, touching no GPU,
    /// camera or network.
    CheckConfig(Flags),
}

/// Why the command line was refused; every refusal exits 2, as an
/// `argparse` error does.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliError {
    /// No command, or a command this worker does not have.
    Command(Option<OsString>),
    /// An argument that is not a known flag.
    UnknownFlag(OsString),
    /// `--config` or `--max-frames-per-camera`.
    RefusedFlag(&'static str),
    /// `--state-dir` without a value.
    MissingValue,
    /// `--heartbeat-on-start=<value>`: the flag takes no value.
    UnexpectedValue,
}

impl CliError {
    pub const fn exit(&self) -> Exit {
        Exit::Config
    }
}

/// Splits `--flag=value` into the flag and its value; an argument that is
/// not UTF-8 is never split.
fn split_flag(argument: &OsStr) -> (&OsStr, Option<&OsStr>) {
    match argument.to_str().and_then(|text| text.split_once('=')) {
        Some((flag, value)) => (OsStr::new(flag), Some(OsStr::new(value))),
        None => (argument, None),
    }
}

/// Parses the flags that follow a command. A value is either
/// `--state-dir=<path>` or the next argument, which must not start with `-`;
/// a repeated `--state-dir` keeps the last value, as `argparse` does.
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
        } else if flag == STATE_DIR {
            let value = match value {
                Some(value) => value,
                None => match remaining.next() {
                    Some(next) if !next.as_encoded_bytes().starts_with(b"-") => next,
                    _ => return Err(CliError::MissingValue),
                },
            };
            flags.state_dir = Some(PathBuf::from(value));
        } else {
            return Err(CliError::UnknownFlag(argument.clone()));
        }
    }
    Ok(flags)
}

/// Parses the arguments after the program name.
pub fn parse(arguments: &[OsString]) -> Result<Command, CliError> {
    match arguments.split_first() {
        Some((command, flags)) if command == CHECK_CONFIG => {
            parse_flags(flags).map(Command::CheckConfig)
        }
        first => Err(CliError::Command(first.map(|(command, _)| command.clone()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };
        assert_eq!(parsed, Ok(Command::CheckConfig(flags)));
        let bare = parse(&arguments(&["check-config"]));
        assert_eq!(bare, Ok(Command::CheckConfig(Flags::default())));
    }

    #[test]
    fn refusals_exit_two() {
        let refused = [
            vec![],
            vec!["run"],
            vec!["--check-config"],
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
}
