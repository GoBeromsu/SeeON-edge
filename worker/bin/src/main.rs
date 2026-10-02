//! `ml-worker` entry point: parses the command line, hands the process env
//! to the command as a map, and maps each outcome to its exit code.

use std::collections::BTreeMap;
use std::process::ExitCode;

use seeon_ml_worker::cli::{self, Command};
use seeon_ml_worker::config;
use seeon_ml_worker::engine_build::execute_engine_build;
use seeon_ml_worker::exit::Exit;
use seeon_ml_worker::run;

/// The process env; an entry whose name or value is not UTF-8 is left out.
fn environment() -> BTreeMap<String, String> {
    std::env::vars_os()
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
        .collect()
}

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let command = match cli::parse(&arguments) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("ml-worker: {error:?}");
            return error.exit().into();
        }
    };
    match command {
        Command::CheckConfig(flags) => check_config(flags),
        Command::EngineBuild(flags) => match execute_engine_build(&environment(), *flags) {
            Ok(()) => Exit::CleanShutdown.into(),
            Err(error) => {
                eprintln!("ml-worker: {error}");
                Exit::Runtime.into()
            }
        },
        Command::Run(flags) => match run::execution::execute(environment(), flags) {
            Ok(()) => Exit::CleanShutdown.into(),
            Err(error) => {
                eprintln!("ml-worker: {error}");
                error.exit().into()
            }
        },
    }
}

fn check_config(flags: cli::Flags) -> ExitCode {
    match config::check_config(&environment(), flags.state_dir.as_deref()) {
        Ok(checked) => {
            if let Err(error) = &checked.last_known_good {
                eprintln!("ml-worker: last-known-good config unreadable: {error:?}");
            }
            println!(
                "ml-worker: config validation passed (static check; no relay, camera or GPU touched); state dir {}",
                checked.state_dir.display()
            );
            Exit::CleanShutdown.into()
        }
        Err(error) => {
            eprintln!("ml-worker: {error:?}");
            error.exit().into()
        }
    }
}
