//! `ml-worker` entry point: parses the command line, hands the process env
//! to the command as a map, and maps each outcome to its exit code.

use std::collections::BTreeMap;
use std::process::ExitCode;

use seeon_ml_worker::cli::{self, Command};
use seeon_ml_worker::config;
use seeon_ml_worker::exit::Exit;

/// The process env; an entry whose name or value is not UTF-8 is left out.
fn environment() -> BTreeMap<String, String> {
    std::env::vars_os()
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
        .collect()
}

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let flags = match cli::parse(&arguments) {
        Ok(Command::CheckConfig(flags)) => flags,
        Err(error) => {
            eprintln!("ml-worker: {error:?}");
            return error.exit().into();
        }
    };
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
