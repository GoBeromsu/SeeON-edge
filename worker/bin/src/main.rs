//! `ml-worker` entry point. Commands arrive with the stages that own them;
//! until then every invocation is a CLI error. Exit codes map through `exit`.

use std::process::ExitCode;

use seeon_ml_worker::exit::Exit;

fn main() -> ExitCode {
    match std::env::args_os().nth(1) {
        Some(command) => eprintln!("ml-worker: unknown command {}", command.to_string_lossy()),
        None => eprintln!("ml-worker: no command given"),
    }
    Exit::Config.into()
}
