//! Tool processes under a deadline. Standard output is drained on a helper
//! thread so a chatty tool never blocks on a full pipe; at the deadline the
//! child is killed and reaped before the refusal is returned.

use std::io::{self, Read};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::RenditionError;

/// Interval between exit checks while a tool runs.
const POLL: Duration = Duration::from_millis(5);
/// Output beyond this is a refusal, never a truncation.
pub const MAX_STDOUT: u64 = 8 * 1024 * 1024;

/// Runs `command` to completion within `deadline` and returns its standard
/// output. A spawn failure, a non-zero exit or oversized output is `Probe`;
/// the deadline is `Deadline`, after the child has been killed and reaped.
pub fn run(mut command: Command, deadline: Duration) -> Result<Vec<u8>, RenditionError> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let started = Instant::now();
    let mut child = command.spawn().map_err(|_| RenditionError::Probe)?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(RenditionError::Probe);
    };
    let reader = thread::spawn(move || drain(stdout));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < deadline => thread::sleep(POLL),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(RenditionError::Deadline);
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(RenditionError::Probe);
            }
        }
    };
    let bytes = reader
        .join()
        .map_err(|_| RenditionError::Probe)?
        .map_err(|_| RenditionError::Probe)?;
    let oversized = u64::try_from(bytes.len()).map_or(true, |length| length > MAX_STDOUT);
    if !status.success() || oversized {
        return Err(RenditionError::Probe);
    }
    Ok(bytes)
}

/// Reads up to one byte past the bound, then discards the rest.
fn drain(stdout: impl Read) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut limited = stdout.take(MAX_STDOUT + 1);
    limited.read_to_end(&mut bytes)?;
    io::copy(&mut limited.into_inner(), &mut io::sink())?;
    Ok(bytes)
}
