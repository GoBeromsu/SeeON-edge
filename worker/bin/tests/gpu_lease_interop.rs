//! T21: the Rust and Python GPU leases exclude each other on one state dir
//! (plan row 20). The oracle is the Python `worker.runtime.lease.GpuLease`,
//! run as a real child process through `$SEEON_TEST_PYTHON` with the test's
//! environment inherited (interpreter venv and `PYTHONPATH`).

use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::Duration;

use seeon_ml_worker::gpu::lease::{self, LeaseError};
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::seam::{Clock, SystemClock};

/// Holds the Python lease, reports `held` on stdout, releases on stdin EOF.
const HOLD: &str = "\
import sys
from pathlib import Path
from worker.runtime.lease import GpuLease
lease = GpuLease.acquire(Path(sys.argv[1]))
print('held', flush=True)
sys.stdin.read()
lease.close()
";

/// Tries the Python lease once and prints the refusal class name or `acquired`.
const PROBE: &str = "\
import sys
from pathlib import Path
from worker.runtime.lease import GpuLease
try:
    lease = GpuLease.acquire(Path(sys.argv[1]))
except Exception as exc:
    print(type(exc).__name__)
else:
    lease.close()
    print('acquired')
";

/// Interpreter start plus one import; an upper bound, not a pause.
const HOLDER_WAIT: Duration = Duration::from_secs(30);

fn python() -> PathBuf {
    let value = std::env::var_os("SEEON_TEST_PYTHON")
        .unwrap_or_else(|| panic!("SEEON_TEST_PYTHON is required"));
    assert!(
        !value.is_empty(),
        "SEEON_TEST_PYTHON must be a nonblank path"
    );
    PathBuf::from(value)
}

fn fresh_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gpu_lease_interop");
    match fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => panic!("clear {}: {error}", dir.display()),
    }
    dir
}

/// Starts the Python holder and waits until it reports that it holds.
fn python_holds(python: &Path, dir: &Path) -> Child {
    let mut holder = Command::new(python)
        .arg("-c")
        .arg(HOLD)
        .arg(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the Python holder");
    let stdout = holder.stdout.take().expect("holder stdout");
    let (line_tx, line_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut line = String::new();
        let _ = line_tx.send(BufReader::new(stdout).read_line(&mut line).map(|_| line));
    });
    let mut first_line = None;
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + HOLDER_WAIT;
    poll_until(&clock, deadline, "python lease holder", || {
        match line_rx.try_recv() {
            Ok(line) => first_line = Some(line.ok()),
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => first_line = Some(None),
        }
        true
    })
    .expect("the Python holder answers before its deadline");
    assert_eq!(
        first_line.flatten().as_deref(),
        Some("held\n"),
        "the Python holder took the lease"
    );
    holder
}

/// Runs the Python probe to completion and returns its one stdout token.
fn python_probe(python: &Path, dir: &Path) -> String {
    let output = Command::new(python)
        .arg("-c")
        .arg(PROBE)
        .arg(dir)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .expect("run the Python probe");
    assert!(output.status.success(), "the Python probe exits cleanly");
    String::from_utf8(output.stdout)
        .expect("UTF-8 probe output")
        .trim()
        .to_owned()
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn rust_and_python_leases_exclude_each_other_on_one_state_dir() {
    let python = python();
    let dir = fresh_dir();
    let lease_path = dir.join(".gpu.lease");

    let mut holder = python_holds(&python, &dir);
    assert_eq!(
        lease::acquire(&dir).unwrap_err(),
        LeaseError::Unavailable {
            lease_path: lease_path.clone()
        }
    );
    drop(holder.stdin.take());
    assert!(holder.wait().expect("wait for the holder").success());

    let rust = lease::acquire(&dir).expect("Rust acquires after Python released");
    assert_eq!(python_probe(&python, &dir), "GpuLeaseUnavailableError");
    drop(rust);

    assert_eq!(python_probe(&python, &dir), "acquired");
    let again = lease::acquire(&dir).expect("Rust acquires after both released");
    assert_eq!(again.lease_path(), lease_path);
}
