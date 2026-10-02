//! Owned GPU child of the actual `ml-worker` binary. Included only by `run_gpu`.
//! Drop always kills and reaps that PID, including after a timed-out SIGTERM.
//! A diagnostic SIGTERM from the active-recording precondition uses the same
//! owned PID. It is not acceptance: Drop still kills and reaps if that child lives.
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::time::Duration;

use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::seam::{Clock, SystemClock};

/// stderr line emitted only after the first successful policy turn and delivery.
pub const POLICY_LOOP_READY: &str = "ml-worker: policy loop ready cameras=0";
/// Production shutdown budget. Harness wait adds a small reap allowance, not a new budget.
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(25);
const SHUTDOWN_REAP_ALLOWANCE: Duration = Duration::from_secs(2);
/// CUDA gate plus three model-owner warm gates share one 30s readiness budget.
const IDLE_READY_WAIT: Duration = Duration::from_secs(45);

pub struct GpuChild {
    child: Child,
    pid: u32,
}

impl GpuChild {
    pub(super) fn pid(&self) -> u32 {
        self.pid
    }

    pub(super) fn take_stdout(&mut self) -> ChildStdout {
        self.child.stdout.take().expect("stdout pipe")
    }

    pub(super) fn take_stderr(&mut self) -> ChildStderr {
        self.child.stderr.take().expect("stderr pipe")
    }

    pub(super) fn try_wait(&mut self) -> Option<ExitStatus> {
        self.child.try_wait().expect("process status")
    }

    pub(super) fn kill_owned(&mut self) {
        self.child.kill().expect("kill only owned GPU child");
        self.child.wait().expect("reap owned GPU child");
    }
}

impl Drop for GpuChild {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Spawns one GPU child. Drop reaps it, including after a timed-out SIGTERM.
pub fn spawn_gpu(mut command: Command) -> GpuChild {
    let child = command.spawn().expect("actual ml-worker binary");
    let pid = child.id();
    GpuChild { child, pid }
}
pub struct GpuRefusal {
    pub status: ExitStatus,
    pub stderr: Vec<u8>,
}

/// Waits for this owned child to refuse before the policy-loop marker. A marker or
/// a missed deadline is a failure; Drop still kills and reaps a live child.
pub fn wait_for_refusal(child: &mut GpuChild) -> GpuRefusal {
    let stdout = child.child.stdout.take().expect("stdout pipe");
    let stderr = child.child.stderr.take().expect("stderr pipe");
    let (marker_tx, marker_rx) = mpsc::channel();
    let saw_marker = Arc::new(AtomicBool::new(false));
    let stdout_done = std::thread::spawn(move || drain_pipe(stdout, None, None));
    let stderr_flag = Arc::clone(&saw_marker);
    let stderr_done =
        std::thread::spawn(move || drain_pipe(stderr, Some(marker_tx), Some(stderr_flag)));
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + IDLE_READY_WAIT;
    let mut status = None;
    let stopped = poll_until(&clock, deadline, "hardware identity refusal", || {
        if marker_rx.try_recv().is_ok() {
            return true;
        }
        status = child.child.try_wait().expect("process status");
        status.is_some()
    });
    if stopped.is_err() || saw_marker.load(Ordering::SeqCst) || status.is_none() {
        if status.is_none() {
            let _ = child.child.kill();
            let _ = child.child.wait();
        }
        let captured = stderr_done.join().expect("stderr drain");
        stdout_done.join().expect("stdout drain");
        panic!(
            "hardware mismatch refused late; status={status:?} wait={stopped:?}: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    let captured = stderr_done.join().expect("stderr drain");
    stdout_done.join().expect("stdout drain");
    GpuRefusal {
        status: status.expect("refused"),
        stderr: captured,
    }
}

pub struct IdleShutdown {
    pub status: ExitStatus,
    pub stderr: Vec<u8>,
    pub saw_policy_loop: bool,
}

/// Waits for the real idle marker, sends SIGTERM to this child only, then requires exit 0
/// inside the shared 25s shutdown budget plus a small reap allowance.
pub fn shutdown_after_policy_loop(
    child: &mut GpuChild,
    shutdown_started: &AtomicBool,
) -> IdleShutdown {
    let stdout = child.child.stdout.take().expect("stdout pipe");
    let stderr = child.child.stderr.take().expect("stderr pipe");
    let (marker_tx, marker_rx) = mpsc::channel();
    let saw_marker = Arc::new(AtomicBool::new(false));
    let stdout_done = std::thread::spawn(move || drain_pipe(stdout, None, None));
    let stderr_flag = Arc::clone(&saw_marker);
    let stderr_done =
        std::thread::spawn(move || drain_pipe(stderr, Some(marker_tx), Some(stderr_flag)));
    let clock = SystemClock::new();
    let ready_deadline = clock.monotonic() + IDLE_READY_WAIT;
    let mut early = None;
    let reached = poll_until(&clock, ready_deadline, "policy loop ready", || {
        if marker_rx.try_recv().is_ok() {
            return true;
        }
        early = child.child.try_wait().expect("process status");
        early.is_some()
    });
    if reached.is_err() || early.is_some() || !saw_marker.load(Ordering::SeqCst) {
        if early.is_none() {
            child
                .child
                .kill()
                .expect("kill only owned GPU child after readiness failure");
            child.child.wait().expect("reap owned GPU child");
        }
        let captured = stderr_done.join().expect("stderr drain");
        stdout_done.join().expect("stdout drain");
        panic!(
            "policy loop marker required before SIGTERM; status={early:?} wait={reached:?}: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    shutdown_started.store(true, Ordering::SeqCst);
    signal_child(child.pid);
    let shutdown_deadline = clock.monotonic() + SHUTDOWN_BUDGET + SHUTDOWN_REAP_ALLOWANCE;
    let mut status = None;
    let stopped = poll_until(&clock, shutdown_deadline, "idle SIGTERM shutdown", || {
        status = child.child.try_wait().expect("process status");
        status.is_some()
    });
    if stopped.is_err() {
        child
            .child
            .kill()
            .expect("kill only owned GPU child after shutdown deadline");
        child.child.wait().expect("reap owned GPU child");
        let captured = stderr_done.join().expect("stderr drain");
        stdout_done.join().expect("stdout drain");
        panic!(
            "SIGTERM exceeded shared25s shutdown budget plus reap allowance: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    let captured = stderr_done.join().expect("stderr drain");
    let _stdout = stdout_done.join().expect("stdout drain");
    assert!(
        captured
            .windows(POLICY_LOOP_READY.len())
            .any(|window| window == POLICY_LOOP_READY.as_bytes()),
        "drained stderr lost the policy-loop marker: {}",
        String::from_utf8_lossy(&captured)
    );
    IdleShutdown {
        status: status.expect("exited"),
        stderr: captured,
        saw_policy_loop: true,
    }
}

fn drain_pipe(
    pipe: impl Read + Send,
    marker_tx: Option<Sender<()>>,
    saw: Option<Arc<AtomicBool>>,
) -> Vec<u8> {
    let mut reader = BufReader::new(pipe);
    let mut captured = Vec::new();
    let mut line = Vec::new();
    let mut signaled = false;
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line).expect("child pipe");
        if read == 0 {
            break;
        }
        if !signaled
            && let Some(marker_tx) = &marker_tx
            && line
                .windows(POLICY_LOOP_READY.len())
                .any(|window| window == POLICY_LOOP_READY.as_bytes())
        {
            signaled = true;
            if let Some(flag) = &saw {
                flag.store(true, Ordering::SeqCst);
            }
            let _ = marker_tx.send(());
        }
        captured.extend_from_slice(&line);
    }
    captured
}
pub(super) fn capture_pipe(pipe: impl Read + Send) -> Vec<u8> {
    let mut captured = Vec::new();
    let mut reader = BufReader::new(pipe);
    reader.read_to_end(&mut captured).expect("child pipe");
    captured
}

pub(super) fn signal_owned(pid: u32) {
    signal_child(pid);
}

fn signal_child(pid: u32) {
    let status = Command::new("/bin/kill")
        .args(["-TERM", &pid.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("signal only the spawned GPU child");
    assert!(status.success(), "SIGTERM to owned GPU child failed");
}
