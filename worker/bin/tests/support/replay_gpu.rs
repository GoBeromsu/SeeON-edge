//! Test-only inspection of an opt-in native replay trace. Included only by
//! `run_gpu`. This is not a production writer, decoder, or shutdown framework.
//!
//! The live file may hold one concurrently written trailing fragment. Only
//! newline-terminated lines are inspected. The file itself is never deleted.
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::seam::{Clock, SystemClock};
use seeon_ml_worker::trace_out::{DEFAULT_MAX_BYTES, HEADER_LINE};
use serde_json::{Map, Value};

use super::gpu_process::{GpuChild, capture_pipe, signal_owned};

/// CUDA, model owners, RTSP connect, and the first accepted native frames.
/// Same bound as the idle readiness wait; not a sleep and not a new budget.
const TRACE_READY_WAIT: Duration = Duration::from_secs(45);
/// Production shutdown budget. Sampled before the flag and SIGTERM.
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(25);
const READY_CAMERAS: &str = "ml-worker: policy loop ready cameras=1";
// Canonical convert_frame receives configured mux dimensions, not decoder size.
// The independent Python oracle matches SDK rectangles at 1280x720; using the
// approved video's 640x360 decoder size instead yields zero matched tracks.
const PERCEPTION_WIDTH: u64 = 1280;
const PERCEPTION_HEIGHT: u64 = 720;

pub struct ReplayShutdown {
    pub status: ExitStatus,
    pub stderr: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraceEvidence {
    pub path: PathBuf,
    pub complete_rows: usize,
    pub frame_rows: usize,
    pub nonempty_pose_rows: usize,
    pub distinct_pts: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceInspectError {
    Missing,
    NotRegular,
    Oversized,
    Header,
    Malformed,
    Camera,
    Source,
    Dimensions,
    Track,
    Geometry,
    Insufficient,
}

pub fn trace_path(root: &Path, camera_id: &str) -> PathBuf {
    let digest = sha256_hex(camera_id.as_bytes());
    root.join(format!("{}.jsonl", &digest[..16]))
}

/// Bounded descriptor read of one regular non-symlink file, capped at the
/// existing writer maximum. A trailing fragment without a newline is ignored.
pub fn inspect_trace(path: &Path, camera_id: &str) -> Result<TraceEvidence, TraceInspectError> {
    inspect_trace_bounded(path, camera_id, DEFAULT_MAX_BYTES)
}

pub fn inspect_trace_bounded(
    path: &Path,
    camera_id: &str,
    max_bytes: u64,
) -> Result<TraceEvidence, TraceInspectError> {
    let meta = std::fs::symlink_metadata(path).map_err(|_| TraceInspectError::Missing)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(TraceInspectError::NotRegular);
    }
    if meta.len() > max_bytes {
        return Err(TraceInspectError::Oversized);
    }
    let fd = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::CLOEXEC
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::empty(),
    )
    .map_err(|_| TraceInspectError::NotRegular)?;
    let file = File::from(fd);
    let captured = file.metadata().map_err(|_| TraceInspectError::NotRegular)?;
    if !captured.is_file() || captured.dev() != meta.dev() || captured.ino() != meta.ino() {
        return Err(TraceInspectError::NotRegular);
    }
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| TraceInspectError::NotRegular)?;
    if bytes.len() as u64 > max_bytes {
        return Err(TraceInspectError::Oversized);
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| TraceInspectError::Malformed)?;
    if !text.contains('\n') {
        return Err(TraceInspectError::Header);
    }
    let (complete, _) = text.rsplit_once('\n').ok_or(TraceInspectError::Header)?;
    let mut lines = complete.split('\n');
    let header = lines.next().unwrap_or_default();
    if header != HEADER_LINE.trim_end() {
        return Err(TraceInspectError::Header);
    }
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|_| TraceInspectError::Malformed)?;
        classify_row(&value, camera_id)?;
        rows.push(value);
    }
    evidence(path, rows)
}

fn classify_row(row: &Value, camera_id: &str) -> Result<(), TraceInspectError> {
    let object = row.as_object().ok_or(TraceInspectError::Malformed)?;
    if object.get("camera_id").and_then(Value::as_str) != Some(camera_id) {
        return Err(TraceInspectError::Camera);
    }
    if object.get("source").and_then(Value::as_str) != Some("nvdcf") {
        return Err(TraceInspectError::Source);
    }
    let width = object
        .get("frame_width")
        .and_then(Value::as_u64)
        .ok_or(TraceInspectError::Dimensions)?;
    let height = object
        .get("frame_height")
        .and_then(Value::as_u64)
        .ok_or(TraceInspectError::Dimensions)?;
    if width != PERCEPTION_WIDTH || height != PERCEPTION_HEIGHT {
        return Err(TraceInspectError::Dimensions);
    }
    let tracks = object
        .get("tracks")
        .and_then(Value::as_array)
        .ok_or(TraceInspectError::Malformed)?;
    if object.get("source_event").and_then(Value::as_str) == Some("frame") {
        for track in tracks {
            classify_track(track)?;
        }
    }
    Ok(())
}

fn classify_track(track: &Value) -> Result<(), TraceInspectError> {
    let object = track.as_object().ok_or(TraceInspectError::Track)?;
    if object.get("track_id").and_then(Value::as_u64).is_none() {
        return Err(TraceInspectError::Track);
    }
    let bbox = object
        .get("bbox")
        .and_then(Value::as_array)
        .ok_or(TraceInspectError::Geometry)?;
    if bbox.len() != 5 || !bbox.iter().all(finite_unit) {
        return Err(TraceInspectError::Geometry);
    }
    let points = object
        .get("keypoints")
        .and_then(Value::as_array)
        .ok_or(TraceInspectError::Geometry)?;
    if points.len() != 17 {
        return Err(TraceInspectError::Geometry);
    }
    for point in points {
        let point = point.as_array().ok_or(TraceInspectError::Geometry)?;
        if point.len() != 3 || !point.iter().all(finite_unit) {
            return Err(TraceInspectError::Geometry);
        }
    }
    Ok(())
}

fn finite_unit(value: &Value) -> bool {
    value
        .as_f64()
        .is_some_and(|number| number.is_finite() && (0.0..=1.0).contains(&number))
}

fn evidence(path: &Path, rows: Vec<Value>) -> Result<TraceEvidence, TraceInspectError> {
    let frames: Vec<&Map<String, Value>> = rows
        .iter()
        .filter_map(Value::as_object)
        .filter(|row| row.get("source_event").and_then(Value::as_str) == Some("frame"))
        .collect();
    let nonempty = frames
        .iter()
        .filter(|row| {
            row.get("tracks")
                .and_then(Value::as_array)
                .is_some_and(|tracks| !tracks.is_empty())
        })
        .count();
    let mut distinct_pts = Vec::new();
    for row in &frames {
        let pts = row
            .get("pts_ns")
            .and_then(Value::as_u64)
            .ok_or(TraceInspectError::Malformed)?;
        if !distinct_pts.contains(&pts) {
            distinct_pts.push(pts);
        }
    }
    if frames.len() < 3 || nonempty < 1 || distinct_pts.len() < 2 {
        return Err(TraceInspectError::Insufficient);
    }
    if rows.first().and_then(|row| row["source_event"].as_str()) != Some("open")
        || rows.first().and_then(|row| row["seq"].as_u64()) != Some(0)
    {
        return Err(TraceInspectError::Malformed);
    }
    let mut previous = None;
    for row in &rows {
        let seq = row["seq"].as_u64().ok_or(TraceInspectError::Malformed)?;
        if previous.is_some_and(|last| seq <= last) {
            return Err(TraceInspectError::Malformed);
        }
        previous = Some(seq);
    }
    Ok(TraceEvidence {
        path: path.to_path_buf(),
        complete_rows: rows.len(),
        frame_rows: frames.len(),
        nonempty_pose_rows: nonempty,
        distinct_pts,
    })
}

/// Waits for the exact one-camera marker and a real accepted trace, samples
/// the 25s deadline, then signals only this child. Failure kills and reaps
/// that child before either drain is joined. Success is exit 0 inside the
/// pre-signal deadline; it does not qualify active recording.
pub fn shutdown_after_accepted_trace(
    child: &mut GpuChild,
    trace: &Path,
    camera_id: &str,
    shutdown_started: &AtomicBool,
) -> ReplayShutdown {
    let stdout = child.take_stdout();
    let stderr = child.take_stderr();
    let (marker_tx, marker_rx) = mpsc::channel();
    let stdout_done = std::thread::spawn(move || capture_pipe(stdout));
    let stderr_done = std::thread::spawn(move || drain_stderr(stderr, marker_tx));
    let clock = SystemClock::new();
    let ready_deadline = clock.monotonic() + TRACE_READY_WAIT;
    let mut early = None;
    let mut saw_marker = false;
    let mut last_refusal = None;
    let ready = poll_until(
        &clock,
        ready_deadline,
        "accepted native replay trace",
        || {
            early = child.try_wait();
            if early.is_some() {
                return true;
            }
            if marker_rx.try_recv().is_ok() {
                saw_marker = true;
            }
            if !saw_marker {
                return false;
            }
            match inspect_trace(trace, camera_id) {
                Ok(_) => true,
                Err(error) => {
                    last_refusal = Some(error);
                    false
                }
            }
        },
    );
    if ready.is_err() || early.is_some() || !saw_marker {
        eprintln!(
            "REPLAY_PRECONDITION_FAILURE path={} refusal={last_refusal:?} marker={saw_marker} early={early:?}",
            trace.display()
        );
        fail_owned(
            child,
            early.is_none(),
            stdout_done,
            stderr_done,
            "accepted native replay trace and cameras=1 marker required before SIGTERM",
        );
    }
    let shutdown_deadline = clock.monotonic() + SHUTDOWN_BUDGET;
    shutdown_started.store(true, Ordering::SeqCst);
    signal_owned(child.pid());
    let mut status = None;
    let stopped = poll_until(&clock, shutdown_deadline, "replay SIGTERM shutdown", || {
        status = child.try_wait();
        status.is_some()
    });
    if stopped.is_err() || clock.monotonic() > shutdown_deadline {
        fail_owned(
            child,
            status.is_none(),
            stdout_done,
            stderr_done,
            "SIGTERM exceeded the pre-signal 25s shutdown budget",
        );
    }
    let stderr = join_drains(stdout_done, stderr_done);
    let text = String::from_utf8_lossy(&stderr);
    assert!(
        text.lines().any(|line| line == READY_CAMERAS),
        "drained stderr lost the exact cameras=1 marker: {text}"
    );
    ReplayShutdown {
        status: status.expect("exited"),
        stderr,
    }
}

fn drain_stderr(pipe: impl Read + Send, marker_tx: mpsc::Sender<()>) -> Vec<u8> {
    let mut captured = Vec::new();
    let mut reader = BufReader::new(pipe);
    let mut line = Vec::new();
    let mut signaled = false;
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line).expect("stderr");
        if read == 0 {
            break;
        }
        if !signaled
            && std::str::from_utf8(&line).is_ok_and(|text| text.trim_end() == READY_CAMERAS)
        {
            signaled = true;
            let _ = marker_tx.send(());
        }
        captured.extend_from_slice(&line);
    }
    captured
}

fn fail_owned(
    child: &mut GpuChild,
    live: bool,
    stdout_done: JoinHandle<Vec<u8>>,
    stderr_done: JoinHandle<Vec<u8>>,
    why: &str,
) -> ! {
    if live {
        child.kill_owned();
    }
    let stderr = join_drains(stdout_done, stderr_done);
    panic!("{why}: {}", String::from_utf8_lossy(&stderr));
}

fn join_drains(stdout_done: JoinHandle<Vec<u8>>, stderr_done: JoinHandle<Vec<u8>>) -> Vec<u8> {
    let stderr = stderr_done.join().expect("stderr drain");
    let _stdout = stdout_done.join().expect("stdout drain");
    stderr
}

pub fn assert_python_decodes(python: &std::ffi::OsStr, path: &Path) {
    let mut child = std::process::Command::new(python)
        .arg("-c")
        .arg(PYTHON_DECODE)
        .arg(path)
        .env(
            "PYTHONPATH",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("SEEON_TEST_PYTHON runs");
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout = std::thread::spawn(move || capture_pipe(stdout));
    let stderr = std::thread::spawn(move || capture_pipe(stderr));
    let clock = SystemClock::new();
    let mut status = None;
    let waited = poll_until(
        &clock,
        clock.monotonic() + Duration::from_secs(20),
        "Python replay decoder",
        || {
            status = child.try_wait().expect("Python child status");
            status.is_some()
        },
    );
    if status.is_none() {
        child.kill().expect("kill owned Python decoder");
        child.wait().expect("reap owned Python decoder");
    }
    let stdout = stdout.join().expect("Python stdout");
    let stderr = stderr.join().expect("Python stderr");
    assert!(
        waited.is_ok() && status.is_some_and(|status| status.success()),
        "contracts.replay_trace.decode_jsonl failed: {}",
        String::from_utf8_lossy(&stderr)
    );
    println!("REPLAY_PYTHON_DECODE={}", String::from_utf8_lossy(&stdout));
}

const PYTHON_DECODE: &str = "
import json, sys
from pathlib import Path
from contracts.replay_trace import decode_jsonl
text = Path(sys.argv[1]).read_text(encoding='utf-8')
assert text.endswith('\\n'), 'saved trace has an incomplete final row'
header, rows = decode_jsonl(text)
print(json.dumps({'version': header.version, 'rows': len(rows)}))
";
