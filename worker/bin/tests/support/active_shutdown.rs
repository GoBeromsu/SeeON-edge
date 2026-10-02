//! Active-camera shutdown of the actual `ml-worker` binary. Included only by
//! `run_gpu`. The fixed product clip store stays the parent-owned mount.
//! Prerequisite: `SEEON_TEST_CLIP_STORE_OWNER` equals the nonce in
//! `/var/lib/clip-store/.seeon-test-owner`. Missing ownership fails.
//! This harness never deletes the mount or any product-store output. Parent
//! owns mount cleanup. The before snapshot is assertion evidence only.
//!
//! An active recording is one regular `.mp4` under the owned record directory
//! whose same device and inode grows across observations. That is observed
//! same-file growth immediately before SIGTERM, not native reserved-slot proof.
//! The production 25s budget is sampled before the signal. A later reap
//! allowance only cleans a failed deadline; it never qualifies success.
//! A missed recording wait is still a test failure. Only that precondition
//! attempts the owned-child SIGTERM plus bounded reap so exit diagnostics can
//! drain; it never becomes recording, growth, or 25s acceptance.
use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use seeon_ml_worker::clips::store::PRODUCT_ROOT;
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::seam::{Clock, SystemClock};

use super::gpu_process::{GpuChild, signal_owned};

/// CUDA, model owners, RTSP connect, cache fill, and a real event can all
/// precede the first growing media file. This bounds that wait; it is not a sleep.
const ACTIVE_RECORD_WAIT: Duration = Duration::from_secs(150);
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(25);
const SHUTDOWN_REAP_ALLOWANCE: Duration = Duration::from_secs(2);
/// Diagnostic-only bound after a missed recording precondition. Not the 25s
/// acceptance budget and not evidence that T28 passed.
const DIAGNOSTIC_SHUTDOWN_WAIT: Duration = Duration::from_secs(5);

pub struct ActiveShutdown {
    pub status: ExitStatus,
    pub stderr: Vec<u8>,
}

/// Parent-owned fixed product clip store. The production path is not overridden.
/// Parent prepares `SEEON_TEST_CLIP_STORE_OWNER` as the exact nonce and writes
/// that same nonce to `/var/lib/clip-store/.seeon-test-owner` as a bounded
/// regular file. Missing or mismatched ownership fails. Drop retains every
/// product-store output for parent-owned mount cleanup.
pub struct OwnedProductStore {
    root: PathBuf,
    before: BTreeSet<PathBuf>,
}

impl OwnedProductStore {
    pub const MARKER: &str = ".seeon-test-owner";
    pub const NONCE_ENV: &str = "SEEON_TEST_CLIP_STORE_OWNER";

    pub fn claim() -> Self {
        let root = PathBuf::from(PRODUCT_ROOT);
        let meta = fs::symlink_metadata(&root).unwrap_or_else(|error| {
            panic!(
                "{PRODUCT_ROOT} must be the parent-owned mount; missing store is a failure: {error}"
            )
        });
        assert!(
            meta.is_dir() && !meta.file_type().is_symlink(),
            "{PRODUCT_ROOT} must be a real directory, not a symlink"
        );
        let nonce = std::env::var(Self::NONCE_ENV).unwrap_or_else(|_| {
            panic!(
                "{} required: parent writes this exact nonce to {}/{}",
                Self::NONCE_ENV,
                PRODUCT_ROOT,
                Self::MARKER
            )
        });
        assert!(
            !nonce.is_empty()
                && nonce.len() <= 128
                && nonce
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
            "{} must be one nonempty safe nonce",
            Self::NONCE_ENV
        );
        let marker = root.join(Self::MARKER);
        let marker_meta = fs::symlink_metadata(&marker).unwrap_or_else(|error| {
            panic!(
                "parent ownership marker {} required and must match {}: {error}",
                marker.display(),
                Self::NONCE_ENV
            )
        });
        assert!(
            marker_meta.is_file() && !marker_meta.file_type().is_symlink(),
            "{} must be a regular non-symlink file",
            marker.display()
        );
        assert!(
            marker_meta.len() <= 129,
            "{} exceeds the nonce bound",
            marker.display()
        );
        let recorded = fs::read_to_string(&marker).unwrap_or_else(|error| {
            panic!(
                "parent ownership marker {} required and must match {}: {error}",
                marker.display(),
                Self::NONCE_ENV
            )
        });
        assert_eq!(
            recorded.trim_end_matches(['\n', '\r']),
            nonce,
            "clip-store ownership nonce mismatch; refusing to use {}",
            PRODUCT_ROOT
        );
        Self {
            before: snapshot(&root),
            root,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn paths(&self) -> BTreeSet<PathBuf> {
        self.before.clone()
    }
}

impl Drop for OwnedProductStore {
    fn drop(&mut self) {
        eprintln!(
            "retaining parent-owned product clip store {} for parent mount cleanup",
            self.root.display()
        );
    }
}

/// Waits until one owned `.mp4` grows by device and inode, rechecks that same
/// file, samples the 25s deadline, then signals only this child. Success is
/// exit 0 inside that pre-signal deadline. The reap allowance is cleanup only.
pub fn shutdown_during_active_recording(
    child: &mut GpuChild,
    record_dir: &Path,
    shutdown_started: &AtomicBool,
) -> ActiveShutdown {
    let stdout = child.take_stdout();
    let stderr = child.take_stderr();
    let stdout_done = std::thread::spawn(move || super::gpu_process::capture_pipe(stdout));
    let stderr_done = std::thread::spawn(move || super::gpu_process::capture_pipe(stderr));
    let clock = SystemClock::new();
    let ready_deadline = clock.monotonic() + ACTIVE_RECORD_WAIT;
    let mut early = None;
    let mut tracked: Option<TrackedMedia> = None;
    let active = poll_until(&clock, ready_deadline, "active recording media", || {
        early = child.try_wait();
        if early.is_some() {
            return true;
        }
        match &tracked {
            None => tracked = first_media(record_dir, child.pid()),
            Some(media) => {
                if !open_for_write_by(child.pid(), media) {
                    tracked = None;
                } else if same_file_grew(media) {
                    return true;
                }
            }
        }
        false
    });
    if active.is_err() || early.is_some() || tracked.is_none() {
        if early.is_none() {
            // Preconditions already failed. SIGTERM is diagnostic cleanup of
            // this owned PID only; the panic below remains the test result.
            shutdown_started.store(true, Ordering::SeqCst);
            let diagnostic_deadline = clock.monotonic() + DIAGNOSTIC_SHUTDOWN_WAIT;
            signal_owned(child.pid());
            let _ = poll_until(
                &clock,
                diagnostic_deadline,
                "diagnostic precondition SIGTERM",
                || child.try_wait().is_some(),
            );
            if child.try_wait().is_none() {
                child.kill_owned();
            }
        }
        let captured = stderr_done.join().expect("stderr drain");
        stdout_done.join().expect("stdout drain");
        panic!(
            "same-file recording growth required before SIGTERM; status={early:?} wait={active:?}: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    let media = tracked.expect("tracked recording");
    let before_signal = same_file_size(&media);
    assert!(
        before_signal.is_some_and(|bytes| bytes > media.bytes)
            && open_for_write_by(child.pid(), &media),
        "SIGTERM requires the same growing recording still open for writing by the child: {}",
        media.path.display()
    );
    let signaled_at = clock.monotonic();
    let shutdown_deadline = signaled_at + SHUTDOWN_BUDGET;
    shutdown_started.store(true, Ordering::SeqCst);
    signal_owned(child.pid());
    let mut status = None;
    let mut observed_exit = None;
    let stopped = poll_until(&clock, shutdown_deadline, "active SIGTERM shutdown", || {
        status = child.try_wait();
        if status.is_some() {
            observed_exit = Some(clock.monotonic());
        }
        status.is_some()
    });
    if stopped.is_err() || observed_exit.is_none_or(|time| time > shutdown_deadline) {
        let cleanup_deadline = clock.monotonic() + SHUTDOWN_REAP_ALLOWANCE;
        let _ = poll_until(&clock, cleanup_deadline, "failed shutdown cleanup", || {
            child.try_wait().is_some()
        });
        if child.try_wait().is_none() {
            child.kill_owned();
        }
        let captured = stderr_done.join().expect("stderr drain");
        stdout_done.join().expect("stdout drain");
        panic!(
            "SIGTERM did not produce an observed exit inside the pre-signal 25s budget: {}",
            String::from_utf8_lossy(&captured)
        );
    }
    let captured = stderr_done.join().expect("stderr drain");
    let _stdout = stdout_done.join().expect("stdout drain");
    ActiveShutdown {
        status: status.expect("exited"),
        stderr: captured,
    }
}

fn snapshot(root: &Path) -> BTreeSet<PathBuf> {
    let mut paths = BTreeSet::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("clip-store snapshot {}: {error}", directory.display()));
        for entry in entries {
            let entry = entry.expect("clip-store entry");
            let path = entry.path();
            paths.insert(path.clone());
            if path.is_dir() && !path.is_symlink() {
                pending.push(path);
            }
        }
    }
    paths
}
struct TrackedMedia {
    path: PathBuf,
    device: u64,
    inode: u64,
    bytes: u64,
}

fn first_media(record_dir: &Path, pid: u32) -> Option<TrackedMedia> {
    let entries = fs::read_dir(record_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("mp4") {
            continue;
        }
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };
        if !meta.is_file() || meta.file_type().is_symlink() || meta.len() == 0 {
            continue;
        }
        let media = TrackedMedia {
            path,
            device: meta.dev(),
            inode: meta.ino(),
            bytes: meta.len(),
        };
        if open_for_write_by(pid, &media) {
            return Some(media);
        }
    }
    None
}

fn same_file_size(media: &TrackedMedia) -> Option<u64> {
    let meta = fs::symlink_metadata(&media.path).ok()?;
    (meta.is_file()
        && !meta.file_type().is_symlink()
        && meta.dev() == media.device
        && meta.ino() == media.inode)
        .then_some(meta.len())
}

fn same_file_grew(media: &TrackedMedia) -> bool {
    same_file_size(media).is_some_and(|bytes| bytes > media.bytes)
}

fn open_for_write_by(pid: u32, media: &TrackedMedia) -> bool {
    use std::io::Read;
    let Ok(entries) = fs::read_dir(format!("/proc/{pid}/fd")) else {
        return false;
    };
    entries.take(4096).flatten().any(|entry| {
        let Ok(meta) = fs::metadata(entry.path()) else {
            return false;
        };
        if meta.dev() != media.device || meta.ino() != media.inode {
            return false;
        }
        let info = PathBuf::from(format!("/proc/{pid}/fdinfo")).join(entry.file_name());
        let Ok(file) = fs::File::open(info) else {
            return false;
        };
        let mut text = String::new();
        if file.take(4096).read_to_string(&mut text).is_err() {
            return false;
        }
        text.lines()
            .find_map(|line| line.strip_prefix("flags:"))
            .and_then(|flags| u32::from_str_radix(flags.trim(), 8).ok())
            .is_some_and(|flags| matches!(flags & 0o3, 0o1 | 0o2))
    })
}
