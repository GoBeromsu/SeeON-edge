//! Owned gst-launch process for one live-pose build. The child is its own
//! process group. Failure kills that still-owned group, then reaps its leader.
//! Scratch and the bounded log stay. Grandchildren are not reaped, and a
//! blocked kernel wait is not claimed to be preempted.

use std::io::{self, Read};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use rustix::fs::{self, OFlags};
use rustix::process::Pid;

use super::observation::LibraryIdentity;
use super::{LiveBuildError, Prepared};

mod child;
mod files;

pub(super) const DEFAULT_WAIT: Duration = Duration::from_secs(600);
const OUTPUT_CAP: usize = 32 * 1024 * 1024;
const READ_SLICE: Duration = Duration::from_millis(50);

pub(super) struct Limits {
    pub(super) wait: Duration,
    pub(super) output_cap: usize,
}

impl Limits {
    pub(super) fn production() -> Self {
        Self {
            wait: DEFAULT_WAIT,
            output_cap: OUTPUT_CAP,
        }
    }
}

pub(super) struct ProcessOutput {
    pub(super) status: i32,
    pub(super) merged: String,
}

pub(super) struct Scratch {
    pub(super) directory: PathBuf,
}

impl Scratch {
    pub(super) fn create(parent: &Path) -> Result<Self, LiveBuildError> {
        files::create_scratch(parent)
    }

    pub(super) fn record(&self, bytes: &[u8]) -> Result<(), LiveBuildError> {
        files::record(self, bytes)
    }
}

pub(super) struct Staged {
    pub(super) argv: Vec<String>,
    pub(super) child_engine: PathBuf,
    pub(super) generated: PathBuf,
}

pub(super) fn stage_files(
    scratch: &Scratch,
    prepared: &Prepared<'_>,
) -> Result<Staged, LiveBuildError> {
    files::stage_files(scratch, prepared)
}

pub(super) fn gst_argv(config: &str, batch: u32) -> Vec<String> {
    let mut argv = vec![
        "gst-launch-1.0".into(),
        "-e".into(),
        "nvstreammux".into(),
        "name=mux".into(),
        format!("batch-size={batch}"),
        "width=640".into(),
        "height=640".into(),
        "live-source=0".into(),
        "batched-push-timeout=40000".into(),
        "!".into(),
        "nvinfer".into(),
        format!("config-file-path={config}"),
        "!".into(),
        "fakesink".into(),
        "sync=false".into(),
    ];
    for index in 0..batch {
        argv.extend([
            "videotestsrc".into(),
            "num-buffers=1".into(),
            "pattern=black".into(),
            "!".into(),
            "video/x-raw,format=I420,width=640,height=640,framerate=30/1".into(),
            "!".into(),
            "nvvideoconvert".into(),
            "!".into(),
            "video/x-raw(memory:NVMM),format=NV12".into(),
            "!".into(),
            format!("mux.sink_{index}"),
        ]);
    }
    argv
}

pub(super) fn render_infer_config(
    template: &str,
    onnx: &str,
    engine: &str,
    batch: u32,
) -> Result<String, LiveBuildError> {
    files::render_config(template, onnx, engine, batch)
}

pub(super) fn regular_engine(path: &Path) -> bool {
    files::regular_engine(path)
}

pub(super) fn copy_exclusive(source: &Path, target: &Path) -> Result<(), LiveBuildError> {
    files::copy_exclusive(source, target)
}

pub(super) fn run_bounded(
    argv: &[String],
    scratch: &Scratch,
    limits: Limits,
    library: &LibraryIdentity,
) -> Result<ProcessOutput, LiveBuildError> {
    if argv.first().is_none_or(|name| name != "gst-launch-1.0") {
        return Err(LiveBuildError::Process);
    }
    let mut command = Command::new("gst-launch-1.0");
    if argv.len() > 1 {
        command.args(&argv[1..]);
    }
    capture(command, scratch, limits, |pid| library.mapped(pid))
}

fn capture<F>(
    command: Command,
    scratch: &Scratch,
    limits: Limits,
    observe: F,
) -> Result<ProcessOutput, LiveBuildError>
where
    F: Fn(Pid) -> Result<bool, LiveBuildError>,
{
    let deadline = Instant::now()
        .checked_add(limits.wait)
        .ok_or(LiveBuildError::Process)?;
    let (mut reader, writer) = UnixStream::pair().map_err(|_| LiveBuildError::Process)?;
    set_nonblocking(&reader)?;
    let mut owned = child::OwnedChild::spawn(command, &scratch.directory, writer.into())?;
    let mut collected = Vec::new();
    let mut saw = false;
    let read = read_until(
        &mut owned,
        &mut reader,
        &limits,
        &observe,
        deadline,
        &mut collected,
        &mut saw,
    );
    let identity = if read.is_ok() {
        confirm_library(&owned, &observe, &mut saw)
    } else {
        Ok(())
    };
    let status = if read.is_ok() && identity.is_ok() {
        owned.finish(deadline)
    } else {
        let _ = owned.abort();
        Err(LiveBuildError::Process)
    };
    let drained = drain_ready(&mut reader, &mut collected, limits.output_cap, deadline);
    let recorded = scratch.record(&collected);
    read?;
    identity?;
    let status = status?;
    drained?;
    recorded?;
    if !saw {
        return Err(LiveBuildError::Observer);
    }
    Ok(ProcessOutput {
        status,
        merged: String::from_utf8(collected).map_err(|_| LiveBuildError::Process)?,
    })
}

fn read_until<F>(
    owned: &mut child::OwnedChild,
    reader: &mut UnixStream,
    limits: &Limits,
    observe: &F,
    deadline: Instant,
    collected: &mut Vec<u8>,
    saw: &mut bool,
) -> Result<(), LiveBuildError>
where
    F: Fn(Pid) -> Result<bool, LiveBuildError>,
{
    let mut buffer = [0_u8; 8192];
    loop {
        if Instant::now() >= deadline {
            return Err(LiveBuildError::Process);
        }
        match reader.read(&mut buffer) {
            Ok(0) => {
                return wait_after_eof(owned, reader, limits, observe, deadline, collected, saw);
            }
            Ok(count) => note_chunk(
                owned,
                observe,
                collected,
                saw,
                &buffer[..count],
                limits.output_cap,
            )?,
            Err(error) if blocked(&error) => {
                if owned.exited(deadline)? {
                    return drain_ready(reader, collected, limits.output_cap, deadline);
                }
                thread::sleep(READ_SLICE.min(deadline.saturating_duration_since(Instant::now())));
            }
            Err(_) => return Err(LiveBuildError::Process),
        }
    }
}

fn wait_after_eof<F>(
    owned: &mut child::OwnedChild,
    reader: &mut UnixStream,
    limits: &Limits,
    observe: &F,
    deadline: Instant,
    collected: &mut Vec<u8>,
    saw: &mut bool,
) -> Result<(), LiveBuildError>
where
    F: Fn(Pid) -> Result<bool, LiveBuildError>,
{
    loop {
        if Instant::now() >= deadline {
            return Err(LiveBuildError::Process);
        }
        drain_ready(reader, collected, limits.output_cap, deadline)?;
        if owned.exited(deadline)? {
            return confirm_library(owned, observe, saw);
        }
        thread::sleep(READ_SLICE.min(deadline.saturating_duration_since(Instant::now())));
    }
}

fn drain_ready(
    reader: &mut UnixStream,
    collected: &mut Vec<u8>,
    cap: usize,
    deadline: Instant,
) -> Result<(), LiveBuildError> {
    let mut buffer = [0_u8; 8192];
    loop {
        if Instant::now() >= deadline {
            return Err(LiveBuildError::Process);
        }
        match reader.read(&mut buffer) {
            Ok(0) => return Ok(()),
            Ok(count) => append_bounded(collected, &buffer[..count], cap)?,
            Err(error) if blocked(&error) => return Ok(()),
            Err(_) => return Err(LiveBuildError::Process),
        }
    }
}

fn note_chunk<F>(
    owned: &child::OwnedChild,
    observe: &F,
    collected: &mut Vec<u8>,
    saw: &mut bool,
    chunk: &[u8],
    cap: usize,
) -> Result<(), LiveBuildError>
where
    F: Fn(Pid) -> Result<bool, LiveBuildError>,
{
    append_bounded(collected, chunk, cap)?;
    if !*saw {
        *saw = observe(owned.pid()?)?;
    }
    Ok(())
}

fn confirm_library<F>(
    owned: &child::OwnedChild,
    observe: &F,
    saw: &mut bool,
) -> Result<(), LiveBuildError>
where
    F: Fn(Pid) -> Result<bool, LiveBuildError>,
{
    if *saw {
        return Ok(());
    }
    match observe(owned.pid()?) {
        Ok(mapped) => {
            *saw = mapped;
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn blocked(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock || error.kind() == io::ErrorKind::Interrupted
}

fn set_nonblocking(reader: &UnixStream) -> Result<(), LiveBuildError> {
    let flags = fs::fcntl_getfl(reader).map_err(|_| LiveBuildError::Process)?;
    fs::fcntl_setfl(reader, flags | OFlags::NONBLOCK).map_err(|_| LiveBuildError::Process)
}

fn append_bounded(collected: &mut Vec<u8>, chunk: &[u8], cap: usize) -> Result<(), LiveBuildError> {
    let room = cap.saturating_sub(collected.len());
    if chunk.len() > room {
        collected.extend_from_slice(&chunk[..room]);
        return Err(LiveBuildError::Process);
    }
    collected.extend_from_slice(chunk);
    Ok(())
}

#[cfg(test)]
mod cpu_tests {
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::UnixStream;
    use std::path::Path;
    use std::process::Command;
    use std::time::{Duration, Instant};

    use rustix::process::{
        Pid, Signal, WaitId, WaitIdOptions, kill_process_group, test_kill_process, waitid,
    };

    use super::super::LiveBuildError;
    use super::{Limits, Scratch, capture};
    use crate::seam::IdSource;

    struct Fixture(std::path::PathBuf);

    impl Fixture {
        fn new() -> Self {
            let name = crate::seam::RandomIds.uuid4().expect("fixture id");
            let root = std::env::temp_dir().join(format!("live-process-{name}"));
            std::fs::create_dir(&root).expect("exclusive fixture");
            let mut mode = std::fs::metadata(&root).expect("mode").permissions();
            mode.set_mode(0o700);
            std::fs::set_permissions(&root, mode).expect("private fixture");
            Self(root)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    fn run(
        script: &str,
        scratch: &Scratch,
        wait: Duration,
        cap: usize,
    ) -> Result<super::ProcessOutput, LiveBuildError> {
        capture(
            shell(script),
            scratch,
            Limits {
                wait,
                output_cap: cap,
            },
            |_| Ok(true),
        )
    }

    fn gone(pid: Pid) -> bool {
        matches!(test_kill_process(pid), Err(rustix::io::Errno::SRCH))
    }

    fn assert_reaped(pid: Pid) {
        let options = WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT;
        assert!(
            matches!(
                waitid(WaitId::Pid(pid), options),
                Err(rustix::io::Errno::CHILD)
            ),
            "production owner must reap before the test checks it"
        );
    }

    fn published(path: &Path) -> Pid {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(raw) = std::fs::read_to_string(path)
                && let Ok(value) = raw.trim().parse::<i32>()
                && let Some(pid) = Pid::from_raw(value)
            {
                return pid;
            }
            assert!(Instant::now() < deadline, "child did not publish its pid");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn nonzero_exit_reaches_eof_and_keeps_its_code() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let started = Instant::now();
        let output = run(
            "printf live-bytes; exit 7",
            &scratch,
            Duration::from_secs(5),
            1024,
        )
        .expect("captured");
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(output.status, 7);
        assert_eq!(output.merged, "live-bytes");
        assert_eq!(
            std::fs::read(scratch.directory.join("build.log")).expect("log"),
            b"live-bytes"
        );
    }

    #[test]
    fn timeout_kills_the_owned_leader_before_cleanup() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let started = Instant::now();
        let error = run(
            "echo $$ > leader.pid; while :; do :; done",
            &scratch,
            Duration::from_millis(200),
            1024,
        );
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(matches!(error, Err(LiveBuildError::Process)));
        let pid = published(&scratch.directory.join("leader.pid"));
        assert_reaped(pid);
        assert!(gone(pid));
        assert!(scratch.directory.join("build.log").is_file());
    }

    #[test]
    fn output_cap_keeps_the_prefix_and_reaps_the_leader() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let script = "echo $$ > leader.pid; printf 0123456789abcdef; exit 0";
        assert!(matches!(
            run(script, &scratch, Duration::from_secs(5), 4),
            Err(LiveBuildError::Process)
        ));
        assert_eq!(
            std::fs::read(scratch.directory.join("build.log")).expect("prefix"),
            b"0123"
        );
        let pid = published(&scratch.directory.join("leader.pid"));
        assert_reaped(pid);
        assert!(gone(pid));
    }

    #[test]
    fn group_is_signaled_while_the_leader_is_still_owned() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let (reader, writer) = UnixStream::pair().expect("pair");
        let mut owned = super::child::OwnedChild::spawn(
            shell("echo $$ > leader.pid; sleep 30"),
            &scratch.directory,
            writer.into(),
        )
        .expect("spawn");
        let pid = published(&scratch.directory.join("leader.pid"));
        assert_eq!(pid, owned.pid().expect("owned pid"));
        let deadline = Instant::now() + Duration::from_secs(2);
        assert!(!owned.exited(deadline).expect("still owned"));
        owned
            .finish(deadline)
            .expect_err("killed leader has no exit code");
        drop(reader);
        assert_reaped(pid);
        assert!(gone(pid));
    }

    #[test]
    fn lost_ownership_revokes_group_before_drop() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let (reader, writer) = UnixStream::pair().expect("pair");
        let mut owned = super::child::OwnedChild::spawn(
            shell("echo $$ > leader.pid; sleep 30"),
            &scratch.directory,
            writer.into(),
        )
        .expect("spawn");
        let pid = published(&scratch.directory.join("leader.pid"));
        let blocking = WaitIdOptions::EXITED;
        assert!(
            waitid(WaitId::Pid(pid), blocking | WaitIdOptions::NOHANG)
                .expect("probe")
                .is_none()
        );
        kill_process_group(pid, Signal::KILL)
            .expect("stop owned fixture group before external reap");
        let reaped = waitid(WaitId::Pid(pid), WaitIdOptions::EXITED).expect("external reap");
        assert!(reaped.is_some());
        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(owned.exited(deadline).is_err());
        assert!(owned.pid().is_err());
        drop(owned);
        drop(reader);
        assert!(gone(pid));
    }

    #[test]
    fn eof_while_running_waits_for_the_same_deadline() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let script = "echo $$ > leader.pid; exec 1>&- 2>&-; sleep 30";
        let started = Instant::now();
        let error = run(script, &scratch, Duration::from_millis(400), 1024);
        assert!(started.elapsed() >= Duration::from_millis(300));
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(matches!(error, Err(LiveBuildError::Process)));
        let pid = published(&scratch.directory.join("leader.pid"));
        assert_reaped(pid);
        assert!(gone(pid));
    }

    #[test]
    fn late_output_past_cap_keeps_prefix_and_fails() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(&fixture.0).expect("scratch");
        let script = "printf ab; while [ ! -f release ]; do :; done; printf cdef; exit 0";
        let error = capture(
            shell(script),
            &scratch,
            Limits {
                wait: Duration::from_secs(2),
                output_cap: 4,
            },
            |_| {
                std::fs::File::create_new(scratch.directory.join("release"))
                    .expect("release second chunk after first observation");
                Ok(true)
            },
        );
        assert!(matches!(error, Err(LiveBuildError::Process)));
        assert_eq!(
            std::fs::read(scratch.directory.join("build.log")).expect("prefix"),
            b"abcd"
        );
    }

    #[test]
    fn copy_refuses_symlink_fifo_and_preserves_target() {
        let fixture = Fixture::new();
        let source = fixture.0.join("source.bin");
        std::fs::write(&source, b"engine").expect("source");
        let link = fixture.0.join("source.link");
        std::os::unix::fs::symlink(&source, &link).expect("link");
        assert!(super::copy_exclusive(&link, &fixture.0.join("from-link.engine")).is_err());
        assert!(!fixture.0.join("from-link.engine").exists());
        let fifo = fixture.0.join("source.fifo");
        assert!(
            Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .expect("mkfifo")
                .success()
        );
        assert!(super::copy_exclusive(&fifo, &fixture.0.join("from-fifo.engine")).is_err());
        assert!(!fixture.0.join("from-fifo.engine").exists());
        let target = fixture.0.join("kept.engine");
        std::fs::write(&target, b"kept").expect("target");
        assert!(super::copy_exclusive(&source, &target).is_err());
        assert_eq!(std::fs::read(&target).expect("preserved"), b"kept");
        let adopted = fixture.0.join("adopted.engine");
        super::copy_exclusive(&source, &adopted).expect("regular copy");
        assert_eq!(std::fs::read(&adopted).expect("copied"), b"engine");
    }

    #[test]
    fn sparse_regular_source_is_copied_without_inventing_short_read() {
        let fixture = Fixture::new();
        let source = fixture.0.join("short.bin");
        let file = std::fs::File::create(&source).expect("source");
        file.set_len(4).expect("declared");
        drop(file);
        let copied = fixture.0.join("copied.engine");
        super::copy_exclusive(&source, &copied).expect("four real zero bytes");
        assert_eq!(std::fs::read(&copied).unwrap(), [0_u8; 4]);
    }

    #[test]
    fn scratch_parent_is_absolute_and_existing_log_stays() {
        let fixture = Fixture::new();
        let scratch = Scratch::create(Path::new(".")).expect("relative parent");
        assert!(scratch.directory.is_absolute());
        scratch.record(b"first").expect("log");
        std::fs::rename(
            scratch.directory.join("build.log"),
            scratch.directory.join("saved.log"),
        )
        .expect("move");
        std::os::unix::fs::symlink("saved.log", scratch.directory.join("build.log"))
            .expect("log link");
        assert!(scratch.record(b"second").is_err());
        assert_eq!(
            std::fs::read(scratch.directory.join("saved.log")).expect("original"),
            b"first"
        );
        std::fs::remove_dir_all(&scratch.directory).expect("remove owned scratch");
        assert!(fixture.0.is_dir());
    }

    #[test]
    fn render_replaces_only_the_three_keys() {
        let template =
            "[property]\nkeep=1\nonnx-file=old\nmodel-engine-file=old\nbatch-size=9\nkeep-end=1\n";
        let rendered = super::render_infer_config(template, "model.onnx", "child.engine", 1)
            .expect("rendered");
        assert_eq!(
            rendered,
            "[property]\nkeep=1\nonnx-file=model.onnx\nmodel-engine-file=child.engine\nbatch-size=1\nkeep-end=1\n"
        );
        assert!(super::gst_argv("/tmp/private.txt", 1).ends_with(&["mux.sink_0".to_owned()]));
    }
}
