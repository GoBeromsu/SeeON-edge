//! T33: the media thread over the RTSP relay (ADR-0009 L96-98: buffers are
//! released on every consume, drop, shutdown and error path). The leases are
//! DeepStream buffers, so there is no CPU stand-in. After each fault the
//! native `callbacks_active` and `records_reserved` return to 0, and after
//! faults (a) and (b) the owner keeps processing frames at the rate the relay
//! source sends them.
//!
//! Faults: (a) a corrupt-stream burst is a publisher restart at the relay,
//! which cuts the stream mid-GOP while a recording is open; the source
//! reconnects rather than failing, so the owner stays up. (b) A full delivery
//! queue is a stalled `pose_rx` consumer. (c) SIGTERM is the stop flag
//! in-process, with a recording still open. (d) is that stop followed by a
//! second owner in the same process: close over a reserved record slot
//! corrupts the heap, which that owner would meet as an abort, so release
//! withholds close while a slot is reserved and says so.
//!
//! The publisher reports its output time with `-progress` into
//! `SEEON_TEST_RELAY_DIR`, which is where the relay-side frame count comes
//! from; a request file there asks the host watcher to restart the publisher.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use seeon_deepstream_native::{
    MediaBinding, MediaConfig, MediaPoll, MediaResult, RecordTicket, SourceConfig,
};
use seeon_ml_worker::media::diagnostics::{Diagnostics, Snapshot};
use seeon_ml_worker::media::owner::{self, MediaParams};
use seeon_ml_worker::media::{COMMAND_CAPACITY, Command};
use seeon_ml_worker::msg::{
    POSE_PER_CAMERA, PREVIEW_CAPACITY, PosePacket, PreviewPacket, RECORD_CAPACITY, RecordReceipt,
};
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::seam::{Clock, SystemClock};

static GPU: Mutex<()> = Mutex::new(());

const BINDING: MediaBinding = MediaBinding {
    token: 73,
    generation: 7,
    epoch: 11,
};
const SHUTDOWN_BUDGET_MS: u32 = 5_000;
/// Engine deserialisation, pipeline start and the first RTSP connect.
const READY_WAIT: Duration = Duration::from_secs(120);
const REPLY_WAIT: Duration = Duration::from_secs(5);
const JOIN_WAIT: Duration = Duration::from_secs(15);
/// Upper bounds for a condition to hold; none of them is a pause.
const RECORD_WAIT: Duration = Duration::from_secs(30);
const RISE_WAIT: Duration = Duration::from_secs(30);
const RELEASE_WAIT: Duration = Duration::from_secs(60);
const RESTART_WAIT: Duration = Duration::from_secs(60);
/// A recording still open when the burst hits; it ends within the waits.
const RECORD_FORWARD_SECONDS: u32 = 5;
/// The clip's frame rate, which `-c copy` keeps.
const RELAY_FPS: u64 = 30;
/// Three seconds of relayed frames per comparison window.
const RELAY_WINDOW_FRAMES: u64 = 90;
/// One second of packets the stalled consumer must have cost.
const STALL_DROPS: u64 = 30;

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn env_text(var: &str) -> String {
    let value = std::env::var(var).unwrap_or_else(|_| panic!("{var} is required as UTF-8"));
    assert!(!value.trim().is_empty(), "{var} must be nonblank");
    value
}

fn env_file(var: &str) -> PathBuf {
    let path = PathBuf::from(env_text(var));
    assert!(path.is_file(), "{var} must name an existing file");
    path
}

fn relay_uri() -> String {
    let uri = env_text("SEEON_TEST_RTSP_URI");
    assert!(
        uri.starts_with("rtsp://") && uri.len() > "rtsp://".len(),
        "SEEON_TEST_RTSP_URI must be an rtsp URI"
    );
    uri
}

fn config(record_directory: PathBuf) -> MediaConfig {
    MediaConfig {
        sources: vec![SourceConfig {
            source_id: 0,
            binding: BINDING,
            uri: relay_uri(),
            record_prefix: "synthetic".to_owned(),
        }],
        infer_config_path: env_file("SEEON_TEST_MEDIA_INFER"),
        tracker_config_path: env_file("SEEON_TEST_MEDIA_TRACKER"),
        tracker_library_path: PathBuf::from(
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        ),
        record_directory,
        record_cache_seconds: 30,
        record_capacity: 4,
        mux_width: 640,
        mux_height: 360,
        mux_batch_timeout_us: 40_000,
        mux_live_source: true,
        tracker_width: 960,
        tracker_height: 544,
        queue_max_buffers: 4,
        preview_enabled: true,
        max_preview_bytes: 1024 * 1024,
        allow_file_uris: false,
    }
}

/// The relay's source side: the publisher's `-progress` output and the
/// restart handshake with the host watcher.
struct Relay {
    directory: PathBuf,
}

impl Relay {
    fn from_env() -> Self {
        let directory = PathBuf::from(env_text("SEEON_TEST_RELAY_DIR"));
        assert!(
            directory.is_dir(),
            "SEEON_TEST_RELAY_DIR must be a directory"
        );
        Self { directory }
    }

    /// Frames sent since the publisher last started, from the last complete
    /// `out_time_us=` report; `None` before the first one.
    fn frames(&self) -> Option<u64> {
        let text = fs::read_to_string(self.directory.join("progress.txt")).ok()?;
        let mut lines: Vec<&str> = text.split('\n').collect();
        lines.pop();
        lines.iter().rev().find_map(|line| {
            let micros: u64 = line.strip_prefix("out_time_us=")?.trim().parse().ok()?;
            Some(micros * RELAY_FPS / 1_000_000)
        })
    }

    /// Asks the watcher to stop and start the publisher; returns once the
    /// publisher is running again. Its frame count restarts from zero.
    fn restart_publisher(&self, clock: &dyn Clock) {
        let ack = self.directory.join("restart.ack");
        fs::remove_file(&ack).ok();
        fs::write(self.directory.join("restart.request"), b"media_lease\n")
            .expect("the relay directory is writable");
        let acknowledged = poll_until(
            clock,
            clock.monotonic() + RESTART_WAIT,
            "publisher restart",
            || ack.exists(),
        );
        assert_eq!(
            acknowledged,
            Ok(()),
            "the relay watcher restarts the publisher"
        );
        fs::remove_file(&ack).expect("the restart acknowledgement is removable");
    }
}

/// One media thread with its consumer ends.
struct Session {
    thread: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    clock: Arc<SystemClock>,
    diagnostics: Arc<Diagnostics>,
    commands: SyncSender<Command>,
    poses: Receiver<PosePacket>,
    _previews: Receiver<PreviewPacket>,
    records: Receiver<RecordReceipt>,
    receipts: Vec<RecordReceipt>,
    next_request: u64,
}

impl Session {
    fn start(name: &str) -> Self {
        let record_directory =
            PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("media_lease-{name}"));
        fs::remove_dir_all(&record_directory).ok();
        fs::create_dir_all(&record_directory).expect("the record directory is creatable");
        let stop = Arc::new(AtomicBool::new(false));
        let clock = Arc::new(SystemClock::new());
        let diagnostics = Arc::new(Diagnostics::new(1));
        let (pose_tx, poses) = mpsc::sync_channel(POSE_PER_CAMERA);
        let (preview_tx, previews) = mpsc::sync_channel(PREVIEW_CAPACITY);
        let (record_tx, records) = mpsc::sync_channel(RECORD_CAPACITY);
        let (commands, command_rx) = mpsc::sync_channel(COMMAND_CAPACITY);
        let (thread, readiness) = owner::spawn(MediaParams {
            config: config(record_directory),
            shutdown_budget_ms: SHUTDOWN_BUDGET_MS,
            stop: Arc::clone(&stop),
            clock: clock.clone(),
            pose_tx,
            preview_tx,
            record_tx,
            commands: command_rx,
            diagnostics: Arc::clone(&diagnostics),
        })
        .expect("the media thread spawns");
        assert_eq!(readiness.recv_timeout(READY_WAIT), Ok(Ok(())));
        Self {
            thread: Some(thread),
            stop,
            clock,
            diagnostics,
            commands,
            poses,
            _previews: previews,
            records,
            receipts: Vec::new(),
            next_request: 1,
        }
    }

    fn published(&self) -> u64 {
        self.diagnostics
            .published_frames(0)
            .expect("camera 0 is on the roster")
    }

    /// Consumes every delivered packet; returns how many pose packets came.
    fn drain(&mut self) -> u64 {
        let mut packets = 0;
        while self.poses.try_recv().is_ok() {
            packets += 1;
        }
        while let Ok(receipt) = self.records.try_recv() {
            self.receipts.push(receipt);
        }
        packets
    }

    /// Starts a recording once the source has frames, and waits until native
    /// holds its slot.
    fn start_record(&mut self, forward_seconds: u32) -> RecordTicket {
        let clock = Arc::clone(&self.clock);
        let mut ticket = None;
        let admitted = poll_until(
            clock.as_ref(),
            clock.monotonic() + RECORD_WAIT,
            "record start",
            || {
                self.drain();
                let (reply, answer) = mpsc::sync_channel(1);
                let request_id = self.next_request;
                self.next_request += 1;
                let command = Command::RecordStart {
                    source_id: 0,
                    binding: BINDING,
                    request_id,
                    lookback_seconds: 0,
                    forward_seconds,
                    reply,
                };
                if self.commands.try_send(command).is_err() {
                    return false;
                }
                match answer.recv_timeout(REPLY_WAIT) {
                    Ok(Ok(MediaPoll::Ready(admitted))) => {
                        ticket = Some(admitted);
                        true
                    }
                    Ok(Ok(MediaPoll::Status(status))) if status.result == MediaResult::Busy => {
                        false
                    }
                    Ok(Ok(MediaPoll::Status(status))) => panic!("record start refused: {status:?}"),
                    Ok(Err(error)) => panic!("record start failed: {error:?}"),
                    Err(error) => panic!("no record start reply: {error:?}"),
                }
            },
        );
        assert_eq!(admitted, Ok(()), "{:?}", self.diagnostics.snapshot());
        let reserved = poll_until(
            clock.as_ref(),
            clock.monotonic() + RECORD_WAIT,
            "record reserved",
            || {
                self.drain();
                self.diagnostics.snapshot().records_reserved >= 1
            },
        );
        assert_eq!(reserved, Ok(()), "{:?}", self.diagnostics.snapshot());
        ticket.expect("an admitted record has a ticket")
    }

    /// Over the next `RELAY_WINDOW_FRAMES` the relay sends, the owner
    /// publishes at least half as many frames and pose packets arrive.
    fn frames_keep_rising(&mut self, relay: &Relay) {
        let clock = Arc::clone(&self.clock);
        let mut start = None;
        let based = poll_until(
            clock.as_ref(),
            clock.monotonic() + RISE_WAIT,
            "relay and media baseline",
            || {
                self.drain();
                start = relay.frames().map(|relayed| (relayed, self.published()));
                start.is_some_and(|(_, processed)| processed > 0)
            },
        );
        assert_eq!(based, Ok(()), "{:?}", self.diagnostics.snapshot());
        let (relay_start, processed_start) = start.expect("a baseline was read");
        let mut end = (relay_start, processed_start);
        let mut packets = 0;
        let windowed = poll_until(
            clock.as_ref(),
            clock.monotonic() + RISE_WAIT,
            "relay window",
            || {
                packets += self.drain();
                if let Some(relayed) = relay.frames() {
                    end = (relayed, self.published());
                }
                end.0 >= relay_start + RELAY_WINDOW_FRAMES
            },
        );
        assert_eq!(windowed, Ok(()), "{:?}", self.diagnostics.snapshot());
        let relayed = end.0 - relay_start;
        let processed = end.1 - processed_start;
        assert!(
            processed * 2 >= relayed,
            "processed {processed} of {relayed} relayed frames"
        );
        assert!(packets > 0, "no pose packet arrived");
    }

    /// Both lease counters at 0 in one native status.
    fn assert_released(&self, fault: &str) {
        let mut last = Snapshot::default();
        let released = poll_until(
            self.clock.as_ref(),
            self.clock.monotonic() + RELEASE_WAIT,
            "media leases released",
            || {
                last = self.diagnostics.snapshot();
                last.callbacks_active == 0 && last.records_reserved == 0
            },
        );
        assert_eq!(released, Ok(()), "{fault}: {last:?}");
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.join(JOIN_WAIT);
    }

    fn join(&mut self, wait: Duration) {
        let thread = self.thread.take().expect("the media thread is joined once");
        let deadline = self.clock.monotonic() + wait;
        let joined = owner::join(thread, self.clock.as_ref(), deadline);
        assert_eq!(joined, Ok(()), "{:?}", self.diagnostics.snapshot());
    }

    fn assert_shut_down(&self) {
        let snapshot = self.diagnostics.snapshot();
        assert!(snapshot.stopped && snapshot.closed, "{snapshot:?}");
    }
}

impl Drop for Session {
    /// A failed assertion still ends the thread before the next test.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let deadline = self.clock.monotonic() + JOIN_WAIT;
            owner::join(thread, self.clock.as_ref(), deadline).ok();
        }
    }
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_MEDIA_INFER, SEEON_TEST_MEDIA_TRACKER and SEEON_TEST_RTSP_URI"]
fn a_corrupt_stream_burst_releases_every_lease_while_the_source_reconnects() {
    let _gpu = gpu_lock();
    let relay = Relay::from_env();
    let mut session = Session::start("corrupt-burst");
    session.frames_keep_rising(&relay);
    let ticket = session.start_record(RECORD_FORWARD_SECONDS);
    relay.restart_publisher(session.clock.as_ref());
    let at_burst = session.published();
    let clock = Arc::clone(&session.clock);
    let settled = poll_until(
        clock.as_ref(),
        clock.monotonic() + RELEASE_WAIT,
        "burst record receipt and reconnect",
        || {
            session.drain();
            let reaped = session
                .receipts
                .iter()
                .any(|receipt| receipt.ticket.request_id == ticket.request_id);
            reaped && session.published() > at_burst
        },
    );
    assert_eq!(settled, Ok(()), "{:?}", session.diagnostics.snapshot());
    session.assert_released("corrupt-stream burst");
    session.frames_keep_rising(&relay);
    session.stop_and_join();
    session.assert_released("stop after the burst");
    session.assert_shut_down();
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_MEDIA_INFER, SEEON_TEST_MEDIA_TRACKER and SEEON_TEST_RTSP_URI"]
fn a_stalled_pose_consumer_drops_packets_while_every_lease_is_released() {
    let _gpu = gpu_lock();
    let relay = Relay::from_env();
    let mut session = Session::start("stalled-consumer");
    session.frames_keep_rising(&relay);
    let ticket = session.start_record(2);
    let dropped = |session: &Session| {
        session
            .diagnostics
            .handoff_dropped_frames(0)
            .expect("camera 0 is on the roster")
    };
    let before = dropped(&session);
    let stalled = poll_until(
        session.clock.as_ref(),
        session.clock.monotonic() + RISE_WAIT,
        "stalled consumer drops",
        || dropped(&session) >= before + STALL_DROPS,
    );
    assert_eq!(stalled, Ok(()), "{:?}", session.diagnostics.snapshot());
    session.assert_released("full delivery queue");
    let frame = session
        .poses
        .try_recv()
        .expect("the stalled queue holds pose packets")
        .frame;
    let receipt = session
        .receipts
        .pop()
        .or_else(|| session.records.try_recv().ok())
        .expect("the completed record was delivered");
    assert_eq!(
        (
            receipt.ticket.request_id,
            receipt.ticket.session_valid,
            receipt.result,
            receipt.error
        ),
        (ticket.request_id, 1, MediaResult::Ok, 0)
    );
    assert!(receipt.duration_ms > 0 && receipt.contains_video);
    assert_eq!(
        (receipt.width, receipt.height),
        (frame.source_width, frame.source_height)
    );
    session.frames_keep_rising(&relay);
    session.stop_and_join();
    session.assert_released("stop after the stall");
    session.assert_shut_down();
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_MEDIA_INFER, SEEON_TEST_MEDIA_TRACKER and SEEON_TEST_RTSP_URI"]
fn the_stop_flag_releases_an_open_recording() {
    let _gpu = gpu_lock();
    let relay = Relay::from_env();
    let mut session = Session::start("stop-flag");
    session.frames_keep_rising(&relay);
    session.start_record(20);
    session.stop_and_join();
    session.assert_released("stop flag");
    session.assert_shut_down();
}

#[test]
#[ignore = "requires an actual GPU, SEEON_TEST_MEDIA_INFER, SEEON_TEST_MEDIA_TRACKER and SEEON_TEST_RTSP_URI"]
fn release_closes_only_with_no_record_reserved_and_the_next_owner_starts() {
    let _gpu = gpu_lock();
    let mut first = Session::start("close-gate");
    first.start_record(20);
    first.stop_and_join();
    let released = first.diagnostics.snapshot();
    assert!(released.stopped, "{released:?}");
    let reserved = released.records_reserved > 0;
    assert_eq!(
        (released.closed, released.close_withheld),
        (!reserved, reserved),
        "{released:?}"
    );
    drop(first);
    let mut next = Session::start("close-gate-next");
    next.stop_and_join();
}
