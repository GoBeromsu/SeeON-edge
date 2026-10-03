//! The fixed delivery-sender owner, separate from policy.
//! One named thread owns `SenderState` and the relay client for the life of
//! the queue. Wake signals are coalesced; stop is a separate flag. Neither
//! drop nor `request_stop` is evidence that entries were delivered.

use std::fmt;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};

use crate::delivery::DeliveryQueue;
use crate::delivery::sender::{DrainStop, SENDER_IDLE_WAIT, SenderState, drain_pass};
use crate::relay::RelayClient;
use crate::seam::Clock;
use crate::shutdown::ShutdownDeadline;

const WAKE_CAPACITY: usize = 1;

#[cfg(test)]
#[path = "delivery_backend_tests.rs"]
mod backend_tests;

/// Why the owner thread ended. Not evidence that entries were delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// `request_stop` (or drop) was observed before the next entry started.
    Requested,
    /// The shared shutdown deadline expired. Entries stay durable.
    Cutoff,
}

/// The actual stop reason. Durability is the queue's, not this report's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    pub stop: StopReason,
}

/// `join` before the thread ends keeps the handle. A later join returns the
/// cached result, including a queue failure or a panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JoinError {
    NotFinished,
    Panicked,
    Queue(QueueFault),
}

/// The queue failure that ended the owner, retained as its own diagnostic.
/// No synthetic success is substituted for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueFault {
    diagnostic: String,
}

/// Retain until `is_finished` and `join`. Drop requests stop and wakes the
/// owner; it does not join and is not completion.
#[must_use = "retain the delivery sender until join observes a finished thread"]
pub struct Handle {
    thread: Option<JoinHandle<Result<Report, JoinError>>>,
    stop: Arc<AtomicBool>,
    wake: SyncSender<()>,
    joined: Option<Result<Report, JoinError>>,
}

impl fmt::Display for QueueFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.diagnostic)
    }
}

impl std::error::Error for QueueFault {}

impl fmt::Display for JoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queue(error) => write!(formatter, "{error}"),
            Self::NotFinished => formatter.write_str("delivery sender has not finished"),
            Self::Panicked => formatter.write_str("delivery sender panicked"),
        }
    }
}

impl std::error::Error for JoinError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Queue(error) => Some(error),
            Self::NotFinished | Self::Panicked => None,
        }
    }
}

impl Handle {
    /// Coalesce a wake. A full signal slot already has one pending.
    pub fn wake(&self) {
        match self.wake.try_send(()) {
            Ok(()) | Err(TrySendError::Full(())) | Err(TrySendError::Disconnected(())) => {}
        }
    }

    /// Ask the owner to stop before the next entry. Wakes a waiting owner.
    /// Not evidence the thread has finished or that entries were delivered.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
        self.wake();
    }

    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Non-blocking. A live thread is retained and reported as `NotFinished`.
    /// After the thread has ended, the joined result is cached.
    pub fn join(&mut self) -> Result<Report, JoinError> {
        if let Some(joined) = &self.joined {
            return joined.clone();
        }
        if !self.is_finished() {
            return Err(JoinError::NotFinished);
        }
        let result = match self.thread.take() {
            Some(thread) => match thread.join() {
                Ok(ended) => ended,
                Err(_payload) => {
                    eprintln!("ml-worker: delivery sender panicked");
                    Err(JoinError::Panicked)
                }
            },
            None => Err(JoinError::Panicked),
        };
        self.joined = Some(result.clone());
        result
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.request_stop();
    }
}

/// Start the named delivery-sender thread. The shared queue is the same
/// identity producers publish into. The client and sender state stay on
/// that thread; this function performs no relay call.
pub fn spawn(
    queue: Arc<DeliveryQueue>,
    client: RelayClient,
    clip_export_enabled: bool,
    clock: Arc<dyn Clock>,
    deadline: Arc<ShutdownDeadline>,
) -> io::Result<Handle> {
    let stop = Arc::new(AtomicBool::new(false));
    let (wake_tx, wake_rx) = mpsc::sync_channel(WAKE_CAPACITY);
    let thread_stop = Arc::clone(&stop);
    let thread = thread::Builder::new()
        .name("delivery-sender".to_owned())
        .spawn(move || {
            run(
                queue,
                client,
                clip_export_enabled,
                clock.as_ref(),
                deadline.as_ref(),
                &thread_stop,
                wake_rx,
            )
        })?;
    Ok(Handle {
        thread: Some(thread),
        stop,
        wake: wake_tx,
        joined: None,
    })
}

fn run(
    queue: Arc<DeliveryQueue>,
    client: RelayClient,
    clip_export_enabled: bool,
    clock: &dyn Clock,
    deadline: &ShutdownDeadline,
    stop: &AtomicBool,
    wake: Receiver<()>,
) -> Result<Report, JoinError> {
    let mut state = SenderState::new();
    // The first pass scans the durable backlog without a wake. Cutoff returns
    // here without another queue read.
    let reason = loop {
        if stop.load(Ordering::Acquire) {
            break StopReason::Requested;
        }
        // Sample before reading the deadline. Never freeze its absolute value.
        let now = clock.monotonic();
        if deadline.deadline().is_some_and(|cutoff| now >= cutoff) {
            break StopReason::Cutoff;
        }
        let summary = drain_pass(
            &queue,
            &client,
            &mut state,
            clock,
            deadline,
            stop,
            clip_export_enabled,
        );
        match summary.stop {
            DrainStop::Stopped => break StopReason::Requested,
            DrainStop::Cutoff => break StopReason::Cutoff,
            DrainStop::Queue(error) => {
                eprintln!("ml-worker: delivery sender stopped on a queue failure: {error}");
                return Err(JoinError::Queue(QueueFault {
                    diagnostic: error.to_string(),
                }));
            }
            DrainStop::StepLimit => continue,
            DrainStop::Idle | DrainStop::Unacknowledged => {
                if !wait(clock, deadline, stop, &wake) {
                    break if stop.load(Ordering::Acquire) {
                        StopReason::Requested
                    } else {
                        StopReason::Cutoff
                    };
                }
            }
        }
    };
    Ok(Report { stop: reason })
}

/// Wait out a non-acknowledgement, or until a wake, stop, or cutoff.
/// `true` means the owner should drain again. A disconnected wake still
/// waits out the cadence; it is not a spin and not an early retry.
fn wait(
    clock: &dyn Clock,
    deadline: &ShutdownDeadline,
    stop: &AtomicBool,
    wake: &Receiver<()>,
) -> bool {
    let idle = SENDER_IDLE_WAIT;
    let start = clock.monotonic();
    loop {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        let now = clock.monotonic();
        if deadline.deadline().is_some_and(|cutoff| now >= cutoff) {
            return false;
        }
        let elapsed = now.saturating_sub(start);
        if elapsed >= idle {
            return true;
        }
        let mut slice = idle - elapsed;
        if let Some(cutoff) = deadline.deadline()
            && let Some(remaining) = cutoff.checked_sub(now)
        {
            slice = slice.min(remaining);
        }
        if slice.is_zero() {
            return false;
        }
        match wake.recv_timeout(slice) {
            Ok(()) => return !stop.load(Ordering::Acquire),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => clock.pause(slice),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{ErrorKind, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime};

    use super::{Handle, JoinError, Report, StopReason, spawn};
    use crate::delivery::{
        ClipEntry, ClipFields, DeliveryEntry, DeliveryQueue, EventEntry, EventFields,
    };
    use crate::poll::{POLL_INTERVAL, poll_until};
    use crate::relay::RelayClient;
    use crate::seam::{Clock, SystemClock};
    use crate::shutdown::ShutdownDeadline;

    struct TempDir(PathBuf);
    struct PanicClock;

    impl Clock for PanicClock {
        fn monotonic(&self) -> Duration {
            panic!("delivery sender fault");
        }

        fn wall(&self) -> SystemTime {
            SystemTime::UNIX_EPOCH
        }

        fn pause(&self, _limit: Duration) {}
    }

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "seeon-delivery-owner-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// One loopback relay. `stall` holds each accepted socket open and does
    /// not answer it. Drop stops the listener and joins it.
    struct Held {
        stop: Arc<AtomicBool>,
        accepted: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<String>>>,
        thread: Option<std::thread::JoinHandle<()>>,
        addr: std::net::SocketAddr,
    }

    impl Held {
        fn start(stall: bool) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let accepted = Arc::new(AtomicUsize::new(0));
            let requests = Arc::new(Mutex::new(Vec::new()));
            let thread = std::thread::Builder::new()
                .name("delivery-loopback".to_owned())
                .spawn({
                    let stop = Arc::clone(&stop);
                    let accepted = Arc::clone(&accepted);
                    let requests = Arc::clone(&requests);
                    move || listen(listener, stop, accepted, requests, stall)
                })
                .unwrap();
            Self {
                stop,
                accepted,
                requests,
                thread: Some(thread),
                addr,
            }
        }

        fn url(&self) -> String {
            format!("http://{}", self.addr)
        }

        fn identities(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }

        fn attempts(&self, id: &str) -> usize {
            self.identities().iter().filter(|seen| *seen == id).count()
        }

        fn close(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(thread) = self.thread.take() {
                let joined = thread.join();
                if !std::thread::panicking() {
                    joined.expect("delivery loopback thread failed");
                }
            }
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            self.close();
        }
    }

    fn listen(
        listener: TcpListener,
        stop: Arc<AtomicBool>,
        accepted: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<String>>>,
        stall: bool,
    ) {
        let clock = SystemClock::new();
        listener.set_nonblocking(true).unwrap();
        let deadline = clock.monotonic().saturating_add(Duration::from_secs(20));
        let mut held = Vec::new();
        poll_until(&clock, deadline, "loopback stop", || {
            if stop.load(Ordering::Acquire) {
                return true;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    assert!(
                        accepted.load(Ordering::Acquire) < 16,
                        "fixture request bound"
                    );
                    stream.set_nonblocking(false).unwrap();
                    let id = read_request(&mut stream, &clock, &stop);
                    requests.lock().unwrap().push(id.clone());
                    accepted.fetch_add(1, Ordering::AcqRel);
                    if stall {
                        held.push(stream);
                    } else {
                        write_ack(&mut stream, &id);
                    }
                    false
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => false,
                Err(error) => panic!("loopback accept failed: {error}"),
            }
        })
        .unwrap();
        drop(held);
    }

    /// Read headers, then the full `Content-Length` body, before parsing.
    fn read_request(stream: &mut TcpStream, clock: &SystemClock, stop: &AtomicBool) -> String {
        let deadline = clock.monotonic().saturating_add(Duration::from_secs(2));
        let mut bytes = Vec::new();
        read_bounded(stream, clock, stop, deadline, &mut bytes, |bytes| {
            header_end(bytes).is_some()
        });
        let end = header_end(&bytes).unwrap();
        let length = content_length(&bytes[..end]);
        assert!(length <= 64 * 1024 - end, "fixture body bound");
        read_bounded(stream, clock, stop, deadline, &mut bytes, |bytes| {
            bytes.len() >= end + length
        });
        edge_event_id(&bytes[end..end + length])
    }

    fn read_bounded(
        stream: &mut TcpStream,
        clock: &SystemClock,
        stop: &AtomicBool,
        deadline: Duration,
        bytes: &mut Vec<u8>,
        mut ready: impl FnMut(&[u8]) -> bool,
    ) {
        poll_until(clock, deadline, "loopback request", || {
            if stop.load(Ordering::Acquire) || ready(bytes) {
                return true;
            }
            let now = clock.monotonic();
            if now >= deadline {
                return false;
            }
            stream
                .set_read_timeout(Some((deadline - now).min(POLL_INTERVAL)))
                .unwrap();
            let mut buf = [0_u8; 2048];
            match stream.read(&mut buf) {
                Ok(0) => true,
                Ok(n) => {
                    bytes.extend_from_slice(&buf[..n]);
                    assert!(bytes.len() <= 64 * 1024, "fixture request byte bound");
                    ready(bytes)
                }
                Err(error)
                    if error.kind() == ErrorKind::TimedOut
                        || error.kind() == ErrorKind::WouldBlock =>
                {
                    false
                }
                Err(error) => panic!("loopback request read failed: {error}"),
            }
        })
        .unwrap();
        assert!(
            ready(bytes),
            "loopback request ended before it was complete"
        );
    }

    fn header_end(bytes: &[u8]) -> Option<usize> {
        bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
    }

    fn content_length(headers: &[u8]) -> usize {
        let text = std::str::from_utf8(headers).unwrap();
        text.split("\r\n")
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .expect("loopback request has no content-length")
    }

    fn edge_event_id(body: &[u8]) -> String {
        let value: serde_json::Value = serde_json::from_slice(body).expect("loopback request body");
        value
            .get("edge_event_id")
            .and_then(|id| id.as_str())
            .expect("loopback request edge_event_id")
            .to_owned()
    }

    fn write_ack(stream: &mut TcpStream, id: &str) {
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let body =
            format!(r#"{{"status":"accepted","edge_event_id":"{id}","event_id":"relay-{id}"}}"#);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    }

    fn event(id: &str) -> DeliveryEntry {
        DeliveryEntry::from(
            EventEntry::new(EventFields {
                edge_event_id: id.to_owned(),
                event_type: "fall".to_owned(),
                detected_at: "2026-10-02T00:00:00Z".to_owned(),
                camera_id: "camera-1".to_owned(),
                facility_id: "facility-1".to_owned(),
                decision_trace: b"{}".to_vec(),
                values: format!(r#"{{"edge_event_id":"{id}"}}"#).into_bytes(),
                ..EventFields::default()
            })
            .unwrap(),
        )
    }

    fn clip(id: &str) -> DeliveryEntry {
        DeliveryEntry::from(
            ClipEntry::new(ClipFields {
                clip_id: id.to_owned(),
                event_ids: vec!["event-1".to_owned()],
                camera_id: "camera-1".to_owned(),
                facility_id: "facility-1".to_owned(),
                local_state: "UNAVAILABLE".to_owned(),
                state_version: 1,
                unavailable_reason: Some("MISSING".to_owned()),
                ..ClipFields::default()
            })
            .unwrap(),
        )
    }

    fn admit(queue: &DeliveryQueue, entry: DeliveryEntry) {
        assert!(queue.try_admit(&entry).unwrap().accepted);
    }

    fn client(url: &str) -> RelayClient {
        RelayClient::new(url, "test-token", Duration::from_millis(200)).unwrap()
    }

    fn open_deadline() -> Arc<ShutdownDeadline> {
        Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).unwrap())
    }

    fn until(clock: &SystemClock, limit: Duration) -> Duration {
        clock.monotonic().saturating_add(limit)
    }

    fn wait_until(clock: &SystemClock, what: &'static str, done: impl FnMut() -> bool) {
        poll_until(clock, until(clock, Duration::from_secs(2)), what, done).unwrap();
    }

    fn finish(handle: &mut Handle, clock: &SystemClock) -> Report {
        handle.request_stop();
        wait_until(clock, "delivery sender finished", || handle.is_finished());
        handle.join().unwrap()
    }

    #[test]
    fn initial_backlog_is_sent_without_a_wake() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, event("backlog"));
        let loopback = Held::start(false);
        let clock = Arc::new(SystemClock::new());
        let mut handle = spawn(
            Arc::clone(&queue),
            client(&loopback.url()),
            true,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        wait_until(clock.as_ref(), "backlog delivered", || {
            queue.entries().unwrap().is_empty()
        });
        assert_eq!(loopback.accepted.load(Ordering::Acquire), 1);
        assert!(matches!(handle.join(), Err(JoinError::NotFinished)));
        let report = finish(&mut handle, clock.as_ref());
        assert_eq!(report.stop, StopReason::Requested);
        assert_eq!(handle.join().unwrap(), report);
    }

    #[test]
    fn wake_sends_a_new_durable_entry() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, event("backlog"));
        let loopback = Held::start(false);
        let clock = Arc::new(SystemClock::new());
        let mut handle = spawn(
            Arc::clone(&queue),
            client(&loopback.url()),
            true,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        wait_until(clock.as_ref(), "initial backlog acknowledged", || {
            queue.entries().unwrap().is_empty() && loopback.identities() == ["backlog".to_owned()]
        });
        assert!(!handle.is_finished());
        let idle_at = clock.monotonic();
        admit(&queue, event("fresh"));
        handle.wake();
        // Receipt alone precedes local ACK; require both within the wake bound.
        poll_until(
            clock.as_ref(),
            idle_at + Duration::from_millis(500),
            "wake delivered the fresh entry",
            || {
                loopback.identities() == ["backlog".to_owned(), "fresh".to_owned()]
                    && queue.entries().unwrap().is_empty()
            },
        )
        .unwrap();
        assert!(queue.entries().unwrap().is_empty());
        let report = finish(&mut handle, clock.as_ref());
        assert_eq!(report.stop, StopReason::Requested);
        assert_eq!(
            loopback.identities(),
            vec!["backlog".to_owned(), "fresh".to_owned()]
        );
    }

    #[test]
    fn disabled_clip_stays_queued() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, clip("held"));
        let loopback = Held::start(false);
        let clock = Arc::new(SystemClock::new());
        let mut handle = spawn(
            Arc::clone(&queue),
            client(&loopback.url()),
            false,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        // The disabled clip ends the first pass as unacknowledged. The owner
        // then waits the idle cadence; that wait is the evidence it was
        // considered and left queued. The condition is false throughout.
        let idle = clock.monotonic() + crate::delivery::sender::SENDER_IDLE_WAIT;
        poll_until(clock.as_ref(), idle, "disabled clip considered", || {
            handle.is_finished() || loopback.accepted.load(Ordering::Acquire) != 0
        })
        .unwrap_err();
        assert_eq!(queue.entries().unwrap().len(), 1);
        assert_eq!(loopback.accepted.load(Ordering::Acquire), 0);
        let report = finish(&mut handle, clock.as_ref());
        assert_eq!(queue.entries().unwrap().len(), 1);
        assert_eq!(report.stop, StopReason::Requested);
    }

    #[test]
    fn stop_during_stalled_http_returns_and_keeps_the_suffix() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, event("current"));
        admit(&queue, event("suffix"));
        let suffix = named_entry(&queue, "suffix");
        let mut loopback = Held::start(true);
        let clock = Arc::new(SystemClock::new());
        let mut handle = spawn(
            Arc::clone(&queue),
            client(&loopback.url()),
            true,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        wait_until(clock.as_ref(), "stalled request accepted", || {
            loopback.accepted.load(Ordering::Acquire) >= 1
        });
        let started = Instant::now();
        handle.request_stop();
        assert!(started.elapsed() < Duration::from_millis(200));
        assert!(matches!(handle.join(), Err(JoinError::NotFinished)));
        assert_eq!(queue.entries().unwrap().len(), 2);
        assert_eq!(loopback.accepted.load(Ordering::Acquire), 1);
        let suffix_attempts = loopback.attempts("suffix");
        assert_eq!(suffix_attempts, 0);
        assert_eq!(loopback.identities(), vec!["current".to_owned()]);
        loopback.close();
        wait_until(clock.as_ref(), "stalled sender finished", || {
            handle.is_finished()
        });
        let report = handle.join().unwrap();
        assert_eq!(report.stop, StopReason::Requested);
        assert_eq!(queue.entries().unwrap().len(), 2);
        assert_eq!(named_entry(&queue, "suffix"), suffix);
        assert_eq!(loopback.attempts("suffix"), suffix_attempts);
        assert_eq!(
            edge_ids(&queue),
            vec!["current".to_owned(), "suffix".to_owned()]
        );
    }

    #[test]
    fn expired_deadline_retains_entries_without_a_relay_call() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, event("held"));
        let loopback = Held::start(false);
        let clock = Arc::new(SystemClock::new());
        let deadline = Arc::new(ShutdownDeadline::new(Duration::from_nanos(1)).unwrap());
        deadline.request_at(Duration::ZERO).unwrap();
        let mut handle = spawn(
            Arc::clone(&queue),
            client(&loopback.url()),
            true,
            clock.clone(),
            deadline,
        )
        .unwrap();
        wait_until(clock.as_ref(), "cutoff sender finished", || {
            handle.is_finished()
        });
        let report = handle.join().unwrap();
        assert_eq!(report.stop, StopReason::Cutoff);
        assert_eq!(edge_ids(&queue), vec!["held".to_owned()]);
        assert_eq!(loopback.accepted.load(Ordering::Acquire), 0);
        assert_eq!(handle.join().unwrap(), report);
    }

    #[test]
    fn queue_error_is_observable_and_repeated_join_is_stable() {
        let dir = TempDir::new();
        let queue_dir = dir.path().join("queue");
        let queue = Arc::new(DeliveryQueue::open(&queue_dir, true).unwrap());
        std::fs::remove_dir_all(&queue_dir).unwrap();
        let loopback = Held::start(false);
        let clock = Arc::new(SystemClock::new());
        let mut handle = spawn(
            queue,
            client(&loopback.url()),
            true,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        wait_until(clock.as_ref(), "queue failure finished", || {
            handle.is_finished()
        });
        let error = handle.join().unwrap_err();
        let JoinError::Queue(fault) = &error else {
            panic!("queue failure was not retained");
        };
        assert!(
            fault.diagnostic.contains("delivery queue I/O failed"),
            "{}",
            fault.diagnostic
        );
        match handle.join() {
            Err(JoinError::Queue(again)) => assert_eq!(again.diagnostic, fault.diagnostic),
            other => panic!("repeated join changed the queue failure: {other:?}"),
        }
        assert_eq!(loopback.accepted.load(Ordering::Acquire), 0);
    }

    #[test]
    fn sender_panic_is_observable() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, event("held"));
        let loopback = Held::start(false);
        let clock = Arc::new(PanicClock);
        let mut handle = spawn(
            queue,
            client(&loopback.url()),
            true,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        let started = std::time::Instant::now();
        while !handle.is_finished() {
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "panicked sender did not finish"
            );
            std::thread::yield_now();
        }
        assert!(matches!(handle.join(), Err(JoinError::Panicked)));
        assert!(matches!(handle.join(), Err(JoinError::Panicked)));
        assert_eq!(loopback.accepted.load(Ordering::Acquire), 0);
        assert!(dir.path().exists());
    }

    #[test]
    fn join_before_finish_retains_the_handle() {
        let dir = TempDir::new();
        let queue = Arc::new(DeliveryQueue::open(dir.path(), true).unwrap());
        admit(&queue, event("later"));
        let loopback = Held::start(true);
        let clock = Arc::new(SystemClock::new());
        let mut handle = spawn(
            Arc::clone(&queue),
            client(&loopback.url()),
            true,
            clock.clone(),
            open_deadline(),
        )
        .unwrap();
        wait_until(clock.as_ref(), "owner blocked in http", || {
            loopback.accepted.load(Ordering::Acquire) >= 1
        });
        assert!(matches!(handle.join(), Err(JoinError::NotFinished)));
        assert!(!handle.is_finished());
        drop(loopback);
        handle.request_stop();
        wait_until(clock.as_ref(), "retained sender finished", || {
            handle.is_finished()
        });
        let report = handle.join().unwrap();
        assert_eq!(report.stop, StopReason::Requested);
        assert_eq!(edge_ids(&queue), vec!["later".to_owned()]);
        assert_eq!(handle.join().unwrap(), report);
    }

    fn named_entry(queue: &DeliveryQueue, id: &str) -> serde_json::Value {
        queue
            .entries()
            .unwrap()
            .into_iter()
            .find(|entry| entry.get("edge_event_id").and_then(|value| value.as_str()) == Some(id))
            .unwrap()
    }

    fn edge_ids(queue: &DeliveryQueue) -> Vec<String> {
        queue
            .entries()
            .unwrap()
            .iter()
            .filter_map(|entry| {
                entry
                    .get("edge_event_id")
                    .and_then(|id| id.as_str())
                    .map(str::to_owned)
            })
            .collect()
    }
}
