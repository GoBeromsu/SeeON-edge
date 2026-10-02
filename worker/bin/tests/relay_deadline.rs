//! Deadline-bounded relay calls against a real loopback listener.
//!
//! Cutoff before send accepts nothing. A shutdown-limited stall, in headers
//! or body, returns `Cutoff` well before the longer baseline. A baseline
//! timeout and an early connection failure with a live budget stay transport
//! errors. No test asserts only a configured duration.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::relay::client::{
    ALERT_DELIVERY_TIMEOUT, DeadlineClient, RelayClient, RequestError, TransportError,
};
use seeon_ml_worker::seam::{Clock, SystemClock};
use seeon_ml_worker::shutdown::ShutdownDeadline;

const TOKEN: &str = "relay-deadline-token";
const BODY: &[u8] = b"{\"edge_event_id\":\"deadline\"}";
const SAFETY: Duration = Duration::from_secs(3);
const READ_LIMIT: usize = 64 * 1024;

struct TestClock {
    now: std::sync::atomic::AtomicU64,
    on_read: Mutex<Option<Box<dyn Fn() + Send>>>,
}

impl TestClock {
    fn at(now: Duration) -> Self {
        Self {
            now: std::sync::atomic::AtomicU64::new(nanos(now)),
            on_read: Mutex::new(None),
        }
    }

    fn set(&self, now: Duration) {
        self.now.store(nanos(now), Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        if let Some(hook) = self.on_read.lock().expect("clock hook").take() {
            hook();
        }
        Duration::from_nanos(self.now.load(Ordering::SeqCst))
    }

    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }

    fn pause(&self, _limit: Duration) {}
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).expect("duration fits")
}

fn client(base: &str, timeout: Duration) -> RelayClient {
    RelayClient::new(base, TOKEN, timeout).expect("client config")
}

fn view<'a>(
    client: &'a RelayClient,
    clock: &'a dyn Clock,
    deadline: &'a ShutdownDeadline,
) -> DeadlineClient<'a> {
    client.with_deadline(clock, deadline)
}

fn loopback() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let address = listener.local_addr().expect("loopback address");
    (listener, format!("http://{address}"))
}

struct OwnedListener {
    thread: Option<JoinHandle<io::Result<Vec<String>>>>,
    stop: Arc<AtomicBool>,
}

impl OwnedListener {
    fn spawn(listener: TcpListener) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            let mut heads = Vec::new();
            let clock = SystemClock::new();
            let deadline = clock.monotonic() + SAFETY;
            loop {
                if stopping.load(Ordering::SeqCst) {
                    break;
                }
                let mut accepted = None;
                let waited = poll_until(&clock, deadline, "relay accept", || {
                    if stopping.load(Ordering::SeqCst) {
                        return true;
                    }
                    match listener.accept() {
                        Ok((stream, _)) => {
                            accepted = Some(stream);
                            true
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => false,
                        Err(error) => panic!("relay accept: {error}"),
                    }
                });
                let Some(mut stream) = accepted else {
                    if waited.is_err() || stopping.load(Ordering::SeqCst) {
                        break;
                    }
                    continue;
                };
                heads.push(read_request(&mut stream)?);
            }
            Ok(heads)
        });
        Self {
            thread: Some(thread),
            stop,
        }
    }

    fn finish(mut self) -> Vec<String> {
        self.stop.store(true, Ordering::SeqCst);
        self.thread
            .take()
            .expect("listener thread")
            .join()
            .expect("listener thread panicked")
            .expect("listener io")
    }
}

impl Drop for OwnedListener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct RequestBounds {
    header_len: usize,
    content_length: usize,
}

/// `Some` only after httparse has a complete head and one `Content-Length`.
/// The request line has no colon, so splitting every line on `:` used to
/// return `None` immediately and close the socket before the body arrived.
fn request_bounds(buffer: &[u8]) -> io::Result<Option<RequestBounds>> {
    let mut slots = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut slots);
    let header_len = match request
        .parse(buffer)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?
    {
        httparse::Status::Complete(length) => length,
        httparse::Status::Partial => return Ok(None),
    };
    let mut content_length = None;
    for header in request.headers {
        if header.name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "chunked body"));
        }
        if header.name.eq_ignore_ascii_case("content-length") {
            let text = std::str::from_utf8(header.value)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let length = text
                .trim()
                .parse::<usize>()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            if content_length.replace(length).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "relay request repeated Content-Length",
                ));
            }
        }
    }
    let Some(content_length) = content_length else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "relay request omitted Content-Length",
        ));
    };
    Ok(Some(RequestBounds {
        header_len,
        content_length,
    }))
}

fn read_request(stream: &mut TcpStream) -> io::Result<String> {
    stream.set_nonblocking(true)?;
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    let mut buffer = Vec::new();
    let mut failed = None;
    let mut bounds = None;
    let mut chunk = [0_u8; 1024];
    poll_until(&clock, deadline, "relay request", || {
        if failed.is_some() || buffer.len() > READ_LIMIT {
            return true;
        }
        match stream.read(&mut chunk) {
            Ok(0) => true,
            Ok(read) => {
                buffer.extend_from_slice(&chunk[..read]);
                match request_bounds(&buffer) {
                    Ok(Some(parsed)) => {
                        let total = parsed.header_len.saturating_add(parsed.content_length);
                        if total > READ_LIMIT {
                            failed = Some(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "relay request exceeded fixture bound",
                            ));
                            return true;
                        }
                        bounds = Some(parsed);
                        buffer.len() >= total
                    }
                    Ok(None) => false,
                    Err(error) => {
                        failed = Some(error);
                        true
                    }
                }
            }
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    || error.kind() == io::ErrorKind::Interrupted =>
            {
                false
            }
            Err(error) => {
                failed = Some(error);
                true
            }
        }
    })
    .map_err(|timeout| io::Error::new(io::ErrorKind::TimedOut, timeout.to_string()))?;
    if let Some(error) = failed {
        return Err(error);
    }
    if buffer.len() > READ_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "relay request exceeded fixture bound",
        ));
    }
    let Some(bounds) = bounds else {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "relay request headers were incomplete",
        ));
    };
    let total = bounds.header_len + bounds.content_length;
    if buffer.len() < total {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "relay request body was incomplete",
        ));
    }
    String::from_utf8(buffer[..total].to_vec())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
}

fn write_all_bounded(stream: &mut TcpStream, bytes: &[u8]) -> io::Result<()> {
    stream.set_nonblocking(true)?;
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    let mut written = 0;
    let mut failed = None;
    poll_until(&clock, deadline, "relay response write", || {
        if failed.is_some() || written == bytes.len() {
            return true;
        }
        match stream.write(&bytes[written..]) {
            Ok(0) => true,
            Ok(count) => {
                written += count;
                written == bytes.len()
            }
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    || error.kind() == io::ErrorKind::Interrupted =>
            {
                false
            }
            Err(error) => {
                failed = Some(error);
                true
            }
        }
    })
    .map_err(|timeout| io::Error::new(io::ErrorKind::TimedOut, timeout.to_string()))?;
    if let Some(error) = failed {
        return Err(error);
    }
    if written != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "relay response write stopped early",
        ));
    }
    stream.flush()
}

enum Reply {
    HeadersThenSilence,
    BodyThenSilence {
        body_len: usize,
    },
    Complete {
        status: String,
        body: Vec<u8>,
    },
    AfterRequest {
        status: String,
        body: Vec<u8>,
        publish: Arc<ShutdownDeadline>,
    },
}

struct Server {
    thread: Option<JoinHandle<io::Result<Vec<String>>>>,
    stop: Arc<AtomicBool>,
}

impl Server {
    fn start(listener: TcpListener, reply: Reply) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = thread::spawn(move || serve(listener, reply, stopping));
        Self {
            thread: Some(thread),
            stop,
        }
    }

    fn finish(mut self) -> Vec<String> {
        self.stop.store(true, Ordering::SeqCst);
        self.thread
            .take()
            .expect("server thread")
            .join()
            .expect("server thread panicked")
            .expect("server io")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve(listener: TcpListener, reply: Reply, stop: Arc<AtomicBool>) -> io::Result<Vec<String>> {
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    let mut accepted = None;
    let waited = poll_until(&clock, deadline, "relay connection", || {
        if stop.load(Ordering::SeqCst) {
            return true;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                accepted = Some(stream);
                true
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => false,
            Err(error) => panic!("relay connection: {error}"),
        }
    });
    let Some(mut stream) = accepted else {
        if stop.load(Ordering::SeqCst) || waited.is_err() {
            return Ok(Vec::new());
        }
        return Err(io::Error::other("relay connection ended without a socket"));
    };
    let head = read_request(&mut stream)?;
    match reply {
        Reply::HeadersThenSilence => hold_until(&stream, &stop)?,
        Reply::BodyThenSilence { body_len } => {
            let response = format!(
                "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nContent-Length: {body_len}\r\nConnection: close\r\n\r\n"
            );
            write_all_bounded(&mut stream, response.as_bytes())?;
            hold_until(&stream, &stop)?;
        }
        Reply::Complete { status, body } => write_response(&mut stream, &status, &body)?,
        Reply::AfterRequest {
            status,
            body,
            publish,
        } => {
            publish
                .request_at(Duration::ZERO)
                .map_err(|error| io::Error::other(error.to_string()))?;
            write_response(&mut stream, &status, &body)?;
        }
    }
    Ok(vec![head])
}

fn write_response(stream: &mut TcpStream, status: &str, body: &[u8]) -> io::Result<()> {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    write_all_bounded(stream, response.as_bytes())?;
    write_all_bounded(stream, body)
}

fn hold_until(mut stream: &TcpStream, stop: &AtomicBool) -> io::Result<()> {
    stream.set_nonblocking(true)?;
    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY;
    let mut chunk = [0_u8; 64];
    let result = poll_until(&clock, deadline, "relay stall", || {
        if stop.load(Ordering::SeqCst) {
            return true;
        }
        match stream.read(&mut chunk) {
            Ok(0) => true,
            Ok(_) => false,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    || error.kind() == io::ErrorKind::Interrupted =>
            {
                false
            }
            Err(error) => panic!("relay stall read: {error}"),
        }
    });
    match result {
        Ok(()) => Ok(()),
        Err(timeout) => Err(io::Error::new(io::ErrorKind::TimedOut, timeout.to_string())),
    }
}

fn refused_port() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind refused port");
    let address = listener.local_addr().expect("refused address");
    drop(listener);
    address
}

fn assert_cutoff(error: RequestError) {
    assert!(
        matches!(error, RequestError::Cutoff),
        "expected cutoff, got {error:?}"
    );
}

fn assert_transport(error: RequestError) -> TransportError {
    match error {
        RequestError::Transport(error) => {
            assert!(
                !error.to_string().contains(TOKEN),
                "transport error leaked the relay token"
            );
            error
        }
        RequestError::Cutoff => panic!("expected a transport error, got cutoff"),
    }
}

fn assert_token(head: &str) {
    assert!(head.contains(TOKEN), "request omitted the relay token");
    assert!(
        head.to_ascii_lowercase().contains("x-edge-relay-token"),
        "request omitted the token header"
    );
    let body = std::str::from_utf8(BODY).expect("fixture body is utf-8");
    assert!(
        head.contains(body),
        "request omitted the bounded Content-Length body"
    );
}

#[test]
fn expired_and_equal_deadlines_send_nothing() {
    let (listener, base) = loopback();
    let accepted = OwnedListener::spawn(listener);
    let relay = client(&base, ALERT_DELIVERY_TIMEOUT);
    let deadline = ShutdownDeadline::new(Duration::from_secs(25)).expect("budget");
    let published = deadline
        .request_at(Duration::from_secs(1))
        .expect("observation");
    // Observation 1s plus the 25s budget is absolute 26s. Clock 10s is still live.
    let absolute = Duration::from_secs(26);
    assert_eq!(published, absolute);
    let clock = TestClock::at(absolute + Duration::from_nanos(1));
    let bounded = view(&relay, &clock, &deadline);

    assert_cutoff(bounded.check().expect_err("expired guard"));
    assert_cutoff(bounded.post_alert(BODY).expect_err("expired alert"));
    assert_cutoff(
        bounded
            .post_json("api/v1/relay/alerts", BODY)
            .expect_err("expired post"),
    );
    clock.set(absolute);
    assert_eq!(clock.monotonic(), deadline.deadline().expect("published"));
    assert_cutoff(
        bounded
            .put_json("api/v1/relay/clips", BODY)
            .expect_err("equal"),
    );
    drop(relay);
    assert!(
        accepted.finish().is_empty(),
        "expired cutoff still connected"
    );
}

#[test]
fn earlier_observation_published_during_clock_read_is_respected() {
    let (listener, base) = loopback();
    let accepted = OwnedListener::spawn(listener);
    let relay = client(&base, ALERT_DELIVERY_TIMEOUT);
    let clock = TestClock::at(Duration::from_secs(30));
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).expect("budget"));
    let published = Arc::clone(&shared);
    *clock.on_read.lock().expect("hook") = Some(Box::new(move || {
        published
            .request_at(Duration::from_secs(1))
            .expect("earlier observation");
    }));
    let error = view(&relay, &clock, &shared)
        .post_alert(BODY)
        .expect_err("publication during clock read cuts off");
    assert_cutoff(error);
    assert!(shared.deadline().expect("published") < Duration::from_secs(30));
    drop(relay);
    assert!(
        accepted.finish().is_empty(),
        "clock-read cutoff still connected"
    );
}

#[test]
fn unrequested_deadline_preserves_success_and_transport_failure() {
    let (listener, base) = loopback();
    let reply = b"{\"ok\":true}";
    let server = Server::start(
        listener,
        Reply::Complete {
            status: "202 Accepted".to_owned(),
            body: reply.to_vec(),
        },
    );
    let relay = client(&base, ALERT_DELIVERY_TIMEOUT);
    let clock = TestClock::at(Duration::from_secs(4));
    let deadline = ShutdownDeadline::new(Duration::from_secs(25)).expect("budget");
    let bounded = view(&relay, &clock, &deadline);
    bounded.check().expect("unset deadline is open");
    let response = bounded.post_alert(BODY).expect("normal success");
    let heads = server.finish();
    assert_eq!(heads.len(), 1, "success made one request");
    assert_eq!(response.status, 202);
    assert_eq!(response.body, reply);
    assert_token(&heads[0]);
    assert!(!response.headers.iter().any(|(name, _)| name == "proxy"));

    let refused = refused_port();
    let dead = client(&format!("http://{refused}"), Duration::from_millis(200));
    let error = assert_transport(
        view(&dead, &clock, &deadline)
            .put_json("api/v1/relay/clips", BODY)
            .expect_err("closed port is a transport failure"),
    );
    assert_ne!(error.kind(), "Timeout");
    assert!(!error.is_timeout());
}

#[test]
fn shutdown_capped_stall_is_cutoff_before_baseline() {
    let baseline = Duration::from_secs(8);
    let cap = Duration::from_millis(200);
    let (header_listener, header_base) = loopback();
    let headers = Server::start(header_listener, Reply::HeadersThenSilence);
    let relay = client(&header_base, baseline);
    let clock = TestClock::at(Duration::from_secs(1));
    let deadline = ShutdownDeadline::new(Duration::from_secs(25)).expect("budget");
    deadline.request_at(Duration::ZERO).expect("observation");
    clock.set(deadline.deadline().expect("deadline") - cap);
    let started = Instant::now();
    let error = view(&relay, &clock, &deadline)
        .post_alert(BODY)
        .expect_err("stalled headers");
    let elapsed = started.elapsed();
    assert_cutoff(error);
    assert!(
        elapsed < Duration::from_millis(1500),
        "header stall took {elapsed:?}, not well before the {baseline:?} baseline"
    );
    let heads = headers.finish();
    assert_eq!(heads.len(), 1, "stalled headers accepted one request");
    assert_token(&heads[0]);

    let (body_listener, body_base) = loopback();
    let body = Server::start(body_listener, Reply::BodyThenSilence { body_len: 64 });
    let relay = client(&body_base, baseline);
    let started = Instant::now();
    let error = view(&relay, &clock, &deadline)
        .put_json("api/v1/relay/clips", BODY)
        .expect_err("stalled body");
    let elapsed = started.elapsed();
    assert_cutoff(error);
    assert!(
        elapsed < Duration::from_millis(1500),
        "body stall took {elapsed:?}, not well before the {baseline:?} baseline"
    );
    let heads = body.finish();
    assert_eq!(heads.len(), 1, "stalled body accepted one request");
    assert_token(&heads[0]);
}

#[test]
fn baseline_timeout_before_later_deadline_stays_transport() {
    let baseline = Duration::from_millis(200);
    let (listener, base) = loopback();
    let stall = Server::start(listener, Reply::HeadersThenSilence);
    let relay = client(&base, baseline);
    let clock = TestClock::at(Duration::from_secs(1));
    let deadline = ShutdownDeadline::new(Duration::from_secs(25)).expect("budget");
    deadline
        .request_at(Duration::from_secs(1))
        .expect("observation");
    let started = Instant::now();
    let error = assert_transport(
        view(&relay, &clock, &deadline)
            .post_json("api/v1/relay/alerts", BODY)
            .expect_err("baseline timeout"),
    );
    let elapsed = started.elapsed();
    assert!(error.is_timeout(), "kind {}", error.kind());
    assert!(
        elapsed < Duration::from_secs(2),
        "baseline timeout took {elapsed:?}"
    );
    assert!(
        deadline.deadline().expect("later") > clock.monotonic() + elapsed,
        "the shared deadline was not still later than the observed timeout"
    );
    let heads = stall.finish();
    assert_eq!(heads.len(), 1, "baseline timeout reached the listener");
    assert_token(&heads[0]);
}

#[test]
fn early_connection_failure_with_live_capped_budget_stays_transport() {
    let refused = refused_port();
    let relay = client(&format!("http://{refused}"), ALERT_DELIVERY_TIMEOUT);
    // Remaining 800ms is below the 16s baseline, so the call is shutdown-capped,
    // but still far longer than a loopback refusal. A 20ms pin made that refusal
    // look like a global timeout and therefore cutoff.
    let clock = TestClock::at(Duration::from_millis(200));
    let deadline = ShutdownDeadline::new(Duration::from_secs(1)).expect("budget");
    deadline.request_at(Duration::ZERO).expect("observation");
    assert_eq!(deadline.deadline(), Some(Duration::from_secs(1)));
    assert!(view(&relay, &clock, &deadline).check().is_ok());
    let started = Instant::now();
    let error = assert_transport(
        view(&relay, &clock, &deadline)
            .post_alert(BODY)
            .expect_err("connection failure"),
    );
    assert!(!error.is_timeout(), "kind {}", error.kind());
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "connection failure was not early"
    );
    assert!(clock.monotonic() < deadline.deadline().expect("still live"));
}

#[test]
fn completed_response_is_withheld_when_deadline_expires_while_reading() {
    let (listener, base) = loopback();
    let shared = Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).expect("budget"));
    shared
        .request_at(Duration::from_secs(10))
        .expect("initial observation");
    let server = Server::start(
        listener,
        Reply::AfterRequest {
            status: "200 OK".to_owned(),
            body: b"{}".to_vec(),
            publish: Arc::clone(&shared),
        },
    );
    let relay = client(&base, ALERT_DELIVERY_TIMEOUT);
    let clock = TestClock::at(Duration::from_secs(26));
    let error = view(&relay, &clock, &shared)
        .post_alert(BODY)
        .expect_err("expired before exposure");
    assert_cutoff(error);
    assert_eq!(
        shared.deadline(),
        Some(Duration::from_secs(25)),
        "server published the earlier observation before the response"
    );
    let heads = server.finish();
    assert_eq!(
        heads.len(),
        1,
        "withheld response still had one real request"
    );
    assert_token(&heads[0]);
}
