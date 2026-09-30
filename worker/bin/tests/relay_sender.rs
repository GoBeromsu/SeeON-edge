//! T9 and T16: one drain of the Python-written queue against a loopback
//! relay, the capabilities probe, the 422 dead-letter, the accepted_local
//! snapshot skip and the mapping of the Python sender's per-entry
//! exceptions to typed refusals.
//!
//! Oracles: the Python-recorded goldens under `r/`, their manifest
//! `transport` entries, the reviewed rows of `r/relay-dispositions.json`,
//! the Python-written queue `d/delivery-queue/`, its dead-letter copy
//! `d/delivery-queue-dead-letter/` and the Python sender lines named in
//! `PYTHON_EXCEPTIONS`. No expected value is produced by the code under test.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use seeon_ml_worker::delivery::DeliveryQueue;
use seeon_ml_worker::delivery::sender::{
    BodyError, DrainStop, EntryOutcome, InvalidEntry, SenderState, drain_pass,
};
use seeon_ml_worker::relay::RelayClient;
use seeon_ml_worker::relay::capabilities::{
    BackendCapabilities, CapabilityError, parse_capabilities, probe_capabilities,
};
use seeon_ml_worker::relay::client::ALERT_DELIVERY_TIMEOUT;
use seeon_ml_worker::seam::Clock;

const RELAY_TOKEN: &str = "<relay-token>";
const SAFETY_TIMEOUT: Duration = Duration::from_secs(30);
/// Headers the HTTP stack owns; the manifest records none of them.
const TRANSPORT_OWNED: [&str; 7] = [
    "host",
    "user-agent",
    "accept",
    "accept-encoding",
    "connection",
    "content-length",
    "transfer-encoding",
];
const QUEUE_FIXTURE: &str = "d/delivery-queue";
const DEAD_LETTER_FIXTURE: &str = "d/delivery-queue-dead-letter/queue-dead-letter";
const EVENT_FILE: &str = "event-a5e15ff2-90fd-4764-be74-a7da4f573cc9.json";
const ATTACHMENT_FILE: &str =
    "attachment-05d867440a7fae84651f59fffbea2b57faa3942ecb0f6b79716a6873ba01e9e7.json";
const DISPOSITION_FILE: &str =
    "disposition-d6681ffe537c67cb3880ffbf5edaa522cd5bfe992a5f99a034c3471ba257c405.json";
const CLIP_READY_FILE: &str =
    "clip-fb4271979957cd5b20b1078f7c3a2e13eaba5661c34a0a8d37e4387023d111e8.json";
const CLIP_NO_FRAMES_FILE: &str =
    "clip-3c30d6568aa365b7abbdf561cbb3234bd102fce015112be7010ec16821b30848.json";
const ALERT: &str = "r/alert.json";
const SNAPSHOT_ATTACHMENT: &str = "r/snapshot-attachment.json";
const SNAPSHOT_DISPOSITION: &str = "r/snapshot-disposition.json";
const CLIP_READY: &str = "r/clip-put-ready.json";
const CLIP_READY_RESPONSE: &str = "r/clip-put-ready.response.json";
const CAPABILITIES_REQUEST: &str = "r/capabilities.request.json";
const CAPABILITIES_RESPONSE: &str = "r/capabilities.response.json";
const DISPOSITIONS: &str = "r/relay-dispositions.json";
/// The camera of every queued fixture entry and of the capabilities query.
const CAMERA_ID: &str = "cmsnw6rjc01vhlh01oswn99yq";

/// Reviewed pairing of the Python clip entries with their PUT goldens, from
/// each golden's manifest `input`: (local_state, unavailable_reason, golden).
const CLIP_PAIRING: [(&str, Option<&str>, &str); 3] = [
    ("VERIFIED", None, CLIP_READY),
    (
        "UNAVAILABLE",
        Some("NO_FRAMES"),
        "r/clip-put-unavailable-capture-failed.json",
    ),
    (
        "UNAVAILABLE",
        Some("CORRUPT"),
        "r/clip-put-unavailable-corrupt.json",
    ),
];

/// One per-entry exception of the Python sender. `_send` raises it,
/// `except Exception` (evidence_sender.py:297) catches it, and the entry is
/// deferred and charged one attempt (L302-303) while it stays queued.
struct PythonException {
    exception: &'static str,
    line: u32,
    file: &'static str,
    field: &'static str,
    /// `None` removes the field; `Some` replaces it with this string.
    value: Option<&'static str>,
    expected: InvalidEntry,
}

/// Reviewed table: Python exception class and raising line of
/// `worker/pipeline/output/evidence/evidence_sender.py`, the one-field
/// mutation that raises it, and the Rust refusal it maps to.
const PYTHON_EXCEPTIONS: [PythonException; 7] = [
    PythonException {
        exception: "KeyError",
        line: 527,
        file: EVENT_FILE,
        field: "values_b64",
        value: None,
        expected: InvalidEntry::Body(BodyError::MissingField("values_b64")),
    },
    PythonException {
        exception: "binascii.Error",
        line: 527,
        file: EVENT_FILE,
        field: "values_b64",
        value: Some("e30"),
        expected: InvalidEntry::Body(BodyError::Base64("values_b64")),
    },
    PythonException {
        exception: "json.JSONDecodeError",
        line: 527,
        file: EVENT_FILE,
        field: "values_b64",
        value: Some("bm90IGpzb24="),
        expected: InvalidEntry::Body(BodyError::NotJson("values_b64")),
    },
    PythonException {
        exception: "ValueError (ClipLocalState)",
        line: 539,
        file: CLIP_READY_FILE,
        field: "local_state",
        value: Some("LOST"),
        expected: InvalidEntry::Body(BodyError::LocalState),
    },
    PythonException {
        exception: "ValueError (int)",
        line: 544,
        file: CLIP_READY_FILE,
        field: "state_version",
        value: Some("one"),
        expected: InvalidEntry::Body(BodyError::WrongType("state_version")),
    },
    PythonException {
        exception: "ValueError (EvidenceReasonCode)",
        line: 555,
        file: CLIP_NO_FRAMES_FILE,
        field: "unavailable_reason",
        value: Some("GONE"),
        expected: InvalidEntry::Body(BodyError::UnavailableReason),
    },
    PythonException {
        exception: "ValueError (unknown kind)",
        line: 495,
        file: EVENT_FILE,
        field: "kind",
        value: Some("AUDIO"),
        expected: InvalidEntry::UnknownKind,
    },
];

static SERIAL: AtomicU64 = AtomicU64::new(0);

fn worker_wire(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(relative)
}

fn read_json(relative: &str) -> Value {
    let bytes = fs::read(worker_wire(relative)).expect("golden is readable");
    serde_json::from_slice(&bytes).expect("golden is JSON")
}

/// The manifest `transport` of a golden, header names lower-cased.
fn manifest_transport(path: &str) -> Value {
    let manifest = read_json("manifest.json");
    let entry = manifest["goldens"]
        .as_array()
        .expect("goldens list")
        .iter()
        .find(|entry| entry["path"] == path)
        .unwrap_or_else(|| panic!("{path} is in the manifest"));
    lowercase_headers(entry["transport"].clone())
}

/// A recorded transport with its header names lower-cased.
fn lowercase_headers(mut transport: Value) -> Value {
    let headers: Map<String, Value> = transport["headers"]
        .as_object()
        .expect("transport headers")
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect();
    transport["headers"] = Value::Object(headers);
    transport
}

fn scratch(label: &str) -> PathBuf {
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let name = format!("relay-sender-{}-{label}-{serial}", std::process::id());
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    if directory.exists() {
        fs::remove_dir_all(&directory).expect("stale scratch removed");
    }
    fs::create_dir_all(&directory).expect("scratch dir");
    directory
}

struct FixedClock;

impl Clock for FixedClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
    fn pause(&self, _limit: Duration) {}
}

struct Recorded {
    method: String,
    target: String,
    headers: Vec<(String, Vec<u8>)>,
    body: Option<Vec<u8>>,
}

struct Head {
    length: usize,
    method: String,
    target: String,
    headers: Vec<(String, Vec<u8>)>,
}

fn invalid(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn parse_head(buffer: &[u8]) -> io::Result<Option<Head>> {
    let mut slots = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut slots);
    let length = match request.parse(buffer).map_err(invalid)? {
        httparse::Status::Complete(length) => length,
        httparse::Status::Partial => return Ok(None),
    };
    let method = request.method.ok_or_else(|| invalid("no method"))?;
    let target = request.path.ok_or_else(|| invalid("no target"))?;
    let headers = request
        .headers
        .iter()
        .map(|header| (header.name.to_owned(), header.value.to_vec()))
        .collect();
    Ok(Some(Head {
        length,
        method: method.to_owned(),
        target: target.to_owned(),
        headers,
    }))
}

fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> io::Result<()> {
    let mut chunk = [0_u8; 4096];
    let read = stream.read(&mut chunk)?;
    if read == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "peer closed"));
    }
    buffer.extend_from_slice(&chunk[..read]);
    Ok(())
}

fn read_request(stream: &mut TcpStream) -> io::Result<Recorded> {
    let mut buffer = Vec::new();
    let head = loop {
        if let Some(head) = parse_head(&buffer)? {
            break head;
        }
        read_more(stream, &mut buffer)?;
    };
    let declared = head
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| {
            std::str::from_utf8(value)
                .map_err(invalid)?
                .trim()
                .parse::<usize>()
                .map_err(invalid)
        })
        .transpose()?;
    let body = match declared {
        None => None,
        Some(length) => {
            while buffer.len() < head.length + length {
                read_more(stream, &mut buffer)?;
            }
            Some(buffer[head.length..head.length + length].to_vec())
        }
    };
    Ok(Recorded {
        method: head.method,
        target: head.target,
        headers: head.headers,
        body,
    })
}

fn loopback() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let base = format!("http://{}", listener.local_addr().expect("bound address"));
    (listener, base)
}

/// A base URL nothing listens on: the port was bound and released.
fn dead_relay() -> String {
    let (listener, base) = loopback();
    drop(listener);
    base
}

/// The request as the manifest records it, header names lower-cased.
fn transport_record(recorded: &Recorded, timeout: Duration) -> Value {
    let (path, query) = match recorded.target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (recorded.target.as_str(), ""),
    };
    let mut headers = Map::new();
    for (name, value) in &recorded.headers {
        let name = name.to_ascii_lowercase();
        if TRANSPORT_OWNED.contains(&name.as_str()) {
            continue;
        }
        let value = String::from_utf8(value.clone()).expect("header is UTF-8");
        let previous = headers.insert(name.clone(), Value::String(value));
        assert!(previous.is_none(), "header {name} sent twice");
    }
    let body_sha256 = recorded.body.as_ref().map_or(Value::Null, |body| {
        let digest: String = Sha256::digest(body)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Value::String(digest)
    });
    let mut record = Map::new();
    record.insert("body_sha256".into(), body_sha256);
    record.insert("headers".into(), Value::Object(headers));
    record.insert("method".into(), Value::String(recorded.method.clone()));
    record.insert("path".into(), Value::String(path.to_owned()));
    record.insert("query".into(), Value::String(query.to_owned()));
    record.insert("timeout_sec".into(), Value::from(timeout.as_secs_f64()));
    Value::Object(record)
}

/// Picks the reply for one recorded request: (status code, body).
type Route = Box<dyn Fn(&Recorded) -> (u16, Vec<u8>) + Send>;

/// A loopback relay that answers every request through its route until a
/// connection closes without sending a byte.
struct Relay {
    base: String,
    address: SocketAddr,
    server: JoinHandle<io::Result<Vec<Recorded>>>,
}

impl Relay {
    fn start(route: Route) -> Relay {
        let (listener, base) = loopback();
        let address = listener.local_addr().expect("bound address");
        let server = thread::spawn(move || serve_until_closed(&listener, &route));
        Relay {
            base,
            address,
            server,
        }
    }

    fn client(&self) -> RelayClient {
        RelayClient::new(&self.base, RELAY_TOKEN, ALERT_DELIVERY_TIMEOUT).expect("client config")
    }

    /// Every request served, in arrival order.
    fn stop(self) -> Vec<Recorded> {
        drop(TcpStream::connect(self.address).expect("stop connection"));
        self.server
            .join()
            .expect("server thread")
            .expect("requests served")
    }
}

fn serve_until_closed(listener: &TcpListener, route: &Route) -> io::Result<Vec<Recorded>> {
    let mut served = Vec::new();
    loop {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(SAFETY_TIMEOUT))?;
        let mut probe = [0_u8; 1];
        if stream.peek(&mut probe)? == 0 {
            return Ok(served);
        }
        let recorded = read_request(&mut stream)?;
        let (status, reply) = route(&recorded);
        let head = format!(
            "HTTP/1.1 {status} Relay\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            reply.len()
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(&reply)?;
        stream.flush()?;
        served.push(recorded);
    }
}

fn request_path(recorded: &Recorded) -> &str {
    recorded
        .target
        .split_once('?')
        .map_or(recorded.target.as_str(), |(path, _)| path)
}

fn golden_path(golden: &str) -> String {
    manifest_transport(golden)["path"]
        .as_str()
        .expect("manifest path")
        .to_owned()
}

/// The reviewed relay answer `name` of `r/relay-dispositions.json`.
fn disposition_reply(name: &str) -> (u16, Vec<u8>) {
    let dispositions = read_json(DISPOSITIONS);
    let row = dispositions["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .find(|row| row["name"] == name)
        .unwrap_or_else(|| panic!("row {name} exists"));
    let response = &row["response"];
    let status = response["status"].as_u64().expect("row status");
    let body = response["body"].as_str().expect("row body");
    (
        u16::try_from(status).expect("HTTP status"),
        body.as_bytes().to_vec(),
    )
}

/// Snapshot POSTs are acknowledged the way Python `_parse_relay_acceptance`
/// requires; a clip PUT gets a receipt for its own id and state_version.
fn relay_route(event_reply: (u16, Vec<u8>)) -> Route {
    let alerts = golden_path(ALERT);
    let snapshots = [
        golden_path(SNAPSHOT_ATTACHMENT),
        golden_path(SNAPSHOT_DISPOSITION),
    ];
    let clip_ready = golden_path(CLIP_READY);
    let ready_receipt = fs::read(worker_wire(CLIP_READY_RESPONSE)).expect("clip receipt");
    let clips = clip_ready
        .rsplit_once('/')
        .map(|(prefix, _)| format!("{prefix}/"))
        .expect("clip path prefix");
    Box::new(move |request: &Recorded| {
        let path = request_path(request);
        if path == alerts {
            return event_reply.clone();
        }
        if snapshots.iter().any(|snapshot| snapshot == path) {
            return (200, br#"{"status":"accepted"}"#.to_vec());
        }
        if path == clip_ready {
            return (200, ready_receipt.clone());
        }
        match path.strip_prefix(&clips) {
            Some(clip_id) => (200, clip_receipt(clip_id, request)),
            None => (404, b"{}".to_vec()),
        }
    })
}

fn clip_receipt(clip_id: &str, request: &Recorded) -> Vec<u8> {
    let body = request_body(request);
    let mut receipt = Map::new();
    receipt.insert("clip_id".into(), Value::String(clip_id.to_owned()));
    receipt.insert("state".into(), body["state"].clone());
    receipt.insert("state_version".into(), body["state_version"].clone());
    serde_json::to_vec(&Value::Object(receipt)).expect("receipt JSON")
}

fn request_body(request: &Recorded) -> Value {
    let body = request.body.as_deref().expect("request body");
    serde_json::from_slice(body).expect("request body is JSON")
}

fn fixture_bytes(file: &str) -> Vec<u8> {
    fs::read(worker_wire(QUEUE_FIXTURE).join(file)).expect("queue fixture")
}

fn fixture_files() -> Vec<String> {
    let mut files: Vec<String> = fs::read_dir(worker_wire(QUEUE_FIXTURE))
        .expect("queue fixture dir")
        .map(|entry| entry.expect("dir entry").file_name())
        .map(|name| name.into_string().expect("UTF-8 name"))
        .filter(|name| name.ends_with(".json"))
        .collect();
    files.sort();
    files
}

/// A fresh queue directory named `queue` holding `entries` byte for byte.
fn staged_queue(label: &str, entries: &[(&str, Vec<u8>)]) -> (PathBuf, DeliveryQueue) {
    let root = scratch(label);
    let directory = root.join("queue");
    let queue = DeliveryQueue::open(&directory, true).expect("queue opens");
    for (file, bytes) in entries {
        fs::write(directory.join(file), bytes).expect("entry staged");
    }
    (root, queue)
}

fn entry_id(entry: &Value) -> String {
    entry["entry_id"].as_str().expect("entry_id").to_owned()
}

/// Regular files directly below `directory`, by name.
fn files_by_name(directory: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(directory)
        .expect("directory listing")
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.is_file())
        .map(|path| {
            let name = path.file_name().expect("file name").to_string_lossy();
            (name.into_owned(), fs::read(&path).expect("file bytes"))
        })
        .collect()
}

/// The queued entry's golden: by kind, and for a clip by `CLIP_PAIRING`.
fn golden_for(entry: &Value) -> &'static str {
    match entry["kind"].as_str() {
        Some("EVENT") => ALERT,
        Some("SNAPSHOT_ATTACHMENT") => SNAPSHOT_ATTACHMENT,
        Some("SNAPSHOT_DISPOSITION") => SNAPSHOT_DISPOSITION,
        Some("CLIP") => {
            let reason = entry["unavailable_reason"].as_str();
            CLIP_PAIRING
                .iter()
                .find(|(state, paired, _)| entry["local_state"] == *state && reason == *paired)
                .map(|(_, _, golden)| *golden)
                .expect("clip entry is paired")
        }
        other => panic!("unexpected kind {other:?}"),
    }
}

fn without_body_sha(mut transport: Value) -> Value {
    transport
        .as_object_mut()
        .expect("transport object")
        .remove("body_sha256");
    transport
}

#[test]
fn drain_bodies_equal_relay_goldens() {
    let files = fixture_files();
    let staged: Vec<(&str, Vec<u8>)> = files
        .iter()
        .map(|file| (file.as_str(), fixture_bytes(file)))
        .collect();
    let entries: Vec<Value> = staged
        .iter()
        .map(|(_, bytes)| serde_json::from_slice(bytes).expect("entry JSON"))
        .collect();
    assert_eq!(entries.len(), 6, "the Python queue holds six entries");
    let (_root, queue) = staged_queue("drain", &staged);
    let relay = Relay::start(relay_route(disposition_reply("matched-ack")));
    let client = relay.client();
    let mut state = SenderState::new();

    let summary = drain_pass(&queue, &client, &mut state, &FixedClock, true);
    let requests = relay.stop();

    assert_eq!(requests.len(), entries.len(), "one request per entry");
    for entry in &entries {
        let golden = golden_for(entry);
        let expected = manifest_transport(golden);
        let matching: Vec<&Recorded> = requests
            .iter()
            .filter(|request| {
                expected["method"] == request.method.as_str()
                    && expected["path"] == request_path(request)
            })
            .collect();
        assert_eq!(matching.len(), 1, "{golden} is requested once");
        let request = matching[0];
        assert_eq!(request_body(request), read_json(golden), "{golden} body");
        assert_eq!(
            without_body_sha(transport_record(request, client.timeout())),
            without_body_sha(expected),
            "{golden} transport"
        );
    }
    let outcomes: BTreeMap<String, EntryOutcome> = summary.outcomes.into_iter().collect();
    let acknowledged: BTreeMap<String, EntryOutcome> = entries
        .iter()
        .map(|entry| (entry_id(entry), EntryOutcome::Acknowledged))
        .collect();
    assert_eq!(outcomes, acknowledged);
    assert!(matches!(summary.stop, DrainStop::Idle));
    assert_eq!(queue.entries().expect("queue listing"), Vec::<Value>::new());
}

#[test]
fn capabilities_probe_matches_manifest_and_refuses_partial_answers() {
    let answer = fs::read(worker_wire(CAPABILITIES_RESPONSE)).expect("capabilities golden");
    let relay = Relay::start(Box::new(move |_: &Recorded| (200, answer.clone())));
    let client = relay.client();

    let probed = probe_capabilities(&client, CAMERA_ID, &FixedClock);
    let requests = relay.stop();

    // The manifest `parsed_by_worker` of the response golden:
    // BackendCapabilities(event_idempotency=1, clip_export=1).
    let parsed = BackendCapabilities {
        event_idempotency: 1,
        clip_export: 1,
    };
    assert_eq!(probed, Ok(parsed));
    assert_eq!(requests.len(), 1, "one capabilities request");
    // The request golden is itself the recorded transport of the probe.
    assert_eq!(
        transport_record(&requests[0], client.timeout()),
        lowercase_headers(read_json(CAPABILITIES_REQUEST))
    );
    let refusals = [
        ("clip_export", CapabilityError::ClipExport),
        ("event_idempotency", CapabilityError::EventIdempotency),
    ];
    for (field, refusal) in refusals {
        let mut partial = read_json(CAPABILITIES_RESPONSE);
        partial.as_object_mut().expect("object").remove(field);
        let bytes = serde_json::to_vec(&partial).expect("partial JSON");
        assert_eq!(parse_capabilities(&bytes), Err(refusal), "without {field}");
    }
}

#[test]
fn status_422_dead_letters_python_file() {
    let event = fixture_bytes(EVENT_FILE);
    let id = entry_id(&serde_json::from_slice(&event).expect("event JSON"));
    let (root, queue) = staged_queue("dead-letter", &[(EVENT_FILE, event)]);
    let relay = Relay::start(relay_route(disposition_reply("422-permanent")));
    let mut state = SenderState::new();

    let summary = drain_pass(&queue, &relay.client(), &mut state, &FixedClock, true);
    let requests = relay.stop();

    assert_eq!(requests.len(), 1, "one alert POST");
    assert_eq!(
        summary.outcomes,
        vec![(id, EntryOutcome::DeadLettered { status: 422 })]
    );
    assert_eq!(queue.entries().expect("queue listing"), Vec::<Value>::new());
    assert_eq!(
        files_by_name(&root.join("queue-dead-letter")),
        files_by_name(&worker_wire(DEAD_LETTER_FIXTURE))
    );
}

#[test]
fn accepted_local_skips_snapshot_requests() {
    let files = [EVENT_FILE, ATTACHMENT_FILE, DISPOSITION_FILE];
    let staged: Vec<(&str, Vec<u8>)> = files
        .iter()
        .map(|file| (*file, fixture_bytes(file)))
        .collect();
    let ids: Vec<String> = staged
        .iter()
        .map(|(_, bytes)| entry_id(&serde_json::from_slice(bytes).expect("entry JSON")))
        .collect();
    let (_root, queue) = staged_queue("accepted-local", &staged);
    let relay = Relay::start(relay_route(disposition_reply("accepted-local-no-put")));
    let mut state = SenderState::new();

    let summary = drain_pass(&queue, &relay.client(), &mut state, &FixedClock, true);
    let requests = relay.stop();

    let snapshot_paths = [
        golden_path(SNAPSHOT_ATTACHMENT),
        golden_path(SNAPSHOT_DISPOSITION),
    ];
    let snapshot_requests = requests
        .iter()
        .filter(|request| {
            snapshot_paths
                .iter()
                .any(|path| path == request_path(request))
        })
        .count();
    assert_eq!(snapshot_requests, 0, "no snapshot request is sent");
    let sent: Vec<(&str, &str)> = requests
        .iter()
        .map(|request| (request.method.as_str(), request_path(request)))
        .collect();
    assert_eq!(sent, vec![("POST", golden_path(ALERT).as_str())]);
    let outcomes: BTreeMap<String, EntryOutcome> = summary.outcomes.into_iter().collect();
    let expected = BTreeMap::from([
        (ids[0].clone(), EntryOutcome::Acknowledged),
        (ids[1].clone(), EntryOutcome::SkippedAcceptedLocal),
        (ids[2].clone(), EntryOutcome::SkippedAcceptedLocal),
    ]);
    assert_eq!(outcomes, expected);
    assert_eq!(queue.entries().expect("queue listing"), Vec::<Value>::new());
}

#[test]
fn python_exceptions_map_to_rust_variants() {
    for case in PYTHON_EXCEPTIONS {
        let origin = format!("{} at evidence_sender.py:{}", case.exception, case.line);
        let mut entry: Value =
            serde_json::from_slice(&fixture_bytes(case.file)).expect("entry JSON");
        let object = entry.as_object_mut().expect("entry object");
        match case.value {
            None => object.remove(case.field),
            Some(value) => object.insert(case.field.to_owned(), Value::String(value.to_owned())),
        };
        let bytes = serde_json::to_vec(&entry).expect("entry JSON");
        let (_root, queue) = staged_queue("exception", &[(case.file, bytes)]);
        let client = RelayClient::new(&dead_relay(), RELAY_TOKEN, ALERT_DELIVERY_TIMEOUT)
            .expect("client config");
        let mut state = SenderState::new();
        let id = entry_id(&entry);

        let summary = drain_pass(&queue, &client, &mut state, &FixedClock, true);

        assert_eq!(
            summary.outcomes,
            vec![(id.clone(), EntryOutcome::Invalid(case.expected))],
            "{origin}"
        );
        assert_eq!(
            queue.entries().expect("queue listing"),
            vec![entry],
            "{origin}"
        );
        assert_eq!(state.attempts(&id), 1, "{origin}");
        assert!(state.is_deferred(&id), "{origin}");
    }
}
