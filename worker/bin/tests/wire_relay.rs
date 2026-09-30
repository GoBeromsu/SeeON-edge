//! Relay wire parity with Python `shared/events/evidence_http_transport.py`
//! and `RelayEvidenceClient`. The oracles are the reviewed Python goldens
//! under `tests/fixtures/worker-wire/r/` and the golden manifest; requests are
//! observed on a loopback `httparse` server, never a real relay.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_ml_worker::relay::client::{ALERT_DELIVERY_TIMEOUT, RelayClient};
use seeon_ml_worker::relay::wire::{
    ClipReceipt, ClipState, DeliveryDisposition, DeliveryFailure, EventReceipt, EventStatus,
    Response, classify_http_failure, parse_clip_result, parse_event_result, retry_after,
};
use seeon_ml_worker::seam::{Clock, SystemClock};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

/// The configured token is the golden placeholder itself, so the recorded
/// header value is compared verbatim and no real token exists anywhere.
const RELAY_TOKEN: &str = "<relay-token>";
/// 2026-09-29T12:00:00Z, the `injected_now_utc` of `r/relay-dispositions.json`.
const INJECTED_NOW_UNIX: u64 = 1_790_683_200;
/// Upper bound on any loopback socket read, so a broken client cannot hang the suite.
const SAFETY_TIMEOUT: Duration = Duration::from_secs(30);
/// Headers the HTTP stack owns (urllib and ureq differ); the goldens record
/// only the headers the relay client itself sets.
const TRANSPORT_OWNED: [&str; 7] = [
    "host",
    "user-agent",
    "accept",
    "accept-encoding",
    "connection",
    "content-length",
    "transfer-encoding",
];

fn worker_wire(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(relative)
}

fn read_bytes(relative: &str) -> Vec<u8> {
    std::fs::read(worker_wire(relative)).expect("golden file is readable")
}

fn read_json(relative: &str) -> Value {
    serde_json::from_slice(&read_bytes(relative)).expect("golden file is JSON")
}

fn golden_entry(manifest: &Value, path: &str) -> Value {
    manifest["goldens"]
        .as_array()
        .expect("manifest goldens is an array")
        .iter()
        .find(|entry| entry["path"] == path)
        .cloned()
        .expect("manifest lists the golden")
}

/// The manifest `json-value` comparison: drop the entry's `normalised`
/// top-level keys from a value before comparing it.
fn drop_normalised(mut value: Value, entry: &Value) -> Value {
    assert_eq!(entry["compare"], "json-value");
    let object = value.as_object_mut().expect("compared value is an object");
    for key in entry["normalised"]
        .as_array()
        .expect("normalised is a list")
    {
        object.remove(key.as_str().expect("normalised key is a string"));
    }
    value
}

struct FixedClock {
    wall: SystemTime,
}

impl FixedClock {
    fn injected() -> Self {
        Self {
            wall: UNIX_EPOCH + Duration::from_secs(INJECTED_NOW_UNIX),
        }
    }
}

impl Clock for FixedClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }

    fn wall(&self) -> SystemTime {
        self.wall
    }

    /// Time never moves on this clock, so a pause has nothing to wait for.
    fn pause(&self, _limit: Duration) {}
}

// The hand-written Rust-to-golden table. Every name below is the Python
// spelling from the reviewed goldens, written out here rather than derived.

fn event_status_name(status: EventStatus) -> &'static str {
    match status {
        EventStatus::Accepted => "accepted",
        EventStatus::AcceptedLocal => "accepted_local",
    }
}

fn clip_state_name(state: ClipState) -> &'static str {
    match state {
        ClipState::Ready => "READY",
        ClipState::Unavailable => "UNAVAILABLE",
        ClipState::Expired => "EXPIRED",
    }
}

fn disposition_name(disposition: DeliveryDisposition) -> &'static str {
    match disposition {
        DeliveryDisposition::Retry => "RETRY",
        DeliveryDisposition::Permanent => "PERMANENT",
        DeliveryDisposition::Compatibility => "COMPATIBILITY",
    }
}

fn event_json(receipt: &EventReceipt) -> Value {
    json!({
        "kind": "EventReceipt",
        "status": event_status_name(receipt.status),
        "edge_event_id": receipt.edge_event_id,
        "event_id": receipt.event_id,
    })
}

fn clip_json(receipt: &ClipReceipt) -> Value {
    json!({
        "kind": "ClipReceipt",
        "clip_id": receipt.clip_id,
        "state": clip_state_name(receipt.state),
        "state_version": receipt.state_version,
        "sha256": receipt.sha256,
        "size_bytes": receipt.size_bytes,
    })
}

fn failure_json(failure: &DeliveryFailure) -> Value {
    assert_eq!(
        failure.transport_error, None,
        "no row is a transport failure"
    );
    json!({
        "kind": "failure",
        "code": failure.code,
        "disposition": disposition_name(failure.disposition),
        "status_code": failure.status_code,
        "retry_after_seconds": failure.retry_after_seconds,
    })
}

fn row_str<'a>(row: &'a Value, key: &str) -> &'a str {
    row[key].as_str().expect("row field is a string")
}

#[test]
fn dispositions_match_python() {
    let golden = read_json("r/relay-dispositions.json");
    assert_eq!(golden["injected_now_utc"], "2026-09-29T12:00:00+00:00");
    let clock = FixedClock::injected();
    let rows = golden["rows"].as_array().expect("rows is an array");
    assert_eq!(rows.len(), 18);
    for row in rows {
        let name = row_str(row, "name");
        let (_, parser) = row_str(row, "parser")
            .split_once(':')
            .expect("parser is file:function");
        let response = &row["response"];
        let status = response["status"]
            .as_u64()
            .and_then(|status| u16::try_from(status).ok())
            .expect("status is an HTTP status");
        let headers: Vec<(String, String)> = response["headers"]
            .as_object()
            .expect("headers is an object")
            .iter()
            .map(|(name, value)| {
                let value = value.as_str().expect("header value is a string");
                (name.to_ascii_lowercase(), value.to_owned())
            })
            .collect();
        let body = response["body"]
            .as_str()
            .map(|text| text.as_bytes().to_vec());
        let request = &row["request"];
        let received = || {
            Ok(Response {
                status,
                headers: headers.clone(),
                body: body.clone().unwrap_or_default(),
            })
        };
        let actual = match parser {
            "parse_event_result" => {
                let expected = row_str(request, "expected_edge_event_id");
                match parse_event_result(received(), expected, &clock) {
                    Ok(receipt) => event_json(&receipt),
                    Err(failure) => failure_json(&failure),
                }
            }
            "parse_clip_result" => {
                let expected = row_str(request, "expected_clip_id");
                let version = request["expected_state_version"]
                    .as_i64()
                    .expect("expected_state_version is an integer");
                match parse_clip_result(received(), expected, version, &clock) {
                    Ok(receipt) => clip_json(&receipt),
                    Err(failure) => failure_json(&failure),
                }
            }
            "classify_http_failure" => failure_json(&classify_http_failure(
                status,
                &headers,
                body.as_deref(),
                &clock,
            )),
            other => panic!("row {name} names an unknown parser {other}"),
        };
        assert_eq!(actual, row["result"], "row {name}");
    }

    let direct = golden["retry_after_direct"]
        .as_object()
        .expect("retry_after_direct is an object");
    assert_eq!(direct.len(), 8);
    for (value, expected) in direct {
        let actual =
            retry_after(Some(value.as_str()), clock.wall()).map_or(Value::Null, |s| json!(s));
        assert_eq!(&actual, expected, "Retry-After {value}");
    }
}

/// What the loopback server saw: request line parts, headers in wire order
/// and the body when a `Content-Length` announced one.
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

fn invalid(error: impl std::error::Error + Send + Sync + 'static) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn parse_head(buffer: &[u8]) -> io::Result<Option<Head>> {
    let mut slots = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut slots);
    let httparse::Status::Complete(length) = request.parse(buffer).map_err(invalid)? else {
        return Ok(None);
    };
    Ok(Some(Head {
        length,
        method: request.method.unwrap_or_default().to_owned(),
        target: request.path.unwrap_or_default().to_owned(),
        headers: request
            .headers
            .iter()
            .map(|header| (header.name.to_owned(), header.value.to_vec()))
            .collect(),
    }))
}

fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> io::Result<()> {
    let mut chunk = [0_u8; 4096];
    match stream.read(&mut chunk)? {
        0 => Err(io::ErrorKind::UnexpectedEof.into()),
        read => {
            buffer.extend_from_slice(&chunk[..read]);
            Ok(())
        }
    }
}

fn read_request(stream: &mut TcpStream) -> io::Result<Recorded> {
    let mut buffer = Vec::new();
    let head = loop {
        read_more(stream, &mut buffer)?;
        if let Some(head) = parse_head(&buffer)? {
            break head;
        }
    };
    let header = |wanted: &str| {
        head.headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(wanted))
            .map(|(_, value)| value.clone())
    };
    if header("transfer-encoding").is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "chunked body"));
    }
    let body = match header("content-length") {
        None => None,
        Some(value) => {
            let text = String::from_utf8(value).map_err(invalid)?;
            let size: usize = text.trim().parse().map_err(invalid)?;
            while buffer.len() < head.length + size {
                read_more(stream, &mut buffer)?;
            }
            Some(buffer[head.length..head.length + size].to_vec())
        }
    };
    Ok(Recorded {
        method: head.method,
        target: head.target,
        headers: head.headers,
        body,
    })
}

/// Accepts one connection, records its request and answers with `reply`.
fn serve_once(
    listener: TcpListener,
    status_line: &'static str,
    reply: Vec<u8>,
) -> JoinHandle<io::Result<Recorded>> {
    thread::spawn(move || {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(SAFETY_TIMEOUT))?;
        let recorded = read_request(&mut stream)?;
        let head = format!(
            "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            reply.len()
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(&reply)?;
        stream.flush()?;
        Ok(recorded)
    })
}

fn loopback() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
    let address = listener.local_addr().expect("loopback address");
    (listener, format!("http://{address}"))
}

/// `X-Edge-Relay-Token` from `x-edge-relay-token`: the golden header spelling.
fn title_case(name: &str) -> String {
    name.split('-')
        .map(|part| {
            let mut chars = part.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join("-")
}

/// The manifest `transport` shape of one observed request.
fn transport_record(recorded: &Recorded, timeout: Duration) -> Value {
    let (path, query) = recorded
        .target
        .split_once('?')
        .unwrap_or((recorded.target.as_str(), ""));
    let mut headers = Map::new();
    for (name, value) in &recorded.headers {
        let lower = name.to_ascii_lowercase();
        if TRANSPORT_OWNED.contains(&lower.as_str()) {
            continue;
        }
        let value = String::from_utf8(value.clone()).expect("header value is UTF-8");
        let previous = headers.insert(title_case(&lower), Value::String(value));
        assert!(previous.is_none(), "header {lower} sent twice");
    }
    let body_sha256 = recorded.body.as_deref().map_or(Value::Null, |body| {
        let digest = Sha256::digest(body);
        Value::String(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    });
    json!({
        "body_sha256": body_sha256,
        "headers": headers,
        "method": recorded.method,
        "path": path,
        "query": query,
        "timeout_sec": timeout.as_secs_f64(),
    })
}

#[test]
fn capabilities_request_shape_matches_golden() {
    let manifest = read_json("manifest.json");
    let camera_id = manifest["fixed_inputs"]["camera_id"]
        .as_str()
        .expect("fixed camera id");
    let reply = read_bytes("r/capabilities.response.json");
    let (listener, base) = loopback();
    let server = serve_once(listener, "200 OK", reply.clone());
    let client = RelayClient::new(&base, RELAY_TOKEN, ALERT_DELIVERY_TIMEOUT).expect("config");
    let response = client
        .get_capabilities(camera_id)
        .expect("loopback relay answers");
    let recorded = server.join().expect("server thread").expect("server io");
    assert_eq!(response.status, 200);
    assert_eq!(response.body, reply);

    let entry = golden_entry(&manifest, "r/capabilities.response.json");
    let observed = transport_record(&recorded, client.timeout());
    assert_eq!(
        drop_normalised(observed.clone(), &entry),
        drop_normalised(read_json("r/capabilities.request.json"), &entry)
    );
    assert_eq!(
        drop_normalised(observed, &entry),
        drop_normalised(entry["transport"].clone(), &entry)
    );
}

#[test]
fn alert_post_shape_matches_golden() {
    let manifest = read_json("manifest.json");
    let alert = read_json("r/alert.json");
    let body = serde_json::to_vec(&alert).expect("alert encodes");
    let reply = read_bytes("r/alert.response.json");
    let (listener, base) = loopback();
    let server = serve_once(listener, "202 Accepted", reply.clone());
    let client = RelayClient::new(&base, RELAY_TOKEN, ALERT_DELIVERY_TIMEOUT).expect("config");
    let response = client.post_alert(&body).expect("loopback relay answers");
    let recorded = server.join().expect("server thread").expect("server io");

    let entry = golden_entry(&manifest, "r/alert.json");
    assert_eq!(
        drop_normalised(transport_record(&recorded, client.timeout()), &entry),
        drop_normalised(entry["transport"].clone(), &entry)
    );
    let sent: Value =
        serde_json::from_slice(recorded.body.as_deref().expect("POST has a body")).expect("JSON");
    assert_eq!(
        drop_normalised(sent, &entry),
        drop_normalised(alert.clone(), &entry)
    );

    let edge_event_id = alert["edge_event_id"].as_str().expect("edge_event_id");
    let receipt = parse_event_result(Ok(response), edge_event_id, &FixedClock::injected())
        .expect("matched 202 is an ack");
    let answered = read_json("r/alert.response.json");
    assert_eq!(receipt.status, EventStatus::Accepted);
    assert_eq!(receipt.edge_event_id, edge_event_id);
    assert_eq!(receipt.event_id, answered["event_id"]);
}

#[test]
fn deadline_is_enforced() {
    const TIMEOUT: Duration = Duration::from_millis(800);
    const MARGIN: Duration = Duration::from_millis(400);
    let (listener, base) = loopback();
    let server = thread::spawn(move || -> io::Result<(bool, usize)> {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(SAFETY_TIMEOUT))?;
        let mut received = 0;
        let mut chunk = [0_u8; 4096];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => return Ok((true, received)),
                Ok(read) => received += read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
    });
    let client = RelayClient::new(&base, RELAY_TOKEN, TIMEOUT).expect("config");
    let clock = SystemClock::new();
    let started = clock.monotonic();
    let error = client
        .get_capabilities("camera-deadline")
        .expect_err("a silent relay times out");
    let elapsed = clock.monotonic() - started;

    assert!(error.is_timeout(), "transport error kind {}", error.kind());
    let failure = DeliveryFailure::from(error);
    assert_eq!(failure.disposition, DeliveryDisposition::Retry);
    assert_eq!(failure.code, "NETWORK");
    assert_eq!(failure.status_code, None);
    assert_eq!(failure.retry_after_seconds, None);
    assert!(failure.transport_error.is_some());
    assert!(
        (TIMEOUT..=TIMEOUT + MARGIN).contains(&elapsed),
        "elapsed {elapsed:?}"
    );

    drop(client);
    let (saw_eof, received) = server.join().expect("server thread").expect("server io");
    assert!(saw_eof);
    assert!(received > 0, "the request reached the server");
}
