//! T18: a relay outage is not fatal to the telemetry loops (§2.3). A
//! loopback relay answers 503 and then refuses connections; the loop stays
//! alive, counts every failed tick, and the next status the relay reads
//! carries the missed attempts in `seq` and keeps the golden key set.
//!
//! Oracles: `r/runtime-status-steady.json` (key set), `r/heartbeat.json`
//! (body), `r/runtime-status.response.json` (the relay answer) and the
//! manifest `transport.path` of each golden. Python `_post` raises `seq`
//! before every attempt, so an accepted status after `n` failed ticks has
//! `seq == n + 1`.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

use serde_json::Value;

use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::relay::RelayClient;
use seeon_ml_worker::seam::{Clock, SystemClock};
use seeon_ml_worker::telemetry::gpu::GpuStatus;
use seeon_ml_worker::telemetry::heartbeat::{Heartbeat, HeartbeatSender};
use seeon_ml_worker::telemetry::status::{
    CameraStatus, ClipExportStatus, ClipRecorderStatus, DecodeStatus, DetectionStatus,
    FacilityStatus, StatusSender, WorkerStatus,
};
use seeon_ml_worker::telemetry::{LoopHandle, Publish, Schedule, spawn};

const RELAY_TOKEN: &str = "relay-token";
const SAFETY_TIMEOUT: Duration = Duration::from_secs(30);
/// Failed ticks the loop must survive before the relay comes back: the 503
/// and at least two refused connections.
const FAILED_TICKS: u64 = 3;
/// No pacing, so the test waits on the loop and never on a timer.
const IMMEDIATE: Schedule = Schedule {
    interval: Duration::ZERO,
    initial_backoff: Duration::ZERO,
    max_backoff: Duration::ZERO,
};

fn worker_wire(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(relative)
}

fn golden_bytes(relative: &str) -> Vec<u8> {
    fs::read(worker_wire(relative)).expect("golden is readable")
}

fn golden(relative: &str) -> Value {
    serde_json::from_slice(&golden_bytes(relative)).expect("golden is JSON")
}

fn transport_path(relative: &str) -> String {
    let manifest = golden("manifest.json");
    let entry = manifest["goldens"]
        .as_array()
        .expect("goldens list")
        .iter()
        .find(|entry| entry["path"] == relative)
        .unwrap_or_else(|| panic!("{relative} is in the manifest"));
    entry["transport"]["path"]
        .as_str()
        .expect("transport path")
        .to_owned()
}

/// Every key path of a JSON value; array members share one `[]` segment.
fn key_paths(value: &Value, prefix: &str, paths: &mut BTreeSet<String>) {
    match value {
        Value::Object(members) => {
            for (key, member) in members {
                let path = format!("{prefix}.{key}");
                paths.insert(path.clone());
                key_paths(member, &path, paths);
            }
        }
        Value::Array(items) => {
            for item in items {
                key_paths(item, &format!("{prefix}[]"), paths);
            }
        }
        _ => {}
    }
}

fn key_set(value: &Value) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    key_paths(value, "", &mut paths);
    paths
}

struct Request {
    target: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> io::Result<Request> {
    let invalid = |error: &str| io::Error::new(io::ErrorKind::InvalidData, error.to_owned());
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let mut slots = [httparse::EMPTY_HEADER; 32];
        let mut request = httparse::Request::new(&mut slots);
        if let httparse::Status::Complete(head) =
            request.parse(&buffer).map_err(|_| invalid("bad head"))?
        {
            let length = request
                .headers
                .iter()
                .find(|header| header.name.eq_ignore_ascii_case("content-length"))
                .and_then(|header| std::str::from_utf8(header.value).ok())
                .and_then(|value| value.trim().parse::<usize>().ok())
                .ok_or_else(|| invalid("no content-length"))?;
            let target = request.path.ok_or_else(|| invalid("no target"))?.to_owned();
            while buffer.len() < head + length {
                let read = stream.read(&mut chunk)?;
                if read == 0 {
                    return Err(invalid("peer closed mid-body"));
                }
                buffer.extend_from_slice(&chunk[..read]);
            }
            let body = buffer[head..head + length].to_vec();
            return Ok(Request { target, body });
        }
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(invalid("peer closed mid-head"));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn reply(stream: &mut TcpStream, status: u16, body: &[u8]) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} Relay\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// Answers one request with 503; the listener closes when this returns, so
/// every later connection is refused.
fn serve_503_once(listener: TcpListener) -> io::Result<Request> {
    let (mut stream, _) = listener.accept()?;
    stream.set_read_timeout(Some(SAFETY_TIMEOUT))?;
    let request = read_request(&mut stream)?;
    reply(&mut stream, 503, b"{\"detail\":\"relay unavailable\"}")?;
    Ok(request)
}

/// Accepts the golden answer for every request until a connection closes
/// without a byte. The first request is held until `gate` opens, so the
/// loop's counters are read while that attempt is still in flight.
fn serve_accepting(
    listener: TcpListener,
    requests: Sender<Request>,
    gate: Receiver<()>,
) -> io::Result<usize> {
    let answer = golden_bytes("r/runtime-status.response.json");
    let mut served = 0;
    loop {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(SAFETY_TIMEOUT))?;
        if stream.peek(&mut [0_u8; 1])? == 0 {
            return Ok(served);
        }
        let request = read_request(&mut stream)?;
        let _ = requests.send(request);
        if served == 0 {
            gate.recv_timeout(SAFETY_TIMEOUT)
                .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "gate never opened"))?;
        }
        reply(&mut stream, 200, &answer)?;
        served += 1;
    }
}

/// What the relay saw across the outage and its recovery.
struct Outage {
    refused_request: Request,
    first_recovered: Request,
    second_recovered: Request,
    /// `(attempts, failures, successes)` while the first recovered attempt
    /// is in flight.
    at_recovery: (u64, u64, u64),
    alive_after_outage: bool,
    alive_after_recovery: bool,
    successes_after_recovery: u64,
    clean_stop: bool,
}

fn run_outage<P: Publish>(name: &str, publisher: impl FnOnce(RelayClient) -> P) -> Outage {
    let listener = TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let address: SocketAddr = listener.local_addr().expect("bound address");
    let client = RelayClient::new(&format!("http://{address}"), RELAY_TOKEN, SAFETY_TIMEOUT)
        .expect("client config");
    let outage = thread::spawn(move || serve_503_once(listener));
    let handle: LoopHandle = spawn(name, publisher(client), IMMEDIATE).expect("loop starts");
    let refused_request = outage
        .join()
        .expect("503 server thread")
        .expect("503 served");

    let clock = SystemClock::new();
    let deadline = clock.monotonic() + SAFETY_TIMEOUT;
    poll_until(&clock, deadline, "failed telemetry ticks", || {
        handle.failures() >= FAILED_TICKS || !handle.is_alive()
    })
    .expect("the loop records failed ticks before the deadline");
    let alive_after_outage = handle.is_alive();

    let listener = TcpListener::bind(address).expect("relay rebinds its port");
    let (requests_tx, requests) = mpsc::channel();
    let (gate, gate_rx) = mpsc::channel();
    let recovery = thread::spawn(move || serve_accepting(listener, requests_tx, gate_rx));
    let first_recovered = requests
        .recv_timeout(SAFETY_TIMEOUT)
        .expect("the loop reaches the recovered relay");
    let at_recovery = (handle.attempts(), handle.failures(), handle.successes());
    gate.send(()).expect("gate opens");
    let second_recovered = requests
        .recv_timeout(SAFETY_TIMEOUT)
        .expect("the loop keeps publishing after recovery");
    let alive_after_recovery = handle.is_alive();
    let successes_after_recovery = handle.successes();
    let clean_stop = handle.stop();
    drop(TcpStream::connect(address).expect("stop connection"));
    recovery
        .join()
        .expect("recovery server thread")
        .expect("recovery served");
    Outage {
        refused_request,
        first_recovered,
        second_recovered,
        at_recovery,
        alive_after_outage,
        alive_after_recovery,
        successes_after_recovery,
        clean_stop,
    }
}

fn body_json(request: &Request) -> Value {
    serde_json::from_slice(&request.body).expect("request body is JSON")
}

fn steady_status() -> FacilityStatus {
    FacilityStatus {
        facility_id: "facility-1".into(),
        cameras: vec![CameraStatus {
            camera_id: "cmsnw6rjc01vhlh01oswn99yq".into(),
            decode: DecodeStatus {
                requested: "auto".into(),
                selected: Some("nvdec".into()),
                fallback_count: 0,
                last_reason: None,
                updated_at_sec: 1_787_000_000.0,
            },
            measured_fps: None,
            detection: Some(DetectionStatus {
                expected: false,
                inference_admitted: 0,
                inference_succeeded: 0,
                inference_overwritten: 0,
                decision_completed: 0,
            }),
        }],
        clip_recorder: ClipRecorderStatus {
            available: true,
            dropped_frames: Some(0),
            dropped_events: Some(0),
            failed_writes: Some(0),
            finalized_clips: Some(4),
            video_unavailable_clips: Some(1),
            active_clips: Some(0),
            encoder: Some("h264_nvenc".into()),
        },
        clip_export: ClipExportStatus {
            enabled: true,
            version: 2,
        },
        gpu: Some(GpuStatus {
            nvml_available: true,
            cuda_context_ok: true,
            driver_version: Some("580.65.06".into()),
            device_name: Some("NVIDIA GeForce RTX 5070 Ti".into()),
            captured_at_sec: 1_787_000_000.0,
            nvml_error: None,
        }),
        worker: Some(WorkerStatus::current(true, 1_786_999_940.0, None)),
        delivery_queue: None,
    }
}

#[test]
fn status_loop_survives_outage_and_reports_missed_attempts_in_seq() {
    let status = steady_status();
    let outage = run_outage("runtime-status", move |client| {
        StatusSender::new(client, move || vec![status.clone()])
    });
    let path = transport_path("r/runtime-status-steady.json");
    assert_eq!(outage.refused_request.target, path);
    assert!(
        outage.alive_after_outage,
        "the loop survives 503 and refusals"
    );

    let (attempts, failures, successes) = outage.at_recovery;
    assert!(failures >= FAILED_TICKS);
    assert_eq!(successes, 0);
    assert_eq!(attempts, failures + 1);
    let first = body_json(&outage.first_recovered);
    assert_eq!(outage.first_recovered.target, path);
    assert_eq!(first["seq"], Value::from(failures + 1));
    assert_eq!(first["generation"], Value::Null);
    assert_eq!(
        key_set(&first),
        key_set(&golden("r/runtime-status-steady.json"))
    );

    let accepted = golden("r/runtime-status.response.json");
    let second = body_json(&outage.second_recovered);
    assert_eq!(second["seq"], Value::from(failures + 2));
    assert_eq!(second["generation"], accepted["generation"]);
    assert!(outage.alive_after_recovery);
    assert!(outage.successes_after_recovery >= 1);
    assert!(outage.clean_stop, "the loop thread ends without a panic");
}

#[test]
fn heartbeat_loop_survives_outage_and_resumes_the_golden_body() {
    let heartbeat = Heartbeat {
        camera_id: "cmsnw6rjc01vhlh01oswn99yq".into(),
        facility_id: "facility-1".into(),
        config_version: 7,
    };
    let outage = run_outage("heartbeat", move |client| {
        HeartbeatSender::new(client, move || vec![heartbeat.clone()])
    });
    let path = transport_path("r/heartbeat.json");
    assert_eq!(outage.refused_request.target, path);
    assert!(
        outage.alive_after_outage,
        "the loop survives 503 and refusals"
    );

    let (attempts, failures, successes) = outage.at_recovery;
    assert!(failures >= FAILED_TICKS);
    assert_eq!(successes, 0);
    assert_eq!(attempts, failures + 1);
    assert_eq!(outage.first_recovered.target, path);
    assert_eq!(
        body_json(&outage.first_recovered),
        golden("r/heartbeat.json")
    );
    assert_eq!(
        body_json(&outage.second_recovered),
        golden("r/heartbeat.json")
    );
    assert!(outage.alive_after_recovery);
    assert!(outage.successes_after_recovery >= 1);
    assert!(outage.clean_stop, "the loop thread ends without a panic");
}
