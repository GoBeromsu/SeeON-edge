//! T31 and the 1 MiB cap: execution-record lanes and exporter parity with
//! Python `worker/pipeline/diagnostics/{lanes,exporter}.py`. The oracle is
//! the reviewed recorded run `r/execution-records-overflow.json`; the
//! transport is a loopback httparse server and time is a fake `Clock`.
//! Composition parity with `compose_execution_records` is checked against the
//! reviewed `d/env-refusals.json` `compose_cases`.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_ml_worker::config::env::{Env, EnvKind, ExecutionRecordsSettings};
use seeon_ml_worker::json::Json;
use seeon_ml_worker::records::builder::{
    Counters, FallScore, Frame, Stream, model_score_record, policy_consume_record,
};
use seeon_ml_worker::records::id::canonical;
use seeon_ml_worker::records::{
    ComposeError, Composed, Drained, Exporter, ExporterError, Identities, Lanes, LanesError,
    MAX_BODY_BYTES, Provenance, ProvenanceError, RELAY_TIMEOUT, Receipt, Record, RecordBody,
    compose,
};
use seeon_ml_worker::relay::RelayClient;
use seeon_ml_worker::seam::Clock;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CAMERA: &str = "cmsnw6rjc01vhlh01oswn99yq";
const BOOT: &str = "boot-0001";
/// Recipe `observed_at_ns` base: wall clock 1787000000.0 s.
const WALL_NS: u64 = 1_787_000_000_000_000_000;
const FRAME_NS: u64 = 33_333_333;
/// The golden header value is the placeholder itself; no real token exists.
const RELAY_TOKEN: &str = "<relay-token>";
const SAFETY_TIMEOUT: Duration = Duration::from_secs(30);

fn golden() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/r/execution-records-overflow.json");
    let bytes = std::fs::read(path).expect("golden file is readable");
    serde_json::from_slice(&bytes).expect("golden file is JSON")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn number(value: &Value) -> u64 {
    value.as_u64().expect("golden unsigned integer")
}

fn text(value: &Value) -> &str {
    value.as_str().expect("golden string")
}

fn items(value: &Value) -> &Vec<Value> {
    value.as_array().expect("golden array")
}

/// Recipe provenance: abc123, sha256:deadbeef, model-1, cal-1,
/// pose-bbox56/v1, cfg-1, fall.policy:2.
fn provenance() -> Provenance {
    Provenance {
        worker_build_revision: "abc123".to_owned(),
        worker_image_digest: "sha256:deadbeef".to_owned(),
        model_digest: "model-1".to_owned(),
        calibration_digest: "cal-1".to_owned(),
        preprocessing_identity: "pose-bbox56/v1".to_owned(),
        config_digest: "cfg-1".to_owned(),
        policy_identity: "fall.policy:2".to_owned(),
    }
}

fn frame(index: u64) -> Frame {
    let offset = i64::try_from(index * FRAME_NS).expect("small offset");
    Frame {
        frame_seq: 100 + index,
        source_pts_ns: Some(4_000_000_000 + offset),
    }
}

fn stream(worker_boot_id: &str, stream_epoch: u64) -> Stream {
    Stream {
        camera_id: CAMERA.to_owned(),
        worker_boot_id: worker_boot_id.to_owned(),
        source_generation: 1,
        stream_epoch,
    }
}

fn model(index: u64, stream_epoch: u64, worker_boot_id: &str) -> Record {
    let score = FallScore {
        track_id: 1,
        generation: Some(3),
        fall_transition: 0.25,
        background: 0.5,
        fallen: 0.25,
        evidence: None,
    };
    let stream = stream(worker_boot_id, stream_epoch);
    let observed = WALL_NS + index * FRAME_NS;
    model_score_record(&stream, frame(index), observed, &score, None).expect("recipe record")
}

fn policy(index: u64, pad: Option<usize>) -> Record {
    let before = Counters {
        accepted: 0,
        overwritten: 0,
        late: 0,
    };
    let after = Counters {
        accepted: 1,
        ..before
    };
    let observed = WALL_NS + index * FRAME_NS + 1_000;
    let record = policy_consume_record(
        &stream(BOOT, 1),
        frame(index),
        observed,
        before,
        after,
        index,
    )
    .expect("recipe record");
    let Some(pad) = pad else {
        return record;
    };
    let mut body: RecordBody = record.body().clone();
    body.payload
        .push(("pad".to_owned(), Json::Str("x".repeat(pad))));
    Record::new(body).expect("padded recipe record")
}

/// Decodes a golden input label; `None` for the Python-only non-record input.
fn recipe(label: &str) -> Option<Record> {
    let words: Vec<&str> = label.split(' ').collect();
    let index = |word: &str| -> u64 { word.trim_start_matches('#').parse().expect("index") };
    match words.as_slice() {
        ["model", n] => Some(model(index(n), 1, BOOT)),
        ["model", n, "epoch", e] => Some(model(index(n), e.parse().expect("epoch"), BOOT)),
        ["model", n, boot] if boot.starts_with("boot-") => Some(model(index(n), 1, boot)),
        ["policy", n] => Some(policy(index(n), None)),
        ["policy", n, "pad", p] => Some(policy(index(n), Some(p.parse().expect("pad")))),
        ["not", "a", "WireRecord", ..] => None,
        _ => panic!("unknown recipe label {label}"),
    }
}

/// Emits each golden input and checks the `try_emit` result and queue depth.
fn emit_all(lanes: &Lanes, results: &Value) -> usize {
    let mut emitted = 0;
    for step in items(results) {
        let label = text(&step["input"]);
        let Some(record) = recipe(label) else {
            continue;
        };
        let accepted = lanes.try_emit(record);
        assert_eq!(Value::Bool(accepted), step["try_emit"], "try_emit {label}");
        let queued = u64::try_from(lanes.queued()).expect("small queue");
        assert_eq!(
            queued,
            number(&step["queued_after"]),
            "queued after {label}"
        );
        emitted += 1;
    }
    emitted
}

fn json_value(value: &Json) -> Value {
    serde_json::from_str(&canonical(value).expect("wire value encodes")).expect("canonical JSON")
}

fn drained_value(drained: Option<Drained>) -> Value {
    let Some(drained) = drained else {
        return Value::Null;
    };
    json!({
        "camera_id": drained.camera_id,
        "worker_boot_id": drained.worker_boot_id,
        "records": drained.records.iter().map(|r| json_value(&r.to_json())).collect::<Vec<_>>(),
        "gaps": drained.gaps.iter().map(|g| json_value(&g.to_json())).collect::<Vec<_>>(),
    })
}

/// `drain_for(<boot>, limit=<n>)` with an optional trailing ` again`.
fn drain_call(step: &str) -> (&str, u64) {
    let inner = step
        .strip_prefix("drain_for(")
        .and_then(|rest| rest.split_once(')'))
        .map(|(inner, _)| inner)
        .expect("drain_for step");
    let (boot, limit) = inner.split_once(", limit=").expect("drain_for arguments");
    (boot, limit.parse().expect("drain limit"))
}

#[test]
fn lanes_match_golden_steps() {
    let golden = golden();
    let lanes_golden = &golden["lanes"];
    let lanes = Lanes::new(number(&lanes_golden["lane_capacity"])).expect("golden capacity");
    let mut drains = 0;
    for step in items(&lanes_golden["steps"]) {
        let name = text(&step["step"]);
        if name == "emit" {
            emit_all(&lanes, &step["results"]);
        } else if name == "cameras_with_work" {
            let expected: Vec<(String, String)> = items(&step["result"])
                .iter()
                .map(|pair| (text(&pair[0]).to_owned(), text(&pair[1]).to_owned()))
                .collect();
            assert_eq!(lanes.cameras_with_work(), expected);
        } else {
            let (boot, limit) = drain_call(name);
            let drained = lanes
                .drain_for(CAMERA, boot, limit)
                .expect("positive limit");
            assert_eq!(drained_value(drained), step["result"], "{name}");
            drains += 1;
        }
    }
    assert_eq!(drains, 4, "every golden drain step ran");
}

/// Monotonic time moves only when a wait pauses; wall time is the recipe's.
struct FakeClock {
    now: Mutex<Duration>,
}

impl FakeClock {
    fn new() -> Self {
        Self {
            now: Mutex::new(Duration::ZERO),
        }
    }
}

impl Clock for FakeClock {
    fn monotonic(&self) -> Duration {
        *self.now.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wall(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_nanos(WALL_NS)
    }

    fn pause(&self, limit: Duration) {
        *self.now.lock().unwrap_or_else(PoisonError::into_inner) += limit;
    }
}

/// What the loopback server saw for one POST.
struct Recorded {
    method: String,
    target: String,
    headers: Vec<(String, Vec<u8>)>,
    body: Vec<u8>,
}

impl Recorded {
    fn header(&self, wanted: &str) -> Option<&[u8]> {
        self.headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(wanted))
            .map(|(_, value)| value.as_slice())
    }
}

fn invalid(error: impl std::error::Error + Send + Sync + 'static) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> io::Result<()> {
    let mut chunk = [0_u8; 65_536];
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
    let (length, method, target, headers) = loop {
        read_more(stream, &mut buffer)?;
        let mut slots = [httparse::EMPTY_HEADER; 64];
        let mut request = httparse::Request::new(&mut slots);
        if let httparse::Status::Complete(length) = request.parse(&buffer).map_err(invalid)? {
            let headers: Vec<(String, Vec<u8>)> = request
                .headers
                .iter()
                .map(|header| (header.name.to_owned(), header.value.to_vec()))
                .collect();
            let method = request.method.unwrap_or_default().to_owned();
            let target = request.path.unwrap_or_default().to_owned();
            break (length, method, target, headers);
        }
    };
    let mut recorded = Recorded {
        method,
        target,
        headers,
        body: Vec::new(),
    };
    if recorded.header("transfer-encoding").is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "chunked body"));
    }
    let size = recorded.header("content-length").unwrap_or(b"0");
    let size: usize = std::str::from_utf8(size)
        .map_err(invalid)?
        .trim()
        .parse()
        .map_err(invalid)?;
    while buffer.len() < length + size {
        read_more(stream, &mut buffer)?;
    }
    recorded.body = buffer[length..length + size].to_vec();
    Ok(recorded)
}

/// Serves one connection per reply, in order, recording each request before
/// answering, so every POST of a finished flush is already in the channel.
/// After the last reply the listener closes and further POSTs fail.
fn serve(replies: Vec<Vec<u8>>) -> (String, Receiver<Recorded>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
    let base = format!(
        "http://{}",
        listener.local_addr().expect("loopback address")
    );
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || -> io::Result<()> {
        for reply in replies {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(SAFETY_TIMEOUT))?;
            let recorded = read_request(&mut stream)?;
            if sender.send(recorded).is_err() {
                return Ok(());
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                reply.len()
            );
            stream.write_all(head.as_bytes())?;
            stream.write_all(&reply)?;
            stream.flush()?;
        }
        Ok(())
    });
    (base, receiver)
}

fn exporter(lanes: &Arc<Lanes>, base: &str, batch_max: u64, flush_ms: u64) -> Exporter {
    let client = RelayClient::new(base, RELAY_TOKEN, RELAY_TIMEOUT).expect("relay config");
    Exporter::new(Arc::clone(lanes), client, provenance(), batch_max, flush_ms)
        .expect("golden settings")
}

fn failure_pairs(exporter: &Exporter) -> Value {
    exporter
        .failures()
        .iter()
        .map(|failure| json!([failure.disposition.as_str(), failure.code]))
        .collect()
}

#[test]
fn exporter_run_posts_match_golden() {
    let golden = golden();
    let run = &golden["exporter_run"];
    let replies = items(&run["receipts"])
        .iter()
        .map(|receipt| serde_json::to_vec(receipt).expect("receipt JSON"))
        .collect();
    let (base, posted) = serve(replies);
    let lanes = Arc::new(Lanes::new(number(&run["lane_capacity"])).expect("golden capacity"));
    let flush_ms = number(&run["flush_ms"]);
    let mut exporter = exporter(&lanes, &base, number(&run["batch_max"]), flush_ms);
    let clock = FakeClock::new();
    let waits = items(&run["waits"]);
    let flushes = items(&run["flushes"]);
    assert_eq!(waits.len(), flushes.len(), "one flush per wait");
    for (wait, flush) in waits.iter().zip(flushes) {
        assert_eq!(flush["after_wait"], wait["wait"]);
        emit_all(&lanes, &wait["emitted_before_wait"]);
        let before = clock.monotonic();
        let ready = exporter.wait_for_work(&clock);
        assert_eq!(
            Value::Bool(ready),
            wait["predicate"],
            "wait {}",
            wait["wait"]
        );
        let waited = clock.monotonic() - before;
        let expected = match text(&wait["outcome"]) {
            "returns-immediately" => Duration::ZERO,
            "times-out-after-timeout_sec" => Duration::from_millis(flush_ms),
            other => panic!("unknown wait outcome {other}"),
        };
        assert_eq!(waited, expected, "wait {} on the fake clock", wait["wait"]);
        exporter.flush_once(&clock);
        let posts: Vec<Recorded> = posted.try_iter().collect();
        let golden_posts = items(&flush["posts"]);
        assert_eq!(
            posts.len(),
            golden_posts.len(),
            "posts after wait {}",
            wait["wait"]
        );
        for (post, expected) in posts.iter().zip(golden_posts) {
            let transport = &expected["transport"];
            assert_eq!(post.method, text(&transport["method"]));
            assert_eq!(post.target, text(&transport["path"]));
            for (name, value) in transport["headers"].as_object().expect("headers") {
                assert_eq!(post.header(name), Some(text(value).as_bytes()), "{name}");
            }
            let body: Value = serde_json::from_slice(&post.body).expect("posted JSON");
            assert_eq!(body, expected["body"]);
            assert_eq!(
                post.body.len(),
                usize::try_from(number(&expected["body_bytes"])).expect("size")
            );
            assert_eq!(sha256_hex(&post.body), text(&transport["body_sha256"]));
        }
    }
    let receipts: Vec<Receipt> = items(&run["receipts"])
        .iter()
        .map(|receipt| Receipt::from_json(receipt).expect("golden receipt"))
        .collect();
    assert_eq!(exporter.receipts(), receipts);
    assert_eq!(failure_pairs(&exporter), run["failures"]);
    assert_eq!(lanes.queued(), 0);
}

/// The byte-cap post summary the golden records for each captured POST.
fn post_summary(body: &[u8]) -> Value {
    let value: Value = serde_json::from_slice(body).expect("posted JSON");
    let records: Vec<Value> = items(&value["records"])
        .iter()
        .map(|record| {
            let pad = record["payload"]["pad"].as_str().map_or(0, str::len);
            json!({
                "pad_chars": pad,
                "producer": record["producer"],
                "producer_sequence": record["producer_sequence"],
                "record_id": record["record_id"],
            })
        })
        .collect();
    json!({
        "batch_id": value["batch_id"],
        "body_bytes": body.len(),
        "body_sha256": sha256_hex(body),
        "gaps": value["gaps"],
        "records": records,
        "within_cap": body.len() <= MAX_BODY_BYTES,
    })
}

#[test]
fn byte_cap_never_posts_over_one_mebibyte() {
    let golden = golden();
    let cap = &golden["byte_cap"];
    assert_eq!(
        number(&cap["max_execution_record_body_bytes"]),
        MAX_BODY_BYTES as u64
    );
    let golden_posts = items(&cap["posts"]);
    let replies = golden_posts
        .iter()
        .enumerate()
        .map(|(index, post)| {
            let receipt = json!({
                "accepted": items(&post["records"]).len(),
                "batch_id": post["batch_id"],
                "committed_at_ns": WALL_NS + 1_000_000 * (index as u64 + 1),
                "duplicates": 0,
                "rejected": [],
                "storage_state": "committed",
            });
            serde_json::to_vec(&receipt).expect("receipt JSON")
        })
        .collect();
    let (base, posted) = serve(replies);
    let lanes = Arc::new(Lanes::new(number(&cap["lane_capacity"])).expect("golden capacity"));
    let mut exporter = exporter(&lanes, &base, number(&cap["batch_max"]), 500);
    let emitted = emit_all(&lanes, &cap["emit"]);
    assert_eq!(emitted, 4);
    exporter.flush_once(&FakeClock::new());
    let posts: Vec<Recorded> = posted.try_iter().collect();
    for post in &posts {
        assert!(
            post.body.len() <= MAX_BODY_BYTES,
            "a POST exceeded the byte cap"
        );
    }
    let summaries: Vec<Value> = posts.iter().map(|post| post_summary(&post.body)).collect();
    let expected: Vec<Value> = golden_posts
        .iter()
        .map(|post| {
            let mut post = post.clone();
            post.as_object_mut().expect("post").remove("backend_parser");
            post
        })
        .collect();
    assert_eq!(summaries, expected);
    assert_eq!(failure_pairs(&exporter), cap["failures"]);
    assert_eq!(
        lanes.queued(),
        usize::try_from(number(&cap["queued_after"])).expect("small")
    );
}

/// A typed refusal, or a negative argument the unsigned Rust type rejects.
#[derive(Debug, PartialEq)]
enum Refusal {
    Negative,
    Lanes(LanesError),
    Exporter(ExporterError),
}

fn argument(case: &str, key: &str) -> Result<u64, Refusal> {
    let start = case.find(&format!("{key}=")).expect("argument present") + key.len() + 1;
    let digits: String = case[start..]
        .chars()
        .take_while(|c| *c == '-' || c.is_ascii_digit())
        .collect();
    let value: i64 = digits.parse().expect("integer argument");
    u64::try_from(value).map_err(|_| Refusal::Negative)
}

fn refusal_of(case: &str) -> Result<(), Refusal> {
    if case.starts_with("ExecutionRecordLanes(") {
        let capacity = argument(case, "lane_capacity")?;
        return Lanes::new(capacity).map(drop).map_err(Refusal::Lanes);
    }
    if case.starts_with("drain_for(") {
        let limit = argument(case, "limit")?;
        let lanes = Lanes::new(1).expect("capacity 1");
        return lanes
            .drain_for(CAMERA, BOOT, limit)
            .map(drop)
            .map_err(Refusal::Lanes);
    }
    assert!(
        case.starts_with("ExecutionRecordExporter("),
        "unknown case {case}"
    );
    let batch_max = argument(case, "batch_max")?;
    let flush_ms = argument(case, "flush_ms")?;
    let lanes = Arc::new(Lanes::new(1).expect("capacity 1"));
    let client =
        RelayClient::new("http://127.0.0.1:9", RELAY_TOKEN, RELAY_TIMEOUT).expect("config");
    Exporter::new(lanes, client, provenance(), batch_max, flush_ms)
        .map(drop)
        .map_err(Refusal::Exporter)
}

#[test]
fn refusals_are_typed() {
    let golden = golden();
    let cases = items(&golden["refusals"]);
    assert_eq!(cases.len(), 9);
    for entry in cases {
        let case = text(&entry["case"]);
        let outcome = refusal_of(case);
        let expected = match (text(&entry["verdict"]), case.contains("=-")) {
            ("accepted", _) => Ok(()),
            ("refused", true) => Err(Refusal::Negative),
            ("refused", false) if case.starts_with("ExecutionRecordLanes(") => {
                Err(Refusal::Lanes(LanesError::Capacity))
            }
            ("refused", false) if case.starts_with("drain_for(") => {
                Err(Refusal::Lanes(LanesError::DrainLimit))
            }
            ("refused", false) => Err(Refusal::Exporter(ExporterError::Settings)),
            (other, _) => panic!("unknown verdict {other}"),
        };
        assert_eq!(outcome, expected, "{case}");
    }
}

/// The goldens' provenance `config_digest`; stage 4 chooses the real source.
const CONFIG_DIGEST: &str = "cfg-1";
const RELAY_MISSING: &str = "execution records enabled but relay URL/token missing";
const PROVENANCE_MISSING: &str = "execution-record provenance missing: ";
const REQUIRED_WHEN_ENABLED: &str = " is required when ML_WORKER_EXECUTION_RECORDS_ENABLED=1";

fn compose_cases() -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire/d/env-refusals.json");
    let bytes = std::fs::read(path).expect("golden file is readable");
    let golden: Value = serde_json::from_slice(&bytes).expect("golden file is JSON");
    items(&golden["compose_cases"]).clone()
}

fn named<'a>(cases: &'a [Value], name: &str) -> &'a Value {
    cases
        .iter()
        .find(|case| case["name"] == name)
        .expect("golden compose case")
}

fn case_env(case: &Value) -> Env {
    case["env"]
        .as_object()
        .expect("env object")
        .iter()
        .map(|(key, value)| (key.clone(), text(value).to_owned()))
        .collect()
}

/// A JSON null is an absent identity; an empty string stays empty.
fn identities(case: &Value) -> Identities {
    let member = |name: &str| case["identities"][name].as_str().map(str::to_owned);
    Identities {
        build_revision: member("build_revision"),
        image_digest: member("image_digest"),
        model_digest: member("model_digest"),
        calibration_digest: member("calibration_digest"),
        preprocessing_identity: member("preprocessing_identity"),
        policy_identity: member("policy_identity"),
    }
}

fn golden_settings(value: &Value) -> Option<ExecutionRecordsSettings> {
    (!value.is_null()).then(|| ExecutionRecordsSettings {
        lane_capacity: number(&value["lane_capacity"]),
        batch_max: number(&value["batch_max"]),
        flush_ms: number(&value["flush_ms"]),
    })
}

fn compose_case(case: &Value, relay_url: &str) -> Result<Option<Composed>, ComposeError> {
    let token = text(&case["relay"]["token"]);
    compose(
        &case_env(case),
        relay_url,
        token,
        &identities(case),
        CONFIG_DIGEST,
    )
}

/// The reviewed table from Rust refusals to the Python exception classes the
/// goldens record. `Relay`, `Lanes` and `Exporter` have no golden compose
/// case; Python raises `ValueError` from the relay client and the lanes.
fn python_class(error: &ComposeError) -> &'static str {
    match error {
        ComposeError::Settings(_) => "WorkerConfigError",
        ComposeError::RelayMissing | ComposeError::Provenance(_) => {
            "ExecutionRecordProvenanceError"
        }
        ComposeError::Relay(_) | ComposeError::Lanes(_) | ComposeError::Exporter(_) => "ValueError",
    }
}

/// The structured cause of a refusal, compared field by field.
#[derive(Debug, PartialEq, Eq)]
enum Cause {
    RelayMissing,
    MissingProvenance(Vec<String>),
    Settings(EnvKind, Vec<String>),
}

/// Reads the cause out of the Python message the golden recorded.
fn golden_cause(message: &str) -> Cause {
    if message == RELAY_MISSING {
        return Cause::RelayMissing;
    }
    if let Some(names) = message.strip_prefix(PROVENANCE_MISSING) {
        return Cause::MissingProvenance(names.split(", ").map(str::to_owned).collect());
    }
    let key = message
        .strip_suffix(REQUIRED_WHEN_ENABLED)
        .unwrap_or_else(|| panic!("unmapped golden message {message:?}"));
    Cause::Settings(EnvKind::Required, vec![key.to_owned()])
}

/// Builds the cause from the refusal's typed fields only.
fn rust_cause(error: &ComposeError) -> Cause {
    match error {
        ComposeError::RelayMissing => Cause::RelayMissing,
        ComposeError::Provenance(ProvenanceError::Missing(names)) => {
            Cause::MissingProvenance(names.iter().map(|name| (*name).to_owned()).collect())
        }
        ComposeError::Settings(error) => Cause::Settings(error.kind, error.keys.clone()),
        other => panic!("no golden compose case refuses as {}", python_class(other)),
    }
}

/// An enabled composition waits `flush_ms` on the clock, shares one `Lanes`
/// between producers and the exporter, and posts the case's identities with
/// its relay token. The golden relay url is not routable, so the post check
/// composes the same case against a loopback relay.
fn check_enabled(composed: Composed, case: &Value) {
    let name = text(&case["name"]);
    let Composed {
        settings,
        lanes,
        exporter,
    } = composed;
    let clock = FakeClock::new();
    assert!(!exporter.wait_for_work(&clock), "{name}: nothing emitted");
    assert_eq!(
        clock.monotonic(),
        Duration::from_millis(settings.flush_ms),
        "{name}: an idle wait lasts flush_ms"
    );
    assert!(
        lanes.try_emit(model(0, 1, BOOT)),
        "{name}: the lane accepts"
    );
    assert!(
        exporter.wait_for_work(&clock),
        "{name}: the exporter sees the producers' lanes"
    );

    let (base, posted) = serve(vec![b"{}".to_vec()]);
    let Composed {
        lanes,
        mut exporter,
        ..
    } = compose_case(case, &base)
        .expect("the case composes on loopback")
        .expect("enabled");
    assert!(
        lanes.try_emit(model(0, 1, BOOT)),
        "{name}: the lane accepts"
    );
    exporter.flush_once(&clock);
    let post = posted.recv_timeout(SAFETY_TIMEOUT).expect("one POST");
    let token = text(&case["relay"]["token"]);
    assert_eq!(post.header("X-Edge-Relay-Token"), Some(token.as_bytes()));
    let body: Value = serde_json::from_slice(&post.body).expect("posted JSON");
    let given = &case["identities"];
    assert_eq!(
        body["provenance"],
        json!({
            "worker_build_revision": given["build_revision"],
            "worker_image_digest": given["image_digest"],
            "model_digest": given["model_digest"],
            "calibration_digest": given["calibration_digest"],
            "preprocessing_identity": given["preprocessing_identity"],
            "config_digest": CONFIG_DIGEST,
            "policy_identity": given["policy_identity"],
        }),
        "{name}: the posted provenance"
    );
}

#[test]
fn compose_cases_match_python_verdicts() {
    let cases = compose_cases();
    assert_eq!(cases.len(), 9, "the reviewed golden has nine compose cases");
    for case in &cases {
        let name = text(&case["name"]);
        let result = compose_case(case, text(&case["relay"]["url"]));
        match (text(&case["verdict"]), result) {
            ("accepted", Ok(composed)) => {
                let settings = composed.as_ref().map(|composed| composed.settings);
                assert_eq!(settings, golden_settings(&case["settings"]), "{name}");
                assert_eq!(composed.is_some(), !case["lanes"].is_null(), "{name}");
                assert_eq!(composed.is_some(), !case["exporter"].is_null(), "{name}");
                if let Some(composed) = composed {
                    check_enabled(composed, case);
                }
            }
            ("refused", Err(error)) => {
                assert_eq!(python_class(&error), text(&case["class"]), "{name}");
                let expected = golden_cause(text(&case["message"]));
                assert_eq!(rust_cause(&error), expected, "{name}");
            }
            (verdict, Err(error)) => {
                panic!(
                    "{name}: golden {verdict}, refused as {}",
                    python_class(&error)
                )
            }
            (verdict, Ok(_)) => panic!("{name}: golden {verdict}, accepted"),
        }
    }
}

/// Derived, not golden: `compose_execution_records` checks the relay url and
/// token before it builds the provenance, and no golden case has both
/// faults. The enabled settings with a blank url and the identities of
/// `provenance-missing-several` must refuse for the relay.
#[test]
fn blank_relay_refuses_before_missing_provenance() {
    let cases = compose_cases();
    let enabled = named(&cases, "compose:enabled");
    let several = named(&cases, "compose:provenance-missing-several");
    let refused = compose(
        &case_env(enabled),
        "",
        RELAY_TOKEN,
        &identities(several),
        CONFIG_DIGEST,
    );
    let cause = refused.err().map(|error| rust_cause(&error));
    assert_eq!(cause, Some(Cause::RelayMissing));
}
