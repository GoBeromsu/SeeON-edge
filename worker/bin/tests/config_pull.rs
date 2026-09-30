//! T7: the relay config pull, its last-known-good fallback, the restart
//! check and the release-identity refusal against a loopback relay.
//!
//! Oracles: the Python-recorded goldens `r/worker-config.response.json`,
//! `r/release-identity.response.json`, their manifest `transport` entries
//! and the Python-written store `d/config-lkg/`. No expected value is
//! produced by the code under test.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use seeon_ml_worker::config::lkg::Directive;
use seeon_ml_worker::config::pull::{
    CONFIG_PULL_TIMEOUT, ConfigSource, PullError, check_release_identity, poll_worker_config,
    pull_startup_config,
};
use seeon_ml_worker::config::restart::{RESTART_POLL_INTERVAL, RestartCheck};
use seeon_ml_worker::json::Json;
use seeon_ml_worker::seam::Clock;

const RELAY_TOKEN: &str = "<relay-token>";
const SAFETY_TIMEOUT: Duration = Duration::from_secs(30);
/// Headers the HTTP stack owns; the manifest records none of them.
const TRANSPORT_OWNED: [&str; 6] = [
    "host",
    "user-agent",
    "accept-encoding",
    "connection",
    "content-length",
    "transfer-encoding",
];
const WORKER_CONFIG: &str = "r/worker-config.response.json";
const RELEASE_IDENTITY: &str = "r/release-identity.response.json";
const LKG_STORE: &str = "d/config-lkg";

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
    let mut transport = entry["transport"].clone();
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
    let name = format!("config-pull-{}-{label}-{serial}", std::process::id());
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    if directory.exists() {
        fs::remove_dir_all(&directory).expect("stale scratch removed");
    }
    fs::create_dir_all(&directory).expect("scratch dir");
    directory
}

/// Every file below `root` except the lock, by relative path.
fn store_files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("store dir") {
            let path = entry.expect("store entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().is_some_and(|name| name != ".lock") {
                let relative = path.strip_prefix(root).expect("below root").to_path_buf();
                files.insert(relative, fs::read(&path).expect("store file"));
            }
        }
    }
    files
}

fn copy_python_store(state: &Path) {
    for (relative, body) in store_files(&worker_wire(LKG_STORE)) {
        let target = state.join("config-lkg").join(relative);
        fs::create_dir_all(target.parent().expect("parent")).expect("store dir");
        fs::write(target, body).expect("store file");
    }
}

/// The only revision file name in the Python store.
fn python_revision_name() -> String {
    let names: Vec<String> = fs::read_dir(worker_wire(LKG_STORE).join("revisions"))
        .expect("revisions dir")
        .map(|entry| {
            entry
                .expect("revision")
                .file_name()
                .into_string()
                .expect("utf-8")
        })
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    names.into_iter().next().expect("one revision")
}

/// The revision name layout of `config_lkg.py`: zero-padded epoch,
/// config version, registry version.
fn revision_name(directive: Directive) -> String {
    format!(
        "{:020}-{:020}-{:020}.json",
        directive.generation, directive.version, directive.registry
    )
}

/// The directive the golden payload declares, read field by field.
fn golden_directive(config: &Value) -> Directive {
    let integer = |key: &str| i128::from(config[key].as_i64().expect("directive field"));
    Directive {
        generation: integer("restart_epoch"),
        version: integer("config_version"),
        registry: integer("registry_version"),
    }
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

/// Answers exactly one request with `status_line` and `reply`.
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

fn serve_json(value: &Value) -> (JoinHandle<io::Result<Recorded>>, String) {
    let (listener, base) = loopback();
    let reply = serde_json::to_vec(value).expect("reply JSON");
    (serve_once(listener, "200 OK", reply), base)
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

fn joined(server: JoinHandle<io::Result<Recorded>>) -> Recorded {
    server
        .join()
        .expect("server thread")
        .expect("one request served")
}

#[test]
fn fresh_pull_equals_golden_and_saves_python_lkg() {
    let golden = read_json(WORKER_CONFIG);
    let state = scratch("fresh");
    let (server, base) = serve_json(&golden);
    let pulled = pull_startup_config(&base, RELAY_TOKEN, &state).expect("fresh pull");
    let recorded = joined(server);
    assert_eq!(
        transport_record(&recorded, CONFIG_PULL_TIMEOUT),
        manifest_transport(WORKER_CONFIG)
    );

    assert_eq!((pulled.source, pulled.stale), (ConfigSource::Pulled, false));
    assert_eq!(pulled.payload, Json::from(&golden));
    assert_eq!(pulled.directive, golden_directive(&golden));
    assert_eq!(
        pulled.config.clip_export_enabled(),
        golden["clip_export_enabled"].as_bool().expect("bool")
    );
    assert_eq!(
        pulled.config.clip_export_version(),
        i128::from(golden["clip_export_version"].as_i64().expect("int"))
    );
    assert_eq!(
        pulled.config.clip_store_subdir(),
        golden["clip_store_subdir"].as_str()
    );

    let declared = golden["cameras"].as_array().expect("cameras");
    assert_eq!(pulled.cameras.len(), declared.len());
    for (camera, want) in pulled.cameras.iter().zip(declared) {
        let text = |key: &str| want[key].as_str().map(str::to_owned);
        assert_eq!(
            (
                Some(camera.camera_id.clone()),
                Some(camera.facility_id.clone()),
                Some(camera.rtsp_url.clone()),
                camera.frame_stride,
                camera.decode_backend.clone(),
                camera.label.clone(),
            ),
            (
                text("camera_id"),
                text("facility_id"),
                text("rtsp_url"),
                want["frame_stride"].as_u64().expect("stride"),
                text("decode_backend"),
                text("label"),
            )
        );
        assert_eq!(
            (
                camera.bed_zone_regions.len(),
                camera.bed_zone_image_width,
                camera.bed_zone_image_height,
            ),
            (
                want["bed_zone_regions"].as_array().expect("regions").len(),
                want["bed_zone_image_width"].as_u64(),
                want["bed_zone_image_height"].as_u64(),
            )
        );
    }

    assert_eq!(
        store_files(&state.join("config-lkg")),
        store_files(&worker_wire(LKG_STORE))
    );
}

/// Serves `config` to one `RestartCheck` booted from the golden and
/// returns what the check decided and what the poll pulled.
fn restart_decision(config: &Value) -> (Option<Directive>, Option<Directive>) {
    let golden = read_json(WORKER_CONFIG);
    let state = scratch("restart-boot");
    let (server, base) = serve_json(&golden);
    let boot = pull_startup_config(&base, RELAY_TOKEN, &state).expect("boot pull");
    joined(server);

    let mut check = RestartCheck::new(boot.directive, boot.roster(), RESTART_POLL_INTERVAL);
    let (server, base) = serve_json(config);
    let mut polled = None;
    let decision = check.check(&FixedClock, || {
        let poll = poll_worker_config(&base, RELAY_TOKEN);
        polled = poll.as_ref().map(|poll| poll.directive);
        poll.map(|poll| poll.restart_candidate())
    });
    let recorded = joined(server);
    assert_eq!(
        transport_record(&recorded, CONFIG_PULL_TIMEOUT),
        manifest_transport(WORKER_CONFIG)
    );
    (decision, polled)
}

#[test]
fn changed_restart_epoch_gives_restart_directive() {
    let mut config = read_json(WORKER_CONFIG);
    let epoch = config["restart_epoch"].as_i64().expect("epoch");
    config["restart_epoch"] = Value::from(epoch + 1);
    let (decision, polled) = restart_decision(&config);
    let expected = golden_directive(&config);
    assert_eq!(polled, Some(expected));
    assert_eq!(decision, Some(expected));
}

#[test]
fn unchanged_epoch_with_new_config_version_gives_no_directive() {
    let mut config = read_json(WORKER_CONFIG);
    let version = config["config_version"].as_i64().expect("version");
    config["config_version"] = Value::from(version + 1);
    let (decision, polled) = restart_decision(&config);
    assert_eq!(polled, Some(golden_directive(&config)));
    assert_eq!(decision, None);
}

#[test]
fn relay_down_resolves_to_python_lkg() {
    let state = scratch("lkg");
    copy_python_store(&state);
    let pulled = pull_startup_config(&dead_relay(), RELAY_TOKEN, &state).expect("LKG config");
    assert_eq!((pulled.source, pulled.stale), (ConfigSource::Lkg, true));
    assert_eq!(pulled.payload, Json::from(&read_json(WORKER_CONFIG)));
    assert_eq!(revision_name(pulled.directive), python_revision_name());
    assert_eq!(
        store_files(&state.join("config-lkg")),
        store_files(&worker_wire(LKG_STORE))
    );
}

#[test]
fn relay_down_without_lkg_is_a_typed_refusal() {
    let state = scratch("empty");
    let refused = pull_startup_config(&dead_relay(), RELAY_TOKEN, &state);
    assert_eq!(refused, Err(PullError::NoConfig));
}

#[test]
fn golden_release_identity_is_accepted() {
    let (server, base) = serve_json(&read_json(RELEASE_IDENTITY));
    assert_eq!(check_release_identity(&base), Ok(()));
    let recorded = joined(server);
    assert_eq!(
        transport_record(&recorded, CONFIG_PULL_TIMEOUT),
        manifest_transport(RELEASE_IDENTITY)
    );
}

#[test]
fn other_schema_version_is_refused() {
    let mut identity = read_json(RELEASE_IDENTITY);
    let golden_version = identity["edge_database_schema_version"]
        .as_i64()
        .expect("schema version");
    identity["edge_database_schema_version"] = Value::from(golden_version - 1);
    let (server, base) = serve_json(&identity);
    let refused = check_release_identity(&base);
    joined(server);
    assert_eq!(
        refused,
        Err(PullError::SchemaMismatch {
            schema_version: i128::from(golden_version - 1)
        })
    );
}
