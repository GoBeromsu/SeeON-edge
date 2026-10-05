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

use seeon_ml_worker::config::lkg::{Directive, LkgStore};
use seeon_ml_worker::config::pull::{
    CONFIG_PULL_TIMEOUT, ConfigSource, PullError, PulledConfig, WorkerConfigPoll,
    check_release_identity, poll_worker_config as poll_from_zoneinfo,
    pull_startup_config as pull_startup_from_zoneinfo,
};
use seeon_ml_worker::config::restart::{RESTART_POLL_INTERVAL, RestartCheck};
use seeon_ml_worker::config::windows::{WindowError, ZONEINFO_DIR};
use seeon_ml_worker::json::{Json, Serialiser};
use seeon_ml_worker::relay::cameras::policies::parse_policy_bundle;
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

// Golden transport cases use the image's zone source. Asset-fault cases below
// pass their owned real filesystem source to the same production functions.
fn pull_startup_config(url: &str, token: &str, state: &Path) -> Result<PulledConfig, PullError> {
    pull_startup_from_zoneinfo(url, token, state, Path::new(ZONEINFO_DIR))
}
fn poll_worker_config(url: &str, token: &str) -> Result<Option<WorkerConfigPoll>, WindowError> {
    poll_from_zoneinfo(url, token, Path::new(ZONEINFO_DIR))
}

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
struct OwnedScratch(PathBuf);

impl Drop for OwnedScratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn owned_scratch(label: &str) -> OwnedScratch {
    OwnedScratch(scratch(label))
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

/// Serves `config` to one `RestartCheck` booted from the golden (directive
/// `(2, 7, 5)`) and returns what the check decided and what the poll
/// pulled.
fn restart_decision(config: &Value) -> (Option<Directive>, Option<Directive>) {
    let golden = read_json(WORKER_CONFIG);
    let state = scratch("restart-boot");
    let (server, base) = serve_json(&golden);
    let boot = pull_startup_config(&base, RELAY_TOKEN, &state).expect("boot pull");
    joined(server);

    let mut check = RestartCheck::new(boot.directive, RESTART_POLL_INTERVAL);
    let (server, base) = serve_json(config);
    let mut polled = None;
    let decision = check
        .check(&FixedClock, || {
            let poll = poll_worker_config(&base, RELAY_TOKEN)?;
            polled = poll.as_ref().map(|poll| poll.directive);
            Ok::<_, seeon_ml_worker::config::windows::WindowError>(
                poll.map(|poll| poll.restart_candidate()),
            )
        })
        .expect("no local zone asset fault");
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

// Python `RestartDirectiveTracker.observe` (`restart.py:41-46`) restarts on
// any strict advance of `(generation, version, registry)`.
#[test]
fn config_version_bump_alone_gives_restart_directive() {
    let mut config = read_json(WORKER_CONFIG);
    let version = config["config_version"].as_i64().expect("version");
    config["config_version"] = Value::from(version + 1);
    let (decision, polled) = restart_decision(&config);
    assert_eq!(polled, Some(golden_directive(&config)));
    let expected = Directive {
        generation: 2,
        version: 8,
        registry: 5,
    };
    assert_eq!(decision, Some(expected));
}

#[test]
fn registry_version_bump_alone_gives_restart_directive() {
    let mut config = read_json(WORKER_CONFIG);
    let registry = config["registry_version"].as_i64().expect("registry");
    config["registry_version"] = Value::from(registry + 1);
    let (decision, polled) = restart_decision(&config);
    assert_eq!(polled, Some(golden_directive(&config)));
    let expected = Directive {
        generation: 2,
        version: 7,
        registry: 6,
    };
    assert_eq!(decision, Some(expected));
}

#[test]
fn directive_not_above_boot_gives_no_directive() {
    let golden = read_json(WORKER_CONFIG);
    let field = |config: &Value, key: &str| config[key].as_i64().expect("directive field");
    let mut lower_version = golden.clone();
    lower_version["config_version"] = Value::from(field(&golden, "config_version") - 1);
    // A lower epoch outranks higher config and registry versions.
    let mut lower_epoch = golden.clone();
    lower_epoch["restart_epoch"] = Value::from(field(&golden, "restart_epoch") - 1);
    lower_epoch["config_version"] = Value::from(field(&golden, "config_version") + 10);
    lower_epoch["registry_version"] = Value::from(field(&golden, "registry_version") + 10);
    for (label, config) in [
        ("equal", golden.clone()),
        ("lower config_version", lower_version),
        ("lower restart_epoch", lower_epoch),
    ] {
        let (decision, polled) = restart_decision(&config);
        assert_eq!(polled, Some(golden_directive(&config)), "{label} poll");
        assert_eq!(decision, None, "{label}");
    }
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
fn enrolled_empty_configuration_retains_identity_through_lkg_revalidation() {
    let state = owned_scratch("enrolled-empty");
    let payload = serde_json::json!({
        "registry_version": 1, "cameras": [], "enrolled_facility_id": "genuine-site"
    });
    let (server, base) = serve_json(&payload);
    let fresh = pull_startup_config(&base, RELAY_TOKEN, &state.0).expect("fresh empty config");
    joined(server);
    assert_eq!(fresh.config.enrolled_facility_id(), Some("genuine-site"));
    assert!(fresh.cameras.is_empty());
    let cached =
        pull_startup_config(&dead_relay(), RELAY_TOKEN, &state.0).expect("cached empty config");
    assert_eq!(cached.source, ConfigSource::Lkg);
    assert_eq!(cached.payload, fresh.payload);
    assert_eq!(cached.config.enrolled_facility_id(), Some("genuine-site"));
    assert!(cached.cameras.is_empty());
}

// Python `load_worker_config_from_relay` (`config_pull.py:139-158`): the
// save of an older fresh pull is refused by the newer stored record, the
// stored payload fails re-validation, so `WorkerConfigLkgStore.clear`
// (`lkg_store.py:89-106`) unlinks `current.json` only and the fresh pull
// is used. Revisions stay on disk.
#[test]
fn corrupt_newer_lkg_is_cleared_and_fresh_wins() {
    let golden = read_json(WORKER_CONFIG);
    let state = scratch("corrupt-newer");
    let store = state.join("config-lkg");
    let newer = Directive {
        generation: 99,
        version: 99,
        registry: 99,
    };
    let record =
        br#"{"config_version":99,"generation":99,"payload":{"bogus":true},"registry_version":99}"#;
    fs::create_dir_all(store.join("revisions")).expect("store dir");
    fs::write(store.join("current.json"), record).expect("current record");
    let revision = PathBuf::from("revisions").join(revision_name(newer));
    fs::write(store.join(&revision), record).expect("revision record");

    let (server, base) = serve_json(&golden);
    let pulled = pull_startup_config(&base, RELAY_TOKEN, &state).expect("fresh pull");
    joined(server);

    assert_eq!((pulled.source, pulled.stale), (ConfigSource::Pulled, false));
    assert_eq!(pulled.payload, Json::from(&golden));
    assert_eq!(pulled.directive, golden_directive(&golden));
    assert_eq!(
        store_files(&store),
        BTreeMap::from([(revision, record.to_vec())])
    );
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

// Row 3: the policy bundle validator. Oracle: `r/policy-bundles.json`,
// recorded from Python `parse_policy_bundle`
// (`shared/detection_policies.py:417-449`) by an out-of-repo recorder; its
// bytes are pinned below so an unreviewed re-record fails here.
const POLICY_BUNDLES: &str = "r/policy-bundles.json";
const POLICY_BUNDLES_SHA256_PREFIX: &str = "4991d53789f2";

fn policy_bundle_golden() -> Value {
    let bytes = fs::read(worker_wire(POLICY_BUNDLES)).expect("policy bundle golden");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert!(
        digest.starts_with(POLICY_BUNDLES_SHA256_PREFIX),
        "unreviewed policy bundle golden: {digest}"
    );
    serde_json::from_slice(&bytes).expect("policy bundle golden JSON")
}

fn policy_case(golden: &Value, name: &str) -> Value {
    golden["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("golden case {name}"))
        .clone()
}

/// JSON cannot carry NaN or Infinity, so the golden names the value
/// Python injected at `inject_path` and the test places it the same way.
fn inject_non_finite(bundle: &mut Json, path: &[Value], value: f64) {
    let mut target = bundle;
    for key in path {
        let key = key.as_str().expect("inject path key");
        let Json::Object(members) = target else {
            panic!("inject path {key} is not inside an object");
        };
        target = members
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, member)| member)
            .unwrap_or_else(|| panic!("inject path key {key}"));
    }
    *target = Json::Float(value);
}

/// The golden worker config with `detection_policies` replaced.
fn config_with_policies(bundle: &Value) -> Value {
    let mut config = read_json(WORKER_CONFIG);
    config["detection_policies"] = bundle.clone();
    config
}

#[test]
fn policy_bundle_verdicts_match_python_oracle() {
    let golden = policy_bundle_golden();
    let path = golden["inject_path"].as_array().expect("inject path");
    let cases = golden["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 22);
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let mut bundle = Json::from(&case["bundle"]);
        match case["inject"].as_str() {
            None => {}
            Some("inf") => inject_non_finite(&mut bundle, path, f64::INFINITY),
            Some("nan") => inject_non_finite(&mut bundle, path, f64::NAN),
            Some(other) => panic!("{name}: unknown inject {other}"),
        }
        let parsed = parse_policy_bundle(&bundle);
        let accepted = case["accepted"].as_bool().expect("accepted");
        assert_eq!(parsed.is_ok(), accepted, "{name}: {parsed:?}");
        if let Ok(parsed) = parsed {
            assert_eq!(
                Serialiser::ExecutionRecords
                    .canonical(&parsed.as_json())
                    .expect("canonical"),
                case["canonical_json"].as_str().expect("canonical_json"),
                "{name}"
            );
            assert_eq!(
                parsed.content_sha256().expect("content sha256"),
                case["content_sha256"].as_str().expect("content_sha256"),
                "{name}"
            );
        }
    }
}

#[test]
fn malformed_policy_bundle_on_fresh_pull_keeps_lkg() {
    let refused = policy_case(&policy_bundle_golden(), "refuse-schema-version-2");
    assert_eq!(refused["accepted"], Value::Bool(false));
    let mut config = config_with_policies(&refused["bundle"]);
    // A newer revision, so a wrongly accepted bundle would also be saved.
    let version = config["config_version"].as_i64().expect("version");
    config["config_version"] = Value::from(version + 1);
    let state = scratch("policy-refused-pull");
    copy_python_store(&state);

    let (server, base) = serve_json(&config);
    let pulled = pull_startup_config(&base, RELAY_TOKEN, &state).expect("LKG config");
    let recorded = joined(server);

    assert_eq!(
        transport_record(&recorded, CONFIG_PULL_TIMEOUT),
        manifest_transport(WORKER_CONFIG)
    );
    assert_eq!((pulled.source, pulled.stale), (ConfigSource::Lkg, true));
    assert_eq!(pulled.payload, Json::from(&read_json(WORKER_CONFIG)));
    assert_eq!(
        store_files(&state.join("config-lkg")),
        store_files(&worker_wire(LKG_STORE))
    );
}

#[test]
fn malformed_policy_bundle_on_poll_gives_none() {
    let refused = policy_case(&policy_bundle_golden(), "refuse-schema-version-2");
    assert_eq!(refused["accepted"], Value::Bool(false));
    let (server, base) = serve_json(&config_with_policies(&refused["bundle"]));
    let polled = poll_worker_config(&base, RELAY_TOKEN)
        .expect("valid zone assets")
        .map(|poll| poll.directive);
    let recorded = joined(server);
    assert_eq!(
        transport_record(&recorded, CONFIG_PULL_TIMEOUT),
        manifest_transport(WORKER_CONFIG)
    );
    assert_eq!(polled, None);
}

#[test]
fn valid_policy_bundle_is_accepted_on_pull() {
    let accepted = policy_case(&policy_bundle_golden(), "accept-override-cam-b");
    assert_eq!(accepted["accepted"], Value::Bool(true));
    let config = config_with_policies(&accepted["bundle"]);
    let state = scratch("policy-accepted-pull");

    let (server, base) = serve_json(&config);
    let pulled = pull_startup_config(&base, RELAY_TOKEN, &state).expect("fresh pull");
    joined(server);
    assert_eq!((pulled.source, pulled.stale), (ConfigSource::Pulled, false));
    assert_eq!(
        pulled.policies.content_sha256().expect("content sha256"),
        accepted["content_sha256"].as_str().expect("content_sha256")
    );

    let (server, base) = serve_json(&config);
    let polled = poll_worker_config(&base, RELAY_TOKEN)
        .expect("valid zone assets")
        .map(|poll| poll.directive);
    joined(server);
    assert_eq!(polled, Some(golden_directive(&config)));
}

fn unknown_domain_payload() -> Value {
    let mut config = read_json(WORKER_CONFIG);
    config["cameras"][0]["domains"] =
        Value::Array(vec![Value::from("fall"), Value::from("not-a-domain")]);
    let version = config["config_version"].as_i64().expect("version");
    config["config_version"] = Value::from(version + 1);
    config
}

#[test]
fn unknown_domain_fresh_pull_keeps_valid_lkg() {
    let scratch = owned_scratch("unknown-domain-lkg");
    copy_python_store(&scratch.0);
    let config = unknown_domain_payload();
    let (server, base) = serve_json(&config);
    let pulled = pull_startup_config(&base, RELAY_TOKEN, &scratch.0).expect("LKG config");
    let recorded = joined(server);

    assert_eq!(
        transport_record(&recorded, CONFIG_PULL_TIMEOUT),
        manifest_transport(WORKER_CONFIG)
    );
    assert_eq!((pulled.source, pulled.stale), (ConfigSource::Lkg, true));
    assert_eq!(pulled.payload, Json::from(&read_json(WORKER_CONFIG)));
    assert_eq!(
        store_files(&scratch.0.join("config-lkg")),
        store_files(&worker_wire(LKG_STORE))
    );

    let (server, base) = serve_json(&config);
    let polled = poll_worker_config(&base, RELAY_TOKEN)
        .expect("valid zone assets")
        .map(|poll| poll.directive);
    joined(server);
    assert_eq!(
        polled,
        Some(golden_directive(&config)),
        "Python poll does not run startup domain admission"
    );
}

#[test]
fn unknown_domain_fresh_pull_without_lkg_refuses() {
    let scratch = owned_scratch("unknown-domain-empty");
    let config = unknown_domain_payload();
    let (server, base) = serve_json(&config);
    let refused = pull_startup_config(&base, RELAY_TOKEN, &scratch.0);
    joined(server);
    assert_eq!(refused, Err(PullError::NoConfig));
    assert!(
        !scratch.0.join("config-lkg").exists(),
        "a refused payload must not publish a store"
    );
}

fn window_config(start: &str, tz: &str) -> Value {
    let mut config = read_json(WORKER_CONFIG);
    config["config_version"] = Value::from(config["config_version"].as_i64().unwrap() + 1);
    config["detection_windows"] = serde_json::json!({
        "fall": {"start":start, "end":"02:00", "tz":tz}
    });
    config
}

#[test]
fn strict_windows_refuse_before_cache_but_remain_poll_candidates() {
    for start in ["1:00", "01:0", "١:٠"] {
        let state = owned_scratch("strict-window");
        let config = window_config(start, "UTC");
        let (server, base) = serve_json(&config);
        assert_eq!(
            pull_startup_config(&base, RELAY_TOKEN, &state.0),
            Err(PullError::NoConfig)
        );
        joined(server);
        assert!(!state.0.join("config-lkg").exists());

        let (server, base) = serve_json(&config);
        let polled = poll_worker_config(&base, RELAY_TOKEN)
            .unwrap()
            .expect("canonical poll candidate");
        joined(server);
        assert_eq!(polled.directive, golden_directive(&config));
        assert_eq!(polled.windows["fall"].definition.start, start);
        assert!(
            polled.windows["fall"]
                .window
                .contains(
                    seeon_worker::detection_window::AwareDateTime::from_utc_system_time(
                        SystemTime::UNIX_EPOCH + Duration::from_secs(5400)
                    )
                    .unwrap()
                )
                .unwrap()
        );
    }
}

#[test]
fn poll_uses_registry_cameras_without_startup_camera_or_domain_gates() {
    // Independently checked with BackendWorkerConfigPayload.to_pulled_config
    // versus to_worker_config: only startup applies these three refusals.
    let cases = [
        serde_json::json!([{}]),
        serde_json::json!([{"camera_id":"camera-a","rtsp_url":"rtsp://relay.invalid/live","fps":-1}]),
        serde_json::json!([{"camera_id":"camera-a","rtsp_url":"rtsp://relay.invalid/live","domains":["unknown"]}]),
    ];
    for (index, cameras) in cases.into_iter().enumerate() {
        let state = owned_scratch("poll-camera-boundary");
        let config = serde_json::json!({"config_version":8,"cameras":cameras});
        let (server, base) = serve_json(&config);
        assert_eq!(
            pull_startup_config(&base, RELAY_TOKEN, &state.0),
            Err(PullError::NoConfig)
        );
        joined(server);
        assert!(!state.0.join("config-lkg").exists());
        let (server, base) = serve_json(&config);
        let polled = poll_worker_config(&base, RELAY_TOKEN)
            .unwrap()
            .expect("Python poll candidate");
        joined(server);
        assert_eq!(
            polled.directive,
            Directive {
                generation: 0,
                version: 8,
                registry: 0
            }
        );
        assert_eq!(
            polled.cameras.len(),
            usize::from(index == 2),
            "invalid fps drops that registry entry; poll still admits its empty roster"
        );
        assert!(polled.windows.is_empty());
    }
}

#[test]
fn mixed_unicode_is_cached_verbatim_and_unknown_zone_drops_before_strictness() {
    let state = owned_scratch("mixed-window-cache");
    let config = window_config("0１:00", "UTC");
    let (server, base) = serve_json(&config);
    let admitted = pull_startup_config(&base, RELAY_TOKEN, &state.0).unwrap();
    joined(server);
    assert_eq!(admitted.windows["fall"].definition.start, "0１:00");
    let stored = LkgStore::new(&state.0).load().unwrap().unwrap();
    assert_eq!(stored.payload, Json::from(&config));
    let offline = pull_startup_config(&dead_relay(), RELAY_TOKEN, &state.0).unwrap();
    assert_eq!(offline.windows, admitted.windows);
    assert_eq!(offline.source, ConfigSource::Lkg);

    let state = owned_scratch("loose-unknown-zone");
    let config = window_config("1:00", "Seeon/Not_A_Zone");
    let (server, base) = serve_json(&config);
    let admitted = pull_startup_config(&base, RELAY_TOKEN, &state.0).unwrap();
    joined(server);
    assert!(
        admitted.windows.is_empty(),
        "unknown zone drops before strict width validation"
    );
    assert_eq!(
        admitted.payload,
        Json::from(&config),
        "cache retains source payload, not a normalized rewrite"
    );
}

#[test]
fn strict_fresh_and_stored_windows_cannot_replace_valid_startup_configuration() {
    let state = owned_scratch("strict-window-lkg");
    copy_python_store(&state.0);
    let before = store_files(&state.0.join("config-lkg"));
    let config = window_config("1:00", "UTC");
    let (server, base) = serve_json(&config);
    let pulled = pull_startup_config(&base, RELAY_TOKEN, &state.0).expect("valid LKG fallback");
    joined(server);
    assert_eq!(pulled.source, ConfigSource::Lkg);
    assert_eq!(store_files(&state.0.join("config-lkg")), before);

    let invalid_state = owned_scratch("stored-strict-window");
    let store = LkgStore::new(&invalid_state.0);
    assert!(
        store
            .save(&Json::from(&config), golden_directive(&config))
            .unwrap()
    );
    let before = store_files(&invalid_state.0.join("config-lkg"));
    assert_eq!(
        pull_startup_config(&dead_relay(), RELAY_TOKEN, &invalid_state.0),
        Err(PullError::NoConfig)
    );
    assert_eq!(
        store_files(&invalid_state.0.join("config-lkg")),
        before,
        "offline refusal does not delete cache evidence"
    );

    let golden = read_json(WORKER_CONFIG);
    let (server, base) = serve_json(&golden);
    let fresh = pull_startup_config(&base, RELAY_TOKEN, &invalid_state.0)
        .expect("fresh wins over invalid newer cache");
    joined(server);
    assert_eq!(fresh.source, ConfigSource::Pulled);
    assert_eq!(fresh.payload, Json::from(&golden));
    assert!(!invalid_state.0.join("config-lkg/current.json").exists());
    assert!(
        invalid_state
            .0
            .join("config-lkg/revisions")
            .join(revision_name(golden_directive(&config)))
            .is_file()
    );
}

#[test]
fn local_zone_faults_never_fall_back_to_or_delete_existing_lkg() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    assert_ne!(
        fs::metadata("/proc/self").unwrap().uid(),
        0,
        "permission proof needs unprivileged execution"
    );
    for permission_denied in [false, true] {
        let state = owned_scratch("asset-window-lkg");
        copy_python_store(&state.0);
        let zones = state.0.join("zoneinfo");
        fs::create_dir(&zones).unwrap();
        let asset = zones.join("Broken");
        let bytes = fs::read(Path::new(ZONEINFO_DIR).join("UTC")).unwrap();
        if permission_denied {
            fs::write(&asset, &bytes).unwrap();
            fs::set_permissions(&asset, fs::Permissions::from_mode(0o000)).unwrap();
        } else {
            fs::write(&asset, &bytes[..20]).unwrap();
        }
        let fault = if permission_denied {
            WindowError::Io(io::ErrorKind::PermissionDenied)
        } else {
            WindowError::CorruptTzif
        };
        let config = window_config("01:00", "Broken");
        let before = store_files(&state.0.join("config-lkg"));
        let (server, base) = serve_json(&config);
        let error = pull_startup_from_zoneinfo(&base, RELAY_TOKEN, &state.0, &zones).unwrap_err();
        joined(server);
        assert_eq!(error, PullError::WindowAsset(fault));
        assert_eq!(error.exit(), seeon_ml_worker::exit::Exit::Runtime);
        assert_eq!(store_files(&state.0.join("config-lkg")), before);

        let (server, base) = serve_json(&config);
        assert_eq!(poll_from_zoneinfo(&base, RELAY_TOKEN, &zones), Err(fault));
        joined(server);

        let store = LkgStore::new(&state.0);
        assert!(
            store
                .save(&Json::from(&config), golden_directive(&config))
                .unwrap()
        );
        let before = store_files(&state.0.join("config-lkg"));
        assert_eq!(
            pull_startup_from_zoneinfo(&dead_relay(), RELAY_TOKEN, &state.0, &zones),
            Err(PullError::WindowAsset(fault))
        );
        assert_eq!(store_files(&state.0.join("config-lkg")), before);

        // The race-losing fresh path must not classify a local asset fault
        // as corrupt cache and clear current.json.
        let (server, base) = serve_json(&read_json(WORKER_CONFIG));
        assert_eq!(
            pull_startup_from_zoneinfo(&base, RELAY_TOKEN, &state.0, &zones),
            Err(PullError::WindowAsset(fault))
        );
        joined(server);
        assert_eq!(store_files(&state.0.join("config-lkg")), before);
    }
}

#[test]
fn startup_camera_window_and_domain_refusals_keep_python_priority() {
    let state = owned_scratch("window-priority");
    let zones = state.0.join("zoneinfo");
    fs::create_dir(&zones).unwrap();
    fs::write(zones.join("Broken"), b"TZif\0").unwrap();
    let mut config = window_config("01:00", "Broken");
    config["cameras"][0]["domains"] = serde_json::json!(["unknown"]);
    let (server, base) = serve_json(&config);
    assert_eq!(
        pull_startup_from_zoneinfo(&base, RELAY_TOKEN, &state.0, &zones),
        Err(PullError::WindowAsset(WindowError::CorruptTzif)),
        "canonical window asset fault precedes domain resolution"
    );
    joined(server);
    config["cameras"][0]["fps"] = Value::from(-1);
    let (server, base) = serve_json(&config);
    assert_eq!(
        pull_startup_from_zoneinfo(&base, RELAY_TOKEN, &state.0, &zones),
        Err(PullError::NoConfig),
        "runtime camera validation precedes the window walk"
    );
    joined(server);
    assert!(!state.0.join("config-lkg").exists());
}

#[test]
fn restart_fault_preserves_tracker_and_consumes_the_attempted_interval() {
    let boot = Directive {
        generation: 1,
        version: 2,
        registry: 3,
    };
    let mut checker = RestartCheck::new(boot, RESTART_POLL_INTERVAL);
    assert_eq!(
        checker.check(&FixedClock, || Err::<Option<Directive>, _>(
            WindowError::CorruptTzif
        )),
        Err(WindowError::CorruptTzif)
    );
    assert_eq!(checker.tracker().current(), boot);
    let mut called = false;
    let result = checker.check(&FixedClock, || {
        called = true;
        Ok::<_, WindowError>(Some(Directive {
            version: 99,
            ..boot
        }))
    });
    assert_eq!(result, Ok(None));
    assert!(
        !called,
        "a fault must not cause an unbounded immediate retry"
    );
    assert_eq!(checker.tracker().current(), boot);
}
