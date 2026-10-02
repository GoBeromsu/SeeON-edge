//! Owned binary boot fixtures. The one schema-1 aggregate is the actual
//! four-engine document produced once by the GPU build/CLI union. This fixture
//! does not retrofit a stored-pose FP32 engine as the live engine, invent image
//! or hardware facts, or construct a production identity. Copying the four
//! engines preserves each entry's basename and every recorded receipt field.
//!
//! Current Flow relocation is not a new engine-build receipt and is not
//! final-image attestation. Only the aggregate `flow` fingerprints are updated
//! to the owned served infer config and the real supplied tracker and parser
//! bytes. The copied serving config omits the legacy template ONNX key; that
//! omission is fixture relocation, not a fresh engine receipt. Native engine
//! entries stay byte-semantically equal to the recorded receipts. Runtime
//! `current_dir` is the owned root; process-global cwd and environment are
//! never changed.
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::seam::{Clock, SystemClock};
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);
pub const TOKEN: &str = "boot-binary-fixture-token";
pub const STATUS_PATH: &str = "/api/v1/relay/runtime-status";
pub const CONFIG_PATH: &str = "/api/v1/cameras/worker-config";
pub const IDENTITY_PATH: &str = "/health/release-identity";
pub const ALERTS_PATH: &str = "/api/v1/relay/alerts";
pub const CLIPS_PATH_PREFIX: &str = "/api/v1/relay/clips/";
pub const EXECUTION_RECORDS_PATH: &str = "/api/v1/relay/execution-records";
const ROLES: [&str; 4] = ["live_pose", "stored_pose", "bed", "fall"];
const FLOW_KEYS: [&str; 4] = [
    "infer_config_sha256",
    "tracker_config_sha256",
    "tracker_library_sha256",
    "parser_lib_sha256",
];
const OBSERVER_LIBRARY: &str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so";
const TRACKER_LIBRARY: &str =
    "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
const PARSER_LIBRARY: &str =
    "/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo26_pose.so";

#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Value,
}

pub struct Fixture {
    owned: OwnedDirectory,
    pub state: PathBuf,
    pub env: BTreeMap<String, String>,
    source: TcpListener,
}

fn required(name: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} required")));
    assert!(path.is_file(), "{name} must name a restored fixture");
    fs::canonicalize(path).expect("absolute restored fixture")
}
fn sha256_file(path: &std::path::Path) -> String {
    sha256_hex(&fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())))
}

fn installed(path: &str) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|error| panic!("installed fixture {path}: {error}"))
}

struct OwnedDirectory(PathBuf);
impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("owned boot fixture cleanup failed: {error}");
            assert!(
                std::thread::panicking(),
                "owned fixture cleanup must succeed"
            );
        }
    }
}

impl Fixture {
    pub fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "seeon-boot-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&root).expect("new owned directory");
        let owned = OwnedDirectory(root.clone());
        let state = root.join("state");
        fs::create_dir(&state).expect("state");
        let aggregate = load_aggregate();
        let fall_onnx = required("SEEON_TEST_FALL_ONNX");
        let pose_onnx = required("SEEON_TEST_STORED_POSE_ONNX");
        let bed_onnx = required("SEEON_TEST_BED_ONNX");
        let package = fall_onnx.parent().expect("published bundle directory");
        assert!(
            package.join("bundle-manifest.json").is_file(),
            "real packaged manifest required"
        );
        fs::create_dir_all(root.join("models/fall")).expect("owned model parent");
        fs::create_dir_all(root.join("models/bed")).expect("owned bed parent");
        symlink(package, root.join("models/fall/pose-bbox56-gru"))
            .expect("read-only published package reference");
        symlink(&bed_onnx, root.join("models/bed/yolo26l-seg.onnx"))
            .expect("canonical bed ONNX reference");
        assert_eq!(
            sha256_file(&pose_onnx),
            entry_str(&aggregate.engines, "live_pose", "onnx_sha256")
        );
        assert_eq!(
            sha256_file(&pose_onnx),
            entry_str(&aggregate.engines, "stored_pose", "onnx_sha256")
        );
        assert_eq!(
            sha256_file(&bed_onnx),
            entry_str(&aggregate.engines, "bed", "onnx_sha256")
        );
        assert_eq!(
            sha256_file(&fall_onnx),
            entry_str(&aggregate.engines, "fall", "onnx_sha256")
        );
        let engine_dir = root.join("engines");
        fs::create_dir(&engine_dir).expect("owned engines");
        let mut owned_engines = BTreeMap::new();
        for role in ROLES {
            let basename = entry_str(&aggregate.engines, role, "engine");
            assert!(
                !basename.contains('/')
                    && !basename.contains('\\')
                    && basename != "."
                    && basename != "..",
                "{role} engine must be a sibling basename"
            );
            let source = aggregate.directory.join(basename);
            assert!(
                source.is_file(),
                "sibling engine {basename} required beside the aggregate"
            );
            let destination = engine_dir.join(basename);
            fs::copy(&source, &destination)
                .expect("owned engine copy; never mutate shared engines");
            let recorded = entry_str(&aggregate.engines, role, "engine_sha256");
            assert_eq!(sha256_file(&destination), recorded, "{role} copied sha");
            assert_eq!(sha256_file(&source), recorded, "{role} source sha");
            owned_engines.insert(role, destination);
        }
        let infer_source = required("SEEON_TEST_MEDIA_INFER");
        let tracker = required("SEEON_TEST_MEDIA_TRACKER");
        let parser = installed(PARSER_LIBRARY);
        let observer = installed(OBSERVER_LIBRARY);
        let tracker_library = installed(TRACKER_LIBRARY);
        assert_eq!(
            sha256_file(&observer),
            entry_str(&aggregate.engines, "live_pose", "observer_library_sha256"),
            "observer fact is the recorded live receipt, not a fabricated library"
        );
        let served = root.join("nvinfer-served.txt");
        fs::write(
            &served,
            render_served(
                &fs::read_to_string(&infer_source).expect("served template"),
                &owned_engines["live_pose"].to_string_lossy(),
                aggregate.batch,
            ),
        )
        .expect("owned served config");
        let identity = root.join("engine-identity.json");
        write_relocated(
            &aggregate,
            &identity,
            &[
                ("infer_config_sha256", served.as_path()),
                ("tracker_config_sha256", tracker.as_path()),
                ("tracker_library_sha256", tracker_library.as_path()),
                ("parser_lib_sha256", parser.as_path()),
            ],
        );
        let mut env: BTreeMap<String, String> = [
            ("RELAY_TOKEN", TOKEN.into()),
            ("ML_WORKER_PROFILE", "flow".into()),
            ("ML_WORKER_EXECUTION_RECORDS_ENABLED", "false".into()),
            ("ML_WORKER_IMAGE", aggregate.image.clone()),
            (
                "ML_WORKER_MODEL_SELECTION_PATH",
                root.join("absent-selection.json")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "ML_WORKER_FALL_ENGINE_PATH",
                owned_engines["fall"].to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_BED_ENGINE_PATH",
                owned_engines["bed"].to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_STORED_POSE_ENGINE_PATH",
                owned_engines["stored_pose"].to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_ENGINE_PATH",
                owned_engines["live_pose"].to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_ENGINE_IDENTITY_PATH",
                identity.to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_INFER_CONFIG",
                served.to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_TRACKER_CONFIG",
                tracker.to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_TRACKER_LIBRARY",
                tracker_library.to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_ONNX_PATH",
                pose_onnx.to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_PARSER_LIBRARY",
                parser.to_string_lossy().into_owned(),
            ),
            (
                "ML_WORKER_FLOW_RECORD_DIR",
                root.join("recording").to_string_lossy().into_owned(),
            ),
            ("ML_WORKER_FLOW_RECORD_CACHE_SECONDS", "30".into()),
            ("ML_WORKER_FLOW_FRAME_WIDTH", "1280".into()),
            ("ML_WORKER_FLOW_FRAME_HEIGHT", "720".into()),
            ("ML_WORKER_FLOW_BATCH_SIZE", aggregate.batch.to_string()),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect();
        env.insert("HOME".into(), root.to_string_lossy().into_owned());
        let source = TcpListener::bind("127.0.0.1:0").expect("source activation trap");
        source.set_nonblocking(true).expect("nonblocking trap");
        Self {
            owned,
            state,
            env,
            source,
        }
    }

    pub fn config(&self, configured: bool) -> Value {
        let cameras = if configured {
            vec![json!({"camera_id":"camera-1","facility_id":"facility-1",
            "rtsp_url":format!("rtsp://{}/synthetic",self.source.local_addr().expect("trap address"))})]
        } else {
            vec![]
        };
        json!({"cameras":cameras,"registry_version":7,"restart_epoch":9,"clip_export_enabled":false,"clip_export_version":0})
    }

    pub fn command(&self, subcommand: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ml-worker"));
        command
            .arg(subcommand)
            .arg("--state-dir")
            .arg(&self.state)
            .current_dir(&self.owned.0)
            .env_clear();
        for name in [
            "PATH",
            "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
        ] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .envs(&self.env)
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    pub fn assert_no_source_activation(&self) {
        assert_eq!(
            self.source
                .accept()
                .expect_err("boot refusal must not activate RTSP")
                .kind(),
            std::io::ErrorKind::WouldBlock
        );
    }
}
pub struct Server {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Vec<Request>>>,
}
impl Server {
    pub fn start(config: Option<Value>, shutdown_started: Option<Arc<AtomicBool>>) -> Self {
        let addresses: Vec<_> = ("ml-api", 8000)
            .to_socket_addrs()
            .expect("isolated ml-api alias required")
            .collect();
        assert!(
            !addresses.is_empty()
                && addresses
                    .iter()
                    .all(|a| a.ip() == std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
            "refuse to contact a non-fixture backend; use isolated ml-api=127.0.0.1"
        );
        let listener = TcpListener::bind("127.0.0.1:8000").expect("exclusive fixture relay port");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            let clock = SystemClock::new();
            let mut requests = Vec::new();
            loop {
                let mut connection = None;
                let deadline = clock.monotonic() + Duration::from_secs(180);
                let waited = poll_until(&clock, deadline, "fixture request", || {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            connection = Some(stream);
                            true
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            stopped.load(Ordering::SeqCst)
                        }
                        Err(error) => panic!("fixture accept: {error}"),
                    }
                });
                let Some(stream) = connection else {
                    assert!(stopped.load(Ordering::SeqCst) || waited.is_err());
                    break;
                };
                requests.push(respond(
                    stream,
                    config.as_ref(),
                    shutdown_started.as_deref(),
                ));
            }
            requests
        });
        Self {
            stop,
            thread: Some(thread),
        }
    }
    pub fn finish(mut self) -> Vec<Request> {
        self.stop.store(true, Ordering::SeqCst);
        self.thread
            .take()
            .expect("server handle")
            .join()
            .expect("server completed")
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

fn respond(
    stream: TcpStream,
    config: Option<&Value>,
    shutdown_started: Option<&AtomicBool>,
) -> Request {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("read deadline");
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .expect("write deadline");
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).expect("request line");
    let parts: Vec<_> = line.split_whitespace().collect();
    assert_eq!(parts.len(), 3);
    let method = parts[0].to_owned();
    let path = parts[1].to_owned();
    let route = request_route(&path);
    let mut length = 0;
    let mut token = None;
    loop {
        line.clear();
        assert!(reader.read_line(&mut line).expect("header") > 0);
        if line == "\r\n" {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse::<usize>().expect("length");
            }
            if key.eq_ignore_ascii_case(seeon_ml_worker::relay::client::TOKEN_HEADER) {
                token = Some(value.trim().to_owned());
            }
        }
    }
    assert!(length <= 1_048_576, "fixture request exceeds 1 MiB");
    let mut body = vec![0; length];
    reader.read_exact(&mut body).expect("body");
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).expect("JSON request")
    };
    let (status, response) = match (method.as_str(), route) {
        ("GET", IDENTITY_PATH) => (
            200,
            json!({"edge_database_schema_version":19,"format":"seeon-edge-v1"}),
        ),
        ("GET", CONFIG_PATH) => {
            assert_eq!(token.as_deref(), Some(TOKEN));
            match config {
                Some(value) => (200, value.clone()),
                None => (503, Value::Null),
            }
        }
        ("POST", STATUS_PATH) => {
            assert_eq!(token.as_deref(), Some(TOKEN));
            (503, Value::Null)
        }
        ("POST", ALERTS_PATH) => {
            assert_eq!(token.as_deref(), Some(TOKEN));
            (202, json!({"accepted": true}))
        }
        ("POST", EXECUTION_RECORDS_PATH) => {
            assert_eq!(token.as_deref(), Some(TOKEN));
            (200, execution_record_ack(&body))
        }
        ("PUT", path) if path.starts_with(CLIPS_PATH_PREFIX) => {
            assert_eq!(token.as_deref(), Some(TOKEN));
            (202, json!({"accepted": true}))
        }
        _ => (404, Value::Null),
    };
    let bytes = serde_json::to_vec(&response).expect("response JSON");
    let writer = reader.get_mut();
    let header = write!(
        writer,
        "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    let written = header.and_then(|()| writer.write_all(&bytes));
    if let Err(error) = written {
        assert!(
            shutdown_config_write_cancelled(
                method.as_str(),
                path.as_str(),
                shutdown_started,
                &error
            ),
            "fixture response write failed outside owned SIGTERM config poll: {method} {path}: {error}"
        );
    }
    Request { method, path, body }
}

/// In-memory test relay ACK. `storage_state=committed` is the existing receipt
/// protocol, never durable PostgreSQL proof.
fn execution_record_ack(body: &Value) -> Value {
    let batch_id = body["batch_id"]
        .as_str()
        .filter(|value| seeon_ml_worker::records::id::sha256_hex_field(value, "batch_id").is_ok())
        .expect("execution-record batch_id");
    let accepted = body["records"]
        .as_array()
        .expect("execution-record records")
        .len();
    let committed_at_ns = u64::try_from(
        SystemClock::new()
            .wall()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("wall clock")
            .as_nanos(),
    )
    .expect("committed_at_ns");
    json!({
        "batch_id": batch_id,
        "accepted": accepted,
        "duplicates": 0,
        "rejected": [],
        "storage_state": "committed",
        "committed_at_ns": committed_at_ns,
    })
}

fn request_route(target: &str) -> &str {
    let without_query = target.split_once('?').map_or(target, |(path, _)| path);
    without_query
        .strip_prefix("http://ml-api:8000")
        .unwrap_or(without_query)
}
pub fn shutdown_config_write_cancelled(
    method: &str,
    path: &str,
    shutdown_started: Option<&AtomicBool>,
    error: &std::io::Error,
) -> bool {
    method == "GET"
        && path == CONFIG_PATH
        && shutdown_started.is_some_and(|flag| flag.load(Ordering::SeqCst))
        && matches!(
            error.kind(),
            std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
        )
}
struct Aggregate {
    directory: PathBuf,
    document: Value,
    engines: serde_json::Map<String, Value>,
    image: String,
    batch: u32,
}

fn load_aggregate() -> Aggregate {
    let path = required("SEEON_TEST_ENGINE_IDENTITY");
    let document: Value =
        serde_json::from_slice(&fs::read(&path).expect("actual four-engine aggregate"))
            .expect("aggregate JSON");
    let members = document.as_object().expect("schema-1 aggregate object");
    assert_eq!(
        members.len(),
        4,
        "one schema-1 aggregate, not a flat receipt"
    );
    assert_eq!(document["schema_version"], 1);
    for key in ["schema_version", "engines", "flow", "batch_size"] {
        assert!(members.contains_key(key), "aggregate missing {key}");
    }
    let engines = document["engines"]
        .as_object()
        .expect("four engine receipts")
        .clone();
    assert_eq!(engines.len(), ROLES.len(), "exactly four roles");
    for role in ROLES {
        assert!(engines.contains_key(role), "aggregate missing {role}");
    }
    let flow = document["flow"].as_object().expect("flow fingerprints");
    assert_eq!(
        flow.len(),
        FLOW_KEYS.len(),
        "exactly four current flow fingerprints"
    );
    for key in FLOW_KEYS {
        assert!(flow.contains_key(key), "aggregate flow missing {key}");
    }
    let batch = document["batch_size"]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| (1..=16).contains(value))
        .expect("recorded batch_size");
    let image = engines["live_pose"]["image_digest"]
        .as_str()
        .filter(|value| {
            value.starts_with("sha256:")
                && value.len() == "sha256:".len() + 64
                && value["sha256:".len()..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
        })
        .expect("recorded caller image digest from the live receipt")
        .to_owned();
    for role in ROLES {
        assert_eq!(
            engines[role]["image_digest"], image,
            "{role} image must match the recorded caller digest"
        );
    }
    Aggregate {
        directory: path.parent().expect("aggregate directory").to_path_buf(),
        document,
        engines,
        image,
        batch,
    }
}

fn entry_str<'a>(engines: &'a serde_json::Map<String, Value>, role: &str, key: &str) -> &'a str {
    engines[role][key]
        .as_str()
        .unwrap_or_else(|| panic!("{role}.{key} required"))
}

fn write_relocated(
    aggregate: &Aggregate,
    destination: &std::path::Path,
    flow_files: &[(&str, &std::path::Path)],
) {
    let mut relocated = aggregate.document.clone();
    let native = ["stored_pose", "bed", "fall"];
    for role in native {
        assert_eq!(relocated["engines"][role], aggregate.engines[role]);
    }
    let flow = relocated["flow"].as_object_mut().expect("flow object");
    assert_eq!(flow_files.len(), FLOW_KEYS.len());
    for (key, path) in flow_files {
        assert!(flow.contains_key(*key), "recorded flow missing {key}");
        flow.insert((*key).to_owned(), Value::String(sha256_file(path)));
    }
    for role in native {
        assert_eq!(
            relocated["engines"][role], aggregate.engines[role],
            "{role} receipt must stay byte-semantically equal"
        );
    }
    assert_eq!(
        relocated["engines"]["live_pose"],
        aggregate.engines["live_pose"]
    );
    assert_eq!(
        relocated["schema_version"],
        aggregate.document["schema_version"]
    );
    assert_eq!(relocated["batch_size"], aggregate.document["batch_size"]);
    let mut encoded = serde_json::to_vec(&relocated).expect("relocated aggregate");
    encoded.push(b'\n');
    fs::write(destination, encoded).expect("owned aggregate");
    let reread: Value =
        serde_json::from_slice(&fs::read(destination).expect("written aggregate")).expect("JSON");
    for role in native {
        assert_eq!(reread["engines"][role], aggregate.engines[role]);
    }
    assert_eq!(
        reread["engines"]["live_pose"],
        aggregate.engines["live_pose"]
    );
}

fn render_served(template: &str, engine: &str, batch: u32) -> String {
    assert!(
        !engine.is_empty() && !engine.contains(['\n', '\r']),
        "owned live engine path"
    );
    let mut engine_keys = 0_u8;
    let mut batch_keys = 0_u8;
    let mut onnx_keys = 0_u8;
    let mut rendered = String::new();
    for line in template.split_inclusive('\n') {
        let bare = line.trim_end_matches(['\n', '\r']);
        if bare.starts_with("onnx-file=") {
            onnx_keys = onnx_keys.checked_add(1).expect("one legacy onnx-file");
            continue;
        } else if let Some(value) = bare.strip_prefix("model-engine-file=") {
            assert!(
                !value.is_empty(),
                "template model-engine-file must already exist"
            );
            engine_keys = engine_keys.checked_add(1).expect("one model-engine-file");
            rendered.push_str("model-engine-file=");
            rendered.push_str(engine);
        } else if let Some(value) = bare.strip_prefix("batch-size=") {
            assert!(!value.is_empty(), "template batch-size must already exist");
            batch_keys = batch_keys.checked_add(1).expect("one batch-size");
            rendered.push_str("batch-size=");
            rendered.push_str(&batch.to_string());
        } else {
            rendered.push_str(bare);
        }
        if line.ends_with('\n') {
            rendered.push('\n');
        }
    }
    assert_eq!(
        [engine_keys, batch_keys],
        [1, 1],
        "relocate exactly one engine and batch"
    );
    assert!(onnx_keys <= 1, "ambiguous template ONNX keys");
    assert!(
        !rendered.contains("onnx-file="),
        "copied serving config omits legacy template ONNX; this is fixture relocation, not a fresh engine receipt"
    );
    rendered
}
