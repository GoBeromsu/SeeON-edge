//! Owned binary boot fixtures. TensorRT requires the actual schema-1 four-engine
//! aggregate; explicit CPU auxiliaries require a supplied schema-2 aggregate
//! with the unchanged live GPU receipt and CPU ONNX hashes. This fixture never
//! converts or constructs a qualification identity, invents image or hardware
//! facts, or substitutes a stored-pose engine for the live engine. Copying only
//! the selected engines preserves their basenames and recorded receipt fields.
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
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use seeon_ml_worker::config::model_bundle::identity::AuxiliaryRuntime;
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::records::id::{sha256_hex, sha256_hex_field};
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
const AUXILIARY_ENGINES: [(&str, &str); 3] = [
    ("stored_pose", "ML_WORKER_STORED_POSE_ENGINE_PATH"),
    ("bed", "ML_WORKER_BED_ENGINE_PATH"),
    ("fall", "ML_WORKER_FALL_ENGINE_PATH"),
];
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

#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Value,
}

pub struct Fixture {
    owned: OwnedDirectory,
    auxiliary_runtime: AuxiliaryRuntime,
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
        Self::with_provider(label, AuxiliaryRuntime::TensorRt)
    }

    pub fn with_provider(label: &str, auxiliary_runtime: AuxiliaryRuntime) -> Self {
        let root = std::env::temp_dir().join(format!(
            "seeon-boot-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&root).expect("new owned directory");
        let owned = OwnedDirectory(root.clone());
        let state = root.join("state");
        fs::create_dir(&state).expect("state");
        let aggregate = load_aggregate(auxiliary_runtime);
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
        verify_model_sources(
            &aggregate,
            auxiliary_runtime,
            &pose_onnx,
            &bed_onnx,
            &fall_onnx,
        );
        let engine_dir = root.join("engines");
        fs::create_dir(&engine_dir).expect("owned engines");
        let mut owned_engines = BTreeMap::new();
        for &role in engine_roles(auxiliary_runtime) {
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
        wire_auxiliary_engines(&mut env, &owned_engines, auxiliary_runtime);
        env.insert("HOME".into(), root.to_string_lossy().into_owned());
        let source = TcpListener::bind("127.0.0.1:0").expect("source activation trap");
        source.set_nonblocking(true).expect("nonblocking trap");
        Self {
            owned,
            auxiliary_runtime,
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
        let mut command = Command::new(worker_binary());
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
            "ML_WORKER_BUILD_REVISION",
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
        select_auxiliary_command(&mut command, self.auxiliary_runtime);
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
    thread: Option<JoinHandle<()>>,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Server {
    pub fn start(config: Option<Value>, shutdown_started: Option<Arc<AtomicBool>>) -> Self {
        Self::start_with_identity(
            json!({"edge_database_schema_version":19,"format":"seeon-edge-v1"}),
            config,
            shutdown_started,
        )
    }

    pub fn start_with_identity(
        identity: Value,
        config: Option<Value>,
        shutdown_started: Option<Arc<AtomicBool>>,
    ) -> Self {
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
        let requests = Arc::new(Mutex::new(Vec::new()));
        let publishing = Arc::clone(&requests);
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            let clock = SystemClock::new();
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
                let request = respond(
                    stream,
                    &identity,
                    config.as_ref(),
                    shutdown_started.as_deref(),
                );
                publishing
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(request);
            }
        });
        Self {
            stop,
            thread: Some(thread),
            requests,
        }
    }
    /// Requests already retained, in arrival order. Later `finish` returns
    /// that same sequence plus anything accepted afterwards.
    pub fn observed(&self) -> Vec<Request> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn finish(mut self) -> Vec<Request> {
        self.stop.store(true, Ordering::SeqCst);
        self.thread
            .take()
            .expect("server handle")
            .join()
            .expect("server completed");
        self.observed()
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
    identity: &Value,
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
        ("GET", IDENTITY_PATH) => (200, identity.clone()),
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
const WORKER_BIN_OVERRIDE: &str = "SEEON_TEST_WORKER_BIN";

/// Absent override keeps the Cargo test binary. A present override must already
/// be an absolute regular executable; there is no path repair or fallback.
fn worker_binary() -> PathBuf {
    select_worker_binary(std::env::var_os(WORKER_BIN_OVERRIDE).as_deref())
}

fn select_worker_binary(raw: Option<&std::ffi::OsStr>) -> PathBuf {
    let Some(raw) = raw else {
        return PathBuf::from(env!("CARGO_BIN_EXE_ml-worker"));
    };
    let path = PathBuf::from(raw);
    assert!(path.is_absolute(), "{WORKER_BIN_OVERRIDE} must be absolute");
    let metadata = fs::symlink_metadata(&path).unwrap_or_else(|error| {
        panic!(
            "{WORKER_BIN_OVERRIDE}={:?} is not an absolute regular executable: {error}",
            path
        )
    });
    let executable = metadata.permissions().mode() & 0o111 != 0;
    assert!(
        path.is_absolute() && metadata.file_type().is_file() && executable,
        "{WORKER_BIN_OVERRIDE}={:?} must be an absolute regular executable",
        path
    );
    path
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

pub fn request_route(target: &str) -> &str {
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

fn engine_roles(auxiliary_runtime: AuxiliaryRuntime) -> &'static [&'static str] {
    match auxiliary_runtime {
        AuxiliaryRuntime::TensorRt => &ROLES,
        AuxiliaryRuntime::OnnxRuntimeCpu => &["live_pose"],
    }
}

fn wire_auxiliary_engines(
    env: &mut BTreeMap<String, String>,
    engines: &BTreeMap<&str, PathBuf>,
    auxiliary_runtime: AuxiliaryRuntime,
) {
    if auxiliary_runtime == AuxiliaryRuntime::TensorRt {
        for (role, name) in AUXILIARY_ENGINES {
            env.insert(
                name.to_owned(),
                engines[role].to_string_lossy().into_owned(),
            );
        }
    }
}

fn select_auxiliary_command(command: &mut Command, auxiliary_runtime: AuxiliaryRuntime) {
    if auxiliary_runtime == AuxiliaryRuntime::OnnxRuntimeCpu {
        command
            .arg("--auxiliary-runtime=onnxruntime-cpu")
            .env("ORT_DISABLE_TELEMETRY", "1");
    }
}

fn load_aggregate(auxiliary_runtime: AuxiliaryRuntime) -> Aggregate {
    let path = required("SEEON_TEST_ENGINE_IDENTITY");
    let document: Value =
        serde_json::from_slice(&fs::read(&path).expect("actual provider-specific aggregate"))
            .expect("aggregate JSON");
    aggregate_from_document(
        path.parent().expect("aggregate directory").to_path_buf(),
        document,
        auxiliary_runtime,
    )
}

fn exact_members<'a>(
    document: &'a Value,
    keys: &[&str],
    subject: &str,
) -> &'a serde_json::Map<String, Value> {
    let members = document
        .as_object()
        .unwrap_or_else(|| panic!("{subject} must be an object"));
    assert_eq!(members.len(), keys.len(), "{subject} exact members");
    for key in keys {
        assert!(members.contains_key(*key), "{subject} missing {key}");
    }
    members
}

fn aggregate_from_document(
    directory: PathBuf,
    document: Value,
    auxiliary_runtime: AuxiliaryRuntime,
) -> Aggregate {
    let (version, keys): (u64, &[&str]) = match auxiliary_runtime {
        AuxiliaryRuntime::TensorRt => (1, &["schema_version", "engines", "flow", "batch_size"]),
        AuxiliaryRuntime::OnnxRuntimeCpu => (
            2,
            &[
                "schema_version",
                "engines",
                "flow",
                "batch_size",
                "auxiliary",
            ],
        ),
    };
    exact_members(&document, keys, "provider-specific aggregate");
    assert_eq!(document["schema_version"].as_u64(), Some(version));
    let roles = engine_roles(auxiliary_runtime);
    let engines = exact_members(&document["engines"], roles, "engine receipts").clone();
    if auxiliary_runtime == AuxiliaryRuntime::OnnxRuntimeCpu {
        let auxiliary = exact_members(
            &document["auxiliary"],
            &["runtime", "provider", "models"],
            "CPU auxiliary declaration",
        );
        assert_eq!(auxiliary["runtime"], "onnxruntime");
        assert_eq!(auxiliary["provider"], "cpu");
        let models = exact_members(&auxiliary["models"], &ROLES[1..], "CPU models");
        for role in &ROLES[1..] {
            exact_members(&models[*role], &["onnx_sha256"], role);
            cpu_source_hash(&document, role);
        }
        assert_eq!(
            cpu_source_hash(&document, "stored_pose"),
            entry_str(&engines, "live_pose", "onnx_sha256"),
            "CPU stored pose must retain the live receipt's ONNX source"
        );
    }
    let flow = exact_members(&document["flow"], &FLOW_KEYS, "flow fingerprints");
    if auxiliary_runtime == AuxiliaryRuntime::OnnxRuntimeCpu {
        for key in FLOW_KEYS {
            assert!(
                flow[key]
                    .as_str()
                    .is_some_and(|value| sha256_hex_field(value, key).is_ok()),
                "CPU aggregate must supply the recorded flow SHA-256 for {key}"
            );
        }
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
    for &role in roles {
        assert_eq!(
            engines[role]["image_digest"], image,
            "{role} image must match the recorded caller digest"
        );
    }
    Aggregate {
        directory,
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

fn cpu_source_hash<'a>(document: &'a Value, role: &str) -> &'a str {
    document["auxiliary"]["models"][role]["onnx_sha256"]
        .as_str()
        .filter(|value| sha256_hex_field(value, "onnx_sha256").is_ok())
        .unwrap_or_else(|| panic!("CPU {role}.onnx_sha256 must be a recorded SHA-256"))
}

fn verify_model_sources(
    aggregate: &Aggregate,
    auxiliary_runtime: AuxiliaryRuntime,
    pose: &std::path::Path,
    bed: &std::path::Path,
    fall: &std::path::Path,
) {
    let pose_sha = sha256_file(pose);
    assert_eq!(
        pose_sha,
        entry_str(&aggregate.engines, "live_pose", "onnx_sha256"),
        "supplied pose source must match the unchanged live GPU receipt"
    );
    for (role, sha) in [
        ("stored_pose", pose_sha),
        ("bed", sha256_file(bed)),
        ("fall", sha256_file(fall)),
    ] {
        let recorded = match auxiliary_runtime {
            AuxiliaryRuntime::TensorRt => entry_str(&aggregate.engines, role, "onnx_sha256"),
            AuxiliaryRuntime::OnnxRuntimeCpu => cpu_source_hash(&aggregate.document, role),
        };
        assert_eq!(
            sha, recorded,
            "{role} supplied ONNX must match its recorded hash"
        );
    }
}

fn write_relocated(
    aggregate: &Aggregate,
    destination: &std::path::Path,
    flow_files: &[(&str, &std::path::Path)],
) {
    let mut relocated = aggregate.document.clone();
    let flow = relocated["flow"].as_object_mut().expect("flow object");
    assert_eq!(flow_files.len(), FLOW_KEYS.len());
    for key in FLOW_KEYS {
        assert!(flow.contains_key(key), "recorded flow missing {key}");
        let mut matches = flow_files.iter().filter(|(provided, _)| *provided == key);
        let (_, path) = matches.next().expect("relocation requires each flow file");
        assert!(matches.next().is_none(), "duplicate flow relocation {key}");
        flow.insert(key.to_owned(), Value::String(sha256_file(path)));
    }
    for (key, value) in aggregate.document.as_object().expect("aggregate object") {
        if key == "flow" {
            continue;
        }
        assert_eq!(
            &relocated[key], value,
            "{key} must stay byte-semantically equal; only four Flow hashes relocate"
        );
    }
    let mut encoded = serde_json::to_vec(&relocated).expect("relocated aggregate");
    encoded.push(b'\n');
    fs::write(destination, encoded).expect("owned aggregate");
    let reread: Value =
        serde_json::from_slice(&fs::read(destination).expect("written aggregate")).expect("JSON");
    assert_eq!(
        reread, relocated,
        "persisted relocation preserves every receipt and CPU fact"
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

#[cfg(test)]
mod provider_selection_tests {
    use super::*;

    const POSE: &[u8] = b"unit-private pose bytes, not a genuine ONNX model";
    const BED: &[u8] = b"unit-private bed bytes, not a genuine ONNX model";
    const FALL: &[u8] = b"unit-private fall bytes, not a genuine ONNX model";

    // Synthetic documents are confined to these private helpers. They never
    // enter Fixture::new/new_cpu, load_aggregate, or an executed worker command.
    fn unit_document(auxiliary_runtime: AuxiliaryRuntime) -> Value {
        let image = format!("sha256:{}", sha256_hex(b"unit-private image"));
        let live = json!({
            "engine": "unit-live.engine",
            "engine_sha256": sha256_hex(b"unit-private live engine"),
            "onnx_sha256": sha256_hex(POSE),
            "image_digest": image,
            "device": 0,
            "device_name": "unit-private device, not observed hardware",
            "compute_capability": "8.9",
            "trt_version": 10000,
            "observer_library_sha256": sha256_hex(b"unit-private observer"),
            "precision": "fp16",
            "tf32_enabled": true,
            "input": "images",
            "min_dimensions": [1, 3, 640, 640],
            "opt_dimensions": [2, 3, 640, 640],
            "max_dimensions": [2, 3, 640, 640],
        });
        let mut engines = serde_json::Map::new();
        engines.insert("live_pose".to_owned(), live.clone());
        if auxiliary_runtime == AuxiliaryRuntime::TensorRt {
            for (role, source) in [("stored_pose", POSE), ("bed", BED), ("fall", FALL)] {
                let mut receipt = live.clone();
                receipt["engine"] = json!(format!("unit-{role}.engine"));
                receipt["engine_sha256"] = json!(sha256_hex(role.as_bytes()));
                receipt["onnx_sha256"] = json!(sha256_hex(source));
                engines.insert(role.to_owned(), receipt);
            }
        }
        let flow: serde_json::Map<String, Value> = FLOW_KEYS
            .into_iter()
            .map(|key| (key.to_owned(), json!(sha256_hex(key.as_bytes()))))
            .collect();
        let mut document = json!({
            "schema_version": if auxiliary_runtime == AuxiliaryRuntime::TensorRt { 1 } else { 2 },
            "engines": engines,
            "flow": flow,
            "batch_size": 2,
        });
        if auxiliary_runtime == AuxiliaryRuntime::OnnxRuntimeCpu {
            document["auxiliary"] = json!({
                "runtime": "onnxruntime",
                "provider": "cpu",
                "models": {
                    "stored_pose": {"onnx_sha256": sha256_hex(POSE)},
                    "bed": {"onnx_sha256": sha256_hex(BED)},
                    "fall": {"onnx_sha256": sha256_hex(FALL)},
                },
            });
        }
        document
    }

    fn unit_directory(label: &str) -> OwnedDirectory {
        let path = std::env::temp_dir().join(format!(
            "seeon-unit-provider-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&path).expect("new unit-private directory");
        OwnedDirectory(path)
    }

    fn refuse(document: Value, auxiliary_runtime: AuxiliaryRuntime) {
        assert!(
            std::panic::catch_unwind(|| {
                aggregate_from_document(PathBuf::new(), document, auxiliary_runtime)
            })
            .is_err(),
            "malformed or mismatched provider declaration must fail closed"
        );
    }

    #[test]
    fn providers_require_their_own_schema_and_exact_selected_engine_roles() {
        for provider in [AuxiliaryRuntime::TensorRt, AuxiliaryRuntime::OnnxRuntimeCpu] {
            let document = unit_document(provider);
            let aggregate = aggregate_from_document(PathBuf::new(), document.clone(), provider);
            let expected: &[&str] = match provider {
                AuxiliaryRuntime::TensorRt => &["live_pose", "stored_pose", "bed", "fall"],
                AuxiliaryRuntime::OnnxRuntimeCpu => &["live_pose"],
            };
            assert_eq!(engine_roles(provider), expected);
            assert_eq!(aggregate.engines.len(), expected.len());
            assert!(
                expected
                    .iter()
                    .all(|role| aggregate.engines.contains_key(*role))
            );
            assert_eq!(
                aggregate.document, document,
                "selection never rewrites identity"
            );
            let other = match provider {
                AuxiliaryRuntime::TensorRt => AuxiliaryRuntime::OnnxRuntimeCpu,
                AuxiliaryRuntime::OnnxRuntimeCpu => AuxiliaryRuntime::TensorRt,
            };
            refuse(document.clone(), other);
            let mut changed_version = document.clone();
            changed_version["schema_version"] = json!(3);
            refuse(changed_version, provider);
            let mut missing_role = document.clone();
            missing_role["engines"]
                .as_object_mut()
                .unwrap()
                .remove("live_pose");
            refuse(missing_role, provider);
            let mut renamed_role = document.clone();
            let engines = renamed_role["engines"].as_object_mut().unwrap();
            let live = engines.remove("live_pose").unwrap();
            engines.insert("unexpected".to_owned(), live);
            refuse(renamed_role, provider);
            let mut extra_role = document;
            extra_role["engines"]["unexpected"] = json!({});
            refuse(extra_role, provider);
        }
    }

    #[test]
    fn cpu_refuses_malformed_provider_model_hash_and_batch_declarations() {
        let provider = AuxiliaryRuntime::OnnxRuntimeCpu;
        let original = unit_document(provider);
        for (pointer, value) in [
            ("/schema_version", json!("2")),
            ("/auxiliary", Value::Null),
            ("/auxiliary/runtime", json!("onnxruntime-cpu")),
            ("/auxiliary/runtime", json!("tensorrt")),
            ("/auxiliary/provider", json!("cuda")),
            ("/auxiliary/provider", json!("CPU")),
            ("/auxiliary/models", json!([])),
            ("/auxiliary/models/fall", json!("not a hash entry")),
            ("/auxiliary/models/bed/onnx_sha256", json!("a".repeat(63))),
            ("/auxiliary/models/bed/onnx_sha256", json!("A".repeat(64))),
            ("/auxiliary/models/bed/onnx_sha256", json!(7)),
            (
                "/auxiliary/models/stored_pose/onnx_sha256",
                json!("0".repeat(64)),
            ),
            ("/engines/live_pose/onnx_sha256", json!("0".repeat(64))),
            (
                "/engines/live_pose/image_digest",
                json!("not an image digest"),
            ),
            ("/flow/infer_config_sha256", Value::Null),
            (
                "/flow/tracker_config_sha256",
                json!("bad recorded fingerprint"),
            ),
            ("/batch_size", json!(0)),
            ("/batch_size", json!(17)),
            ("/batch_size", json!(1.5)),
        ] {
            let mut malformed = original.clone();
            *malformed.pointer_mut(pointer).expect("unit document field") = value;
            refuse(malformed, provider);
        }
        for pointer in [
            "",
            "/auxiliary",
            "/auxiliary/models",
            "/auxiliary/models/fall",
            "/engines",
            "/flow",
        ] {
            let mut extra = original.clone();
            extra
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unexpected".to_owned(), json!(true));
            refuse(extra, provider);
            let mut missing = original.clone();
            let members = missing
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap();
            let key = members.keys().next().unwrap().clone();
            members.remove(&key);
            refuse(missing, provider);
        }
    }

    #[test]
    fn cpu_sources_must_match_each_supplied_model_without_repairing_hashes() {
        let provider = AuxiliaryRuntime::OnnxRuntimeCpu;
        let owned = unit_directory("model-hashes");
        let pose = owned.0.join("pose.onnx");
        let bed = owned.0.join("bed.onnx");
        let fall = owned.0.join("fall.onnx");
        for (path, bytes) in [(&pose, POSE), (&bed, BED), (&fall, FALL)] {
            fs::write(path, bytes).unwrap();
        }
        let original = unit_document(provider);
        let aggregate = aggregate_from_document(owned.0.clone(), original.clone(), provider);
        verify_model_sources(&aggregate, provider, &pose, &bed, &fall);
        for (path, bytes) in [(&pose, POSE), (&bed, BED), (&fall, FALL)] {
            fs::write(path, b"unit-private changed source").unwrap();
            assert!(
                std::panic::catch_unwind(|| {
                    verify_model_sources(&aggregate, provider, &pose, &bed, &fall)
                })
                .is_err(),
                "{} must remain bound to its recorded hash",
                path.display()
            );
            assert_eq!(aggregate.document, original, "no hash repair on refusal");
            fs::write(path, bytes).unwrap();
        }
        for role in ["stored_pose", "bed", "fall"] {
            let mut changed = original.clone();
            changed["auxiliary"]["models"][role]["onnx_sha256"] = json!("0".repeat(64));
            if role == "stored_pose" {
                // Even mutually consistent live/CPU declarations must match real bytes.
                changed["engines"]["live_pose"]["onnx_sha256"] = json!("0".repeat(64));
            }
            let aggregate = aggregate_from_document(owned.0.clone(), changed.clone(), provider);
            assert!(
                std::panic::catch_unwind(|| {
                    verify_model_sources(&aggregate, provider, &pose, &bed, &fall)
                })
                .is_err()
            );
            assert_eq!(aggregate.document, changed);
        }
    }

    #[test]
    fn provider_command_selects_cpu_and_omits_unused_auxiliary_gpu_wiring() {
        for provider in [AuxiliaryRuntime::TensorRt, AuxiliaryRuntime::OnnxRuntimeCpu] {
            let engines: BTreeMap<_, _> = engine_roles(provider)
                .iter()
                .map(|&role| (role, PathBuf::from(format!("/unit-private/{role}.engine"))))
                .collect();
            let mut env = BTreeMap::new();
            wire_auxiliary_engines(&mut env, &engines, provider);
            let mut command = Command::new("/unit-private/not-executed-worker");
            command
                .arg("run")
                .arg("--state-dir=/unit-private/state")
                .env(
                    "ORT_DISABLE_TELEMETRY",
                    "image-setting-cleared-before-start",
                )
                .env_clear()
                .envs(&env)
                .env("CUDA_VISIBLE_DEVICES", "");
            select_auxiliary_command(&mut command, provider);
            let arguments = command
                .get_args()
                .map(std::ffi::OsString::from)
                .collect::<Vec<_>>();
            let seeon_ml_worker::cli::Command::Run(flags) =
                seeon_ml_worker::cli::parse(&arguments).expect("provider command contract")
            else {
                panic!("provider selection must stay on the run command");
            };
            assert_eq!(flags.auxiliary_runtime, provider);
            let command_env: BTreeMap<_, _> = command.get_envs().collect();
            let cpu = provider == AuxiliaryRuntime::OnnxRuntimeCpu;
            for (role, name) in AUXILIARY_ENGINES {
                let actual = command_env
                    .get(std::ffi::OsStr::new(name))
                    .copied()
                    .flatten();
                let expected = (!cpu).then(|| engines[role].as_os_str());
                assert_eq!(actual, expected, "provider-specific {name} wiring");
            }
            assert_eq!(
                command_env
                    .get(std::ffi::OsStr::new("ORT_DISABLE_TELEMETRY"))
                    .copied()
                    .flatten(),
                cpu.then(|| std::ffi::OsStr::new("1")),
                "CPU child restores telemetry opt-out after env_clear"
            );
            assert_eq!(
                command_env
                    .get(std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES"))
                    .copied()
                    .flatten(),
                Some(std::ffi::OsStr::new("")),
                "negative fixture must never force device 0"
            );
        }
    }

    #[test]
    fn relocation_changes_only_four_flow_fingerprints_for_each_provider() {
        let owned = unit_directory("relocation");
        let files: Vec<_> = FLOW_KEYS
            .into_iter()
            .map(|key| {
                let path = owned.0.join(key);
                fs::write(&path, format!("unit-private relocated {key}")).unwrap();
                (key, path)
            })
            .collect();
        let flow_files: Vec<_> = files
            .iter()
            .map(|(key, path)| (*key, path.as_path()))
            .collect();
        let destination = owned.0.join("unit-relocated.json");
        for provider in [AuxiliaryRuntime::TensorRt, AuxiliaryRuntime::OnnxRuntimeCpu] {
            let original = unit_document(provider);
            let aggregate = aggregate_from_document(owned.0.clone(), original.clone(), provider);
            write_relocated(&aggregate, &destination, &flow_files);
            let relocated: Value =
                serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
            let mut expected = original.clone();
            for (key, path) in &flow_files {
                expected["flow"][*key] = json!(sha256_file(path));
            }
            assert_eq!(
                relocated, expected,
                "all native, image, batch and CPU facts stay intact"
            );
            assert_eq!(
                aggregate.document, original,
                "source document is never mutated"
            );
            for bad_key in [FLOW_KEYS[1], "onnx_sha256"] {
                let mut invalid = flow_files.clone();
                invalid[0].0 = bad_key;
                assert!(
                    std::panic::catch_unwind(|| {
                        write_relocated(&aggregate, &destination, &invalid)
                    })
                    .is_err(),
                    "duplicate or non-Flow relocation must refuse"
                );
                let reread: Value =
                    serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
                assert_eq!(reread, relocated, "refused relocation must not write");
            }
        }
    }
}

#[cfg(test)]
mod binary_selection_tests {
    use super::*;

    #[test]
    fn explicit_binary_override_refuses_invalid_paths_without_fallback() {
        let root = std::env::temp_dir().join(format!(
            "seeon-binary-selection-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&root).expect("new owned binary-selection directory");
        let owned = OwnedDirectory(root);
        let executable = owned.0.join("worker");
        fs::write(&executable, b"test-only executable path; never executed").unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            select_worker_binary(Some(executable.as_os_str())),
            executable
        );
        let link = owned.0.join("link");
        symlink(&executable, &link).unwrap();
        let not_executable = owned.0.join("not-executable");
        fs::write(&not_executable, b"not executable").unwrap();
        fs::set_permissions(&not_executable, fs::Permissions::from_mode(0o600)).unwrap();
        for path in [
            PathBuf::from("relative-worker"),
            owned.0.join("missing"),
            owned.0.clone(),
            link,
            not_executable,
        ] {
            assert!(
                std::panic::catch_unwind(|| select_worker_binary(Some(path.as_os_str()))).is_err(),
                "invalid explicit override must refuse, not use the Cargo binary: {path:?}"
            );
        }
    }
}
