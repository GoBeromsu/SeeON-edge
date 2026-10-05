//! Ignored actual-GPU `engine-build` CLI. The parent owns an empty
//! `SEEON_TEST_ENGINE_BUILD_ROOT` and runs this once at the connected feature
//! boundary. This test exclusively creates that root's contents and retains
//! the four genuine engines, `engine-identity.json`, and `nvinfer-served.txt`
//! for subsequent `boot_gpu` / `run_gpu` consumers. It never recursively
//! cleans the root, on failure or success.
//!
//! Proof scope is the real `ml-worker engine-build` binary only. Native build,
//! observer, and parser are not mocked. A second identical invocation without
//! `--force` must be a cache hit: engine and identity bytes, hashes, inodes,
//! and mtimes stay unchanged, and no additional `.engine-stage-*` directory
//! appears. `check-config` then proves the file-only gate, including with
//! `CUDA_VISIBLE_DEVICES=""`, and a copied identity missing `engines.fall`
//! must exit 3. The original identity bytes remain unchanged. There is no
//! production override, forged native fact, final-image claim, or numerical
//! parity claim.
//!
//! Hardware comparison uses the existing pinned runtime GPU manifest. The
//! actual `hardware_identity(0)` must match its device name and compute
//! capability; TensorRT version is absent there, so it is compared to that
//! native identity rather than invented. Missing selected prerequisites fail;
//! this test does not soft-skip.
//! `--infer-config` is the repo template. Its `custom-lib-path` must already
//! name the mounted parser; this test does not rewrite that template.

use std::fs::{self, File};
use std::io::Write;
use std::os::unix::fs::{MetadataExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;

use seeon_deepstream_native::hardware_identity;
use seeon_ml_worker::poll::poll_until;
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::seam::{Clock, SystemClock};
use serde_json::Value;

const BUILD_WAIT: Duration = Duration::from_secs(300);
const CHECK_WAIT: Duration = Duration::from_secs(30);
const OBSERVER_LIBRARY: &str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer.so";
const TRACKER_LIBRARY: &str =
    "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so";
const PARSER_LIBRARY: &str =
    "/opt/nvidia/deepstream/deepstream/lib/libnvdsinfer_custom_yolo26_pose.so";
const MANIFEST: &str = include_str!("../../runtime/inference/tests/fixtures/gpu/manifest.json");
const ROLES: [&str; 4] = ["live_pose", "stored_pose", "bed", "fall"];
const ENGINES: [&str; 4] = [
    "live-pose.engine",
    "stored-pose.engine",
    "bed.engine",
    "fall.engine",
];
const KEPT_ENV: [&str; 6] = [
    "PATH",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "CUDA_VISIBLE_DEVICES",
    "GST_PLUGIN_PATH",
];

struct OwnedBinary {
    child: Option<Child>,
}

impl OwnedBinary {
    fn spawn(mut command: Command) -> Self {
        command.stdin(Stdio::null());
        let child = command.spawn().expect("owned ml-worker binary");
        Self { child: Some(child) }
    }
}

impl Drop for OwnedBinary {
    fn drop(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => {}
            Err(error) => {
                eprintln!("owned binary status unavailable; refusing to signal: {error}");
                assert!(std::thread::panicking(), "owned binary status failed");
                return;
            }
        }
        if let Err(error) = child.kill() {
            eprintln!("owned binary kill failed: {error}");
        }
        if let Err(error) = child.wait() {
            eprintln!("owned binary reap failed: {error}");
            assert!(std::thread::panicking(), "owned binary must be reaped");
        }
    }
}

struct Stamp {
    bytes: Vec<u8>,
    sha256: String,
    dev: u64,
    ino: u64,
    mtime: i64,
    mtime_nsec: i64,
}

fn required_file(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required"));
    let path = PathBuf::from(value);
    assert!(path.is_file(), "{name} must name an existing file");
    fs::canonicalize(&path).unwrap_or_else(|error| panic!("{name} unreadable: {error}"))
}

fn required_text(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn sha256_file(path: &Path) -> String {
    sha256_hex(&fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())))
}

fn stamp(path: &Path) -> Stamp {
    let metadata = fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("stat {}: {error}", path.display()));
    assert!(
        metadata.is_file(),
        "{} must stay a regular file",
        path.display()
    );
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    Stamp {
        sha256: sha256_hex(&bytes),
        bytes,
        dev: metadata.dev(),
        ino: metadata.ino(),
        mtime: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
    }
}

fn engine_stage_dirs(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        {
            let entry = entry.expect("owned root entry");
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if entry.file_type().expect("entry type").is_dir() {
                if name.starts_with(".engine-stage-") {
                    found.push(path);
                } else if !name.starts_with(".identity-stage-") {
                    pending.push(path);
                }
            }
        }
    }
    found.sort();
    found
}

fn capture_stdio(root: &Path, label: &str) -> (File, File, PathBuf, PathBuf) {
    let stdout_path = root.join(format!("{label}.stdout"));
    let stderr_path = root.join(format!("{label}.stderr"));
    let stdout = File::options()
        .write(true)
        .create_new(true)
        .open(&stdout_path)
        .expect("exclusive stdout capture");
    let stderr = File::options()
        .write(true)
        .create_new(true)
        .open(&stderr_path)
        .expect("exclusive stderr capture");
    (stdout, stderr, stdout_path, stderr_path)
}

fn await_owned(owned: &mut OwnedBinary, deadline: Duration, what: &'static str) -> ExitStatus {
    let clock = SystemClock::new();
    let started = clock.monotonic();
    let finished = poll_until(&clock, started + deadline, what, || {
        owned
            .child
            .as_mut()
            .expect("owned direct binary")
            .try_wait()
            .expect("owned binary status")
            .is_some()
    });
    let mut child = owned.child.take().expect("owned direct binary");
    if finished.is_err() {
        if let Err(error) = child.kill() {
            eprintln!("owned timed-out binary kill failed: {error}");
        }
        child.wait().expect("reap owned timed-out binary");
        panic!("{what} exceeded {deadline:?}; killed and reaped only the owned binary");
    }
    let status = child.try_wait().expect("reaped status").expect("exited");
    owned.child = Some(child);
    status
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key].as_str().unwrap_or_else(|| panic!("{key} text"))
}

fn prepare_root(root: &Path, fall_onnx: &Path, bed_onnx: &Path) -> PathBuf {
    assert!(
        root.is_dir(),
        "SEEON_TEST_ENGINE_BUILD_ROOT must be an existing directory"
    );
    assert_eq!(
        fs::read_dir(root).expect("parent-owned root").count(),
        0,
        "SEEON_TEST_ENGINE_BUILD_ROOT must be empty; this test creates its contents"
    );
    let package = fall_onnx.parent().expect("published fall bundle directory");
    assert!(
        package.join("bundle-manifest.json").is_file(),
        "real packaged fall bundle-manifest required beside {}",
        fall_onnx.display()
    );
    fs::create_dir_all(root.join("models/fall")).expect("owned fall parent");
    fs::create_dir_all(root.join("models/bed")).expect("owned bed parent");
    symlink(package, root.join("models/fall/pose-bbox56-gru")).expect("packaged fall reference");
    symlink(bed_onnx, root.join("models/bed/yolo26l-seg.onnx")).expect("bed ONNX reference");
    let state = root.join("state");
    fs::create_dir(&state).expect("owned state dir");
    state
}

fn build_command(root: &Path) -> Command {
    let pose = required_file("SEEON_TEST_STORED_POSE_ONNX");
    let bed = required_file("SEEON_TEST_BED_ONNX");
    let image = required_text("ML_WORKER_IMAGE");
    let parser = fs::canonicalize(PARSER_LIBRARY).expect("installed parser");
    let infer = fs::canonicalize(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../adapters/deepstream/configs/nvinfer-yolo26-pose.txt"),
    )
    .expect("absolute repo infer template");
    let tracker = fs::canonicalize(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../adapters/deepstream/configs/config_tracker_NvDCF_perf.yml"),
    )
    .expect("absolute repo tracker config");
    let mut command = Command::new(env!("CARGO_BIN_EXE_ml-worker"));
    command
        .current_dir(root)
        .env_clear()
        .arg("engine-build")
        .arg("--onnx")
        .arg(pose)
        .arg("--engine")
        .arg(root.join("live-pose.engine"))
        .arg("--identity")
        .arg(root.join("engine-identity.json"))
        .arg("--parser-lib")
        .arg(&parser)
        .arg("--infer-config")
        .arg(&infer)
        .arg("--served-infer-config")
        .arg(root.join("nvinfer-served.txt"))
        .arg("--tracker-config")
        .arg(&tracker)
        .arg("--tracker-library")
        .arg(TRACKER_LIBRARY)
        .arg("--stored-pose-engine")
        .arg(root.join("stored-pose.engine"))
        .arg("--bed-onnx")
        .arg(bed)
        .arg("--bed-engine")
        .arg(root.join("bed.engine"))
        .arg("--fall-engine")
        .arg(root.join("fall.engine"))
        .arg("--image-digest")
        .arg(image)
        .arg("--batch-size")
        .arg("2");
    for name in KEPT_ENV {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command.env(
        "ML_WORKER_MODEL_SELECTION_PATH",
        root.join("absent-selection.json"),
    );
    command
}
fn assert_exit(status: ExitStatus, expected: i32, stderr: &Path) {
    assert_eq!(
        status.code(),
        Some(expected),
        "{}",
        fs::read_to_string(stderr).unwrap_or_else(|error| format!("stderr unreadable: {error}"))
    );
}

fn assert_aggregate(root: &Path, image: &str, pose: &Path, bed: &Path, fall: &Path) {
    let identity = root.join("engine-identity.json");
    let document: Value = serde_json::from_slice(&fs::read(&identity).expect("identity"))
        .expect("schema-1 aggregate");
    let members = document.as_object().expect("schema-1 object");
    assert_eq!(members.len(), 4, "one schema-1 aggregate");
    for key in ["schema_version", "engines", "flow", "batch_size"] {
        assert!(members.contains_key(key), "aggregate missing {key}");
    }
    assert_eq!(document["schema_version"], 1);
    assert_eq!(document["batch_size"], 2);
    let engines = document["engines"]
        .as_object()
        .expect("four engine entries");
    assert_eq!(engines.len(), 4);
    let expected_image = image.rsplit_once('@').map_or(image, |(_, digest)| digest);
    let sources = [
        ("live_pose", sha256_file(pose), "live-pose.engine"),
        ("stored_pose", sha256_file(pose), "stored-pose.engine"),
        ("bed", sha256_file(bed), "bed.engine"),
        ("fall", sha256_file(fall), "fall.engine"),
    ];
    let mut engine_shas = Vec::new();
    for (role, onnx_sha, basename) in sources {
        let entry = &document["engines"][role];
        assert_eq!(text(entry, "engine"), basename);
        assert_eq!(text(entry, "onnx_sha256"), onnx_sha);
        let engine = root.join(basename);
        let actual = sha256_file(&engine);
        assert_eq!(text(entry, "engine_sha256"), actual);
        assert_ne!(
            actual, onnx_sha,
            "{role} engine must not echo its input digest"
        );
        engine_shas.push(actual);
        assert_eq!(text(entry, "image_digest"), expected_image);
        assert_eq!(entry["device"], 0);
    }
    engine_shas.sort();
    engine_shas.dedup();
    assert_eq!(engine_shas.len(), 4, "four independent engine digests");
    let live = &document["engines"]["live_pose"];
    assert_eq!(text(live, "precision"), "fp16");
    assert!(
        live["tf32_enabled"].is_boolean(),
        "live tf32 is a recorded bool"
    );
    assert_eq!(
        text(live, "observer_library_sha256"),
        sha256_file(Path::new(OBSERVER_LIBRARY))
    );
    for role in ["stored_pose", "bed", "fall"] {
        let entry = &document["engines"][role];
        assert_eq!(text(entry, "precision"), "fp32");
        assert_eq!(entry["tf32_enabled"], false);
    }
    let native = hardware_identity(0).expect("actual native hardware_identity(0)");
    let pinned: Value = serde_json::from_str(MANIFEST).expect("pinned gpu manifest");
    let pinned_name = text(&pinned["gpu"], "name");
    let pinned_compute = text(&pinned["gpu"], "compute_capability");
    assert_eq!(
        native.device_name, pinned_name,
        "native device must match pinned gpu manifest"
    );
    assert_eq!(
        format!("{}.{}", native.compute_major, native.compute_minor),
        pinned_compute,
        "native compute capability must match pinned gpu manifest"
    );
    for role in ROLES {
        let entry = &document["engines"][role];
        assert_eq!(text(entry, "device_name"), pinned_name);
        assert_eq!(text(entry, "compute_capability"), pinned_compute);
        assert_eq!(entry["trt_version"], i64::from(native.trt_version));
    }
    let first = &document["engines"][ROLES[0]];
    for role in &ROLES[1..] {
        let entry = &document["engines"][role];
        assert_eq!(entry["trt_version"], first["trt_version"]);
        assert_eq!(entry["device_name"], first["device_name"]);
        assert_eq!(entry["compute_capability"], first["compute_capability"]);
    }
    assert!(root.join("nvinfer-served.txt").is_file());
    println!("ENGINE_BUILD_ROOT={}", root.display());
    println!("ENGINE_BUILD_IDENTITY={}", identity.display());
}

fn check_config_command(root: &Path, state: &Path, image: &str, identity: &Path) -> Command {
    let pose = required_file("SEEON_TEST_STORED_POSE_ONNX");
    let parser = fs::canonicalize(PARSER_LIBRARY).expect("installed parser");
    let tracker = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../adapters/deepstream/configs/config_tracker_NvDCF_perf.yml");
    let mut command = Command::new(env!("CARGO_BIN_EXE_ml-worker"));
    command
        .current_dir(root)
        .env_clear()
        .arg("check-config")
        .arg("--state-dir")
        .arg(state)
        .env("RELAY_TOKEN", "engine-build-cli-check")
        .env("ML_WORKER_IMAGE", image)
        .env("ML_WORKER_PROFILE", "flow")
        .env(
            "ML_WORKER_MODEL_SELECTION_PATH",
            root.join("absent-selection.json"),
        )
        .env("ML_WORKER_EXECUTION_RECORDS_ENABLED", "false")
        .env("ML_WORKER_FLOW_BATCH_SIZE", "2")
        .env("ML_WORKER_FLOW_RECORD_DIR", root.join("records"))
        .env("ML_WORKER_FLOW_RECORD_CACHE_SECONDS", "30")
        .env("ML_WORKER_FLOW_FRAME_WIDTH", "1280")
        .env("ML_WORKER_FLOW_FRAME_HEIGHT", "720")
        .env("ML_WORKER_FLOW_ENGINE_PATH", root.join("live-pose.engine"))
        .env(
            "ML_WORKER_STORED_POSE_ENGINE_PATH",
            root.join("stored-pose.engine"),
        )
        .env("ML_WORKER_BED_ENGINE_PATH", root.join("bed.engine"))
        .env("ML_WORKER_FALL_ENGINE_PATH", root.join("fall.engine"))
        .env("ML_WORKER_FLOW_ENGINE_IDENTITY_PATH", identity)
        .env("ML_WORKER_FLOW_ONNX_PATH", pose)
        .env(
            "ML_WORKER_FLOW_INFER_CONFIG",
            root.join("nvinfer-served.txt"),
        )
        .env(
            "ML_WORKER_FLOW_TRACKER_CONFIG",
            fs::canonicalize(&tracker).expect("absolute tracker config"),
        )
        .env("ML_WORKER_FLOW_TRACKER_LIBRARY", TRACKER_LIBRARY)
        .env("ML_WORKER_FLOW_PARSER_LIBRARY", parser)
        .env("CUDA_VISIBLE_DEVICES", "");
    for name in ["PATH", "LD_LIBRARY_PATH", "GST_PLUGIN_PATH"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
}

fn run_captured(
    root: &Path,
    label: &'static str,
    command: Command,
    deadline: Duration,
) -> (ExitStatus, PathBuf) {
    let (stdout, stderr, _, stderr_path) = capture_stdio(root, label);
    let mut command = command;
    command.stdout(stdout).stderr(stderr);
    let mut owned = OwnedBinary::spawn(command);
    let status = await_owned(&mut owned, deadline, label);
    (status, stderr_path)
}

#[test]
#[ignore = "requires an actual GPU, empty SEEON_TEST_ENGINE_BUILD_ROOT, SEEON_TEST_STORED_POSE_ONNX, SEEON_TEST_BED_ONNX, SEEON_TEST_FALL_ONNX, ML_WORKER_IMAGE, mounted custom parser, and installed DeepStream observer/tracker libraries; retains the four engines for later boot tests"]
fn engine_build_cli_publishes_four_engines_then_reuses_cache_and_checks_files() {
    for path in [OBSERVER_LIBRARY, TRACKER_LIBRARY] {
        assert!(Path::new(path).is_file(), "{path} is required");
    }
    let root = PathBuf::from(
        std::env::var_os("SEEON_TEST_ENGINE_BUILD_ROOT")
            .expect("SEEON_TEST_ENGINE_BUILD_ROOT is required"),
    );
    let fall = required_file("SEEON_TEST_FALL_ONNX");
    let bed = required_file("SEEON_TEST_BED_ONNX");
    let pose = required_file("SEEON_TEST_STORED_POSE_ONNX");
    let image = required_text("ML_WORKER_IMAGE");
    assert!(
        seeon_ml_worker::config::model_bundle::identity::deployment_image_digest(&image).is_some(),
        "ML_WORKER_IMAGE must be an actual caller image reference"
    );
    let state = prepare_root(&root, &fall, &bed);
    let (stdout, stderr, _, stderr_path) = capture_stdio(&root, "engine-build");
    let mut command = build_command(&root);
    command.stdout(stdout).stderr(stderr);
    let mut owned = OwnedBinary::spawn(command);
    let status = await_owned(&mut owned, BUILD_WAIT, "engine-build");
    assert_exit(status, 0, &stderr_path);
    assert_aggregate(&root, &image, &pose, &bed, &fall);
    let stages_before = engine_stage_dirs(&root);
    let before: Vec<_> = ENGINES
        .into_iter()
        .map(|name| stamp(&root.join(name)))
        .chain(std::iter::once(stamp(&root.join("engine-identity.json"))))
        .collect();

    let (status, stderr_path) = run_captured(
        &root,
        "engine-build-cache",
        build_command(&root),
        BUILD_WAIT,
    );
    assert_exit(status, 0, &stderr_path);
    for (path, earlier) in ENGINES
        .into_iter()
        .map(|name| root.join(name))
        .chain(std::iter::once(root.join("engine-identity.json")))
        .zip(&before)
    {
        let later = stamp(&path);
        assert_eq!(
            later.bytes,
            earlier.bytes,
            "{} bytes changed",
            path.display()
        );
        assert_eq!(
            later.sha256,
            earlier.sha256,
            "{} hash changed",
            path.display()
        );
        assert_eq!((later.dev, later.ino), (earlier.dev, earlier.ino));
        assert_eq!(
            (later.mtime, later.mtime_nsec),
            (earlier.mtime, earlier.mtime_nsec)
        );
    }
    assert_eq!(
        engine_stage_dirs(&root),
        stages_before,
        "cache hit must not create an engine-stage directory"
    );

    let (status, stderr_path) = run_captured(
        &root,
        "check-config",
        check_config_command(&root, &state, &image, &root.join("engine-identity.json")),
        CHECK_WAIT,
    );
    assert_exit(status, 0, &stderr_path);

    let identity = root.join("engine-identity.json");
    let original = fs::read(&identity).expect("identity bytes");
    let mut copied: Value = serde_json::from_slice(&original).expect("identity json");
    copied["engines"]
        .as_object_mut()
        .expect("engines")
        .remove("fall")
        .expect("fall entry");
    let copy = root.join("engine-identity-missing-fall.json");
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&copy)
        .expect("exclusive owned identity copy");
    file.write_all(serde_json::to_vec(&copied).expect("copy json").as_slice())
        .and_then(|()| file.write_all(b"\n"))
        .expect("write identity copy");
    let (status, stderr_path) = run_captured(
        &root,
        "check-config-missing-fall",
        check_config_command(&root, &state, &image, &copy),
        CHECK_WAIT,
    );
    assert_exit(status, 3, &stderr_path);
    assert_eq!(fs::read(&identity).expect("unchanged identity"), original);
    assert!(root.join("nvinfer-served.txt").is_file());
    for name in ENGINES {
        assert!(root.join(name).is_file(), "{name} retained");
    }
}
