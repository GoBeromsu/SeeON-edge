//! T4 part 1: `ml-worker check-config` exit codes as a process, against the
//! bundle Python published at 030aaf1
//! (`tests/fixtures/worker-wire/d/model-bundle-admission/publication.json`).
//! Env refusals and refused flags exit 2 (`Exit::Config`), a bundle that
//! fails admission exits 3 (`Exit::RefuseToStart`), and a valid config exits
//! 0 without the relay, which the env points at an unroutable address.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use seeon_ml_worker::json::{Json, Serialiser};
use serde_json::Value;
use sha2::{Digest, Sha256};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

/// A scratch root holding an empty `HOME`, a `--state-dir` that does not
/// exist yet, the published models tree and the selection document.
struct Fixture {
    root: PathBuf,
    member: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
        let name = format!("cli-exit-{}-{serial}", std::process::id());
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
        if root.exists() {
            fs::remove_dir_all(&root).expect("stale scratch removed");
        }
        fs::create_dir_all(root.join("home")).expect("home");
        let publication = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/worker-wire/d/model-bundle-admission/publication.json");
        let publication = fs::read_to_string(publication).expect("publication golden");
        let publication: Value = serde_json::from_str(&publication).expect("publication JSON");
        let bodies = publication["bodies"].as_object().expect("bodies");
        let bodies = bodies.values().chain([&publication["manifest_json"]]);
        let bodies: Vec<&str> = bodies
            .map(|body| body["text"].as_str().expect("text"))
            .collect();
        let mut member = None;
        for node in publication["published_tree"].as_array().expect("tree") {
            let path = root.join(node["path"].as_str().expect("node path"));
            if node["kind"] == "dir" {
                fs::create_dir_all(&path).expect("dir");
                continue;
            }
            assert_eq!(node["kind"], "file", "published trees hold dirs and files");
            let body = bodies
                .iter()
                .map(|body| body.as_bytes())
                .find(|body| sha256_hex(body) == node["sha256"].as_str().expect("sha256"))
                .unwrap_or_else(|| panic!("no body for {node}"));
            fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
            fs::write(&path, body).expect("member");
            if !path.ends_with("bundle-manifest.json") {
                member = Some(path);
            }
        }
        let selection = Json::from(&publication["selection_document"]);
        let selection = Serialiser::FetchModelsManifest
            .canonical(&selection)
            .expect("selection canonical");
        fs::write(root.join("model-selection.json"), selection).expect("selection");
        let member = member.expect("a member besides the manifest");
        Self { root, member }
    }

    fn state_dir(&self) -> PathBuf {
        self.root.join("state")
    }

    /// `check-config --state-dir <state> <extra>` under a cleared env that
    /// has `env` plus `HOME` and the native library search path; returns the exit code.
    fn run(&self, env: &[(&str, &str)], extra: &[&str]) -> i32 {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ml-worker"));
        command
            .arg("check-config")
            .arg("--state-dir")
            .arg(self.state_dir())
            .args(extra)
            .env_clear()
            .env("HOME", self.root.join("home"))
            .envs(env.iter().copied());
        // The executable now links the real native owners, even for static
        // check-config. Preserve loader configuration, not worker settings.
        if let Some(path) = std::env::var_os("LD_LIBRARY_PATH") {
            command.env("LD_LIBRARY_PATH", path);
        }
        let status = command.status().expect("ml-worker runs");
        status.code().expect("exited, not signalled")
    }

    /// The env of a valid config, the relay at an unroutable address.
    fn valid_env(&self) -> Vec<(String, String)> {
        let path = |relative: &str| self.root.join(relative).display().to_string();
        vec![
            ("RELAY_TOKEN".to_owned(), "ci-boot-smoke-test".to_owned()),
            (
                "ML_WORKER_RELAY_ENDPOINT_UNUSED".to_owned(),
                "http://relay.invalid:8000".to_owned(),
            ),
            (
                "ML_WORKER_MODEL_SELECTION_PATH".to_owned(),
                path("model-selection.json"),
            ),
            ("ML_WORKER_MODELS_ROOT".to_owned(), path("models")),
        ]
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn borrowed(env: &[(String, String)]) -> Vec<(&str, &str)> {
    env.iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect()
}

/// Every path below `root`, directories included.
fn paths(root: &Path) -> BTreeSet<PathBuf> {
    let mut found = BTreeSet::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("scratch dir") {
            let path = entry.expect("scratch entry").path();
            if path.is_dir() {
                pending.push(path.clone());
            }
            found.insert(path);
        }
    }
    found
}

#[test]
fn missing_env_exits_config() {
    let fixture = Fixture::new();
    let mut env = fixture.valid_env();
    env.retain(|(key, _)| key != "RELAY_TOKEN");
    assert_eq!(fixture.run(&borrowed(&env), &[]), 2);
}

#[test]
fn refused_flags_exit_config() {
    let fixture = Fixture::new();
    let env = fixture.valid_env();
    for extra in [
        &["--unknown"][..],
        &["--config", "worker.yaml"],
        &["--max-frames-per-camera", "1"],
    ] {
        assert_eq!(fixture.run(&borrowed(&env), extra), 2, "{extra:?}");
    }
}

#[test]
fn valid_config_exits_zero_and_writes_only_state() {
    let fixture = Fixture::new();
    let before = paths(&fixture.root);
    assert_eq!(fixture.run(&borrowed(&fixture.valid_env()), &[]), 0);
    let created: Vec<PathBuf> = paths(&fixture.root).difference(&before).cloned().collect();
    let state = fixture.state_dir();
    assert!(
        created.iter().all(|path| path.starts_with(&state)),
        "{created:?}"
    );
    let home = fs::read_dir(fixture.root.join("home")).expect("home");
    assert_eq!(home.count(), 0);
}

#[test]
fn flipped_member_byte_refuses_to_start() {
    let fixture = Fixture::new();
    let mut body = fs::read(&fixture.member).expect("member");
    body[0] ^= 1;
    fs::write(&fixture.member, body).expect("flip");
    let env = fixture.valid_env();
    assert_eq!(fixture.run(&borrowed(&env), &[]), 3);
    let mut no_gpu = borrowed(&env);
    no_gpu.push(("CUDA_VISIBLE_DEVICES", ""));
    assert_eq!(fixture.run(&no_gpu, &[]), 3);
}
