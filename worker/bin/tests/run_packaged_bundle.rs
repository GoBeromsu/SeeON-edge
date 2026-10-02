//! Real filesystem admission; opaque model bytes do not claim GPU/model loading.
use std::fs;
use std::io::Write;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use seeon_ml_worker::config::model_bundle::AdmissionKind;
use seeon_ml_worker::config::model_bundle::packaged::admit_packaged_bundle;
use seeon_ml_worker::records::id::sha256_hex;
use serde_json::{Value, json};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[test]
fn a_named_pipe_is_refused_without_waiting_for_a_writer() {
    let fixture = Fixture::new();
    let path = fixture.root.join("model.onnx");
    fs::remove_file(&path).unwrap();
    rustix::fs::mknodat(
        rustix::fs::CWD,
        &path,
        rustix::fs::FileType::Fifo,
        rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        0,
    )
    .expect("owned FIFO");
    let (sender, receiver) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        sender.send(admit_packaged_bundle(&fixture.root)).unwrap();
    });
    let result = receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("nonregular member must not block admission");
    reader.join().expect("reader joined");
    assert_eq!(result.unwrap_err().kind, AdmissionKind::NotRegularFile);
}

struct Fixture {
    base: PathBuf,
    root: PathBuf,
    manifest: Value,
}
impl Fixture {
    fn new() -> Self {
        let base = std::env::temp_dir().join(format!(
            "seeon-packaged-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&base).expect("own fresh temp directory");
        let root = base.join("bundle");
        fs::create_dir_all(root.join("conformance")).unwrap();
        // The parser's shape is valid; runtime domain validation is separately
        // covered by run_conformance. No fake inference session is constructed.
        let conformance = serde_json::to_vec(&json!({
            "preprocessing_identity":"fixture", "vector":{"length":56,"tail_indices":{}},
            "confidence":{"gate":0.5}, "temporal":{"window_frames":30,"stride_frames":5,"fps":15},
            "coordinate_system":{}, "keypoint_order":[]
        }))
        .unwrap();
        let bodies: [(&str, &[u8]); 4] = [
            ("model.onnx", b"o"),
            ("model.pt", b"weights"),
            ("calibration.json", b"{\"temperature\":1.5}"),
            ("conformance/case.json", &conformance),
        ];
        let mut entries = Vec::new();
        for (relative, bytes) in bodies {
            fs::write(root.join(relative), bytes).unwrap();
            entries.push(
                json!({"relative_path":relative,"sha256":sha256_hex(bytes),"size":bytes.len()}),
            );
        }
        let fixture = Self {
            base,
            root,
            manifest: json!({"files":entries}),
        };
        fixture.save();
        fixture
    }
    fn save(&self) {
        fs::write(
            self.root.join("bundle-manifest.json"),
            serde_json::to_vec(&self.manifest).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.base).expect("remove owned fixture");
    }
}

fn cases() -> Vec<(&'static str, Fixture, bool)> {
    let mut cases = Vec::new();
    for (name, accepted) in [
        ("verified members", true),
        ("unlisted extra file", true),
        ("duplicate ordinary member", true),
        ("bool integer size", true),
        ("normalized conformance path", true),
        ("linked root", true),
        ("linked member outside root", true),
        ("linked directory outside root", true),
        ("tampered model", false),
        ("bad digest", false),
        ("wrong size", false),
        ("float size", false),
        ("unlisted model", false),
        ("unlisted weights", false),
        ("unlisted calibration", false),
        ("duplicate conformance", false),
        ("unlisted conformance", false),
        ("parent path", false),
        ("absolute path", false),
        ("bad duplicate", false),
        ("missing member", false),
    ] {
        let mut f = Fixture::new();
        match name {
            "verified members" => {}
            "unlisted extra file" => {
                fs::write(
                    f.root.join("evaluation-receipt.json"),
                    b"not part of this manifest",
                )
                .unwrap();
            }
            "duplicate ordinary member" | "bad duplicate" => {
                let mut duplicate = f.manifest["files"][0].clone();
                if name == "bad duplicate" {
                    duplicate["sha256"] = json!("f".repeat(64));
                }
                f.manifest["files"].as_array_mut().unwrap().push(duplicate);
            }
            "bool integer size" => f.manifest["files"][0]["size"] = json!(true),
            "normalized conformance path" => {
                f.manifest["files"][3]["relative_path"] = json!("./conformance/./case.json")
            }
            "linked root" => {
                let alias = f.base.join("alias");
                symlink(&f.root, &alias).unwrap();
                f.root = alias;
            }
            "linked member outside root" => {
                let target = f.base.join("model.onnx");
                fs::write(&target, b"o").unwrap();
                fs::remove_file(f.root.join("model.onnx")).unwrap();
                symlink(&target, f.root.join("model.onnx")).unwrap();
            }
            "linked directory outside root" => {
                let target = f.base.join("conformance-target");
                fs::rename(f.root.join("conformance"), &target).unwrap();
                symlink(&target, f.root.join("conformance")).unwrap();
            }
            "tampered model" => fs::write(f.root.join("model.onnx"), b"x").unwrap(),
            "bad digest" => f.manifest["files"][0]["sha256"] = json!("f".repeat(64)),
            "wrong size" => f.manifest["files"][0]["size"] = json!(2),
            "float size" => f.manifest["files"][0]["size"] = json!(1.0),
            "unlisted model"
            | "unlisted weights"
            | "unlisted calibration"
            | "unlisted conformance" => {
                let index = match name {
                    "unlisted model" => 0,
                    "unlisted weights" => 1,
                    "unlisted calibration" => 2,
                    _ => 3,
                };
                f.manifest["files"].as_array_mut().unwrap().remove(index);
            }
            "duplicate conformance" => {
                let duplicate = f.manifest["files"][3].clone();
                f.manifest["files"].as_array_mut().unwrap().push(duplicate);
            }
            "parent path" => {
                fs::write(f.base.join("model.onnx"), b"o").unwrap();
                let mut extra = f.manifest["files"][0].clone();
                extra["relative_path"] = json!("../model.onnx");
                f.manifest["files"].as_array_mut().unwrap().push(extra);
            }
            "absolute path" => {
                let mut extra = f.manifest["files"][0].clone();
                extra["relative_path"] = json!(f.root.join("model.onnx"));
                f.manifest["files"].as_array_mut().unwrap().push(extra);
            }
            "missing member" => fs::remove_file(f.root.join("model.onnx")).unwrap(),
            _ => unreachable!(),
        }
        f.save();
        cases.push((name, f, accepted));
    }
    cases
}

#[test]
fn admitted_bytes_and_member_identities_survive_later_file_changes() {
    let f = Fixture::new();
    let expected_calibration = fs::read(f.root.join("calibration.json")).unwrap();
    let expected_conformance = fs::read(f.root.join("conformance/case.json")).unwrap();
    let proof = admit_packaged_bundle(&f.root).expect("verified files");
    fs::write(f.root.join("calibration.json"), b"changed").unwrap();
    fs::write(f.root.join("conformance/case.json"), b"changed").unwrap();
    assert_eq!(proof.calibration, expected_calibration);
    assert_eq!(
        proof.conformance,
        ("conformance/case.json".to_owned(), expected_conformance)
    );
    assert_eq!(proof.member_digests["model.onnx"], sha256_hex(b"o"));
    assert_eq!(proof.member_digests["model.pt"], sha256_hex(b"weights"));
    assert!(
        admit_packaged_bundle(&f.root).is_err(),
        "fresh admission detects changed bytes"
    );
}

#[test]
fn file_integrity_and_legacy_packaged_path_semantics_are_preserved() {
    for (name, f, expected) in cases() {
        assert_eq!(admit_packaged_bundle(&f.root).is_ok(), expected, "{name}");
    }
}

const ORACLE: &str = r#"
import json, sys
from pathlib import Path
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import read_json, verify_bundle, member_digest
from worker.adapters.model.ort_pose_bbox56 import _packaged_conformance
results = []
for directory in json.load(sys.stdin):
    root = Path(directory).resolve()
    try:
        manifest = read_json(root / 'bundle-manifest.json')
        verify_bundle(root, manifest)
        for member in ['model.onnx', 'model.pt', 'calibration.json']:
            member_digest(manifest, member)
        _packaged_conformance(root, manifest)
    except ModelLoadError:
        results.append(False)
    else:
        results.append(True)
print(json.dumps(results))
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with original packaged bundle verifier"]
fn filesystem_admission_matches_actual_python_helpers() {
    let cases = cases();
    let paths: Vec<_> = cases.iter().map(|(_, f, _)| &f.root).collect();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child =
        Command::new(std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON required"))
            .arg("-c")
            .arg(ORACLE)
            .current_dir(&repo)
            .env("PYTHONPATH", &repo)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("oracle starts");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&paths).unwrap())
        .unwrap();
    let output = child.wait_with_output().expect("oracle completes");
    assert!(output.status.success());
    let results: Vec<bool> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    assert_eq!(results.len(), cases.len());
    for ((name, f, expected), oracle) in cases.iter().zip(results) {
        assert_eq!(oracle, *expected, "Python case {name}");
        assert_eq!(
            admit_packaged_bundle(&f.root).is_ok(),
            oracle,
            "Python parity {name}"
        );
    }
}
