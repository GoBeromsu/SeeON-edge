//! Publisher documents are validated against the actual runner and domain contract.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use seeon_ml_worker::config::model_bundle::conformance::validate;
use seeon_ml_worker::json::Json;
use serde_json::{Value, json};

#[path = "support/conformance_document.rs"]
mod conformance_document;
use conformance_document::document;

fn cases() -> Vec<(&'static str, Value, bool)> {
    let mut cases = vec![("publisher contract", document(), true)];
    for (name, pointer, value, accepted) in [
        ("identity", "/preprocessing_identity", json!("other"), false),
        ("length", "/vector/length", json!(55), false),
        ("float length", "/vector/length", json!(56.0), true),
        ("tail index", "/vector/tail_indices/valid", json!(54), false),
        ("float tail", "/vector/tail_indices/x1", json!(51.0), false),
        ("bool tail", "/vector/tail_indices/x1", json!(true), false),
        (
            "keypoint order",
            "/keypoint_order/0",
            json!("left_eye"),
            false,
        ),
        ("nonstr keypoint", "/keypoint_order/0", json!(0), false),
        ("gate", "/confidence/gate", json!(0.25), false),
        ("bool gate", "/confidence/gate", json!(true), false),
        ("window", "/temporal/window_frames", json!(29), false),
        ("float window", "/temporal/window_frames", json!(30.0), true),
        ("stride", "/temporal/stride_frames", json!(4), false),
        ("float stride", "/temporal/stride_frames", json!(5.0), false),
        ("bool stride", "/temporal/stride_frames", json!(true), false),
        ("fps", "/temporal/fps", json!(30.0), false),
        ("int fps", "/temporal/fps", json!(15), true),
        ("bool fps", "/temporal/fps", json!(true), false),
        (
            "origin",
            "/coordinate_system/origin",
            json!("bottom_left"),
            false,
        ),
        (
            "x denominator",
            "/coordinate_system/xy_normalization_denominators/x",
            json!("frame_height"),
            false,
        ),
        (
            "normalization",
            "/coordinate_system/xy_normalization_rule",
            json!("subtract half pixel"),
            false,
        ),
        ("missing temporal", "/temporal", Value::Null, false),
        ("missing confidence", "/confidence", Value::Null, false),
    ] {
        let mut doc = document();
        *doc.pointer_mut(pointer).expect("existing case field") = value;
        cases.push((name, doc, accepted));
    }
    let mut extra = document();
    extra["coordinate_system"]["publisher_note"] = json!("한글 extra metadata");
    cases.push(("extra coordinate metadata", extra, true));
    let mut extra = document();
    extra["coordinate_system"]["xy_normalization_denominators"]["z"] = json!("depth");
    cases.push(("extra denominator", extra, false));
    let mut extra = document();
    extra["vector"]["tail_indices"]["extra"] = json!(56);
    cases.push(("extra tail index", extra, false));
    let mut missing = document();
    missing["vector"]["tail_indices"]
        .as_object_mut()
        .unwrap()
        .remove("x1");
    cases.push(("missing tail index", missing, false));
    cases.push(("nonobject document", json!([]), false));
    cases
}

fn accepted(doc: &Value) -> bool {
    validate(&Json::from(doc)).is_ok()
}

#[test]
fn changed_feature_contracts_refuse_without_rejecting_allowed_metadata() {
    for (name, doc, expected) in cases() {
        assert_eq!(accepted(&doc), expected, "{name}");
    }
}

const ORACLE: &str = r#"
import json, sys, tempfile
from pathlib import Path
from worker.adapters.model.ort_pose_bbox56 import _parse_conformance, _validate_runner_conformance
from worker.adapters.model.errors import ModelLoadError
from worker.runtime.worker import _validate_fall_bundle_conformance
results = []
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for document in json.load(sys.stdin):
        (root / 'conformance.json').write_text(json.dumps(document), encoding='utf-8')
        try:
            conformance = _parse_conformance(root, 'conformance.json')
            _validate_runner_conformance(conformance)
            _validate_fall_bundle_conformance(conformance)
        except ModelLoadError:
            results.append(False)
        else:
            results.append(True)
print(json.dumps(results))
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with canonical model and domain conformance validators"]
fn all_admissions_match_actual_python_runner_and_domain() {
    let cases = cases();
    let documents: Vec<_> = cases.iter().map(|(_, doc, _)| doc).collect();
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
            .expect("Python oracle starts");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(&serde_json::to_vec(&documents).unwrap())
        .expect("oracle input");
    let output = child.wait_with_output().expect("oracle finishes");
    assert!(
        output.status.success(),
        "canonical Python validators must execute"
    );
    let expected: Vec<bool> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    assert_eq!(expected.len(), cases.len());
    for ((name, doc, expected_case), oracle) in cases.iter().zip(expected) {
        assert_eq!(oracle, *expected_case, "Python case {name}");
        assert_eq!(accepted(doc), oracle, "Python parity {name}");
    }
}
