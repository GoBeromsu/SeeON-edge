//! Metadata after bundle admission; wire_config covers selected file admission.
//! The Python oracle invokes real runner constructors, replacing only ORT execution.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use seeon_ml_worker::config::model_bundle::bundle::BundleProof;
use seeon_ml_worker::config::model_bundle::packaged::PackagedProof;
use seeon_ml_worker::config::selection::{ModelSelection, Publication};
use seeon_ml_worker::records::id::sha256_hex;
use seeon_ml_worker::run::IdentityError;
use seeon_ml_worker::run::fall_evidence::{self, FallEvidence};
use serde_json::{Value, json};

#[path = "support/conformance_document.rs"]
mod conformance_document;
const PREPROCESSING: &str = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1";
const MODEL: &[u8] = b"opaque-onnx";
const WEIGHTS: &[u8] = b"opaque-weights";

#[derive(Clone)]
struct Case {
    name: &'static str,
    packaged: bool,
    source: &'static str,
    threshold: f64,
    classes: i128,
    preprocessing: &'static str,
    calibration: Value,
    conformance: Value,
    accepted: bool,
}
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (name, accepted) in [
        ("packaged", true),
        ("selected default", true),
        ("selected receipt", true),
        ("default missing promotion", true),
        ("default malformed promotion", true),
        ("default near threshold", true),
        ("default wrong threshold", false),
        ("receipt near grant", true),
        ("receipt far grant", false),
        ("receipt missing promotion", false),
        ("receipt false promotion", false),
        ("receipt malformed promotion", false),
        ("receipt missing grant", false),
        ("packaged missing promotion", false),
        ("wrong output count", false),
        ("wrong source", false),
        ("wrong coordinates", false),
        ("selection preprocessing mismatch", false),
        ("calibration preprocessing mismatch", false),
        ("bad temperature", false),
        ("wrong class order", false),
    ] {
        let mut case = Case {
            name,
            packaged: name.starts_with("packaged"),
            source: if name.contains("default") {
                "default"
            } else {
                "receipt"
            },
            threshold: if name.contains("default") { 0.5 } else { 0.7 },
            classes: 2,
            preprocessing: PREPROCESSING,
            accepted,
            calibration: json!({"class_order":["non_fall","fall_transition_proxy"],
                "preprocessing_identity_digest":sha256_hex(PREPROCESSING.as_bytes()),
                "temperature":2.5,"threshold":0.7,"promotion_eligible":true,
                "temporal_rule":{"m":2,"n":4}}),
            conformance: conformance_document::document(),
        };
        match name {
            "default missing promotion"
            | "receipt missing promotion"
            | "packaged missing promotion" => {
                case.calibration
                    .as_object_mut()
                    .unwrap()
                    .remove("promotion_eligible");
            }
            "default malformed promotion" | "receipt malformed promotion" => {
                case.calibration["promotion_eligible"] = json!("yes")
            }
            "default near threshold" => case.threshold = 0.5 + 4e-10,
            "default wrong threshold" => case.threshold = 0.6,
            "receipt near grant" => case.threshold = 0.7 + 6e-10,
            "receipt far grant" => case.threshold = 0.7 + 8e-10,
            "receipt false promotion" => case.calibration["promotion_eligible"] = json!(false),
            "receipt missing grant" => {
                case.calibration
                    .as_object_mut()
                    .unwrap()
                    .remove("threshold");
            }
            "wrong output count" => case.classes = 3,
            "wrong source" => case.source = "operator",
            "wrong coordinates" => {
                case.conformance["coordinate_system"]["origin"] = json!("bottom_left")
            }
            "selection preprocessing mismatch" => case.preprocessing = "other",
            "calibration preprocessing mismatch" => {
                case.calibration["preprocessing_identity_digest"] = json!("0".repeat(64))
            }
            "bad temperature" => case.calibration["temperature"] = json!(0),
            "wrong class order" => {
                case.calibration["class_order"] = json!(["fall_transition_proxy", "non_fall"])
            }
            _ => {}
        }
        cases.push(case);
    }
    cases
}
fn proof(case: &Case) -> PackagedProof {
    let calibration = serde_json::to_vec(&case.calibration).unwrap();
    let conformance = serde_json::to_vec(&case.conformance).unwrap();
    PackagedProof {
        member_digests: [
            ("model.onnx".into(), sha256_hex(MODEL)),
            ("model.pt".into(), sha256_hex(WEIGHTS)),
            ("calibration.json".into(), sha256_hex(&calibration)),
            ("conformance/case.json".into(), sha256_hex(&conformance)),
        ]
        .into(),
        calibration,
        conformance: ("conformance/case.json".into(), conformance),
    }
}
fn selected_inputs(case: &Case, proof: PackagedProof) -> (ModelSelection, BundleProof) {
    // Opaque publication/receipt fields are not fabricated runtime evidence:
    // this test starts at the post-admission boundary and never emits them.
    let publication = Publication {
        source_locator: "https://models.invalid/fixture".into(),
        revision: "fixture".into(),
        content: "a".repeat(64),
    };
    let selection = ModelSelection {
        model_publication: publication.clone(),
        dataset_publication: publication,
        bundle_members_digest: "b".repeat(64),
        evaluation_receipt_digest: "c".repeat(64),
        field_evaluation_receipt_digest: "d".repeat(64),
        input_observation_schema: "e".repeat(64),
        output_class_semantics_digest: "f".repeat(64),
        policy_digest: "0".repeat(64),
        runtime_format: "onnx".into(),
        bundle_format: "fixture".into(),
        calibration_digest: proof.member_digests["calibration.json"].clone(),
        conformance_digest: proof.member_digests["conformance/case.json"].clone(),
        output_class_count: case.classes,
        preprocessing_identity: case.preprocessing.into(),
        transition_threshold: case.threshold,
        threshold_source: case.source.into(),
    };
    let bundle = BundleProof {
        bundle_sha256: selection.model_publication.content.clone(),
        members: proof.member_digests.keys().cloned().collect(),
        receipts: Vec::new(),
        member_digests: proof.member_digests,
        identities: BTreeMap::new(),
        calibration: proof.calibration,
        conformance: proof.conformance,
    };
    (selection, bundle)
}
fn admit(case: &Case) -> Result<FallEvidence, IdentityError> {
    let proof = proof(case);
    if case.packaged {
        fall_evidence::packaged(&proof)
    } else {
        let (selection, proof) = selected_inputs(case, proof);
        fall_evidence::selected(&selection, &proof)
    }
}
fn snapshot(evidence: FallEvidence) -> Value {
    let c = evidence.calibration;
    json!({"temperature":c.temperature,"threshold":c.receipt_threshold,
        "votes":c.transition_votes,"window":c.transition_window,"promoted":c.promotion_eligible,
        "model_version":evidence.model_version,"weights":evidence.published_weights_digest,
        "calibration_digest":evidence.calibration_digest,"preprocessing":evidence.preprocessing_identity})
}

#[test]
fn runtime_metadata_refuses_incompatible_contracts_before_model_creation() {
    for case in cases() {
        assert_eq!(admit(&case).is_ok(), case.accepted, "{}", case.name);
    }
}

#[test]
fn selected_source_not_publisher_flag_owns_promotion_and_declared_threshold() {
    for case in cases().into_iter().filter(|c| c.accepted) {
        let evidence = admit(&case).unwrap();
        assert_eq!(evidence.model_version, sha256_hex(MODEL));
        assert_eq!(evidence.published_weights_digest, sha256_hex(WEIGHTS));
        assert_eq!(
            evidence.calibration_digest,
            sha256_hex(&serde_json::to_vec(&case.calibration).unwrap())
        );
        assert_eq!(evidence.calibration.temperature, 2.5);
        assert_eq!(
            (
                evidence.calibration.transition_votes,
                evidence.calibration.transition_window
            ),
            (2, 4)
        );
        if !case.packaged {
            assert_eq!(
                evidence.calibration.promotion_eligible,
                case.source == "receipt"
            );
            assert_eq!(evidence.calibration.receipt_threshold, Some(case.threshold));
        }
    }
}

const ORACLE: &str = r#"
import hashlib, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from worker.adapters.model.ort_pose_bbox56 import OrtPoseBbox56Runner
from worker.adapters.model.errors import ModelLoadError
from worker.adapters.model.pose_bbox56_bundle_support import member_digest
from worker.runtime.worker import _validate_fall_bundle_conformance
class Session:
    def run(self, outputs, inputs):
        assert outputs is None and inputs['window'].shape == (1,30,56)
        return [np.zeros((1,1), dtype=np.float32)]
results=[]
for case in json.load(sys.stdin):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); (root/'conformance').mkdir()
        bodies={'model.onnx':b'opaque-onnx','model.pt':b'opaque-weights',
            'calibration.json':case['calibration_raw'].encode(),
            'conformance/case.json':case['conformance_raw'].encode()}
        digests={p:hashlib.sha256(b).hexdigest() for p,b in bodies.items()}
        manifest={'files':[{'relative_path':p,'sha256':digests[p],'size':len(b)} for p,b in bodies.items()]}
        for p,b in bodies.items(): (root/p).write_bytes(b)
        (root/'bundle-manifest.json').write_text(json.dumps(manifest))
        def factory(path, providers):
            assert Path(path)==root/'model.onnx' and providers==['CPUExecutionProvider']
            return Session()
        try:
            if case['packaged']:
                runner=OrtPoseBbox56Runner.from_artifact_dir(root,session_factory=factory)
            else:
                selection=SimpleNamespace(output_class_count=case['classes'],threshold_source=case['source'],
                    transition_threshold=case['threshold'],preprocessing_identity=case['preprocessing'],
                    conformance_digest=digests['conformance/case.json'])
                runner=OrtPoseBbox56Runner.from_admitted_bundle(root,
                    SimpleNamespace(observed={'member_digests':digests}),selection,session_factory=factory)
            _validate_fall_bundle_conformance(runner.conformance)
        except ModelLoadError:
            results.append(None)
        else:
            results.append({'temperature':runner._temperature,'threshold':runner.receipt_threshold,
                'votes':runner.receipt_transition_votes,'window':runner.receipt_transition_window,
                'promoted':runner.promotion_eligible,'model_version':runner.artifact_digest,
                'weights':member_digest(manifest,'model.pt'),'calibration_digest':digests['calibration.json'],
                'preprocessing':runner.preprocessing_identity})
print(json.dumps(results))
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with actual selected and packaged runner constructors"]
fn metadata_matches_actual_python_runners_with_only_ort_execution_replaced() {
    let cases = cases();
    let input: Vec<_> = cases
        .iter()
        .map(|c| {
            json!({"packaged":c.packaged,"classes":c.classes,
        "source":c.source,"threshold":c.threshold,"preprocessing":c.preprocessing,
        "calibration_raw":serde_json::to_string(&c.calibration).unwrap(),
        "conformance_raw":serde_json::to_string(&c.conformance).unwrap()})
        })
        .collect();
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
            .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&input).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "real runner oracle must run");
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    assert_eq!(expected.len(), cases.len());
    for (case, expected) in cases.iter().zip(expected) {
        let actual = admit(case).map(snapshot).unwrap_or(Value::Null);
        assert_eq!(actual, expected, "{}", case.name);
    }
}
