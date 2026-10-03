//! T5, T6, T8: the stage-1 config ports against the goldens recorded from the
//! Python worker at 030aaf1 under `tests/fixtures/worker-wire/`. Python
//! exception messages map to Rust variants only through the reviewed tables
//! in this file; no case is named and no recorder digest is read.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use seeon_ml_worker::config::lkg::{Directive, LkgStore};
use seeon_ml_worker::config::model_bundle::AdmissionKind;
use seeon_ml_worker::config::model_bundle::bundle::admit_model_bundle;
use seeon_ml_worker::config::model_bundle::identity::{
    EnginePaths, IdentityInputs, IdentityKind, verify_aggregate,
};
use seeon_ml_worker::config::model_bundle::layout::{LayoutError, parse_manifest, verify_layout};
use seeon_ml_worker::config::selection::{SelectionKind, parse_model_selection};
use seeon_ml_worker::json::{Json, Serialiser};
use seeon_ml_worker::relay::cameras::{
    CameraConfigError, DomainSelection, ResolvedDomains, WorkerConfigPayload,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

/// Retired flat identity keys and the per-case file each one hashed.
const LEGACY_ARTIFACTS: [(&str, &str); 6] = [
    ("engine_sha256", "cache/model.engine"),
    ("infer_config_sha256", "config/infer.yml"),
    ("tracker_config_sha256", "config/tracker.yml"),
    ("tracker_library_sha256", "models/libtracker.so"),
    ("onnx_sha256", "models/pose.onnx"),
    ("parser_lib_sha256", "models/libparser.so"),
];

fn wire(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(path)
}

fn golden(path: &str) -> Value {
    let text = fs::read_to_string(wire(path)).expect("golden is readable");
    serde_json::from_str(&text).expect("golden is JSON")
}

fn scratch(label: &str) -> PathBuf {
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let name = format!("wire-config-{}-{label}-{serial}", std::process::id());
    let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    if directory.exists() {
        fs::remove_dir_all(&directory).expect("stale scratch removed");
    }
    fs::create_dir_all(&directory).expect("scratch dir");
    directory
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn write(path: &Path, body: &[u8]) {
    fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
    fs::write(path, body).expect("fixture file");
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is a string"))
}

fn canonical(json: &Json) -> String {
    Serialiser::ModelSelection
        .canonical(json)
        .expect("canonical JSON")
}
/// `WorkerConfig.enabled_domains` for one payload, from the Python owner.
const DOMAIN_ORACLE: &str = r#"
import json, sys
from pydantic import ValidationError
from worker.runtime.config.pull_models import BackendWorkerConfigPayload

payload = json.loads(sys.argv[1])
try:
    config = BackendWorkerConfigPayload.model_validate(payload).to_worker_config(
        "http://relay.invalid", "token"
    )
except ValidationError as error:
    result = {"error": "ValidationError", "locations": [
        list(item["loc"]) for item in error.errors(include_input=False, include_url=False)
    ]}
else:
    result = {"enabled": list(config.enabled_domains)}
print(json.dumps(result, separators=(",", ":")))
"#;
/// Canonical clock masks from `detection_window_validation_error`.
///
/// Stdout stays under 4 KiB: 68 pinned zeroes, then ten ASCII and ten
/// non-ASCII acceptance bitmasks. Each digit has six bits for the
/// positions `0{c}:00`, `2{c}:00`, `09:0{c}`,
/// `{c}:00`, `{c}{c}:00`, and `09:{c}`. The child derives every bit by
/// calling the validator; it does not copy a regex.
const CLOCK_ORACLE: &str = r#"
import json, sys, unicodedata
from contracts.worker_config import detection_window_validation_error

assert unicodedata.unidata_version == "15.0.0", unicodedata.unidata_version
assert detection_window_validation_error("01:00", "02:00", "UTC") is None, "UTC unavailable"

def accepted(start):
    return detection_window_validation_error(start, "23:59", "UTC") is None

def mask_for(digit):
    probes = (
        f"0{digit}:00",
        f"2{digit}:00",
        f"09:0{digit}",
        f"{digit}:00",
        f"{digit}{digit}:00",
        f"09:{digit}",
    )
    bits = 0
    for index, start in enumerate(probes):
        if accepted(start):
            bits |= 1 << index
    return bits

zeroes = [
    code for code in range(sys.maxunicode + 1)
    if unicodedata.decimal(chr(code), None) == 0
]
assert len(zeroes) == 68
ascii_masks = [mask_for(str(value)) for value in range(10)]
sample = zeroes[1]
other = [mask_for(chr(sample + value)) for value in range(10)]
for zero in zeroes:
    expected = ascii_masks if zero == 0x30 else other
    for value in range(10):
        assert unicodedata.decimal(chr(zero + value), None) == value
        assert mask_for(chr(zero + value)) == expected[value], (hex(zero), value)
print(json.dumps({"z": zeroes, "a": ascii_masks, "n": other}, separators=(",", ":")))
"#;

fn python_script_outcome(script: &str, input: &str, label: &'static str) -> Value {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is required");
    assert!(input.len() < 65_536, "bounded oracle input");
    let mut child = OracleChild(Some(
        Command::new(python)
            .arg("-c")
            .arg(script)
            .arg(input)
            .current_dir(repo_root())
            .env("PYTHONPATH", repo_root())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("python oracle"),
    ));
    let process = child.0.as_mut().expect("owned oracle");
    let clock = seeon_ml_worker::seam::SystemClock::new();
    let finished = seeon_ml_worker::poll::poll_until(
        &clock,
        std::time::Duration::from_secs(20),
        label,
        || process.try_wait().expect("oracle status").is_some(),
    );
    assert!(finished.is_ok(), "Python {label} exceeded its deadline");
    let output = child
        .0
        .take()
        .expect("exited oracle")
        .wait_with_output()
        .expect("oracle output");
    assert!(
        output.status.success(),
        "{label} refused: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() < 4_096, "{label} stdout exceeds 4 KiB");
    serde_json::from_slice(&output.stdout).expect("oracle JSON")
}

fn python_domain_outcome(payload: &Value) -> Value {
    let rendered = serde_json::to_string(payload).expect("payload JSON");
    python_script_outcome(DOMAIN_ORACLE, &rendered, "domain oracle")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct OracleChild(Option<std::process::Child>);
impl Drop for OracleChild {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn worker_config() -> Value {
    golden("r/worker-config.response.json")
}

fn resolved(payload: &Value) -> (DomainSelection, ResolvedDomains) {
    let config = WorkerConfigPayload::parse(&Json::from(payload)).expect("payload parses");
    let selection = config.domain_selection();
    let resolved = selection.resolve().expect("selection resolves");
    assert!(config.runtime_cameras().is_ok(), "legal roster admits");
    (selection, resolved)
}

fn assert_matches_python(payload: &Value, fall: bool, bed_exit: bool) {
    let outcome = python_domain_outcome(payload);
    let enabled: Vec<String> = serde_json::from_value(outcome["enabled"].clone())
        .unwrap_or_else(|_| panic!("Python refused a legal domain selection: {outcome}"));
    assert_eq!(enabled.contains(&"fall".to_owned()), fall);
    assert_eq!(enabled.contains(&"bed_exit".to_owned()), bed_exit);
    let (_, domains) = resolved(payload);
    assert_eq!((domains.fall, domains.bed_exit), (fall, bed_exit));
}

#[test]
fn selection_golden_round_trips_bytes() {
    let raw = fs::read(wire("d/model-selection.json")).expect("selection golden");
    let document: Value = serde_json::from_slice(&raw).expect("selection JSON");
    let selection = parse_model_selection(&Json::from(&document)).expect("golden is valid");
    let rendered = format!("{}\n", canonical(&selection.as_json()));
    assert_eq!(rendered.as_bytes(), raw.as_slice());
}

/// Reviewed map from a `ContractError` message fragment to the check.
fn selection_kind(message: &str) -> SelectionKind {
    let table = [
        (" must be an object", SelectionKind::NotObject),
        (" keys differ: ", SelectionKind::Keys),
        (" must be 2", SelectionKind::SchemaVersion),
        (" must be a non-empty string", SelectionKind::NotString),
        (" must be an owner/repository name", SelectionKind::Locator),
        (
            " must be a lowercase 40-hex immutable ref",
            SelectionKind::Revision,
        ),
        (
            " must be a lowercase 64-hex SHA-256 digest",
            SelectionKind::Digest,
        ),
        (" must be a positive integer", SelectionKind::PositiveInt),
        (" must be a probability", SelectionKind::Probability),
        (" must be in [0, 1]", SelectionKind::Range),
        (
            " must be default or receipt",
            SelectionKind::ThresholdSource,
        ),
    ];
    let found = table.into_iter().find(|(part, _)| message.contains(part));
    found
        .unwrap_or_else(|| panic!("unmapped message {message:?}"))
        .1
}

/// The case document, with each `{"$nonfinite": ...}` marker the golden
/// lists replaced by the float JSON cannot carry.
fn selection_document(case: &Value) -> Json {
    let mut document = Json::from(&case["document"]);
    let paths = case["document_nonfinite_paths"].as_array();
    for key in paths
        .into_iter()
        .flatten()
        .map(|key| key.as_str().expect("path"))
    {
        let value = match case["document"][key]["$nonfinite"].as_str() {
            Some("nan") => f64::NAN,
            Some("inf") => f64::INFINITY,
            marker => panic!("nonfinite marker {marker:?}"),
        };
        let Json::Object(entries) = &mut document else {
            panic!("nonfinite document is an object");
        };
        let entry = entries.iter_mut().find(|(name, _)| name == key);
        entry.expect("nonfinite key present").1 = Json::Float(value);
    }
    document
}

#[test]
fn selection_refusals_match_python() {
    let golden = golden("d/model-selection-refusals.json");
    let (mut accepted, mut refused) = (0, 0);
    for case in golden["cases"].as_array().expect("cases") {
        let (id, outcome) = (
            &case["id"],
            parse_model_selection(&selection_document(case)),
        );
        match (text(case, "verdict"), outcome) {
            ("ok", Ok(selection)) => {
                let bytes = text(case, "canonical_bytes");
                assert_eq!(canonical(&selection.as_json()), bytes, "{id}");
                assert_eq!(canonical(&Json::from(&case["selection"])), bytes, "{id}");
                let digest = selection.digest().expect("digest");
                assert_eq!(digest, text(case, "digest"), "{id}");
                accepted += 1;
            }
            ("refused", Err(error)) => {
                let expected = (
                    text(case, "named_field"),
                    selection_kind(text(case, "message")),
                );
                assert_eq!((error.field.as_str(), error.kind), expected, "{id}");
                refused += 1;
            }
            (verdict, outcome) => panic!("{id}: {verdict} vs {outcome:?}"),
        }
    }
    assert!(accepted > 0 && refused > 0);
}

/// Every file below `root` except the `.lock` both stores leave, by path.
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

/// The directive the Python store recorded in `current.json`.
fn python_directive() -> Directive {
    let current = golden("d/config-lkg/current.json");
    let integer = |key: &str| i128::from(current[key].as_i64().expect("directive field"));
    Directive {
        generation: integer("generation"),
        version: integer("config_version"),
        registry: integer("registry_version"),
    }
}

#[test]
fn lkg_writes_python_bytes() {
    let payload = Json::from(&golden("r/worker-config.response.json"));
    let state = scratch("lkg-write");
    let saved = LkgStore::new(&state).save(&payload, python_directive());
    assert!(matches!(saved, Ok(true)), "{saved:?}");
    let python = store_files(&wire("d/config-lkg"));
    assert!(!python.is_empty());
    assert_eq!(store_files(&state.join("config-lkg")), python);
}

#[test]
fn lkg_reads_python_store() {
    let state = scratch("lkg-read");
    for (relative, body) in store_files(&wire("d/config-lkg")) {
        write(&state.join("config-lkg").join(relative), &body);
    }
    let stored = LkgStore::new(&state).load().expect("store loads");
    let stored = stored.expect("current.json present");
    let payload = Json::from(&golden("r/worker-config.response.json"));
    assert_eq!(
        (stored.payload, stored.directive),
        (payload, python_directive())
    );
}

/// Every body a bundle case tree may hold, by SHA-256: the published ones,
/// plus the mutations reconstructed from them (first byte replaced by `S`,
/// the `extra\n` file, the manifest re-indented by `json.dumps(indent=2)`).
fn bundle_bodies(publication: &Value) -> BTreeMap<String, Vec<u8>> {
    let published = publication["bodies"].as_object().expect("bodies");
    let mut texts: Vec<String> = published
        .values()
        .map(|b| text(b, "text").to_owned())
        .collect();
    let manifest = text(&publication["manifest_json"], "text");
    let indented: Value = serde_json::from_str(manifest).expect("manifest JSON");
    let indented = serde_json::to_string_pretty(&indented).expect("indent");
    let flipped: Vec<String> = texts.iter().map(|t| format!("S{}", &t[1..])).collect();
    texts.extend(flipped);
    texts.extend([
        manifest.to_owned(),
        "extra\n".to_owned(),
        format!("{indented}\n"),
    ]);
    texts
        .into_iter()
        .map(|body| (sha256_hex(body.as_bytes()), body.into_bytes()))
        .collect()
}

/// Reviewed map from a `ModelBundleAdmissionError` message to the kind and
/// what it names: the member path, or the `_read_regular` label (e.g.
/// `member model.onnx`) of `{label} is unavailable`.
fn admission_refusal(message: &str) -> (AdmissionKind, String) {
    if message == "bundle path is unavailable" {
        return (AdmissionKind::PathUnavailable, String::new());
    }
    let named = [
        ("member mismatch: ", "", AdmissionKind::MemberMismatch),
        (
            "bundle contains unsafe path: ",
            "",
            AdmissionKind::UnsafePath,
        ),
        ("", " is unavailable", AdmissionKind::Unavailable),
    ];
    for (prefix, suffix, kind) in named {
        let path = message
            .strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(suffix));
        if let Some(path) = path {
            return (kind, path.to_owned());
        }
    }
    let kind = match message {
        "bundle path escapes its root" => AdmissionKind::PathEscapes,
        "bundle tree contains missing or extra filesystem nodes" => AdmissionKind::TreeMismatch,
        "bundle contains symlink path" => AdmissionKind::SymlinkPath,
        "bundle manifest is not canonical" => AdmissionKind::ManifestNotCanonical,
        _ if message.starts_with("bundle format mismatch: ") => AdmissionKind::BundleFormat,
        _ => panic!("unmapped message {message:?}"),
    };
    (kind, String::new())
}

fn strings(value: &Value) -> Vec<String> {
    serde_json::from_value(value.clone()).expect("string list")
}

fn string_map(value: &Value) -> BTreeMap<String, String> {
    serde_json::from_value(value.clone()).expect("string map")
}

#[test]
fn bundle_admission_matches_python() {
    let directory = wire("d/model-bundle-admission");
    let (mut publications, mut cases) = (Vec::new(), Vec::new());
    for entry in fs::read_dir(&directory).expect("admission goldens") {
        let path = entry.expect("entry").path();
        let document: Value =
            serde_json::from_slice(&fs::read(&path).expect("golden")).expect("golden is JSON");
        match document.get("case") {
            Some(case) => cases.push(case.clone()),
            None => publications.push(document),
        }
    }
    let [publication] = publications.as_slice() else {
        panic!("one publication, {} found", publications.len());
    };
    let bodies = bundle_bodies(publication);
    assert!(!cases.is_empty());
    for case in &cases {
        let (id, base) = (&case["id"], scratch("bundle"));
        for node in case["tree"].as_array().expect("tree") {
            let path = base.join(text(node, "path"));
            fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
            match text(node, "kind") {
                "dir" => fs::create_dir_all(&path).expect("dir"),
                "symlink" => symlink(text(node, "target"), &path).expect("symlink"),
                "file" => {
                    let body = bodies.get(text(node, "sha256"));
                    let body = body.unwrap_or_else(|| panic!("{id}: no body for {node}"));
                    assert_eq!(Some(body.len() as u64), node["size"].as_u64(), "{id}");
                    fs::write(&path, body).expect("file");
                }
                kind => panic!("{id}: node kind {kind}"),
            }
        }
        let mut document = publication["selection_document"].clone();
        let changes = case["selection_changes"].as_object().into_iter().flatten();
        for (key, value) in changes {
            document[key] = value.clone();
        }
        let desired = parse_model_selection(&Json::from(&document)).expect("case selection");
        match (
            text(case, "verdict"),
            admit_model_bundle(&base.join("models"), &desired),
        ) {
            ("ok", Ok(proof)) => {
                let observed = &case["proof"]["observed"];
                assert_eq!(proof.bundle_sha256, text(observed, "bundle_sha256"), "{id}");
                assert_eq!(proof.members, strings(&observed["members"]), "{id}");
                assert_eq!(proof.receipts, strings(&observed["receipts"]), "{id}");
                let digests = string_map(&observed["member_digests"]);
                assert_eq!(proof.member_digests, digests, "{id}");
                assert_eq!(
                    &proof.calibration,
                    bodies
                        .get(&desired.calibration_digest)
                        .expect("calibration golden"),
                    "{id}"
                );
                assert_eq!(
                    &proof.conformance.1,
                    bodies
                        .get(&desired.conformance_digest)
                        .expect("conformance golden"),
                    "{id}"
                );
                assert_eq!(
                    proof.member_digests.get(&proof.conformance.0),
                    Some(&desired.conformance_digest),
                    "{id}"
                );
                let identities = string_map(&observed["identities"]);
                assert_eq!(proof.identities, identities, "{id}");
                let root = base
                    .join("models/bundles")
                    .join(&desired.model_publication.content);
                fs::write(root.join("calibration.json"), b"changed after admission")
                    .expect("replace");
                assert_eq!(
                    &proof.calibration,
                    bodies
                        .get(&desired.calibration_digest)
                        .expect("captured calibration"),
                    "{id}"
                );
                assert!(
                    admit_model_bundle(&base.join("models"), &desired).is_err(),
                    "{id}"
                );
                let calibration_path = root.join("calibration.json");
                fs::remove_file(&calibration_path).expect("remove owned calibration");
                rustix::fs::mknodat(
                    rustix::fs::CWD,
                    &calibration_path,
                    rustix::fs::FileType::Fifo,
                    rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
                    0,
                )
                .expect("owned selected FIFO");
                let (sender, receiver) = std::sync::mpsc::channel();
                let models = base.join("models");
                let desired = desired.clone();
                let reader = std::thread::spawn(move || {
                    sender.send(admit_model_bundle(&models, &desired)).unwrap();
                });
                let refused = receiver
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .expect("selected FIFO cannot block boot");
                reader.join().expect("selected reader joined");
                assert_eq!(
                    refused.unwrap_err().kind,
                    AdmissionKind::NotRegularFile,
                    "{id}"
                );
            }
            ("refused", Err(error)) => {
                let expected = admission_refusal(text(case, "message"));
                assert_eq!((error.kind, error.subject), expected, "{id}");
            }
            (verdict, outcome) => panic!("{id}: {verdict} vs {outcome:?}"),
        }
    }
}

/// Schema1 reader inputs for one retired flat document. The four engine paths
/// are explicit fixture files; they are not a claim that a flat document
/// describes four engines.
fn legacy_probe(dir: &Path, identity: &Path) -> Result<(), (IdentityKind, String)> {
    let engines = [
        dir.join("cache/model.engine"),
        dir.join("stored_pose.engine"),
        dir.join("bed.engine"),
        dir.join("fall.engine"),
    ];
    for (path, body) in engines
        .iter()
        .zip([b"flat" as &[u8], b"stored", b"bed", b"fall"])
    {
        write(path, body);
    }
    let flow_paths = [
        dir.join("config/infer.yml"),
        dir.join("config/tracker.yml"),
        dir.join("models/libtracker.so"),
        dir.join("models/libparser.so"),
    ];
    for path in &flow_paths {
        if !path.exists() {
            write(path, b"legacy-flow");
        }
    }
    let observer = dir.join("observer.so");
    write(&observer, b"synthetic-observer");
    let flow = [
        ("infer_config_sha256", flow_paths[0].clone()),
        ("tracker_config_sha256", flow_paths[1].clone()),
        ("tracker_library_sha256", flow_paths[2].clone()),
        ("parser_lib_sha256", flow_paths[3].clone()),
    ];
    let source = "0".repeat(64);
    verify_aggregate(
        identity,
        IdentityInputs {
            engines: EnginePaths::TensorRt {
                live_pose: &engines[0],
                stored_pose: &engines[1],
                bed: &engines[2],
                fall: &engines[3],
            },
            pose_onnx_sha256: &source,
            bed_onnx_sha256: &source,
            fall_onnx_sha256: &source,
            flow: &flow,
            observer_library: &observer,
            image_digest: &format!("sha256:{}", "e".repeat(64)),
            configured_batch: Some(1),
            deployed_batch: None,
        },
    )
    .map(|_| ())
    .map_err(|error| (error.kind, error.subject))
}

#[test]
fn retired_flat_identity_golden_is_refused() {
    let golden_path = wire("d/engine-identity.json");
    let before = fs::read(&golden_path).expect("frozen flat golden");
    let dir = scratch("identity-golden");
    for (key, relative) in LEGACY_ARTIFACTS {
        write(
            &dir.join(relative),
            format!("synthetic-{}", key.trim_end_matches("_sha256")).as_bytes(),
        );
    }
    let identity = dir.join("cache/identity.json");
    fs::copy(&golden_path, &identity).expect("copy frozen golden; never rewrite it");
    assert_eq!(
        legacy_probe(&dir, &identity),
        Err((IdentityKind::Schema, "identity".to_owned())),
        "schema1 reader refuses the retired flat golden"
    );
    assert_eq!(fs::read(&golden_path).expect("golden"), before);
    assert_eq!(fs::read(&identity).expect("copied golden"), before);
}

#[test]
fn retired_flat_g1_documents_are_not_accepted() {
    let golden = golden("d/engine-identity-refusals.json");
    let cases = golden["g1_identity"].as_array().expect("g1_identity cases");
    assert_eq!(cases.len(), 87, "frozen g1 matrix is unchanged");
    let mut refused = 0_usize;
    for case in cases {
        let description = text(case, "case");
        let batch = case["identity_batch"].as_i64().expect("identity batch");
        let dir = scratch("g1-retired");
        let infer = format!(
            "[property]\nonnx-file={0}/models/pose.onnx\nmodel-engine-file={0}/cache/model.engine\nbatch-size={batch}\ninfer-dims=3;640;640\n",
            dir.display()
        );
        let bodies = [
            format!("engine b{batch}\n"),
            infer,
            "tracker-config: synthetic\n".to_owned(),
            "tracker-lib\n".to_owned(),
            "synthetic-onnx".to_owned(),
            "parser-lib\n".to_owned(),
        ];
        let mut identity = BTreeMap::new();
        for ((key, relative), body) in LEGACY_ARTIFACTS.into_iter().zip(bodies) {
            write(&dir.join(relative), body.as_bytes());
            identity.insert(key.to_owned(), Value::from(sha256_hex(body.as_bytes())));
        }
        identity.insert(
            "image_digest".to_owned(),
            Value::from("sha256:golden-image"),
        );
        identity.insert("batch_size".to_owned(), Value::from(batch.to_string()));
        let identity_path = dir.join("cache/identity.json");
        let rendered = serde_json::to_vec(&identity).expect("retired flat document");
        write(&identity_path, &rendered);
        let outcome = legacy_probe(&dir, &identity_path);
        assert!(
            outcome.is_err(),
            "{description}: retired flat identity must not be accepted"
        );
        assert_eq!(fs::read(&identity_path).expect("identity"), rendered);
        refused += 1;
    }
    assert_eq!(refused, cases.len());
}

#[test]
fn models_layout_flat() {
    let golden = golden("d/models-layout.json");
    let manifest = parse_manifest(&Json::from(&golden["manifest"])).expect("manifest parses");
    let bodies: BTreeMap<String, &str> = ["synthetic-pose-onnx", "synthetic-bed-onnx"]
        .into_iter()
        .map(|body| (sha256_hex(body.as_bytes()), body))
        .collect();
    let root = scratch("layout");
    let artifacts = golden["manifest"]["artifacts"]
        .as_array()
        .expect("artifacts");
    for artifact in artifacts {
        let body = bodies.get(text(artifact, "sha256")).expect("known body");
        write(&root.join(text(artifact, "path")), body.as_bytes());
    }
    let expected: Vec<(String, String, i128)> = golden["models_root_after"]
        .as_array()
        .expect("models_root_after")
        .iter()
        .map(|file| {
            let size = i128::from(file["size"].as_u64().expect("size"));
            (
                text(file, "path").to_owned(),
                text(file, "sha256").to_owned(),
                size,
            )
        })
        .collect();
    assert_eq!(verify_layout(&root, &manifest), Ok(expected));
    let flipped = text(&artifacts[0], "path");
    let mut body = fs::read(root.join(flipped)).expect("artifact");
    body[0] ^= 1;
    fs::write(root.join(flipped), body).expect("flip");
    let absent = LayoutError::Absent(flipped.to_owned());
    assert_eq!(verify_layout(&root, &manifest), Err(absent));
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with worker.runtime.config"]
fn domain_resolution_matches_python_boundaries() {
    let base = worker_config();

    let mut omitted = base.clone();
    omitted["cameras"][0]
        .as_object_mut()
        .expect("camera")
        .remove("domains");
    let (selection, domains) = resolved(&omitted);
    assert_eq!(selection, DomainSelection::Cameras(None));
    assert_eq!((domains.fall, domains.bed_exit), (true, true));
    assert_matches_python(&omitted, true, true);

    let mut empty = base.clone();
    empty["cameras"][0]["domains"] = Value::Array(Vec::new());
    let (selection, domains) = resolved(&empty);
    assert_eq!(selection, DomainSelection::Cameras(Some(Vec::new())));
    assert_eq!((domains.fall, domains.bed_exit), (false, false));
    assert_matches_python(&empty, false, false);

    let (selection, domains) = resolved(&base);
    assert_eq!(
        selection,
        DomainSelection::Cameras(Some(vec!["fall".to_owned()]))
    );
    assert_eq!((domains.fall, domains.bed_exit), (true, false));
    assert_matches_python(&base, true, false);

    let mut both = base.clone();
    both["cameras"][0]["domains"] = Value::Array(vec![
        Value::from("bed_exit"),
        Value::from("fall"),
        Value::from("fall"),
    ]);
    let (selection, domains) = resolved(&both);
    assert_eq!(
        selection,
        DomainSelection::Cameras(Some(vec!["bed_exit".to_owned(), "fall".to_owned()]))
    );
    assert_eq!((domains.fall, domains.bed_exit), (true, true));
    assert_matches_python(&both, true, true);

    let mut only_bed = base.clone();
    only_bed["cameras"][0]["domains"] = Value::Array(vec![Value::from("bed_exit")]);
    assert_matches_python(&only_bed, false, true);

    let mut partial = base.clone();
    partial["domains"] = serde_json::json!({"fall": {"enabled": false}});
    let (selection, domains) = resolved(&partial);
    assert_eq!(
        selection,
        DomainSelection::Override {
            fall: Some(false),
            bed_exit: None,
        }
    );
    assert_eq!((domains.fall, domains.bed_exit), (false, true));
    assert_matches_python(&partial, false, true);

    let mut both_off = base.clone();
    both_off["domains"] = serde_json::json!({
        "fall": {"enabled": false},
        "bed_exit": {"enabled": false},
    });
    assert_matches_python(&both_off, false, false);

    let mut unknown_override = base.clone();
    unknown_override["domains"] = serde_json::json!({
        "not-a-domain": {"enabled": true},
        "fall": {"enabled": false},
    });
    unknown_override["cameras"][0]["domains"] = Value::Array(vec![Value::from("also-unknown")]);
    let (selection, domains) = resolved(&unknown_override);
    assert_eq!(
        selection,
        DomainSelection::Override {
            fall: Some(false),
            bed_exit: None,
        }
    );
    assert_eq!((domains.fall, domains.bed_exit), (false, true));
    assert_matches_python(&unknown_override, false, true);

    let mut unknown_list = base.clone();
    unknown_list["cameras"][0]["domains"] =
        Value::Array(vec![Value::from("fall"), Value::from("not-a-domain")]);
    let config = WorkerConfigPayload::parse(&Json::from(&unknown_list)).expect("parses");
    assert_eq!(
        config.domain_selection().resolve(),
        Err(CameraConfigError::UnknownDomain)
    );
    assert!(
        config.runtime_cameras().is_ok(),
        "camera conversion precedes the separate startup domain gate"
    );
    assert_eq!(
        python_domain_outcome(&unknown_list),
        serde_json::json!({"error": "ValidationError", "locations": [["enabled"]]})
    );
}

fn decimal_char(zero: u32, digit: u32) -> char {
    char::from_u32(zero + digit).expect("pinned decimal digit")
}

fn clock_probe(zero: u32, digit: u32, position: usize) -> String {
    let digit = decimal_char(zero, digit).to_string();
    match position {
        0 => format!("0{digit}:00"),
        1 => format!("2{digit}:00"),
        2 => format!("09:0{digit}"),
        3 => format!("{digit}:00"),
        4 => format!("{digit}{digit}:00"),
        5 => format!("09:{digit}"),
        _ => unreachable!("six clock positions"),
    }
}

fn admitted_windows(
    start: &str,
    end: &str,
    tz: &str,
) -> BTreeMap<String, (String, String, String)> {
    let payload = serde_json::json!({
        "config_version": 0,
        "cameras": [],
        "detection_windows": {
            "fall": {"start": start, "end": end, "tz": tz}
        }
    });
    WorkerConfigPayload::parse(&Json::from(&payload))
        .expect("window payload parses")
        .detection_windows()
        .into_iter()
        .map(|(domain, window)| (domain, (window.start, window.end, window.tz)))
        .collect()
}

fn window_admitted(start: &str) -> bool {
    admitted_windows(start, "23:59", "UTC").contains_key("fall")
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with contracts.worker_config"]
fn canonical_wire_clock_matches_python_decimal_digits() {
    let outcome = python_script_outcome(CLOCK_ORACLE, "null", "clock oracle");
    let zeroes: Vec<u32> = serde_json::from_value(outcome["z"].clone()).expect("Python Nd zeroes");
    assert_eq!(zeroes.len(), 68);
    let ascii: Vec<u16> = serde_json::from_value(outcome["a"].clone()).expect("ascii masks");
    let other: Vec<u16> = serde_json::from_value(outcome["n"].clone()).expect("nd masks");
    assert_eq!(ascii.len(), 10);
    assert_eq!(other.len(), 10);

    for (digit, mask) in ascii.iter().enumerate() {
        let digit = u32::try_from(digit).expect("digit");
        for position in 0..6 {
            let admitted = window_admitted(&clock_probe(0x30, digit, position));
            assert_eq!(
                admitted,
                mask & (1 << position) != 0,
                "ascii {digit} pos {position}"
            );
        }
    }
    for zero in zeroes.into_iter().filter(|zero| *zero != 0x30) {
        for digit in 0..10 {
            let start = format!("0{}:00", decimal_char(zero, digit));
            let equivalent = format!("0{digit}:00");
            assert!(
                admitted_windows(&start, &equivalent, "UTC").is_empty(),
                "{zero:#x}+{digit}: exact decimal value must compare equal"
            );
            let mask = other[usize::try_from(digit).expect("digit")];
            for position in 0..6 {
                let admitted = window_admitted(&clock_probe(zero, digit, position));
                assert_eq!(
                    admitted,
                    mask & (1 << position) != 0,
                    "{zero:#x}+{digit} pos {position}"
                );
            }
        }
    }
}

#[test]
fn wire_clock_preserves_unicode_values_and_rejects_non_decimal_numerics() {
    let fullwidth_zero = decimal_char(0xFF10, 0).to_string();
    let fullwidth_one = decimal_char(0xFF10, 1).to_string();
    let arabic_one = decimal_char(0x660, 1).to_string();
    let arabic_zero = decimal_char(0x660, 0).to_string();
    let mixed_kept = format!("0{fullwidth_one}:00");
    let one_digit_kept = format!("01:0{fullwidth_one}");
    let fullwidth_pair =
        format!("{fullwidth_zero}{fullwidth_one}:{fullwidth_zero}{fullwidth_zero}");
    let two_arabic = format!("2{arabic_one}:00");
    let arabic_one_digit = format!("{arabic_one}:{arabic_zero}");
    assert!(
        window_admitted(&mixed_kept),
        "0<fullwidth 1>:00 stays canonical"
    );
    assert!(
        window_admitted(&one_digit_kept),
        "01:0<fullwidth 1> stays canonical"
    );
    assert!(
        !window_admitted(&fullwidth_pair),
        "fullwidth 01:00 is outside %H"
    );
    assert!(
        !window_admitted(&two_arabic),
        "hour 2 requires an ASCII unit"
    );
    assert!(
        window_admitted(&arabic_one_digit),
        "Arabic-Indic 1:0 is canonical here"
    );

    let equal_start = format!("0{fullwidth_one}:0{arabic_zero}");
    let equal_end = "01:00";
    assert!(
        admitted_windows(&equal_start, equal_end, "UTC").is_empty(),
        "mixed-script equal minute values drop"
    );
    let same_minute = admitted_windows("0１:0٠", "01:00", "UTC");
    let shifted = admitted_windows("0１:0١", "01:00", "UTC");
    assert!(same_minute.is_empty(), "parsed minute values compare equal");
    assert!(
        shifted.contains_key("fall"),
        "a different parsed minute stays"
    );
    for malformed in [
        " 01:00", "01:00 ", "+01:00", "-1:00", "01 :00", "01: 00", "001:00", "01:000", "1:2:00",
        "²⁴:00", "①:00", "Ⅻ:00", "Ⅰ:00", "⁹:00",
    ] {
        assert!(!window_admitted(malformed), "{malformed:?} is not Nd HH:MM");
    }

    let raw_end = format!("{}:{arabic_zero}", decimal_char(0x660, 2));
    let kept = admitted_windows(&mixed_kept, &raw_end, "Asia/Seoul");
    assert_eq!(
        kept.get("fall").cloned(),
        Some((mixed_kept, raw_end, "Asia/Seoul".to_owned())),
        "raw start/end/tz strings stay unchanged"
    );
}
