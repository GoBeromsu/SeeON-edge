//! T5, T6, T8: the stage-1 config ports against the goldens recorded from the
//! Python worker at 030aaf1 under `tests/fixtures/worker-wire/`. Python
//! exception messages map to Rust variants only through the reviewed tables
//! in this file; no case is named and no recorder digest is read.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use seeon_ml_worker::config::lkg::{Directive, LkgStore};
use seeon_ml_worker::config::model_bundle::AdmissionKind;
use seeon_ml_worker::config::model_bundle::bundle::admit_model_bundle;
use seeon_ml_worker::config::model_bundle::identity::{
    IdentityKind, identity_for, verify_engine_identity,
};
use seeon_ml_worker::config::model_bundle::layout::{LayoutError, parse_manifest, verify_layout};
use seeon_ml_worker::config::selection::{SelectionKind, parse_model_selection};
use seeon_ml_worker::json::{Json, Serialiser};
use serde_json::Value;
use sha2::{Digest, Sha256};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

/// Identity keys and the per-case file each one hashes (the g1 recipe).
const ARTIFACTS: [(&str, &str); 6] = [
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
                let identities = string_map(&observed["identities"]);
                assert_eq!(proof.identities, identities, "{id}");
            }
            ("refused", Err(error)) => {
                let expected = admission_refusal(text(case, "message"));
                assert_eq!((error.kind, error.subject), expected, "{id}");
            }
            (verdict, outcome) => panic!("{id}: {verdict} vs {outcome:?}"),
        }
    }
}

/// `json.dumps(identity, sort_keys=True) + "\n"`.
fn python_dumps(identity: &BTreeMap<String, Value>) -> String {
    let entries: Vec<String> = identity
        .iter()
        .map(|(key, value)| format!("{}: {value}", Value::from(key.as_str())))
        .collect();
    format!("{{{}}}\n", entries.join(", "))
}

/// The identity files `verify_engine_identity` checks, below `dir`.
fn flow_files(dir: &Path) -> Vec<(&'static str, PathBuf)> {
    ARTIFACTS[1..]
        .iter()
        .map(|(key, relative)| (*key, dir.join(relative)))
        .collect()
}

#[test]
fn engine_identity_golden() {
    let dir = scratch("identity-golden");
    for (key, relative) in ARTIFACTS {
        let body = format!("synthetic-{}", key.trim_end_matches("_sha256"));
        write(&dir.join(relative), body.as_bytes());
    }
    let image = format!("sha256:{}", "e".repeat(64));
    let engine = dir.join(ARTIFACTS[0].1);
    let rendered = identity_for(&engine, &flow_files(&dir), &image, 1).expect("identity");
    let python = fs::read_to_string(wire("d/engine-identity.json")).expect("identity golden");
    assert_eq!(rendered, python);
}

/// One g1 per-case directory: the recipe's files and the identity map.
struct G1 {
    dir: PathBuf,
    actual: BTreeMap<&'static str, String>,
    identity: BTreeMap<String, Value>,
}

impl G1 {
    fn new(batch: i64) -> Self {
        let dir = scratch("g1");
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
        let mut actual = BTreeMap::new();
        for ((key, relative), body) in ARTIFACTS.into_iter().zip(bodies) {
            write(&dir.join(relative), body.as_bytes());
            actual.insert(key, sha256_hex(body.as_bytes()));
        }
        let mut identity: BTreeMap<String, Value> = actual
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Value::from(v.as_str())))
            .collect();
        identity.insert(
            "image_digest".to_owned(),
            Value::from("sha256:golden-image"),
        );
        identity.insert("batch_size".to_owned(), Value::from(batch.to_string()));
        let g1 = Self {
            dir,
            actual,
            identity,
        };
        g1.write_identity();
        g1
    }

    fn identity_path(&self) -> PathBuf {
        self.dir.join("cache/identity.json")
    }

    fn write_identity(&self) {
        write(
            &self.identity_path(),
            python_dumps(&self.identity).as_bytes(),
        );
    }

    /// A Python literal from the case text, placeholders resolved for `key`.
    fn literal(&self, key: &str, raw: &str) -> Value {
        let quoted = raw.strip_prefix('\'').and_then(|r| r.strip_suffix('\''));
        let quoted = quoted.or_else(|| raw.strip_prefix('"').and_then(|r| r.strip_suffix('"')));
        let Some(quoted) = quoted else {
            return match raw {
                "None" => Value::Null,
                "True" => Value::Bool(true),
                digits => Value::from(digits.parse::<i64>().expect("integer literal")),
            };
        };
        let actual = || self.actual.get(key).expect("placeholder names a file key");
        Value::from(match quoted {
            "<zeros-64>" => "0".repeat(64),
            "<uppercase-of-actual>" => actual().to_uppercase(),
            "<63-chars: actual[:63]>" => actual()[..63].to_owned(),
            "<65-chars: actual + '0'>" => format!("{}0", actual()),
            plain => plain.to_owned(),
        })
    }

    /// One atomic edit of the reviewed recipe vocabulary.
    fn apply(&mut self, edit: &str) {
        if let Some((key, raw)) = edit.split_once(" set to ") {
            let value = self.literal(key, raw);
            self.identity.insert(key.to_owned(), value);
            return self.write_identity();
        }
        if let Some(key) = edit.strip_suffix(" removed") {
            self.identity.remove(key);
            return self.write_identity();
        }
        if let Some(key) = edit.strip_suffix(" artifact absent") {
            let found = ARTIFACTS.iter().find(|(name, _)| *name == key);
            let relative = found.expect("artifact key").1;
            return fs::remove_file(self.dir.join(relative)).expect("artifact removed");
        }
        if let Some(bytes) = edit
            .strip_prefix("identity bytes b'")
            .and_then(|r| r.strip_suffix('\''))
        {
            return fs::write(self.identity_path(), python_bytes(bytes)).expect("identity bytes");
        }
        let identity = self.identity_path();
        match edit {
            "engine file absent" => fs::remove_file(self.dir.join(ARTIFACTS[0].1)).expect("engine"),
            "identity file absent" => fs::remove_file(&identity).expect("identity removed"),
            "identity file is a directory" => {
                fs::remove_file(&identity).expect("identity removed");
                fs::create_dir(&identity).expect("identity dir");
            }
            "identity file mode 000 (non-root reader)" => {
                fs::set_permissions(&identity, fs::Permissions::from_mode(0o000)).expect("chmod");
                assert!(
                    fs::read(&identity).is_err(),
                    "mode 000 needs a non-root runner"
                );
            }
            other => panic!("edit outside the reviewed vocabulary: {other:?}"),
        }
    }
}

/// The body of a Python `b'...'` literal (`\xNN`, `\n`, plain ASCII).
fn python_bytes(literal: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut rest = literal;
    while let Some(c) = rest.chars().next() {
        if let Some(hex) = rest.strip_prefix("\\x") {
            bytes.push(u8::from_str_radix(&hex[..2], 16).expect("hex escape"));
            rest = &hex[2..];
        } else if let Some(after) = rest.strip_prefix("\\n") {
            bytes.push(b'\n');
            rest = after;
        } else {
            bytes.push(u8::try_from(c).expect("ASCII literal"));
            rest = &rest[1..];
        }
    }
    bytes
}

/// Reviewed rewrite of each case text into atomic edits. The `order:` cases
/// combine two edits; `baseline` and `identity batch N, deployed roster batch
/// M` change nothing on disk (the batches come from the case fields).
fn g1_edits(case: &str) -> Vec<&str> {
    match case {
        "order: batch_size '0' and engine_sha256 zeros-64" => {
            vec!["batch_size set to '0'", "engine_sha256 set to '<zeros-64>'"]
        }
        "order: tracker_config_sha256 and engine_sha256 both zeros-64" => vec![
            "tracker_config_sha256 set to '<zeros-64>'",
            "engine_sha256 set to '<zeros-64>'",
        ],
        "order: onnx_sha256 63 chars and the ONNX absent" => vec![
            "onnx_sha256 set to '<63-chars: actual[:63]>'",
            "onnx_sha256 artifact absent",
        ],
        "order: image_digest '' and tracker_library_sha256 zeros-64" => vec![
            "image_digest set to ''",
            "tracker_library_sha256 set to '<zeros-64>'",
        ],
        "order: batch_size 'abc' and deployed roster batch -1" => vec!["batch_size set to 'abc'"],
        "order: engine absent and identity bytes '{'" => {
            vec!["engine file absent", "identity bytes b'{'"]
        }
        "extra key 'extra': 5 is returned stringified" => vec!["extra set to 5"],
        _ if case.starts_with("baseline: ") || case.starts_with("identity batch ") => vec![],
        edit => vec![edit],
    }
}

/// Reviewed map from an `EngineIdentityError` (or uncaught) class and
/// message to the kind and the identity key it names.
fn identity_refusal(class: &str, message: &str) -> (IdentityKind, String) {
    if class == "UnicodeDecodeError" {
        return (IdentityKind::IdentityNotUtf8, String::new());
    }
    let keyed = [
        (
            "Flow artifact digest mismatch for ",
            IdentityKind::DigestMismatch,
        ),
        ("Flow artifact is absent for ", IdentityKind::ArtifactAbsent),
    ];
    for (prefix, kind) in keyed {
        if let Some(rest) = message.strip_prefix(prefix) {
            return (kind, rest.split(':').next().unwrap_or_default().to_owned());
        }
    }
    let kind = match message {
        "Flow engine identity lacks valid batch_size" => IdentityKind::BatchSize,
        "Flow engine identity lacks image_digest" => IdentityKind::ImageDigest,
        "Flow engine identity must be a JSON object" => IdentityKind::NotObject,
        "deployed Flow roster batch must not be negative" => IdentityKind::NegativeDeployedBatch,
        _ if message.starts_with("Flow engine is absent:") => IdentityKind::EngineAbsent,
        _ if message.starts_with("Flow engine identity is absent:") => IdentityKind::IdentityAbsent,
        _ if message.starts_with("Flow engine identity is unreadable:") => {
            IdentityKind::IdentityUnreadable
        }
        _ if message.starts_with("Flow engine batch ") && message.contains("does not cover") => {
            IdentityKind::BatchNotCovering
        }
        _ => match message.strip_prefix("Flow engine identity lacks valid ") {
            Some(key) => return (IdentityKind::DigestInvalid, key.to_owned()),
            None => panic!("unmapped message {message:?}"),
        },
    };
    (kind, String::new())
}

#[test]
fn engine_identity_refusals_match_python() {
    let golden = golden("d/engine-identity-refusals.json");
    let cases = golden["g1_identity"].as_array().expect("g1_identity cases");
    assert!(!cases.is_empty());
    for case in cases {
        let description = text(case, "case");
        let mut g1 = G1::new(case["identity_batch"].as_i64().expect("identity batch"));
        for edit in g1_edits(description) {
            g1.apply(edit);
        }
        let deployed = case["deployed_batch"].as_i64().map(i128::from);
        let engine = g1.dir.join(ARTIFACTS[0].1);
        let files = flow_files(&g1.dir);
        let outcome = verify_engine_identity(&engine, &g1.identity_path(), &files, deployed);
        match (text(case, "verdict"), outcome) {
            ("accepted", Ok(returned)) => {
                let mut expected = string_map(&case["returned"]);
                for (key, value) in &mut expected {
                    if value.starts_with("<sha256:") {
                        value.clone_from(&g1.actual[key.as_str()]);
                    }
                }
                assert_eq!(returned, expected, "{description}");
            }
            ("refused", Err(error)) => {
                let expected = identity_refusal(text(case, "class"), text(case, "message"));
                assert_eq!((error.kind, error.subject), expected, "{description}");
            }
            (verdict, outcome) => panic!("{description}: {verdict} vs {outcome:?}"),
        }
    }
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
