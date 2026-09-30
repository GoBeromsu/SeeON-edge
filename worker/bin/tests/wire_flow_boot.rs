//! Flow boot verification, ONNX input dims and identity placeholders against
//! the `flow_boot`, `onnx_input_dims` and `g1_placeholders` groups of
//! `d/engine-identity-refusals.json`, recorded from the Python worker at
//! 030aaf1. Python exception messages map to Rust variants only through the
//! reviewed tables in this file, and each ONNX artifact is written by the
//! test-side protobuf writer below from the golden `recipe`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use seeon_ml_worker::config::env::Env;
use seeon_ml_worker::config::model_bundle::composition::{
    Attr, Attrs, Binding, Compiled, PlaceholderKind, verified_identity_field,
};
use seeon_ml_worker::config::model_bundle::flow_boot::{FlowBootKind, verify_flow_boot_inputs};
use seeon_ml_worker::config::model_bundle::identity::{IdentityKind, identity_for};
use seeon_ml_worker::config::model_bundle::onnx_shape::{
    Dim, OnnxShapeKind, batch_axis_is_dynamic, input_dims,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const GOLDEN: &str = "d/engine-identity-refusals.json";

static SERIAL: AtomicUsize = AtomicUsize::new(0);

/// The recipe's per-case files and their fixed bodies (`pose.onnx`,
/// `infer.yml` and the engine depend on the case).
const FIXED_FILES: [(&str, &[u8]); 3] = [
    ("models/libparser.so", b"parser-lib\n"),
    ("models/libtracker.so", b"tracker-lib\n"),
    ("config/tracker.yml", b"tracker-config: synthetic\n"),
];

/// Identity keys and the per-case file each one hashes.
const IDENTITY_FILES: [(&str, &str); 5] = [
    ("infer_config_sha256", "config/infer.yml"),
    ("tracker_config_sha256", "config/tracker.yml"),
    ("tracker_library_sha256", "models/libtracker.so"),
    ("onnx_sha256", "models/pose.onnx"),
    ("parser_lib_sha256", "models/libparser.so"),
];

/// The recipe's `boot_env` paths, relative to the case directory.
const BOOT_PATHS: [(&str, &str); 8] = [
    ("ML_WORKER_FLOW_ENGINE_PATH", "cache/model.engine"),
    ("ML_WORKER_FLOW_ENGINE_IDENTITY_PATH", "cache/identity.json"),
    ("ML_WORKER_FLOW_INFER_CONFIG", "config/infer.yml"),
    ("ML_WORKER_FLOW_TRACKER_CONFIG", "config/tracker.yml"),
    ("ML_WORKER_FLOW_TRACKER_LIBRARY", "models/libtracker.so"),
    ("ML_WORKER_FLOW_ONNX_PATH", "models/pose.onnx"),
    ("ML_WORKER_FLOW_PARSER_LIBRARY", "models/libparser.so"),
    ("ML_WORKER_FLOW_RECORD_DIR", "records"),
];

/// The recipe's `boot_env` literals (`BATCH_SIZE` comes from each case).
const BOOT_VALUES: [(&str, &str); 3] = [
    ("ML_WORKER_FLOW_RECORD_CACHE_SECONDS", "30"),
    ("ML_WORKER_FLOW_FRAME_WIDTH", "640"),
    ("ML_WORKER_FLOW_FRAME_HEIGHT", "640"),
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
    let name = format!("wire-flow-boot-{}-{label}-{serial}", std::process::id());
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

fn string_map(value: &Value) -> BTreeMap<String, String> {
    serde_json::from_value(value.clone()).expect("string map")
}

fn cases<'a>(golden: &'a Value, group: &str) -> &'a [Value] {
    let cases = golden[group].as_array().expect("case group");
    assert!(!cases.is_empty(), "{group} has cases");
    cases
}

// Protobuf writer: `onnx.helper` as `tests/test_edge_engine_build.py`
// `_write_onnx` uses it, one field at a time.

fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while value >= 0x80 {
        out.push((value & 0x7f) as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
    out
}

fn varint_field(number: u64, value: u64) -> Vec<u8> {
    let mut out = varint(number << 3);
    out.extend(varint(value));
    out
}

fn bytes_field(number: u64, body: &[u8]) -> Vec<u8> {
    let mut out = varint((number << 3) | 2);
    out.extend(varint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

/// `make_tensor_value_info(name, FLOAT, dims)`: `None` dims leave the shape
/// unset; `[]` sets an empty shape; a `None` dim is a `Dimension` with
/// neither `dim_value` nor `dim_param`.
fn value_info(name: &str, dims: Option<&[Value]>) -> Vec<u8> {
    let mut tensor = varint_field(1, 1);
    if let Some(dims) = dims {
        let mut shape = Vec::new();
        for dim in dims {
            let body = match dim {
                Value::Null => Vec::new(),
                Value::String(param) => bytes_field(2, param.as_bytes()),
                other => varint_field(1, other.as_u64().expect("dim_value")),
            };
            shape.extend(bytes_field(1, &body));
        }
        tensor.extend(bytes_field(2, &shape));
    }
    let mut info = bytes_field(1, name.as_bytes());
    info.extend(bytes_field(2, &bytes_field(1, &tensor)));
    info
}

/// `make_model(graph, opset_imports=[make_opsetid("", 13)], ir_version=9)`.
fn model(graph: &[u8]) -> Vec<u8> {
    let mut opset = bytes_field(1, b"");
    opset.extend(varint_field(2, 13));
    let mut out = varint_field(1, 9);
    out.extend(bytes_field(7, graph));
    out.extend(bytes_field(8, &opset));
    out
}

/// The recipe's variant bytes: one `Identity` node `frames -> output0` whose
/// input and output share the declared dims, or the two special variants.
fn onnx_bytes(recipe: &Value, variant: &str) -> Vec<u8> {
    let mut graph = Vec::new();
    match variant {
        "unloadable" => return b"not-an-onnx-model\n".to_vec(),
        "no-inputs" => {
            let mut tensor = varint_field(1, 1);
            tensor.extend(varint_field(2, 1));
            tensor.extend(bytes_field(4, &[0, 0, 0, 0]));
            tensor.extend(bytes_field(8, b"value"));
            let mut attribute = bytes_field(1, b"value");
            attribute.extend(bytes_field(5, &tensor));
            attribute.extend(varint_field(20, 4));
            let mut node = bytes_field(2, b"output0");
            node.extend(bytes_field(4, b"Constant"));
            node.extend(bytes_field(5, &attribute));
            graph.extend(bytes_field(1, &node));
            graph.extend(bytes_field(2, b"pose"));
            let one = [Value::from(1)];
            graph.extend(bytes_field(12, &value_info("output0", Some(&one))));
        }
        _ => {
            let declared = &recipe["onnx_variants"][variant];
            let dims = match declared {
                Value::Null => None,
                Value::Array(dims) => Some(dims.as_slice()),
                other => panic!("unmapped variant {variant}: {other}"),
            };
            let mut node = bytes_field(1, b"frames");
            node.extend(bytes_field(2, b"output0"));
            node.extend(bytes_field(4, b"Identity"));
            graph.extend(bytes_field(1, &node));
            graph.extend(bytes_field(2, b"pose"));
            graph.extend(bytes_field(11, &value_info("frames", dims)));
            graph.extend(bytes_field(12, &value_info("output0", dims)));
        }
    }
    model(&graph)
}

/// One boot case laid out as the recipe's `case_layout`.
struct BootCase {
    work: PathBuf,
    dir: PathBuf,
}

impl BootCase {
    fn new(recipe: &Value, work: &Path, case: &Value) -> Self {
        let dir = work.join("item10").join(text(case, "dir"));
        let batch = case["identity_batch"].as_i64().expect("identity batch");
        for (path, body) in FIXED_FILES {
            write(&dir.join(path), body);
        }
        write(
            &dir.join("models/pose.onnx"),
            &onnx_bytes(recipe, text(case, "variant")),
        );
        let infer = [
            "[property]".to_owned(),
            format!("onnx-file={}", dir.join("models/pose.onnx").display()),
            format!(
                "model-engine-file={}",
                dir.join("cache/model.engine").display()
            ),
            format!("batch-size={batch}"),
            "infer-dims=3;640;640".to_owned(),
        ];
        write(&dir.join("config/infer.yml"), infer.join("\n").as_bytes());
        let engine = dir.join("cache/model.engine");
        write(&engine, format!("engine b{batch}\n").as_bytes());
        fs::create_dir_all(dir.join("records")).expect("records dir");
        let files: Vec<(&str, PathBuf)> = IDENTITY_FILES
            .iter()
            .map(|(key, path)| (*key, dir.join(path)))
            .collect();
        let identity = identity_for(&engine, &files, "sha256:golden-image", i128::from(batch))
            .expect("identity renders");
        write(&dir.join("cache/identity.json"), identity.as_bytes());
        let built = Self {
            work: work.to_path_buf(),
            dir,
        };
        for (variant, path) in boot_edits(text(case, "case")) {
            write(&built.dir.join(path), &onnx_bytes(recipe, variant));
        }
        built
    }

    fn env(&self, case: &Value) -> Env {
        let mut env: Env = BOOT_PATHS
            .iter()
            .map(|(key, path)| (key.to_string(), self.dir.join(path).display().to_string()))
            .chain(
                BOOT_VALUES
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string())),
            )
            .collect();
        if let Some(batch) = case["env_batch"].as_str() {
            env.insert("ML_WORKER_FLOW_BATCH_SIZE".to_owned(), batch.to_owned());
        }
        for key in case["env_removed"].as_array().expect("env_removed") {
            env.remove(key.as_str().expect("removed key"));
        }
        env
    }

    /// The digest a `<sha256:ROLE>` placeholder of the recipe names.
    fn role_digest(&self, recipe: &Value, placeholder: &str) -> String {
        let role = placeholder
            .strip_prefix("<sha256:")
            .and_then(|rest| rest.strip_suffix('>'))
            .unwrap_or_else(|| panic!("unmapped placeholder {placeholder}"));
        if let Some(batch) = role
            .strip_prefix("engine(b")
            .and_then(|r| r.strip_suffix(')'))
        {
            return sha256_hex(format!("engine b{batch}\n").as_bytes());
        }
        if let Some(variant) = role.strip_prefix("onnx[").and_then(|r| r.strip_suffix(']')) {
            return sha256_hex(&onnx_bytes(recipe, variant));
        }
        if let Some(dir) = role
            .strip_prefix("infer-config[")
            .and_then(|r| r.strip_suffix(']'))
        {
            let path = self.work.join("item10").join(dir).join("config/infer.yml");
            return sha256_hex(&fs::read(path).expect("infer config"));
        }
        let body: &[u8] = match role {
            "parser-lib" => FIXED_FILES[0].1,
            "tracker-lib" => FIXED_FILES[1].1,
            "tracker-config" => FIXED_FILES[2].1,
            _ => panic!("unmapped digest role {role}"),
        };
        sha256_hex(body)
    }
}

/// Reviewed rewrite of case texts that edit a file after the identity is
/// written: the variant written over the file. Every other case is laid out
/// from its fields alone.
fn boot_edits(case: &str) -> Vec<(&'static str, &'static str)> {
    match case {
        "ONNX replaced by fixed-1 after the identity was written" => {
            vec![("fixed-1", "models/pose.onnx")]
        }
        _ => vec![],
    }
}

/// Reviewed map from an `EngineIdentityError` (or uncaught) class and
/// message to the kind and the identity key it names (as `wire_config.rs`).
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

/// Reviewed map from an `input_dims` refusal to the kind and the ONNX path
/// it names. onnxruntime's `InvalidProtobuf` names no path; the subject is
/// the path that was read. `<work>` is the scratch root.
fn shape_refusal(class: &str, message: &str, work: &Path, onnx: &Path) -> (OnnxShapeKind, String) {
    let work = work.display().to_string();
    match (class, message.strip_prefix("ONNX artifact has no inputs: ")) {
        ("OnnxShapeError" | "EngineIdentityError", Some(path)) => {
            (OnnxShapeKind::NoInputs, path.replace("<work>", &work))
        }
        ("InvalidProtobuf", _) => (OnnxShapeKind::Unloadable, onnx.display().to_string()),
        _ => panic!("unmapped shape refusal {class}: {message:?}"),
    }
}

/// Reviewed map from a `verify_flow_boot_inputs` refusal to the kind and
/// what the message names.
fn flow_refusal(class: &str, message: &str, work: &Path, onnx: &Path) -> (FlowBootKind, String) {
    if let Some(keys) = message.strip_prefix("flow profile wiring is missing: ") {
        return (FlowBootKind::WiringMissing, keys.to_owned());
    }
    if message == "flow profile batch size must be a positive integer" {
        return (FlowBootKind::BatchNotPositive, String::new());
    }
    if let Some(rest) = message.strip_prefix("Flow engine batch ")
        && let Some((built, configured)) = rest.split_once(" does not match configured batch ")
    {
        return (FlowBootKind::BatchMismatch, format!("{built} {configured}"));
    }
    if let Some(rest) = message.strip_prefix("Flow ONNX has fixed batch dimension ") {
        let dim = rest.split(';').next().unwrap_or_default();
        return (FlowBootKind::FixedBatch, dim.to_owned());
    }
    if (class, message) == ("IndexError", "tuple index out of range") {
        return (FlowBootKind::ShapeRankZero, String::new());
    }
    if class == "InvalidProtobuf" || message.starts_with("ONNX artifact has no inputs: ") {
        let (kind, subject) = shape_refusal(class, message, work, onnx);
        return (FlowBootKind::Shape(kind), subject);
    }
    let (kind, subject) = identity_refusal(class, message);
    (FlowBootKind::Identity(kind), subject)
}

#[test]
fn flow_boot_matches_python() {
    let golden = golden(GOLDEN);
    let recipe = &golden["recipe"];
    let work = scratch("boot");
    for case in cases(&golden, "flow_boot") {
        let description = text(case, "case");
        let boot = BootCase::new(recipe, &work, case);
        let deployed = case["deployed_batch"].as_i64().map(i128::from);
        let outcome = verify_flow_boot_inputs(&boot.env(case), deployed);
        match (text(case, "verdict"), outcome) {
            ("accepted", Ok(returned)) => {
                let mut expected = string_map(&case["returned"]);
                for value in expected.values_mut() {
                    if value.starts_with("<sha256:") {
                        *value = boot.role_digest(recipe, value);
                    }
                }
                assert_eq!(returned, expected, "{description}");
            }
            ("refused", Err(error)) => {
                let onnx = boot.dir.join("models/pose.onnx");
                let expected =
                    flow_refusal(text(case, "class"), text(case, "message"), &work, &onnx);
                assert_eq!((error.kind, error.subject), expected, "{description}");
            }
            (verdict, outcome) => panic!("{description}: {verdict} vs {outcome:?}"),
        }
    }
}

/// The golden `input_dims` list: an `int` is `Value`, a `str` is `Param` and
/// `None` is `Unknown`.
fn golden_dims(dims: &Value) -> Vec<Dim> {
    let dims = dims.as_array().expect("input_dims list");
    dims.iter()
        .map(|dim| match dim {
            Value::Null => Dim::Unknown,
            Value::String(param) => Dim::Param(param.clone()),
            other => Dim::Value(other.as_i64().expect("int dim")),
        })
        .collect()
}

/// The golden `batch_axis_is_dynamic`: a `bool`, or the `IndexError` Python
/// raises on an empty dims tuple (`None`).
fn golden_dynamic(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(dynamic) => Some(*dynamic),
        refused if text(refused, "class") == "IndexError" => None,
        other => panic!("unmapped batch_axis_is_dynamic {other}"),
    }
}

#[test]
fn onnx_input_dims_match_python() {
    let golden = golden(GOLDEN);
    let work = scratch("onnx");
    for case in cases(&golden, "onnx_input_dims") {
        let variant = text(case, "variant");
        let onnx = work.join("item10/onnx").join(format!("{variant}.onnx"));
        write(&onnx, &onnx_bytes(&golden["recipe"], variant));
        let outcome = input_dims(&onnx);
        match (text(case, "verdict"), outcome) {
            ("accepted", Ok(dims)) => {
                assert_eq!(dims, golden_dims(&case["input_dims"]), "{variant}");
                assert_eq!(
                    batch_axis_is_dynamic(&dims),
                    golden_dynamic(&case["batch_axis_is_dynamic"]),
                    "{variant}"
                );
            }
            ("refused", Err(error)) => {
                let expected =
                    shape_refusal(text(case, "class"), text(case, "message"), &work, &onnx);
                assert_eq!((error.kind, error.subject), expected, "{variant}");
            }
            (verdict, outcome) => panic!("{variant}: {verdict} vs {outcome:?}"),
        }
    }
}

#[test]
fn onnx_reader_bounds_corrupt_lengths() {
    let work = scratch("corrupt");
    let huge = varint(u64::from(u32::MAX));
    let mut graph_too_long = varint(7 << 3 | 2);
    graph_too_long.extend(&huge);
    graph_too_long.extend(b"pose");
    let mut input_too_long = bytes_field(2, b"pose");
    input_too_long.extend(varint(11 << 3 | 2));
    input_too_long.extend(varint(1 << 40));
    input_too_long.extend(b"frames");
    let mut dim_truncated = bytes_field(1, &varint_field(1, 1));
    dim_truncated.extend([0x08, 0x80]);
    let shape = bytes_field(2, &dim_truncated);
    let info = [
        bytes_field(1, b"frames"),
        bytes_field(2, &bytes_field(1, &shape)),
    ]
    .concat();
    let mut eleven_bytes = vec![0x08];
    eleven_bytes.extend([0x80; 10]);
    eleven_bytes.push(0x01);
    let corrupt = [
        ("graph length past the end", graph_too_long),
        ("input length past the end", model(&input_too_long)),
        ("dim varint truncated", model(&bytes_field(11, &info))),
        ("varint of eleven bytes", eleven_bytes),
        ("fixed64 truncated", vec![0x09, 1, 2, 3]),
    ];
    for (serial, (label, bytes)) in corrupt.into_iter().enumerate() {
        let onnx = work.join(format!("corrupt-{serial}.onnx"));
        write(&onnx, &bytes);
        let outcome = input_dims(&onnx).map_err(|error| (error.kind, error.subject));
        let expected = (OnnxShapeKind::Unloadable, onnx.display().to_string());
        assert_eq!(outcome, Err(expected), "{label}");
    }
}

/// `onnx.proto` numbers `GraphProto.input` 11 and `output` 12. Every recipe
/// variant gives both the same dims, so this graph gives them different dims
/// and puts the output first on the wire, which protobuf allows.
#[test]
fn onnx_input_dims_read_graph_input_not_output() {
    let work = scratch("input-field");
    let input = [
        Value::from("batch"),
        Value::from(3),
        Value::from(640),
        Value::from(640),
    ];
    let output = [Value::from(1), Value::from(17), Value::from(3)];
    let mut graph = bytes_field(2, b"pose");
    graph.extend(bytes_field(12, &value_info("output0", Some(&output))));
    graph.extend(bytes_field(11, &value_info("frames", Some(&input))));
    let onnx = work.join("input-field.onnx");
    write(&onnx, &model(&graph));
    let expected = vec![
        Dim::Param("batch".to_owned()),
        Dim::Value(3),
        Dim::Value(640),
        Dim::Value(640),
    ];
    assert_eq!(input_dims(&onnx).map_err(|error| error.kind), Ok(expected));
}

/// A golden compiled value: `null` is `None`, a `<RUNTIME_RESOLVED_...>`
/// string is the marker, any other string is itself and a number is not a
/// string.
fn compiled(value: &Value) -> Compiled<'_> {
    match value {
        Value::Null => Compiled::Absent,
        Value::String(marker) if marker.starts_with("<RUNTIME_RESOLVED_") => {
            Compiled::RuntimeResolved
        }
        Value::String(compiled) => Compiled::Text(compiled),
        Value::Number(_) => Compiled::NotText,
        other => panic!("unmapped compiled value {other}"),
    }
}

/// Reviewed map from a `DetectionModuleActivationError` message to the kind,
/// the component, the label and, for a mismatch, `"{compiled} {resolved}"`.
fn placeholder_refusal(message: &str) -> (PlaceholderKind, String, String, String) {
    let (component, rest) = message
        .strip_prefix("component '")
        .and_then(|rest| rest.split_once("' "))
        .unwrap_or_else(|| panic!("unmapped message {message:?}"));
    let unnamed = [
        ("has no compiled ", PlaceholderKind::NoCompiled),
        ("has no resolved ", PlaceholderKind::NoResolved),
    ];
    for (prefix, kind) in unnamed {
        if let Some(label) = rest
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(" identity"))
        {
            return (kind, component.to_owned(), label.to_owned(), String::new());
        }
    }
    let (label, pair) = rest
        .split_once(" identity mismatch: compiled '")
        .unwrap_or_else(|| panic!("unmapped message {message:?}"));
    let (compiled, resolved) = pair
        .strip_suffix('\'')
        .and_then(|pair| pair.split_once("', resolved '"))
        .unwrap_or_else(|| panic!("unmapped message {message:?}"));
    let subject = format!("{compiled} {resolved}");
    (
        PlaceholderKind::Mismatch,
        component.to_owned(),
        label.to_owned(),
        subject,
    )
}

/// The golden attribute maps hold strings only: a key is `Text`, no key is
/// `Absent`.
fn attr(value: Option<&Value>) -> Attr<'_> {
    value.map_or(Attr::Absent, |v| {
        Attr::Text(v.as_str().expect("string attr"))
    })
}

#[test]
fn placeholders_match_python() {
    let golden = golden(GOLDEN);
    for case in cases(&golden, "g1_placeholders") {
        let description = text(case, "case");
        let binding = &case["binding"];
        let field = text(binding, "field");
        let attrs = &case["component_attrs"];
        let private = format!("_{field}");
        let outcome = verified_identity_field(
            &Binding {
                component_id: text(binding, "id"),
                compiled: compiled(&binding["compiled"]),
            },
            Attrs {
                public: attr(attrs.get(field)),
                private: attr(attrs.get(&private)),
            },
            case["provisioned"].as_str(),
            text(case, "identity_label"),
            case["expected_override"]
                .as_str()
                .map(|_| compiled(&case["expected_override"])),
        );
        match (text(case, "verdict"), outcome) {
            ("accepted", Ok(resolved)) => {
                assert_eq!(resolved, text(case, "returned"), "{description}")
            }
            ("refused", Err(error)) => {
                assert_eq!(text(case, "class"), "DetectionModuleActivationError");
                let actual = (error.kind, error.component, error.label, error.subject);
                assert_eq!(
                    actual,
                    placeholder_refusal(text(case, "message")),
                    "{description}"
                );
            }
            (verdict, outcome) => panic!("{description}: {verdict} vs {outcome:?}"),
        }
    }
}

/// `_verified_identity_field` reads `getattr(component, field, None)`, then
/// `_field` only while the value is `None`. A present non-`str` value such as
/// `7` is not `None`, so the chain stops and the `str` check refuses it
/// (`model_composition.py` L207-211). It must not fall through to the private
/// attribute or to `provisioned`.
#[test]
fn placeholders_non_text_attr_stops_chain() {
    let binding = Binding {
        component_id: "pose",
        compiled: Compiled::RuntimeResolved,
    };
    let run = |public, private, provisioned| {
        verified_identity_field(
            &binding,
            Attrs { public, private },
            provisioned,
            "artifact",
            None,
        )
        .map_err(|error| error.kind)
    };
    let stopped = Err(PlaceholderKind::NoResolved);
    assert_eq!(
        run(Attr::NotText, Attr::Absent, Some("sha256:provisioned")),
        stopped
    );
    assert_eq!(
        run(Attr::NotText, Attr::Text("sha256:private"), None),
        stopped
    );
    assert_eq!(run(Attr::Absent, Attr::NotText, Some("sha256:p")), stopped);
    assert_eq!(
        run(Attr::Absent, Attr::Text("sha256:private"), Some("sha256:p")),
        Ok("sha256:private".to_owned())
    );
    assert_eq!(
        run(Attr::Text("sha256:public"), Attr::NotText, None),
        Ok("sha256:public".to_owned())
    );
    assert_eq!(
        run(Attr::Absent, Attr::Absent, Some("sha256:provisioned")),
        Ok("sha256:provisioned".to_owned())
    );
}

/// What one `ML_WORKER_FLOW_BATCH_SIZE` text does at a boot whose engine was
/// built for batch 13 with a dynamic-batch ONNX.
#[derive(Debug, Eq, PartialEq)]
enum BatchText {
    /// Parsed to 13: the boot succeeds.
    Thirteen,
    /// Parsed to a positive value other than 13: `BatchMismatch`.
    OtherPositive,
    /// `int()` raised or the value is `<= 0`: `BatchNotPositive`.
    Refused,
}

fn batch_text_outcome(boot: &BootCase, case: &Value, text: &str) -> BatchText {
    let mut env = boot.env(case);
    env.insert("ML_WORKER_FLOW_BATCH_SIZE".to_owned(), text.to_owned());
    match verify_flow_boot_inputs(&env, None) {
        Ok(_) => BatchText::Thirteen,
        Err(error) if error.kind == FlowBootKind::BatchMismatch => BatchText::OtherPositive,
        Err(error) if error.kind == FlowBootKind::BatchNotPositive => BatchText::Refused,
        Err(error) => panic!("{text:?}: unexpected refusal {:?}", error.kind),
    }
}

fn batch_boot(label: &str) -> (BootCase, Value) {
    let golden = golden(GOLDEN);
    let case = serde_json::json!({
        "case": "int semantics",
        "dir": format!("int/{label}"),
        "identity_batch": 13,
        "variant": "symbolic",
        "env_removed": [],
    });
    let boot = BootCase::new(&golden["recipe"], &scratch(label), &case);
    (boot, case)
}

/// `cold_start.py:146` calls `int(env[...])` with no `.strip()`. Python 3.14.7
/// `int()` refuses U+001C..U+001F next to digits (`int('\x1c13')`,
/// `int('\x1c 13')` raise ValueError) and skips exactly the code points of
/// `char::is_whitespace`, U+3000 included.
#[test]
fn flow_batch_separator_controls_refused_ideographic_space_accepted() {
    let (boot, case) = batch_boot("separators");
    for text in ["\x1c13", "13\x1f", "\x1d13", "\x1e13", "\x1c 13", "\x1f"] {
        assert_eq!(
            batch_text_outcome(&boot, &case, text),
            BatchText::Refused,
            "{text:?}"
        );
    }
    for text in ["\u{3000}13", "13\u{3000}", "\u{a0}13\u{2028}"] {
        assert_eq!(
            batch_text_outcome(&boot, &case, text),
            BatchText::Thirteen,
            "{text:?}"
        );
    }
}

/// `int()` edge semantics of the batch env, each expectation verified with
/// Python 3.14.7: `+13`, `1_3` and `013` are 13, `13 ` and ` 13` are 13,
/// `_13`, `1__3`, `13_` and `+` raise ValueError, `-0` and `0_0` are 0 (not
/// positive), a 4300 digit string is accepted (also with one interior `_`) and
/// a 4301 digit string raises ValueError. Fullwidth digits (`int('１３')` is 13
/// in Python) are a documented fail-closed deviation: refused here.
#[test]
fn flow_batch_int_edges_match_python() {
    let (boot, case) = batch_boot("int-edges");
    let digits = |count: usize| format!("1{}", "0".repeat(count - 1));
    let with_underscore = format!("1_{}", "0".repeat(4299));
    let table: Vec<(String, BatchText)> = vec![
        ("+13".into(), BatchText::Thirteen),
        ("1_3".into(), BatchText::Thirteen),
        ("013".into(), BatchText::Thirteen),
        ("13 ".into(), BatchText::Thirteen),
        (" 13".into(), BatchText::Thirteen),
        ("_13".into(), BatchText::Refused),
        ("1__3".into(), BatchText::Refused),
        ("13_".into(), BatchText::Refused),
        ("+".into(), BatchText::Refused),
        ("-0".into(), BatchText::Refused),
        ("0_0".into(), BatchText::Refused),
        (digits(4300), BatchText::OtherPositive),
        (digits(4301), BatchText::Refused),
        (with_underscore, BatchText::OtherPositive),
        ("\u{ff11}\u{ff13}".into(), BatchText::Refused),
    ];
    for (text, expected) in table {
        let label: String = text.chars().take(12).collect();
        assert_eq!(
            batch_text_outcome(&boot, &case, &text),
            expected,
            "{label:?} (length {})",
            text.len()
        );
    }
}
