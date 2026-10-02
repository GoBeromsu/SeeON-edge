//! Policy-boundary checks for `verify_with`. This is not whole-worker
//! schema-1 or native provenance proof: the loader closure supplies an
//! in-memory admitted media map, and production wiring, batch, deployed
//! roster and ONNX checks run against that map. Aggregate source and hash
//! admission lives in `aggregate_admission.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::super::identity::IdentityKind;
use super::super::onnx_shape::OnnxShapeKind;
use super::{Boot, FlowBootError, FlowBootKind, verify_with};
use crate::config::env::Env;

#[path = "."]
mod recipe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/flow_recipe.rs"
    ));
}

use recipe::{cases, golden, onnx_bytes, scratch, text, write};

/// The recipe's per-case files and their fixed bodies (`pose.onnx`,
/// `infer.yml` and the engine depend on the case).
const FIXED_FILES: [(&str, &[u8]); 3] = [
    ("models/libparser.so", b"parser-lib\n"),
    ("models/libtracker.so", b"tracker-lib\n"),
    ("config/tracker.yml", b"tracker-config: synthetic\n"),
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

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn string_map(value: &Value) -> BTreeMap<String, String> {
    serde_json::from_value(value.clone()).expect("string map")
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
        let infer = infer_config(&dir, batch);
        write(&dir.join("config/infer.yml"), infer.as_bytes());
        write(
            &dir.join("cache/model.engine"),
            format!("engine b{batch}\n").as_bytes(),
        );
        std::fs::create_dir_all(dir.join("records")).expect("records dir");
        // Wiring requires the identity path to exist as a non-empty env
        // value. The file is not a schema-1 aggregate and is not read by
        // the policy verifier.
        write(&dir.join("cache/identity.json"), b"{}\n");
        let built = Self {
            work: work.to_path_buf(),
            dir,
        };
        for (variant, path) in boot_edits(text(case, "case")) {
            write(&built.dir.join(path), &onnx_bytes(recipe, variant));
        }
        built
    }

    fn env(&self, case: &Value) -> BTreeMap<String, String> {
        let mut env: BTreeMap<String, String> = BOOT_PATHS
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
            return sha256_hex(&std::fs::read(&path).unwrap_or_else(|error| {
                panic!("frozen infer config {} is missing: {error}", path.display())
            }));
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

fn infer_config(dir: &Path, batch: i64) -> String {
    [
        "[property]".to_owned(),
        format!("onnx-file={}", dir.join("models/pose.onnx").display()),
        format!(
            "model-engine-file={}",
            dir.join("cache/model.engine").display()
        ),
        format!("batch-size={batch}"),
        "infer-dims=3;640;640".to_owned(),
    ]
    .join("\n")
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

const GOLDEN: &str = "d/engine-identity-refusals.json";
const ONNX_REPLACED: &str = "ONNX replaced by fixed-1 after the identity was written";

/// In-memory admitted media map for one policy row. Digests are the current
/// fixture files, except the post-write ONNX replacement, whose admitted
/// digest is the file that existed before the replacement.
struct PolicyBoot {
    layout: BootCase,
    admitted: BTreeMap<String, String>,
    admitted_onnx: String,
}

impl PolicyBoot {
    fn new(recipe: &Value, work: &Path, case: &Value) -> Self {
        let description = text(case, "case");
        let layout = BootCase::new(recipe, work, case);
        let current_onnx =
            std::fs::read(layout.dir.join("models/pose.onnx")).expect("current ONNX fixture");
        let admitted_onnx = if description == ONNX_REPLACED {
            assert_eq!(
                boot_edits(description),
                vec![("fixed-1", "models/pose.onnx")]
            );
            sha256_hex(&onnx_bytes(recipe, text(case, "variant")))
        } else {
            sha256_hex(&current_onnx)
        };
        let batch = case["identity_batch"].as_i64().expect("identity batch");
        let mut admitted = BTreeMap::new();
        admitted.insert("batch_size".to_owned(), batch.to_string());
        admitted.insert("image_digest".to_owned(), "sha256:golden-image".to_owned());
        admitted.insert(
            "engine_sha256".to_owned(),
            sha256_hex(format!("engine b{batch}\n").as_bytes()),
        );
        admitted.insert("onnx_sha256".to_owned(), admitted_onnx.clone());
        admitted.insert(
            "infer_config_sha256".to_owned(),
            sha256_hex(&std::fs::read(layout.dir.join("config/infer.yml")).expect("infer config")),
        );
        admitted.insert("parser_lib_sha256".to_owned(), sha256_hex(FIXED_FILES[0].1));
        admitted.insert(
            "tracker_library_sha256".to_owned(),
            sha256_hex(FIXED_FILES[1].1),
        );
        admitted.insert(
            "tracker_config_sha256".to_owned(),
            sha256_hex(FIXED_FILES[2].1),
        );
        Self {
            layout,
            admitted,
            admitted_onnx,
        }
    }

    fn verify(&self, case: &Value) -> Boot<BTreeMap<String, String>> {
        let env = Env::from(self.layout.env(case));
        let deployed = case["deployed_batch"].as_i64().map(i128::from);
        let admitted = self.admitted.clone();
        let admitted_onnx = self.admitted_onnx.clone();
        let onnx = self.layout.dir.join("models/pose.onnx");
        verify_with(&env, deployed, || {
            if text(case, "case") == ONNX_REPLACED {
                let current = sha256_hex(&std::fs::read(&onnx).expect("current ONNX"));
                if current != admitted_onnx {
                    return Err(FlowBootError {
                        kind: FlowBootKind::Identity(IdentityKind::DigestMismatch),
                        subject: "onnx_sha256".to_owned(),
                    });
                }
            }
            Ok(admitted)
        })
    }
}

/// Reviewed map from a `verify_flow_boot_inputs` refusal to the kind and
/// what the message names. Identity-file refusals are not re-proved here.
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
        return shape_subject(class, message, work, onnx);
    }
    if message.starts_with("Flow artifact digest mismatch for ") {
        let subject = message
            .strip_prefix("Flow artifact digest mismatch for ")
            .and_then(|rest| rest.split(':').next())
            .unwrap_or_default();
        return (
            FlowBootKind::Identity(IdentityKind::DigestMismatch),
            subject.to_owned(),
        );
    }
    if message.starts_with("Flow engine batch ") && message.contains("does not cover") {
        // `verify_with` names no field for this refusal. The aggregate
        // reader's own `batch_size` subject is a separate contract.
        return (
            FlowBootKind::Identity(IdentityKind::BatchNotCovering),
            String::new(),
        );
    }
    panic!("unmapped flow policy refusal {class}: {message:?}");
}

fn shape_subject(class: &str, message: &str, work: &Path, onnx: &Path) -> (FlowBootKind, String) {
    let work = work.display().to_string();
    match (class, message.strip_prefix("ONNX artifact has no inputs: ")) {
        ("OnnxShapeError" | "EngineIdentityError", Some(path)) => (
            FlowBootKind::Shape(OnnxShapeKind::NoInputs),
            path.replace("<work>", &work),
        ),
        ("InvalidProtobuf", _) => (
            FlowBootKind::Shape(OnnxShapeKind::Unloadable),
            onnx.display().to_string(),
        ),
        _ => panic!("unmapped shape refusal {class}: {message:?}"),
    }
}

#[test]
fn flow_boot_policy_matches_frozen_matrix() {
    let document = golden(GOLDEN);
    let recipe = &document["recipe"];
    let rows = cases(&document, "flow_boot");
    assert_eq!(rows.len(), 35, "frozen flow_boot matrix");
    let work = scratch("policy");
    for case in rows {
        let description = text(case, "case");
        let boot = PolicyBoot::new(recipe, &work, case);
        let outcome = boot.verify(case);
        match (text(case, "verdict"), outcome) {
            ("accepted", Ok(returned)) => {
                let mut expected = string_map(&case["returned"]);
                for value in expected.values_mut() {
                    if value.starts_with("<sha256:") {
                        *value = boot.layout.role_digest(recipe, value);
                    }
                }
                assert_eq!(returned, expected, "{description}");
            }
            ("refused", Err(error)) => {
                let onnx = boot.layout.dir.join("models/pose.onnx");
                let expected =
                    flow_refusal(text(case, "class"), text(case, "message"), &work, &onnx);
                assert_eq!((error.kind, error.subject), expected, "{description}");
            }
            (verdict, outcome) => panic!("{description}: {verdict} vs {outcome:?}"),
        }
    }
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

fn batch_text_outcome(boot: &PolicyBoot, case: &Value, text: &str) -> BatchText {
    let mut row = case.clone();
    row["env_batch"] = Value::String(text.to_owned());
    match boot.verify(&row) {
        Ok(_) => BatchText::Thirteen,
        Err(error) if error.kind == FlowBootKind::BatchMismatch => BatchText::OtherPositive,
        Err(error) if error.kind == FlowBootKind::BatchNotPositive => BatchText::Refused,
        Err(error) => panic!("{text:?}: unexpected refusal {:?}", error.kind),
    }
}

fn batch_boot(label: &str) -> (PolicyBoot, Value) {
    let document = golden(GOLDEN);
    let case = serde_json::json!({
        "case": "int semantics",
        "dir": format!("int/{label}"),
        "identity_batch": 13,
        "variant": "symbolic",
        "env_removed": [],
    });
    let boot = PolicyBoot::new(&document["recipe"], &scratch(label), &case);
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
