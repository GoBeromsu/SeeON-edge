//! ONNX input dims, reconnect interval and identity placeholders against the
//! `flow_rtsp_reconnect_interval`, `onnx_input_dims` and `g1_placeholders`
//! groups of `d/engine-identity-refusals.json`, recorded from the Python
//! worker at 030aaf1. Python exception messages map to Rust variants only
//! through the reviewed tables in this file.
//!
//! The frozen `flow_boot` matrix and the two batch-text edge tests are
//! policy-boundary checks. They live in the private `flow_boot::policy_tests`
//! child and call `verify_with` with an injected admitted media map. They are
//! not whole-worker schema-1 or native provenance proof.

use std::path::Path;

use seeon_ml_worker::config::env::Env;
use seeon_ml_worker::config::model_bundle::composition::{
    Attr, Attrs, Binding, Compiled, PlaceholderKind, verified_identity_field,
};
use seeon_ml_worker::config::model_bundle::flow_boot::{FlowBootKind, rtsp_reconnect_interval_sec};
use seeon_ml_worker::config::model_bundle::onnx_shape::{
    Dim, OnnxShapeKind, batch_axis_is_dynamic, input_dims, input_dims_bytes,
};
use serde_json::Value;

mod recipe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/flow_recipe.rs"
    ));
}

use recipe::{
    bytes_field, cases, golden, model, onnx_bytes, scratch, text, value_info, varint, varint_field,
    write,
};

const GOLDEN: &str = "d/engine-identity-refusals.json";

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

/// Reviewed map from a reconnect-interval refusal message to its kind; the
/// subject is the raw value the row carries.
fn reconnect_refusal(message: &str, raw: &str) -> (FlowBootKind, String) {
    assert_eq!(
        message,
        "ML_WORKER_FLOW_RTSP_RECONNECT_INTERVAL_SEC must be a non-negative integer \
         (0-86400, digits only)",
        "unmapped reconnect-interval refusal"
    );
    (FlowBootKind::ReconnectInterval, raw.to_owned())
}

#[test]
fn rtsp_reconnect_interval_matches_python() {
    let document = golden(GOLDEN);
    for case in cases(&document, "flow_rtsp_reconnect_interval") {
        let description = text(case, "case");
        let mut env = Env::new();
        if let Some(raw) = case["raw"].as_str() {
            env.insert(
                "ML_WORKER_FLOW_RTSP_RECONNECT_INTERVAL_SEC".to_owned(),
                raw.to_owned(),
            );
        }
        match (text(case, "verdict"), rtsp_reconnect_interval_sec(&env)) {
            ("accepted", Ok(interval)) => {
                let expected = case["interval_sec"].as_u64().expect("accepted interval");
                assert_eq!(u64::from(interval), expected, "{description}");
            }
            ("refused", Err(error)) => {
                let raw = case["raw"].as_str().expect("a refused row sets the key");
                let expected = reconnect_refusal(text(case, "message"), raw);
                assert_eq!((error.kind, error.subject), expected, "{description}");
            }
            (verdict, outcome) => panic!("{description}: {verdict} vs {outcome:?}"),
        }
    }
}

/// The golden `input_dims` list: an `int` is `Value`, a `str` is `Param` and
/// `None` is `Unknown`.
fn golden_dims(dims: &Value) -> Vec<Dim> {
    dims.as_array()
        .expect("input_dims list")
        .iter()
        .cloned()
        .map(|dim| match dim {
            Value::Null => Dim::Unknown,
            Value::String(param) => Dim::Param(param),
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
    let document = golden(GOLDEN);
    let work = scratch("onnx");
    for case in cases(&document, "onnx_input_dims") {
        let variant = text(case, "variant");
        let onnx = work.join("item10/onnx").join(format!("{variant}.onnx"));
        write(&onnx, &onnx_bytes(&document["recipe"], variant));
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

/// The same frozen matrix, read from captured bytes. The path is the
/// diagnostic subject only; the bytes API does not open it.
#[test]
fn onnx_input_dims_bytes_match_python() {
    let document = golden(GOLDEN);
    let work = scratch("onnx");
    for case in cases(&document, "onnx_input_dims") {
        let variant = text(case, "variant");
        let onnx = work.join("item10/onnx").join(format!("{variant}.onnx"));
        let bytes = onnx_bytes(&document["recipe"], variant);
        let outcome = input_dims_bytes(&bytes, &onnx);
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

/// Captured bytes stay the model after the source path is overwritten or
/// removed. The file API still reads whatever is at the path.
#[test]
fn onnx_input_dims_bytes_keep_captured_model_after_source_changes() {
    let document = golden(GOLDEN);
    let work = scratch("onnx-captured");
    let onnx = work.join("item10/onnx/symbolic.onnx");
    let captured = onnx_bytes(&document["recipe"], "symbolic");
    let expected = golden_dims(&document["recipe"]["onnx_variants"]["symbolic"]);
    write(&onnx, &captured);
    write(&onnx, b"not-an-onnx-model\n");
    assert_eq!(input_dims_bytes(&captured, &onnx), Ok(expected.clone()));
    let overwritten = input_dims(&onnx).map_err(|error| (error.kind, error.subject));
    assert_eq!(
        overwritten,
        Err((OnnxShapeKind::Unloadable, onnx.display().to_string()))
    );
    std::fs::remove_file(&onnx).expect("owned scratch file removed");
    assert!(!onnx.exists());
    assert_eq!(input_dims_bytes(&captured, &onnx), Ok(expected));
}

/// Malformed and input-less captured bytes name the diagnostic subject,
/// including a source path that was never created.
#[test]
fn onnx_input_dims_bytes_name_malformed_and_missing_input_subjects() {
    let document = golden(GOLDEN);
    let work = scratch("onnx-bytes-refuse");
    let absent = work.join("item10/onnx/never-written.onnx");
    assert!(!absent.exists());
    let unloadable = onnx_bytes(&document["recipe"], "unloadable");
    let no_inputs = onnx_bytes(&document["recipe"], "no-inputs");
    let malformed =
        input_dims_bytes(&unloadable, &absent).map_err(|error| (error.kind, error.subject));
    assert_eq!(
        malformed,
        Err((OnnxShapeKind::Unloadable, absent.display().to_string()))
    );
    let missing =
        input_dims_bytes(&no_inputs, &absent).map_err(|error| (error.kind, error.subject));
    assert_eq!(
        missing,
        Err((OnnxShapeKind::NoInputs, absent.display().to_string()))
    );
    let huge = varint(u64::from(u32::MAX));
    let mut graph_too_long = varint(7 << 3 | 2);
    graph_too_long.extend(&huge);
    graph_too_long.extend(b"pose");
    let truncated =
        input_dims_bytes(&graph_too_long, &absent).map_err(|error| (error.kind, error.subject));
    assert_eq!(
        truncated,
        Err((OnnxShapeKind::Unloadable, absent.display().to_string()))
    );
    assert!(!absent.exists());
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
    let document = golden(GOLDEN);
    for case in cases(&document, "g1_placeholders") {
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
