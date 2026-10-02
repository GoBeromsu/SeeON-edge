// Shared flow-recipe fixtures included by private policy and integration tests.
// Callers pass paths explicitly and never change process cwd or environment.
// Scratch roots are exclusive; a preexisting path is never deleted.
// Policy-only boot layout lives with the policy tests that consume it.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

static SCRATCH_SERIAL: AtomicU64 = AtomicU64::new(0);

fn wire(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(path)
}

pub(super) fn golden(path: &str) -> Value {
    let text = fs::read_to_string(wire(path)).unwrap_or_else(|error| {
        panic!("frozen golden {path} is missing or unreadable: {error}")
    });
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("frozen golden {path} is not JSON: {error}"))
}

/// An exclusive scratch directory. Never removes a preexisting path.
pub(super) fn scratch(label: &str) -> PathBuf {
    let tick = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let serial = SCRATCH_SERIAL.fetch_add(1, Ordering::Relaxed);
    let name = format!(
        "flow-recipe-{}-{tick}-{serial}-{label}",
        std::process::id()
    );
    let root = std::env::temp_dir().join(name);
    if root.exists() {
        panic!(
            "scratch path already exists and is not deleted: {}",
            root.display()
        );
    }
    fs::create_dir(&root).unwrap_or_else(|error| {
        panic!("exclusive scratch {} was not created: {error}", root.display())
    });
    root
}

pub(super) fn write(path: &Path, body: &[u8]) {
    fs::create_dir_all(path.parent().expect("parent")).expect("parent dir");
    fs::write(path, body).expect("fixture file");
}

pub(super) fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is a string"))
}

pub(super) fn cases<'a>(golden: &'a Value, group: &str) -> &'a [Value] {
    let cases = golden[group].as_array().expect("case group");
    assert!(!cases.is_empty(), "{group} has cases");
    cases
}

// Protobuf writer: `onnx.helper` as `tests/test_edge_engine_build.py`
// `_write_onnx` uses it, one field at a time.

pub(super) fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while value >= 0x80 {
        out.push((value & 0x7f) as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
    out
}

pub(super) fn varint_field(number: u64, value: u64) -> Vec<u8> {
    let mut out = varint(number << 3);
    out.extend(varint(value));
    out
}

pub(super) fn bytes_field(number: u64, body: &[u8]) -> Vec<u8> {
    let mut out = varint((number << 3) | 2);
    out.extend(varint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

/// `make_tensor_value_info(name, FLOAT, dims)`: `None` dims leave the shape
/// unset; `[]` sets an empty shape; a `None` dim is a `Dimension` with
/// neither `dim_value` nor `dim_param`.
pub(super) fn value_info(name: &str, dims: Option<&[Value]>) -> Vec<u8> {
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
pub(super) fn model(graph: &[u8]) -> Vec<u8> {
    let mut opset = bytes_field(1, b"");
    opset.extend(varint_field(2, 13));
    let mut out = varint_field(1, 9);
    out.extend(bytes_field(7, graph));
    out.extend(bytes_field(8, &opset));
    out
}

/// The recipe's variant bytes: one `Identity` node `frames -> output0` whose
/// input and output share the declared dims, or the two special variants.
pub(super) fn onnx_bytes(recipe: &Value, variant: &str) -> Vec<u8> {
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
