//! Engine-only nvinfer text. Serving may name an already-built engine; the
//! seven construction keys native admission rejects must not be present.

const CONSTRUCTION: [&str; 7] = [
    "onnx-file",
    "model-file",
    "proto-file",
    "uff-file",
    "tlt-encoded-model",
    "custom-network-config",
    "engine-create-func-name",
];

pub(crate) fn engine_only(text: &str) -> bool {
    if text.bytes().any(|byte| byte == 0) {
        return false;
    }
    let mut property = false;
    for line in text.split('\n') {
        let line = line.trim_end_matches('\r').trim_matches(ascii_space);
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(name) = section(line) {
            property = name == "property";
            continue;
        }
        if !property {
            continue;
        }
        let Some(key) = line.split_once('=').map(|(key, _)| key) else {
            continue;
        };
        if CONSTRUCTION.contains(&key.trim_matches(ascii_space)) {
            return false;
        }
    }
    true
}

fn section(line: &str) -> Option<&str> {
    let body = line.strip_prefix('[')?.strip_suffix(']')?;
    let name = body.trim_matches(ascii_space);
    (!name.is_empty()).then_some(name)
}

fn ascii_space(byte: char) -> bool {
    matches!(byte, ' ' | '\t' | '\u{000B}' | '\u{000C}')
}
