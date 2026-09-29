//! T2: `b64.rs` against the queue files Python `delivery_queue._serialize`
//! wrote with `base64.b64encode` (`d/codec-edges.json` `queue.entries`).

use std::path::PathBuf;

use seeon_ml_worker::b64;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn wire_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/worker-wire")
}

fn bytes_from_hex(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "{hex}: odd hex length");
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).expect("hex byte"))
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn encode_matches_python_queue_files() {
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(wire_dir().join("d/codec-edges.json"))
            .expect("golden is readable"),
    )
    .expect("golden is JSON");
    let entries = golden["queue"]["entries"]
        .as_array()
        .expect("queue.entries");
    assert_eq!(entries.len(), 6);
    for entry in entries {
        let case = entry["case"].as_str().expect("case");
        let file = &entry["file"];
        let recorded = std::fs::read(wire_dir().join(file["path"].as_str().expect("file.path")))
            .expect("recorded queue file is readable");
        assert_eq!(
            Some(recorded.len() as u64),
            file["size_bytes"].as_u64(),
            "{case}: size"
        );
        assert_eq!(
            Some(sha256_hex(&recorded).as_str()),
            file["sha256"].as_str(),
            "{case}: sha256"
        );
        let written: Value = serde_json::from_slice(&recorded).expect("queue file is JSON");
        for (hex_field, b64_field) in [
            ("decision_trace_hex", "decision_trace_b64"),
            ("values_hex", "values_b64"),
        ] {
            let raw = bytes_from_hex(entry["fields"][hex_field].as_str().expect("hex field"));
            let encoded = b64::encode(&raw);
            assert_eq!(
                Some(encoded.as_str()),
                entry["written_b64"][b64_field].as_str(),
                "{case}: {b64_field}"
            );
            assert_eq!(
                Some(encoded.as_str()),
                written[b64_field].as_str(),
                "{case}: file {b64_field}"
            );
        }
    }
}
