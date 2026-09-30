//! T19: execution-record wire parity with Python
//! `shared/events/execution_records.py`. The oracles are the reviewed Python
//! goldens `r/execution-records*.json` and their manifest transport entries.

use std::path::PathBuf;

use seeon_deepstream_native::GpuMetrics;
use seeon_ml_worker::json::Json;
use seeon_ml_worker::records::builder::{
    FallScore, Frame, ModelEvidence, Stream, model_score_record,
};
use seeon_ml_worker::records::id::canonical;
use seeon_ml_worker::records::{
    Batch, ContractError, Provenance, Receipt, Record, RecordBody, StorageState,
};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CAMERA: &str = "cmsnw6rjc01vhlh01oswn99yq";
const BOOT: &str = "boot-0001";
/// Manifest `fixed_inputs.wall_clock_sec` in nanoseconds.
const WALL_NS: u64 = 1_787_000_000_000_000_000;

fn worker_wire(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/worker-wire")
        .join(relative)
}

fn read_json(relative: &str) -> Value {
    let bytes = std::fs::read(worker_wire(relative)).expect("golden file is readable");
    serde_json::from_slice(&bytes).expect("golden file is JSON")
}

fn golden_entry(path: &str) -> Value {
    read_json("manifest.json")["goldens"]
        .as_array()
        .expect("manifest goldens is an array")
        .iter()
        .find(|entry| entry["path"] == path)
        .cloned()
        .expect("manifest lists the golden")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn provenance(model: &str, calibration: &str, preprocessing: &str) -> Provenance {
    Provenance {
        worker_build_revision: "abc123".to_owned(),
        worker_image_digest: "sha256:deadbeef".to_owned(),
        model_digest: model.to_owned(),
        calibration_digest: calibration.to_owned(),
        preprocessing_identity: preprocessing.to_owned(),
        config_digest: "cfg-1".to_owned(),
        policy_identity: "fall.policy:2".to_owned(),
    }
}

fn stream() -> Stream {
    Stream {
        camera_id: CAMERA.to_owned(),
        worker_boot_id: BOOT.to_owned(),
        source_generation: 1,
        stream_epoch: 1,
    }
}

/// The lane assigns the real producer sequence; the builder stamps 0.
fn sequenced(record: &Record, producer_sequence: u64) -> Record {
    Record::new(RecordBody {
        producer_sequence,
        ..record.body().clone()
    })
    .expect("renumbered record is valid")
}

/// The golden recipe: two model.score records, frame_seq 100 and 101, one
/// 30 fps frame apart, for fall-classifier generation 3 of track 1.
fn scored(index: u64, fall_transition: f64, fallen: f64) -> Record {
    let frame = Frame {
        frame_seq: 100 + index,
        source_pts_ns: Some(4_000_000_000 + i64::try_from(index).expect("small") * 33_333_333),
    };
    let score = FallScore {
        track_id: 1,
        generation: Some(3),
        fall_transition,
        background: 0.8,
        fallen,
        evidence: None,
    };
    let record = model_score_record(&stream(), frame, WALL_NS + index * 33_333_333, &score, None)
        .expect("golden record is valid");
    sequenced(&record, index)
}

/// Parses a C99 `%a` literal as `float.hex()` prints it (`0x1.<13 hex>p<exp>`),
/// independently of any decimal float formatting.
fn from_c99_hex(text: &str) -> f64 {
    let rest = text.strip_prefix("0x1.").expect("normalised hex float");
    let (fraction, exponent) = rest.split_once('p').expect("binary exponent");
    let mantissa = u64::from_str_radix(&format!("1{fraction}"), 16).expect("hex mantissa");
    let exponent: i32 = exponent.parse().expect("decimal exponent");
    let shift = exponent - 4 * i32::try_from(fraction.len()).expect("short fraction");
    assert!(mantissa < 1 << 53, "the mantissa is exact in a double");
    mantissa as f64 * 2f64.powi(shift)
}

fn text(value: &Value) -> String {
    value.as_str().expect("golden string").to_owned()
}

fn number(value: &Value) -> f64 {
    value.as_f64().expect("golden number")
}

/// The model-evidence recipe: frame 129 of track 1, generation 0, scored by
/// the runner in `runner_output`, with the logit taken from its float32 hex.
fn evidence_record(accelerator: Option<&AcceleratorEvidence>) -> Record {
    let golden = read_json("r/execution-records.model-evidence.json");
    let runner = &golden["runner_output"];
    let raw_logit = from_c99_hex(runner["raw_logit_float32_hex"].as_str().expect("hex"));
    assert_eq!(
        f64::from(raw_logit as f32),
        raw_logit,
        "the logit is a float32 value"
    );
    let score = FallScore {
        track_id: 1,
        generation: Some(0),
        fall_transition: number(&runner["fall_transition"]),
        background: number(&runner["background"]),
        fallen: number(&runner["fallen"]),
        evidence: Some(ModelEvidence {
            raw_logit,
            applied_temperature: number(&runner["applied_temperature"]),
            class_origins: runner["class_origins"]
                .as_array()
                .expect("class origins")
                .iter()
                .map(text)
                .collect(),
        }),
    };
    let frame = Frame {
        frame_seq: 129,
        source_pts_ns: Some(4_966_666_657),
    };
    model_score_record(&stream(), frame, WALL_NS, &score, accelerator)
        .expect("model-evidence record is valid")
}

fn metrics(attempted: u64, h2d: u64, d2h: u64, elapsed_ns: u64) -> GpuMetrics {
    GpuMetrics {
        attempted,
        succeeded: attempted,
        failed: 0,
        host_to_device_bytes: h2d,
        device_to_host_bytes: d2h,
        elapsed_ns,
        device: 0,
    }
}

fn record_value(record: &Record) -> Value {
    let text = canonical(&record.to_json()).expect("record encodes");
    serde_json::from_str(&text).expect("record JSON")
}

fn body_value(batch: &Batch) -> (Vec<u8>, Value) {
    let body = batch.encode().expect("golden batch encodes");
    let value = serde_json::from_slice(&body).expect("encoded body is JSON");
    (body, value)
}

#[test]
fn model_score_batch_matches_python_golden() {
    let golden = read_json("r/execution-records.json");
    let records = vec![scored(0, 0.12, 0.08), scored(1, 0.13, 0.07)];
    for (record, expected) in records
        .iter()
        .zip(golden["records"].as_array().expect("records"))
    {
        assert_eq!(record.record_id(), expected["record_id"]);
    }
    let prov = provenance("model-1", "cal-1", "pose-bbox56/v1");
    let batch = Batch::new(CAMERA, BOOT, prov, records, Vec::new()).expect("golden batch");
    assert_eq!(batch.batch_id(), golden["batch_id"]);
    let (body, value) = body_value(&batch);
    assert_eq!(value, golden);
    let transport = &golden_entry("r/execution-records.json")["transport"];
    assert_eq!(sha256_hex(&body), transport["body_sha256"]);
}

#[test]
fn model_evidence_batch_matches_python_golden() {
    let golden = read_json("r/execution-records.model-evidence.json");
    let posted = &golden["batch"];
    let record = evidence_record(None);
    assert_eq!(record.record_id(), posted["records"][0]["record_id"]);
    let prov = posted["provenance"].clone();
    let prov = Provenance {
        worker_build_revision: text(&prov["worker_build_revision"]),
        worker_image_digest: text(&prov["worker_image_digest"]),
        model_digest: text(&prov["model_digest"]),
        calibration_digest: text(&prov["calibration_digest"]),
        preprocessing_identity: text(&prov["preprocessing_identity"]),
        config_digest: text(&prov["config_digest"]),
        policy_identity: text(&prov["policy_identity"]),
    };
    let batch = Batch::new(CAMERA, BOOT, prov, vec![record], Vec::new()).expect("golden batch");
    assert_eq!(batch.batch_id(), posted["batch_id"]);
    let (body, value) = body_value(&batch);
    assert_eq!(&value, posted);
    let transport = &golden_entry("r/execution-records.model-evidence.json")["transport"];
    assert_eq!(sha256_hex(&body), transport["body_sha256"]);
}

#[test]
fn accelerator_receipt_is_hashed_into_the_record() {
    let before = metrics(4, 1_000, 200, 4_000);
    let after = metrics(5, 1_600, 260, 4_789);
    let receipt = AcceleratorEvidence::from_delta(
        &before,
        &after,
        0,
        EngineDigest::new([0xab; 32]),
        Precision::Fp32,
    )
    .expect("one successful call with both copies");
    let with = evidence_record(Some(&receipt));
    let without = evidence_record(None);
    let mut value = record_value(&with);
    let payload = value["payload"].as_object_mut().expect("payload object");
    let accelerator = payload.remove("accelerator").expect("payload.accelerator");
    assert_eq!(
        accelerator,
        json!({
            "provider": "tensorrt",
            "precision": "fp32",
            "device_ordinal": 0,
            "engine_sha256": "ab".repeat(32),
            "call_seq": 5,
            "attempted": 1,
            "succeeded": 1,
            "failed": 0,
            "h2d_bytes": 600,
            "d2h_bytes": 60,
            "elapsed_ns": 789,
        })
    );
    let mut expected = record_value(&without);
    for record in [&mut value, &mut expected] {
        record.as_object_mut().expect("record").remove("record_id");
    }
    assert_eq!(value, expected, "the accelerator is the only difference");
    assert_ne!(with.record_id(), without.record_id());
    let rebuilt = Record::new(with.body().clone()).expect("rebuilt record is valid");
    assert_eq!(
        rebuilt.record_id(),
        with.record_id(),
        "the id covers the accelerator"
    );
}

#[test]
fn receipts_parse_and_echo_their_batch() {
    for (response, batch) in [
        (
            "r/execution-records.response.json",
            "r/execution-records.json",
        ),
        (
            "r/execution-records.model-evidence.response.json",
            "r/execution-records.model-evidence.json",
        ),
    ] {
        let golden = read_json(batch);
        let posted = golden.get("batch").unwrap_or(&golden);
        let receipt = Receipt::from_json(&read_json(response)).expect("golden receipt parses");
        assert_eq!(receipt.batch_id, posted["batch_id"]);
        let count = posted["records"].as_array().expect("records").len();
        assert_eq!(receipt.accepted, u64::try_from(count).expect("small"));
        assert_eq!((receipt.duplicates, receipt.rejected.len()), (0, 0));
        assert_eq!(receipt.storage_state, StorageState::Committed);
    }
}

#[test]
fn non_finite_payload_is_a_typed_refusal() {
    let frame = Frame {
        frame_seq: 100,
        source_pts_ns: None,
    };
    for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let score = FallScore {
            track_id: 1,
            generation: Some(3),
            fall_transition: 0.12,
            background: 0.8,
            fallen: number,
            evidence: None,
        };
        let refused = model_score_record(&stream(), frame, WALL_NS, &score, None);
        assert_eq!(refused, Err(ContractError::NonFinite));
        let mut body = scored(0, 0.12, 0.08).body().clone();
        body.payload
            .push(("raw_logit".to_owned(), Json::Float(number)));
        assert_eq!(Record::new(body), Err(ContractError::NonFinite));
    }
}
