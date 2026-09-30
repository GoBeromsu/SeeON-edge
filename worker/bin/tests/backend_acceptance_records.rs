//! T20: a Rust-encoded model.score batch carrying an accelerator receipt is
//! accepted by the real backend ingest route on PostgreSQL, and the stored
//! `payload.accelerator` equals the one the worker sent. The oracle is the
//! backend itself (`WireBatch.from_json` recomputes every record id), driven
//! through `fastapi.testclient` in a spawned Python interpreter.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use seeon_deepstream_native::GpuMetrics;
use seeon_ml_worker::records::builder::{FallScore, Frame, Stream, model_score_record};
use seeon_ml_worker::records::{Batch, Provenance, Receipt, Record, RecordBody, StorageState};
use seeon_worker_runtime::evidence::{AcceleratorEvidence, EngineDigest, Precision};
use serde_json::Value;

const CAMERA: &str = "cmsnw6rjc01vhlh01oswn99yq";
const BOOT: &str = "boot-t20";
const WALL_NS: u64 = 1_787_000_000_000_000_000;
const RESULT_PREFIX: &str = "T20RESULT ";

/// Installs the store by hand on a no-lifespan app, the way the backend's own
/// execution-record API tests do, then posts the body three times: without a
/// store and a wrong token, without a store and the right token, and with the
/// store and the right token. It prints the statuses, the receipt and every
/// stored record's id and accelerator.
const SCRIPT: &str = r#"
import json, os, sys
import psycopg
from fastapi.testclient import TestClient
from backend.app.core.config import get_settings
from backend.app.features.diagnostics.retention import RetentionBudget
from backend.app.features.diagnostics.store import ExecutionRecordStore
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_diagnostics_sandbox import diagnostics_database_for
from tests_support.postgres_sandbox import open_product_sandbox, product_audit_runtime

with open(sys.argv[1], "rb") as handle:
    body = handle.read()
os.environ["ML_API_EXECUTION_RECORDS_ENABLED"] = "true"
os.environ["ML_API_EXECUTION_RECORDS_BUDGET_BYTES"] = str(2**20)
os.environ["ML_API_BUILD_REVISION"] = "backend-t20"
get_settings.cache_clear()
dsn = os.environ["SEEON_TEST_POSTGRES_DSN"]
url = "/api/v1/relay/execution-records"
def headers(token):
    return {"X-Edge-Relay-Token": token, "Content-Type": "application/json"}
out = {}
admin = psycopg.connect(dsn, autocommit=True)
try:
    with open_product_sandbox(admin, dsn) as sandbox:
        app = postgres_api_app(sandbox, product_audit_runtime(sandbox))
        app.state.edge_relay_token = "relay-token"
        client = TestClient(app)
        out["no_store_wrong_token"] = client.post(url, content=body, headers=headers("wrong-token")).status_code
        out["no_store"] = client.post(url, content=body, headers=headers("relay-token")).status_code
        with diagnostics_database_for(sandbox.schema) as db:
            app.state.backend_build_revision = "backend-t20"
            app.state.execution_record_store = ExecutionRecordStore(db, RetentionBudget(total_bytes=2**20))
            response = client.post(url, content=body, headers=headers("relay-token"))
            out["status"] = response.status_code
            out["receipt"] = response.json()
            rows = db.read(lambda c: c.execute(
                "SELECT record_id, payload FROM execution_records ORDER BY producer_sequence"
            ).fetchall())
            stored = []
            for record_id, payload in rows:
                payload = payload if isinstance(payload, dict) else json.loads(payload)
                stored.append({"record_id": record_id, "accelerator": payload.get("accelerator")})
            out["stored"] = stored
finally:
    admin.close()
print("T20RESULT " + json.dumps(out))
"#;

fn python() -> PathBuf {
    let value = std::env::var_os("SEEON_TEST_PYTHON")
        .unwrap_or_else(|| panic!("SEEON_TEST_PYTHON is required"));
    assert!(
        !value.is_empty(),
        "SEEON_TEST_PYTHON must be a nonblank path"
    );
    PathBuf::from(value)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
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

/// One TensorRT fp32 call on device 0 that copied both ways.
fn receipt() -> AcceleratorEvidence {
    AcceleratorEvidence::from_delta(
        &metrics(11, 9_000, 400, 70_000),
        &metrics(12, 9_602, 464, 71_234),
        0,
        EngineDigest::new([0x5c; 32]),
        Precision::Fp32,
    )
    .expect("one successful call with both copies")
}

fn accelerated_batch() -> Batch {
    let stream = Stream {
        camera_id: CAMERA.to_owned(),
        worker_boot_id: BOOT.to_owned(),
        source_generation: 1,
        stream_epoch: 1,
    };
    let frame = Frame {
        frame_seq: 42,
        source_pts_ns: Some(1_400_000_000),
    };
    let score = FallScore {
        track_id: 7,
        generation: Some(2),
        fall_transition: 0.61,
        background: 0.25,
        fallen: 0.14,
        evidence: None,
    };
    let built = model_score_record(&stream, frame, WALL_NS, &score, Some(&receipt()))
        .expect("accelerated record is valid");
    let record = Record::new(RecordBody {
        producer_sequence: 3,
        ..built.body().clone()
    })
    .expect("sequenced record is valid");
    let provenance = Provenance {
        worker_build_revision: "worker-t20".to_owned(),
        worker_image_digest: "sha256:t20".to_owned(),
        model_digest: "model-t20".to_owned(),
        calibration_digest: "cal-t20".to_owned(),
        preprocessing_identity: "pose-bbox56/v1".to_owned(),
        config_digest: "cfg-t20".to_owned(),
        policy_identity: "fall.policy:2".to_owned(),
    };
    Batch::new(CAMERA, BOOT, provenance, vec![record], Vec::new()).expect("batch is valid")
}

fn fresh_body_file(body: &[u8]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("t20-records-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("stale body dir is removable");
    }
    std::fs::create_dir_all(&dir).expect("body dir is creatable");
    let path = dir.join("batch.json");
    std::fs::write(&path, body).expect("body file is writable");
    path
}

fn post_through_backend(body: &[u8]) -> Value {
    let path = fresh_body_file(body);
    let output = Command::new(python())
        .arg("-c")
        .arg(SCRIPT)
        .arg(&path)
        .env("PYTHONPATH", repo_root())
        .current_dir(repo_root())
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .expect("the Python interpreter starts");
    std::fs::remove_dir_all(path.parent().expect("body dir")).expect("body dir is removable");
    assert!(output.status.success(), "the backend script completes");
    let stdout = String::from_utf8(output.stdout).expect("stdout is UTF-8");
    let line = stdout
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix(RESULT_PREFIX))
        .expect("the script prints its result line");
    serde_json::from_str(line).expect("result line is JSON")
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON and SEEON_TEST_POSTGRES_DSN"]
fn backend_accepts_rust_accelerator_record_and_stores_it_verbatim() {
    let batch = accelerated_batch();
    let body = batch.encode().expect("batch encodes");
    let sent: Value = serde_json::from_slice(&body).expect("body is JSON");
    let sent_record = &sent["records"][0];
    let sent_accelerator = &sent_record["payload"]["accelerator"];
    assert!(
        sent_accelerator.is_object(),
        "the body carries an accelerator"
    );

    let result = post_through_backend(&body);

    // Auth runs before the store check: a wrong token is refused even while
    // execution records are not installed, and the right token then reaches
    // the "disabled" answer.
    assert_eq!(result["no_store_wrong_token"], 403);
    assert_eq!(result["no_store"], 503);
    assert_eq!(result["status"], 200, "the backend accepts the Rust batch");
    let receipt = Receipt::from_json(&result["receipt"]).expect("receipt parses");
    assert_eq!(receipt.batch_id, batch.batch_id());
    assert_eq!(
        (receipt.accepted, receipt.duplicates, receipt.rejected.len()),
        (1, 0, 0)
    );
    assert_eq!(receipt.storage_state, StorageState::Committed);
    let stored = result["stored"].as_array().expect("stored rows");
    assert_eq!(stored.len(), 1, "exactly the posted record is stored");
    assert_eq!(stored[0]["record_id"], sent_record["record_id"]);
    assert_eq!(&stored[0]["accelerator"], sent_accelerator);
}
