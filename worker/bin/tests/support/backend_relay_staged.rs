//! Frozen event staging and independent SQL/wire assertions shared by both gates.

use super::backend_relay_fixture::{OwnedDir, repo_root};
use super::worker::delivery::DeliveryQueue;
use serde_json::{Value, json};
use std::fs;
use std::sync::Arc;

pub(super) const EVENT_FILE: &str = "event-a5e15ff2-90fd-4764-be74-a7da4f573cc9.json";

pub(super) struct Staged {
    pub root: OwnedDir,
    pub queue: Arc<DeliveryQueue>,
    pub alert: Value,
    pub original: Vec<u8>,
}

impl Staged {
    pub fn new() -> Self {
        let root = OwnedDir::new();
        let fixtures = repo_root().join("tests/fixtures/worker-wire");
        let alert_path = root.0.join("alert.json");
        fs::copy(fixtures.join("r/alert.json"), &alert_path).expect("copy frozen alert oracle");
        let alert = serde_json::from_slice(&fs::read(&alert_path).expect("frozen alert bytes"))
            .expect("frozen alert JSON");
        let queue = Arc::new(
            DeliveryQueue::open(&root.0.join("queue"), true).expect("real durable queue opens"),
        );
        let entry_path = queue.directory().join(EVENT_FILE);
        fs::copy(
            fixtures.join("d/delivery-queue").join(EVENT_FILE),
            &entry_path,
        )
        .expect("copy exact frozen Python queue entry into owned directory");
        fs::File::open(&entry_path)
            .expect("published entry")
            .sync_all()
            .expect("durable entry bytes");
        fs::File::open(queue.directory())
            .expect("queue directory")
            .sync_all()
            .expect("durable entry name");
        let original = fs::read(&entry_path).expect("original durable bytes");
        Self {
            root,
            queue,
            alert,
            original,
        }
    }
}

pub(super) fn assert_committed(snapshot: &Value, alert: &Value) {
    for table in ["incidents", "outbox", "relay_alert_audit"] {
        assert_eq!(
            snapshot[table].as_array().expect("SQL rows").len(),
            1,
            "{table}"
        );
    }
    let incident = &snapshot["incidents"][0];
    for field in [
        "edge_event_id",
        "camera_id",
        "facility_id",
        "event_type",
        "probability",
        "detected_at",
    ] {
        assert_eq!(incident[field], alert[field], "committed incident {field}");
    }
    assert_eq!(
        incident["incident_id"],
        format!("incident:{}", alert["edge_event_id"].as_str().unwrap())
    );
    let outbox = &snapshot["outbox"][0];
    assert_eq!(outbox["edge_event_id"], alert["edge_event_id"]);
    assert_eq!(outbox["backend_camera_id"], alert["camera_id"]);
    assert_eq!(
        outbox["state"], "LOCAL_ONLY",
        "no central client is installed"
    );
    assert_eq!(outbox["attempt_count"], 0, "no Hub delivery attempt");
    let envelope: Value =
        serde_json::from_str(outbox["envelope"].as_str().expect("stored envelope"))
            .expect("actual committed envelope JSON");
    let mut expected_envelope = alert.clone();
    expected_envelope["resident_id"] = Value::Null;
    expected_envelope["evidence"] = Value::Null;
    assert_eq!(
        envelope, expected_envelope,
        "identity, values and frozen audit trace survive admission"
    );
    let audit = &snapshot["relay_alert_audit"][0];
    assert_eq!(audit["action"], "relay.alert");
    assert_eq!(audit["target_id"], alert["edge_event_id"]);
    assert_eq!(audit["actor_id"], "worker-relay");
    assert_eq!(audit["outcome"], "success");
}

pub(super) fn assert_requests(snapshot: &Value, alert: &Value, count: usize) {
    assert_eq!(
        snapshot["proxy_errors"],
        json!([]),
        "proxy must observe a real successful backend return"
    );
    assert_eq!(snapshot["connections"], count);
    let requests = snapshot["requests"]
        .as_array()
        .expect("actual TCP observations");
    assert_eq!(requests.len(), count, "no implicit HTTP retries");
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(
            request["request_line"],
            "POST /api/v1/relay/alerts HTTP/1.1"
        );
        assert_eq!(
            request["request_body"], *alert,
            "original frozen event body"
        );
        assert_eq!(request["backend_status"], 202);
        assert_eq!(
            request["backend_body"],
            json!({"status": "accepted_local", "edge_event_id": alert["edge_event_id"]})
        );
        assert_eq!(
            request["queue_unchanged_before_response"], true,
            "no worker ACK before receiving a reply"
        );
        assert_eq!(request["dropped"], index == 0);
    }
}
