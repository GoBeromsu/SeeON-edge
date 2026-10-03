//! CPU-only real Rust queue -> TCP proxy -> FastAPI relay -> PostgreSQL.
//! The first genuine accepted_local response is observed, then lost on TCP.
//! No Hub, model inference, native/GPU provenance, deployment or cutover is tested.

#[path = "support/backend_relay_fixture.rs"]
mod backend_relay_fixture;

use std::fs;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime};

use backend_relay_fixture::{Backend, OwnedDir, RELAY_TOKEN, repo_root};
use seeon_ml_worker::delivery::DeliveryQueue;
use seeon_ml_worker::delivery::sender::{
    DrainStop, EntryOutcome, SENDER_IDLE_WAIT, SenderState, drain_pass,
};
use seeon_ml_worker::relay::RelayClient;
use seeon_ml_worker::relay::client::ALERT_DELIVERY_TIMEOUT;
use seeon_ml_worker::relay::wire::DeliveryDisposition;
use seeon_ml_worker::seam::Clock;
use seeon_ml_worker::shutdown::ShutdownDeadline;
use serde_json::{Value, json};

const EVENT_FILE: &str = "event-a5e15ff2-90fd-4764-be74-a7da4f573cc9.json";

struct At(Duration);

impl Clock for At {
    fn monotonic(&self) -> Duration {
        self.0
    }
    fn wall(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
    fn pause(&self, _limit: Duration) {}
}

fn assert_committed(snapshot: &Value, alert: &Value) {
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

fn assert_requests(snapshot: &Value, alert: &Value, count: usize) {
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
            json!({
                "status": "accepted_local", "edge_event_id": alert["edge_event_id"],
            })
        );
        assert_eq!(
            request["queue_unchanged_before_response"], true,
            "no worker ACK before receiving a reply"
        );
        assert_eq!(request["dropped"], index == 0);
    }
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON and SEEON_TEST_POSTGRES_DSN"]
fn lost_backend_ack_retries_the_original_durable_event_once() {
    let root = OwnedDir::new();
    let fixtures = repo_root().join("tests/fixtures/worker-wire");
    let alert_path = root.0.join("alert.json");
    fs::copy(fixtures.join("r/alert.json"), &alert_path).expect("copy frozen alert oracle");
    let alert: Value = serde_json::from_slice(&fs::read(&alert_path).expect("frozen alert bytes"))
        .expect("frozen alert JSON");
    let queue = DeliveryQueue::open(&root.0.join("queue"), true).expect("real durable queue opens");
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
    let entry: Value = serde_json::from_slice(&original).expect("frozen queue entry JSON");
    let id = entry["entry_id"]
        .as_str()
        .expect("original entry ID")
        .to_owned();
    assert_eq!(entry["edge_event_id"], alert["edge_event_id"]);
    assert_eq!(
        queue.entries().expect("published entries"),
        vec![entry.clone()]
    );

    let mut backend = Backend::start(&alert_path, &entry_path);
    let empty = backend.snapshot();
    for table in [
        "incidents",
        "outbox",
        "relay_alert_audit",
        "requests",
        "proxy_errors",
    ] {
        assert_eq!(empty[table], json!([]), "fresh sandbox {table}");
    }
    assert_eq!(empty["connections"], 0);
    // RelayClient's existing defaults disable proxying, redirects and implicit retries.
    let client = RelayClient::new(backend.base(), RELAY_TOKEN, ALERT_DELIVERY_TIMEOUT)
        .expect("actual Rust relay client");
    let mut state = SenderState::new();
    let deadline = ShutdownDeadline::new(Duration::from_secs(25)).expect("unrequested cutoff");
    let stop = AtomicBool::new(false);
    let first = drain_pass(
        &queue,
        &client,
        &mut state,
        &At(Duration::ZERO),
        &deadline,
        &stop,
        true,
    );
    assert!(matches!(first.stop, DrainStop::Unacknowledged));
    assert!(
        first.relay_unreachable(),
        "no response bytes reached the worker"
    );
    assert_eq!(first.outcomes.len(), 1);
    assert_eq!(first.outcomes[0].0, id);
    match &first.outcomes[0].1 {
        EntryOutcome::Failed { failure, counted } => {
            assert_eq!(failure.disposition, DeliveryDisposition::Retry);
            assert_eq!(failure.code, "NETWORK");
            assert_eq!(failure.status_code, None);
            assert!(
                !*counted,
                "existing RETRY contract spends no counted-attempt budget"
            );
        }
        outcome => panic!("lost ACK must not acknowledge or retire the entry: {outcome:?}"),
    }
    // One real HTTP attempt, but zero charged attempts for RETRY/NETWORK.
    assert_eq!(state.attempts(&id), 0);
    assert!(state.is_deferred(&id));
    assert_eq!(state.blocked_until(&id), None);
    assert_eq!(
        fs::read(&entry_path).expect("lost response must retain durable event"),
        original
    );
    assert_eq!(queue.entries().expect("retained entry"), vec![entry]);
    assert!(!queue.dead_letter_directory().exists());
    let committed_without_ack = backend.snapshot();
    assert_requests(&committed_without_ack, &alert, 1);
    assert_committed(&committed_without_ack, &alert);

    // Advance only the injected Clock by the product's existing sender idle wait.
    let second = drain_pass(
        &queue,
        &client,
        &mut state,
        &At(SENDER_IDLE_WAIT),
        &deadline,
        &stop,
        true,
    );
    assert_eq!(
        second.outcomes,
        vec![(id.clone(), EntryOutcome::Acknowledged)]
    );
    assert!(matches!(second.stop, DrainStop::Idle));
    assert!(
        !entry_path.exists(),
        "remove only after receiving actual accepted_local reply"
    );
    assert_eq!(
        queue.entries().expect("acknowledged queue"),
        Vec::<Value>::new()
    );
    assert_eq!(queue.accepted_count().expect("queue count"), 0);
    assert_eq!(state.attempts(&id), 0);
    assert!(!state.is_deferred(&id));
    assert_eq!(state.blocked_until(&id), None);
    assert!(!queue.dead_letter_directory().exists());
    let after_retry = backend.snapshot();
    assert_requests(&after_retry, &alert, 2);
    assert_committed(&after_retry, &alert);
    assert_eq!(
        after_retry["requests"][0],
        committed_without_ack["requests"][0]
    );
    assert_eq!(
        after_retry["requests"][1]["request_bytes"],
        after_retry["requests"][0]["request_bytes"]
    );
    for table in ["incidents", "outbox", "relay_alert_audit"] {
        assert_eq!(
            after_retry[table], committed_without_ack[table],
            "duplicate acceptance cannot mutate or append {table}"
        );
    }
    backend.stop();
}
