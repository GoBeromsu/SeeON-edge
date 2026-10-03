//! CPU-only real Rust queue -> TCP proxy -> FastAPI relay -> PostgreSQL.
//! The first genuine accepted_local response is observed, then lost on TCP.
//! No Hub, model inference, native/GPU provenance, deployment or cutover is tested.

#[path = "support/backend_relay_fixture.rs"]
mod backend_relay_fixture;
#[path = "support/backend_relay_staged.rs"]
mod backend_relay_staged;

use std::fs;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime};

use backend_relay_fixture::{Backend, RELAY_TOKEN};
use backend_relay_staged::{EVENT_FILE, Staged, assert_committed, assert_requests};
use seeon_ml_worker as worker;
use seeon_ml_worker::delivery::sender::{
    DrainStop, EntryOutcome, SENDER_IDLE_WAIT, SenderState, drain_pass,
};
use seeon_ml_worker::relay::RelayClient;
use seeon_ml_worker::relay::client::ALERT_DELIVERY_TIMEOUT;
use seeon_ml_worker::relay::wire::DeliveryDisposition;
use seeon_ml_worker::seam::Clock;
use seeon_ml_worker::shutdown::ShutdownDeadline;
use serde_json::{Value, json};

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

#[test]
#[ignore = "requires SEEON_TEST_PYTHON and SEEON_TEST_POSTGRES_DSN"]
fn lost_backend_ack_retries_the_original_durable_event_once() {
    let Staged {
        root,
        queue,
        alert,
        original,
    } = Staged::new();
    let alert_path = root.0.join("alert.json");
    let entry_path = queue.directory().join(EVENT_FILE);
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
