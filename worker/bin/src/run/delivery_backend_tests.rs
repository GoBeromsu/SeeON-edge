//! Actual private delivery owner retries with its real clock and unchanged cadence.

#[path = "../../tests/support/backend_relay_fixture.rs"]
mod backend_relay_fixture;
#[path = "../../tests/support/backend_relay_staged.rs"]
mod backend_relay_staged;

use serde_json::json;
use std::fs;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::{Handle, Report, StopReason, spawn};
use crate as worker;
use crate::relay::RelayClient;
use crate::relay::client::ALERT_DELIVERY_TIMEOUT;
use crate::seam::SystemClock;
use crate::shutdown::ShutdownDeadline;
use backend_relay_fixture::{Backend, RELAY_TOKEN};
use backend_relay_staged::{EVENT_FILE, Staged, assert_committed, assert_requests};

struct Owner(Handle);

impl Owner {
    fn stop_and_join(&mut self) -> Report {
        self.0.request_stop();
        let deadline = Instant::now() + ALERT_DELIVERY_TIMEOUT + Duration::from_secs(4);
        while !self.0.is_finished() {
            assert!(
                Instant::now() < deadline,
                "delivery owner did not stop within its bound"
            );
            thread::sleep(Duration::from_millis(5));
        }
        let report = self.0.join().expect("actual delivery owner join");
        assert_eq!(self.0.join(), Ok(report), "joined outcome remains cached");
        report
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.0.request_stop();
        let deadline = Instant::now() + ALERT_DELIVERY_TIMEOUT + Duration::from_secs(4);
        while !self.0.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if let Err(error) = self.0.join() {
            eprintln!("delivery owner cleanup did not establish successful join: {error}");
        }
    }
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON and SEEON_TEST_POSTGRES_DSN"]
fn autonomous_owner_retries_lost_ack_with_real_clock_and_joins() {
    let staged = Staged::new();
    let entry_path = staged.queue.directory().join(EVENT_FILE);
    assert_eq!(
        fs::read(&entry_path).expect("original queue entry"),
        staged.original
    );
    let mut backend = Backend::start(&staged.root.0.join("alert.json"), &entry_path);
    let initial = backend.snapshot();
    assert_eq!(initial["connections"], 0);
    for table in ["incidents", "outbox", "relay_alert_audit"] {
        assert_eq!(initial[table], json!([]));
    }
    let client = RelayClient::new(backend.base(), RELAY_TOKEN, ALERT_DELIVERY_TIMEOUT)
        .expect("actual relay client");
    let mut owner = Owner(
        spawn(
            Arc::clone(&staged.queue),
            client,
            true,
            Arc::new(SystemClock::new()),
            Arc::new(ShutdownDeadline::new(Duration::from_secs(25)).expect("unrequested cutoff")),
        )
        .expect("production delivery owner spawns"),
    );
    // No explicit drain, wake, or clock advancement drives either HTTP attempt.
    let deadline = Instant::now() + Duration::from_secs(30);
    while entry_path.exists() {
        assert!(
            Instant::now() < deadline,
            "autonomous delivery did not acknowledge"
        );
        assert!(
            !owner.0.is_finished(),
            "delivery owner exited before acknowledgment"
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        owner.stop_and_join(),
        Report {
            stop: StopReason::Requested
        }
    );
    assert!(staged.queue.entries().expect("queue scan").is_empty());
    assert_eq!(staged.queue.accepted_count().expect("accepted count"), 0);
    assert!(!staged.queue.dead_letter_directory().exists());
    let observed = backend.snapshot();
    assert_requests(&observed, &staged.alert, 2);
    assert_committed(&observed, &staged.alert);
    assert_eq!(
        observed["requests"][0]["request_bytes"],
        observed["requests"][1]["request_bytes"]
    );
    drop(owner);
    backend.stop();
}
