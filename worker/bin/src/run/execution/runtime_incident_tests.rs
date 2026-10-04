//! Genuine fall admission over the real queue; no native media or crash oracle.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use crate::clips::manifest::Contributor;
use crate::clips::recorder::{MAX_PENDING_ALERTS, RecorderError};
use crate::clips::time::Utc;
use crate::delivery::{DeliveryQueue, sender::event_body};
use crate::policy::incident::IncidentError;
use crate::run::events::EventDeliveryError;
use crate::run::execution::publication::PublicationError;
use crate::seam::Clock;

use super::*;

#[path = "runtime_incident_retention_tests.rs"]
mod retention_tests;

struct WallFailure {
    queue: Arc<DeliveryQueue>,
    after: usize,
    skip: usize,
}

#[derive(Default)]
struct TestClock {
    now: AtomicU64,
    samples: AtomicUsize,
    wall_step: AtomicU64,
    failure: Mutex<Option<WallFailure>>,
}

impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        self.samples.fetch_add(1, Ordering::SeqCst);
        Duration::from_secs(self.now.load(Ordering::SeqCst))
    }
    fn wall(&self) -> SystemTime {
        self.now
            .fetch_add(self.wall_step.load(Ordering::SeqCst), Ordering::SeqCst);
        if let Some(failure) = self.failure.lock().unwrap().as_mut()
            && failure.queue.accepted_count().unwrap() >= failure.after
        {
            if failure.skip == 0 {
                return SystemTime::UNIX_EPOCH - Duration::from_secs(1);
            }
            failure.skip -= 1;
        }
        SystemTime::UNIX_EPOCH
            + Duration::from_secs(1_700_000_000 + self.now.load(Ordering::SeqCst))
    }
    fn pause(&self, limit: Duration) {
        self.now.fetch_add(limit.as_secs(), Ordering::SeqCst);
    }
}

fn genuine(clock: Arc<dyn Clock>) -> LiveSink {
    let mut sink = LiveSink::new(clock);
    let mut stage = stage(19);
    let mut sequence = 0;
    let requests = next_due(&mut stage, 19, &mut sequence, &[9, 7, 2]);
    let snapshots = consume(&mut stage, &requests, 2.0, &mut sink);
    assert_eq!(snapshots.len(), 3);
    assert!(snapshots.iter().all(|snapshot| snapshot.triggered));
    assert_eq!(sink.triggered.len(), 3);
    sink
}

fn deliveries(fixture: &Fixture) -> Vec<serde_json::Value> {
    of_kind(&fixture.drain(19), RecordKind::EventDelivery)
        .into_iter()
        .map(wire)
        .collect()
}

fn unstarted(fixture: &mut Fixture) -> Vec<Contributor> {
    fixture.session.publications.recorders[0]
        .take_unstarted()
        .unwrap()
}

#[test]
fn captured_update_time_and_opaque_sources_reach_uuid_queue_and_recording_without_journal() {
    let clock = Arc::new(TestClock::default());
    clock.now.store(17, Ordering::SeqCst);
    let mut fixture = Fixture::with_clock(64, clock.clone());
    let before = clock.samples.load(Ordering::SeqCst);
    let mut sink = genuine(clock.clone());
    assert_eq!(clock.samples.load(Ordering::SeqCst), before + 1);
    let raw = sink.triggered.clone();
    assert!(
        raw.iter()
            .all(|item| item.admission_time == Duration::from_secs(17))
    );
    assert!(raw.iter().all(|item| item.admitted.is_none()));
    assert!(
        raw.iter()
            .all(|item| !crate::run::event_payload::is_uuid(&item.event.identity))
    );
    clock.wall_step.store(31, Ordering::SeqCst);
    fixture.deliver(&mut sink).unwrap();
    assert!(clock.now.load(Ordering::SeqCst) >= 30);
    let receipts = deliveries(&fixture);
    let queue = fixture.session.publications.queue.clone();
    let entries = queue.entries().unwrap();
    assert_eq!(receipts.len(), 3);
    assert_eq!(entries.len(), 3);
    let contributors = unstarted(&mut fixture);
    assert_eq!(contributors.len(), 3);
    for (source, receipt) in raw.iter().zip(receipts) {
        let uuid = receipt["payload"]["edge_event_id"].as_str().unwrap();
        assert!(crate::run::event_payload::is_uuid(uuid));
        assert_ne!(uuid, source.event.identity);
        let entry = entries
            .iter()
            .find(|entry| entry["edge_event_id"] == uuid)
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&event_body(entry).unwrap()).unwrap();
        assert_eq!(body["evidence"]["identity"], uuid);
        assert_eq!(
            body["evidence"]["person_id"],
            source.event.person_id.unwrap()
        );
        assert!(body.get("audit").is_none());
        assert!(body["evidence"].get("source_identity").is_none());
        assert!(contributors.iter().any(|item| item.event_ref == uuid));
    }
    // Duplicate owned source receipts retain their update time, not publication time.
    sink.triggered = raw;
    fixture.deliver(&mut sink).unwrap();
    assert!(sink.triggered.is_empty());
    assert!(fixture.drain(19).is_empty());
    assert_eq!(fixture.session.publications.recorders[0].pending(), 0);
    assert_eq!(queue.entries().unwrap(), entries);
    assert_eq!(
        DeliveryQueue::open(queue.directory(), true)
            .unwrap()
            .entries()
            .unwrap(),
        entries
    );
    let names: Vec<_> = std::fs::read_dir(queue.directory().parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(names, vec![std::ffi::OsString::from("delivery-queue")]);
}

#[test]
fn incident_regression_is_typed_visible_and_leaves_the_source_unpublished() {
    let clock = Arc::new(TestClock::default());
    clock.now.store(10, Ordering::SeqCst);
    let mut fixture = Fixture::with_clock(64, clock.clone());
    fixture.deliver(&mut genuine(clock.clone())).unwrap();
    let entries = fixture.session.publications.queue.entries().unwrap();
    fixture.drain(19);
    clock.now.store(9, Ordering::SeqCst);
    let mut sink = genuine(clock);
    let raw = sink.triggered.clone();
    let error = fixture.deliver(&mut sink).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("incident monotonic clock regressed")
    );
    assert_eq!(error.exit(), crate::exit::Exit::Runtime);
    assert!(matches!(
        error,
        RuntimeError::Publication(PublicationError::Incident(
            IncidentError::ClockRegression { .. }
        ))
    ));
    assert_eq!(sink.triggered, raw);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        entries
    );
    assert!(deliveries(&fixture).is_empty());
}
