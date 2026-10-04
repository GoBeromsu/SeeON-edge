//! Retained incident ownership across preparation, staging and recorder failures.

use super::*;

#[test]
fn preparation_failure_retains_uuid_and_retires_only_the_successful_prefix() {
    failed_after_prefix(false);
}

#[test]
fn post_acceptance_stage_failure_reuses_uuid_even_after_queue_ack() {
    failed_after_prefix(true);
}

fn failed_after_prefix(stage_failure: bool) {
    let clock = Arc::new(TestClock::default());
    let mut fixture = Fixture::with_clock(64, clock.clone());
    let mut sink = genuine(clock.clone());
    let sources = sink.triggered.clone();
    let queue = fixture.session.publications.queue.clone();
    // Prepare: permit the prefix receipt, then fail the next capture. Stage:
    // permit the second preparation, then fail its post-acceptance receipt.
    *clock.failure.lock().unwrap() = Some(WallFailure {
        queue: queue.clone(),
        after: if stage_failure { 2 } else { 1 },
        skip: usize::from(!stage_failure),
    });
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Publication(PublicationError::Event(
            EventDeliveryError::WallTime
        )))
    ));
    assert_eq!(sink.triggered.len(), 2);
    assert_eq!(sink.triggered[0].event, sources[1].event);
    assert_eq!(sink.triggered[1], sources[2]);
    let admitted = sink.triggered[0].admitted.clone().unwrap();
    let mut expected = sources[1].event.clone();
    expected.identity = admitted.event.identity.clone();
    assert_eq!(admitted.event, expected);
    assert_eq!(admitted.audit.source_identity, sources[1].event.identity);
    assert_eq!(admitted.audit.edge_event_id, admitted.event.identity);
    assert_eq!(sink.triggered[0].prepared.is_some(), stage_failure);
    assert!(sink.triggered[0].staged.is_none());
    let retained = sink.triggered.clone();
    assert_eq!(deliveries(&fixture).len(), 1);
    assert_eq!(
        queue.entries().unwrap().len(),
        if stage_failure { 2 } else { 1 }
    );
    assert!(fixture.deliver(&mut sink).is_err());
    assert_eq!(sink.triggered, retained);
    assert!(fixture.drain(19).is_empty());
    if stage_failure {
        assert!(
            queue
                .acknowledge_backend(&format!("event-{}", admitted.event.identity), 204)
                .unwrap()
        );
    }
    *clock.failure.lock().unwrap() = None;
    clock.now.store(120, Ordering::SeqCst);
    fixture.deliver(&mut sink).unwrap();
    assert!(sink.triggered.is_empty());
    let resumed = deliveries(&fixture);
    assert_eq!(resumed.len(), 2);
    assert_eq!(
        resumed[0]["payload"]["edge_event_id"],
        admitted.event.identity
    );
    assert_eq!(
        queue.entries().unwrap().len(),
        if stage_failure { 2 } else { 3 }
    );
    assert_eq!(
        queue
            .entries()
            .unwrap()
            .iter()
            .any(|entry| entry["edge_event_id"] == admitted.event.identity),
        !stage_failure
    );
    let contributors = unstarted(&mut fixture);
    assert_eq!(contributors.len(), 3);
    assert!(
        contributors
            .iter()
            .any(|item| item.event_ref == admitted.event.identity)
    );
}

#[test]
fn recorder_failure_retains_original_uuid_and_staged_prefix_without_restage() {
    let clock = Arc::new(TestClock::default());
    let mut fixture = Fixture::with_clock(64, clock.clone());
    let detected = Utc::parse("2026-01-01T00:00:00Z").unwrap();
    for index in 0..MAX_PENDING_ALERTS - 1 {
        fixture.session.publications.recorders[0]
            .admit(&format!("prior-{index}"), detected)
            .unwrap();
    }
    let mut sink = genuine(clock);
    let sources = sink.triggered.clone();
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Publication(PublicationError::Recorder(
            RecorderError::PendingFull
        )))
    ));
    assert_eq!(sink.triggered.len(), 2);
    assert_eq!(sink.triggered[0].event, sources[1].event);
    assert_eq!(sink.triggered[1], sources[2]);
    let admitted = sink.triggered[0].admitted.clone().unwrap();
    let staged = sink.triggered[0].staged.clone().unwrap();
    assert_eq!(staged.event_ref, admitted.event.identity);
    assert!(staged.admission.accepted);
    let retained = sink.triggered.clone();
    assert_eq!(deliveries(&fixture).len(), 2);
    let queue = fixture.session.publications.queue.clone();
    assert!(
        queue
            .acknowledge_backend(&format!("event-{}", admitted.event.identity), 204)
            .unwrap()
    );
    assert!(matches!(
        fixture.deliver(&mut sink),
        Err(RuntimeError::Publication(PublicationError::Recorder(
            RecorderError::PendingFull
        )))
    ));
    assert_eq!(sink.triggered, retained);
    assert!(fixture.drain(19).is_empty());
    assert_eq!(unstarted(&mut fixture).len(), MAX_PENDING_ALERTS);
    fixture.deliver(&mut sink).unwrap();
    assert!(sink.triggered.is_empty());
    assert_eq!(deliveries(&fixture).len(), 1);
    let entries = queue.entries().unwrap();
    assert_eq!(entries.len(), 2);
    assert!(
        !entries
            .iter()
            .any(|entry| entry["edge_event_id"] == admitted.event.identity)
    );
    let contributors = unstarted(&mut fixture);
    assert_eq!(contributors.len(), 2);
    assert!(
        contributors
            .iter()
            .any(|item| item.event_ref == admitted.event.identity)
    );
}
