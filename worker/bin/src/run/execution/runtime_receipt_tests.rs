//! CPU composition regressions with unit-constructed replies and no-media receipts.
//! These are not SDK timing, encoded-media, or production incident acceptance evidence.

#[path = "runtime_receipt_tests/assertions.rs"]
mod assertions;
#[path = "runtime_receipt_tests/cutoff_tests.rs"]
mod cutoff_tests;
#[path = "runtime_receipt_tests/fixture.rs"]
mod fixture;
#[path = "runtime_receipt_tests/publication_failure_tests.rs"]
mod publication_failure_tests;
#[path = "runtime_receipt_tests/shutdown_tests.rs"]
mod shutdown_tests;

use seeon_deepstream_native::{MediaBinding, MediaResult, RecordTicket};

use super::super::{LiveSink, PendingEvent, advance_recordings, apply_sink};
use crate::clips::recorder::State;
use assertions::{clip_id, overdue, unavailable};
use fixture::{FIRST, Fixture, NEXT, SUFFIX, WAITING, event};

#[test]
fn queued_matching_receipt_at_150s_is_drained_before_watchdog() {
    let mut fixture = Fixture::new();
    let admitted = fixture.start(41, FIRST, 71);
    fixture.delivery(41, FIRST);
    let receipt = fixture.receipt(admitted, MediaResult::Ok);
    assert_eq!(receipt.ticket.binding, admitted.binding);
    assert_eq!(
        (receipt.ticket.source_id, receipt.ticket.request_id),
        (19, 71)
    );
    assert_eq!((admitted.session_id, admitted.session_valid), (u32::MAX, 0));
    fixture.clock.at(150);
    fixture.receipts.try_send(receipt).unwrap();
    advance_recordings(&mut fixture.session, fixture.clock.as_ref()).unwrap();
    assert_eq!(fixture.snapshot().0, State::Idle);
    assert_eq!(fixture.snapshot().2, 0);
    unavailable(&fixture, admitted, FIRST, "NO_FRAMES");
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        2
    );
    fixture.clock.at(151);
    advance_recordings(&mut fixture.session, fixture.clock.as_ref()).unwrap();
    fixture.no_commands();
}

#[test]
fn missing_receipt_at_150s_propagates_original_ticket_and_runtime_exit() {
    let mut fixture = Fixture::new();
    let admitted = fixture.start(41, FIRST, 71);
    fixture.clock.at(149);
    advance_recordings(&mut fixture.session, fixture.clock.as_ref()).unwrap();
    let before = fixture.snapshot();
    let queued = fixture.session.publications.queue.entries().unwrap();
    fixture.clock.at(150);
    overdue(
        advance_recordings(&mut fixture.session, fixture.clock.as_ref()),
        admitted,
    );
    assert_eq!(fixture.snapshot(), before);
    assert_eq!(fixture.snapshot().0, State::Recording);
    assert!(
        !fixture
            .session
            .publications
            .store
            .clip_dir(&clip_id(admitted))
            .exists()
    );
    let observed = fixture.receipt(admitted, MediaResult::Ok).ticket;
    for wrong in [
        RecordTicket {
            request_id: 999,
            ..observed
        },
        RecordTicket {
            source_id: 20,
            ..observed
        },
        RecordTicket {
            binding: MediaBinding {
                generation: observed.binding.generation + 1,
                ..observed.binding
            },
            ..observed
        },
        RecordTicket {
            session_valid: 2,
            ..observed
        },
    ] {
        let mut receipt = fixture.receipt(admitted, MediaResult::Ok);
        receipt.ticket = wrong;
        fixture.receipts.try_send(receipt).unwrap();
        overdue(
            advance_recordings(&mut fixture.session, fixture.clock.as_ref()),
            admitted,
        );
        assert_eq!(fixture.snapshot(), before);
        assert_eq!(
            fixture.session.publications.queue.entries().unwrap(),
            queued
        );
    }
    fixture.no_commands();
}

#[test]
fn apply_sink_at_150s_drains_completion_before_new_recording_admission() {
    let mut fixture = Fixture::new();
    let first = fixture.start(41, FIRST, 71);
    fixture.delivery(41, FIRST);
    fixture.clock.at(150);
    fixture
        .receipts
        .try_send(fixture.receipt(first, MediaResult::Ok))
        .unwrap();
    let mut held = super::sink();
    held.triggered.push(event(42, NEXT));
    let next = fixture.apply_with_start(&mut held, 72);
    assert!(held.triggered.is_empty());
    fixture.delivery(42, NEXT);
    unavailable(&fixture, first, FIRST, "NO_FRAMES");
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        3
    );
    assert_eq!(fixture.snapshot().0, State::Recording);
    assert_eq!(fixture.snapshot().2, 0);
    assert_ne!(next.request_id, first.request_id);
    advance_recordings(&mut fixture.session, fixture.clock.as_ref()).unwrap();
    fixture.clock.at(299);
    advance_recordings(&mut fixture.session, fixture.clock.as_ref()).unwrap();
    fixture.clock.at(300);
    overdue(
        advance_recordings(&mut fixture.session, fixture.clock.as_ref()),
        next,
    );
    fixture.no_commands();
}

#[test]
fn overdue_admission_retains_staged_event_and_suffix_without_restage_or_restart() {
    let mut fixture = Fixture::new();
    let admitted = fixture.start(41, FIRST, 71);
    fixture.delivery(41, FIRST);
    fixture.clock.at(120);
    let mut waiting = super::sink();
    waiting.triggered.push(event(42, WAITING));
    apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut waiting).unwrap();
    assert!(waiting.triggered.is_empty());
    fixture.delivery(42, WAITING);
    fixture.clock.at(150);
    let mut held = super::sink();
    held.triggered = vec![event(43, NEXT), event(44, SUFFIX)];
    let original = held.triggered.clone();
    overdue(
        apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut held),
        admitted,
    );
    assert_eq!(held.triggered.len(), 2);
    assert_eq!(held.triggered[0].frame, original[0].frame);
    assert_eq!(held.triggered[0].event, original[0].event);
    assert_eq!(held.triggered[1], original[1]);
    assert!(held.triggered[0].prepared.is_some());
    let staged = held.triggered[0].staged.clone().unwrap();
    assert_eq!(staged.event_ref, NEXT);
    assert!(staged.admission.accepted);
    assert!(!staged.admission.already_admitted);
    let retained = held.triggered.clone();
    let before = fixture.snapshot();
    assert_eq!(before.0, State::Recording);
    assert_eq!(before.2, 1);
    let entries = fixture.session.publications.queue.entries().unwrap();
    assert_eq!(entries.len(), 3);
    fixture.delivery(43, NEXT);
    // Unit queue acknowledgement makes any accidental re-stage observable.
    let accepted = entries
        .iter()
        .find(|entry| entry["edge_event_id"] == NEXT)
        .unwrap();
    assert!(
        fixture
            .session
            .publications
            .queue
            .acknowledge(accepted["entry_id"].as_str().unwrap(),)
            .unwrap()
    );
    let queued = fixture.session.publications.queue.entries().unwrap();
    fixture.clock.at(151);
    overdue(
        apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut held),
        admitted,
    );
    assert_eq!(held.triggered, retained);
    assert_eq!(fixture.snapshot(), before);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        queued
    );
    assert!(!fixture.lanes.has_work());
    fixture.no_commands();
    fixture
        .receipts
        .try_send(fixture.receipt(admitted, MediaResult::Stale))
        .unwrap();
    overdue(
        apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut held),
        admitted,
    );
    assert_eq!(held.triggered, retained);
    assert_eq!(fixture.snapshot().0, State::Idle);
    assert_eq!(fixture.snapshot().2, 1);
    assert_eq!(fixture.snapshot().3, before.3);
    unavailable(&fixture, admitted, FIRST, "ENCODER_FAILED");
    let published = fixture.session.publications.queue.entries().unwrap();
    assert_eq!(published.len(), 3);
    assert!(published.iter().all(|entry| entry["edge_event_id"] != NEXT));
    fixture
        .receipts
        .try_send(fixture.receipt(admitted, MediaResult::Stale))
        .unwrap();
    overdue(
        advance_recordings(&mut fixture.session, fixture.clock.as_ref()),
        admitted,
    );
    overdue(
        apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut held),
        admitted,
    );
    assert_eq!(held.triggered, retained);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        published
    );
    assert!(!fixture.lanes.has_work());
    fixture.no_commands();
}
