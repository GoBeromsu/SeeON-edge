//! CPU ownership checks; constructed receipts are not native media evidence.

use seeon_deepstream_native::MediaResult;

use super::assertions::{clip_id, unavailable};
use super::fixture::{FIRST, Fixture, NEXT, event};
use super::{LiveSink, apply_sink};
use crate::clips::recorder::State;
use crate::run::execution::lifecycle::drain_policy;

#[test]
fn shutdown_drain_rejects_late_admission_without_consuming_or_fabricating_a_clip() {
    let mut fixture = Fixture::new();
    fixture.session.request_media_stop();
    let mut sink = LiveSink::new(fixture.clock.clone());
    assert!(drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut sink
    ));
    sink.triggered.push(event(41, FIRST));
    assert!(!drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut sink
    ));
    assert!(sink.triggered.is_empty());
    assert_eq!(fixture.snapshot().0, State::Idle);
    assert_eq!(fixture.snapshot().2, 1);
    fixture.delivery(41, FIRST);
    let queued = fixture.session.publications.queue.entries().unwrap();
    assert_eq!(queued.len(), 1);
    let retained = fixture.snapshot();
    assert!(!drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut sink
    ));
    assert_eq!(fixture.snapshot(), retained);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        queued
    );
    fixture.no_commands();
}

#[test]
fn shutdown_drain_rejects_active_receipt_ownership_even_without_pending_alerts() {
    let mut fixture = Fixture::new();
    fixture.start(41, FIRST, 71);
    fixture.session.request_media_stop();
    let retained = fixture.snapshot();
    assert_eq!(retained.0, State::Recording);
    assert_eq!(retained.2, 0);
    let queued = fixture.session.publications.queue.entries().unwrap();
    assert!(!drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut LiveSink::new(fixture.clock.clone()),
    ));
    assert_eq!(fixture.snapshot(), retained);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        queued
    );
    fixture.no_commands();
}

#[test]
fn shutdown_drain_preserves_cap_race_pending_after_prior_receipt_publication() {
    let mut fixture = Fixture::new();
    let admitted = fixture.start(41, FIRST, 71);
    fixture.clock.at(120);
    let mut sink = LiveSink::new(fixture.clock.clone());
    sink.triggered.push(event(42, NEXT));
    apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut sink).unwrap();
    assert!(sink.triggered.is_empty());
    assert_eq!(fixture.snapshot().2, 1);
    fixture.session.request_media_stop();
    fixture
        .receipts
        .try_send(fixture.receipt(admitted, MediaResult::Ok))
        .unwrap();
    assert!(!drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut sink
    ));
    assert_eq!(fixture.snapshot().0, State::Idle);
    assert_eq!(fixture.snapshot().2, 1);
    unavailable(&fixture, admitted, FIRST, "NO_FRAMES");
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        3
    );
    fixture.no_commands();
}

#[test]
fn shutdown_drain_retains_failed_save_attribution_in_finalizing() {
    let mut fixture = Fixture::new();
    let admitted = fixture.start(41, FIRST, 71);
    fixture.session.request_media_stop();
    let id = clip_id(admitted);
    let conflict = fixture.session.publications.store.clip_dir(&id);
    std::fs::create_dir_all(&conflict).unwrap();
    std::fs::write(
        conflict.join("manifest.json"),
        b"conflicting-owned-test-manifest\n",
    )
    .unwrap();
    let queued = fixture.session.publications.queue.entries().unwrap();
    assert_eq!(queued.len(), 1, "the admitted event remains queued");
    let mut receipt = fixture.receipt(admitted, MediaResult::Ok);
    receipt.duration_ms = 0;
    fixture.receipts.try_send(receipt).unwrap();
    let mut sink = LiveSink::new(fixture.clock.clone());
    assert!(!drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut sink
    ));
    assert_eq!(fixture.snapshot().0, State::Finalizing);
    assert_eq!(fixture.snapshot().2, 0);
    assert_eq!(
        std::fs::read(conflict.join("manifest.json")).unwrap(),
        b"conflicting-owned-test-manifest\n"
    );
    let sidecars = fixture
        .session
        .publications
        .sidecars
        .pending("camera-19")
        .unwrap();
    assert!(sidecars.malformed.is_empty());
    assert_eq!(sidecars.recoveries.len(), 1);
    let retained = &sidecars.recoveries[0];
    assert_eq!(retained.sealed.clip_id(), id.as_str());
    assert!(matches!(
        &retained.sealed,
        crate::clips::sealed::SealedObservation::Unavailable(unavailable)
            if unavailable.duration_ms == 0 && !unavailable.contains_video
    ));
    assert_eq!(retained.sealed.contributors().len(), 1);
    assert_eq!(retained.sealed.contributors()[0].event_ref, FIRST);
    assert_eq!(retained.events.get(FIRST).unwrap().identity, FIRST);
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        queued
    );
    assert!(!drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut sink
    ));
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        queued
    );
    fixture.no_commands();
}

#[test]
fn shutdown_drain_accepts_completed_recording_without_requiring_empty_delivery_queue() {
    let mut fixture = Fixture::new();
    let admitted = fixture.start(41, FIRST, 71);
    fixture.session.request_media_stop();
    fixture
        .receipts
        .try_send(fixture.receipt(admitted, MediaResult::Ok))
        .unwrap();
    assert!(drain_policy(
        &mut fixture.session,
        fixture.clock.as_ref(),
        &mut LiveSink::new(fixture.clock.clone()),
    ));
    assert_eq!(fixture.snapshot().0, State::Idle);
    assert_eq!(fixture.snapshot().2, 0);
    unavailable(&fixture, admitted, FIRST, "NO_FRAMES");
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        2
    );
    fixture.no_commands();
}
