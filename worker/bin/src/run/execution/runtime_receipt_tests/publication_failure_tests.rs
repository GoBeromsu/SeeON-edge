//! CPU queue/attribution regressions, not SDK completion or crash-durability proof.

use seeon_deepstream_native::MediaResult;

use super::assertions::{clip_id, unavailable};
use super::fixture::{FIRST, Fixture, NEXT, SUFFIX, WAITING, event_on};
use super::{LiveSink, apply_sink};
use crate::clips::publish::PublishError;
use crate::clips::recorder::State;
use crate::run::clip_output::ClipOutputError;
use crate::run::execution::publication::PublicationError;

#[test]
fn failed_publication_still_hands_off_later_camera_receipt() {
    drain_after_failure(false);
}

#[test]
fn rejected_media_path_still_hands_off_later_camera_receipt() {
    drain_after_failure(true);
}

fn drain_after_failure(bad_path: bool) {
    let mut fixture = Fixture::two();
    let first = fixture.start(41, FIRST, 71);
    let next = fixture.start_on(23, 42, NEXT, 72);
    fixture.clock.at(120);
    let mut pending = LiveSink::new(fixture.clock.clone());
    pending.triggered = vec![event_on(19, 43, WAITING), event_on(23, 44, SUFFIX)];
    apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut pending).unwrap();
    assert!(pending.triggered.is_empty());
    for recorder in &fixture.session.publications.recorders {
        assert_eq!(recorder.pending(), 1);
    }
    assert!(
        fixture
            .session
            .publications
            .recorders
            .iter()
            .all(|recorder| !recorder.is_quiesced())
    );
    let retained = fixture.session.publications.queue.entries().unwrap();
    let mut failed = fixture.receipt(first, MediaResult::Ok);
    let first_dir = fixture.session.publications.store.clip_dir(&clip_id(first));
    if bad_path {
        // No media exists. This is a malformed unit receipt, not measured video.
        failed.contains_video = true;
    } else {
        // A real conflicting filesystem entry refuses the first publication.
        std::fs::create_dir_all(&first_dir).unwrap();
        std::fs::write(
            first_dir.join("manifest.json"),
            b"conflicting-owned-test-manifest\n",
        )
        .unwrap();
    }
    fixture.receipts.try_send(failed).unwrap();
    if !bad_path {
        // A distinct later error must not replace the first conflict. Its
        // rejected path does not consume the following valid unit receipt.
        let mut invalid = fixture.receipt(next, MediaResult::Ok);
        invalid.contains_video = true;
        fixture.receipts.try_send(invalid).unwrap();
    }
    fixture
        .receipts
        .try_send(fixture.receipt(next, MediaResult::Ok))
        .unwrap();
    let error = fixture
        .session
        .publications
        .drain_records(fixture.clock.as_ref())
        .unwrap_err();
    if bad_path {
        assert!(matches!(error, PublicationError::MediaPath));
        assert!(!first_dir.exists());
        assert_eq!(
            fixture.session.publications.recorders[0].state(),
            State::Recording
        );
    } else {
        assert!(matches!(
            error,
            PublicationError::Clip(ClipOutputError::Publish(PublishError::Conflict))
        ));
        assert_eq!(
            std::fs::read(first_dir.join("manifest.json")).unwrap(),
            b"conflicting-owned-test-manifest\n"
        );
    }
    assert_eq!(
        fixture.session.publications.recorders[1].state(),
        State::Idle
    );
    assert!(
        fixture
            .session
            .publications
            .recorders
            .iter()
            .all(|recorder| recorder.is_quiesced())
    );
    for recorder in &fixture.session.publications.recorders {
        assert_eq!(recorder.pending(), 1);
    }
    unavailable(&fixture, next, NEXT, "NO_FRAMES");
    let entries = fixture.session.publications.queue.entries().unwrap();
    assert_eq!(entries.len(), 5); // Four accepted events, one unit terminal.
    for accepted in retained {
        assert!(entries.contains(&accepted));
    }
    fixture.no_commands();
    assert!(
        !fixture
            .session
            .publications
            .drain_records(fixture.clock.as_ref())
            .unwrap()
    );
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap(),
        entries
    );
    fixture.no_commands();
}
