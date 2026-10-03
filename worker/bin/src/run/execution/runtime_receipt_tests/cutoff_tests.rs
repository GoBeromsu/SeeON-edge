//! CPU regressions separating receipt observation from fresh content scheduling.

use std::sync::Arc;
use std::time::Duration;

use seeon_deepstream_native::{MediaCallStatus, MediaResult};

use super::super::super::{advance_recordings, apply_sink, attributed, observed_frame};
use super::assertions::{clip_id, unavailable};
use super::fixture::{FIRST, Fixture, NEXT, WAITING, event_on};
use crate::clips::recorder::State;
use crate::media::Command;
use crate::seam::Clock;

#[test]
fn blocking_camera_tick_cannot_expire_another_cameras_queued_receipt() {
    let mut fixture = Fixture::two();
    let b = fixture.start_on(23, 41, FIRST, 71);
    fixture.clock.at(100);
    let a = fixture.start(41, NEXT, 72);
    fixture.clock.at(149);
    let counters = fixture.session.publications.recorders[1].counters();
    let receipt = fixture.receipt(b, MediaResult::Ok);
    let inbox = fixture.commands.take().unwrap();
    let clock = Arc::clone(&fixture.clock);
    let receipts = fixture.receipts.clone();
    let worker = std::thread::spawn(move || {
        let Command::RecordStop { ticket, reply } =
            inbox.recv_timeout(Duration::from_secs(5)).unwrap()
        else {
            panic!("expected camera A's unit stop");
        };
        assert_eq!(ticket, a);
        clock.at(151);
        receipts.try_send(receipt).unwrap();
        reply
            .send(Ok(MediaCallStatus {
                result: MediaResult::Ok,
                fatal: None,
                warning: None,
                required_bytes: None,
            }))
            .unwrap();
        inbox
    });
    let result = advance_recordings(&mut fixture.session, fixture.clock.as_ref());
    fixture.commands = Some(worker.join().unwrap());
    result.unwrap();
    assert_eq!(fixture.clock.monotonic(), Duration::from_secs(151));
    assert_eq!(fixture.snapshot().0, State::Stopping);
    assert_eq!(
        fixture.session.publications.recorders[1].state(),
        State::Recording
    );
    assert_eq!(
        fixture.session.publications.recorders[1].counters(),
        counters
    );
    assert!(
        !fixture
            .session
            .publications
            .store
            .clip_dir(&clip_id(b))
            .exists()
    );
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        2
    );
    advance_recordings(&mut fixture.session, fixture.clock.as_ref()).unwrap();
    assert_eq!(
        fixture.session.publications.recorders[1].state(),
        State::Idle
    );
    unavailable(&fixture, b, FIRST, "NO_FRAMES");
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        3
    );
    fixture.no_commands();
}

#[test]
fn content_cap_admission_uses_fresh_time_after_slow_receipt_drain() {
    let mut fixture = Fixture::two();
    let a = fixture.start(41, FIRST, 71);
    fixture.start_on(23, 41, NEXT, 72);
    fixture.clock.at(119);
    let mut pending = event_on(23, 42, WAITING);
    let stream = attributed(&fixture.session, &pending.frame).unwrap();
    let mut prepared = fixture
        .session
        .publications
        .prepare(
            1,
            &pending.event,
            &stream,
            observed_frame(&pending.frame).unwrap(),
            None,
        )
        .unwrap();
    pending.staged = Some(
        fixture
            .session
            .publications
            .stage(1, &mut prepared, &mut |record| {
                assert!(fixture.lanes.try_emit(record));
            })
            .unwrap(),
    );
    pending.prepared = Some(prepared);
    let mut held = super::super::sink();
    held.triggered.push(pending);
    // All staging is done; the next wall read is A's receipt publication.
    fixture.clock.jump_on_wall(121);
    fixture
        .receipts
        .try_send(fixture.receipt(a, MediaResult::Ok))
        .unwrap();
    apply_sink(&mut fixture.session, fixture.clock.as_ref(), &mut held).unwrap();
    assert!(held.triggered.is_empty());
    assert_eq!(fixture.clock.monotonic(), Duration::from_secs(121));
    let b = &fixture.session.publications.recorders[1];
    assert_eq!(b.state(), State::Recording);
    assert_eq!(b.pending(), 1);
    assert_eq!(b.counters().extended, 0);
    assert_eq!(b.counters().raced, 1);
    unavailable(&fixture, a, FIRST, "NO_FRAMES");
    assert_eq!(
        fixture.session.publications.queue.entries().unwrap().len(),
        4
    );
    fixture.no_commands();
}
