//! Unit coverage of recorder admission, receipt bounds, and quiescence.
//! The counted RecordPlane and manual clock are not real SDK evidence.
//!
//! `quiesce` only closes the start/extend door. Native stop, finalize, and
//! receipt delivery stay with the media owner. Never-started alerts stay
//! pending so the parent can publish a real NO_FRAMES entry; this component
//! does not invent tickets, media, duration, codec, or terminal publications.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use seeon_deepstream_native::{MediaBinding, MediaResult, RecordTicket};
use seeon_ml_worker::clips::manifest::Contributor;
use seeon_ml_worker::clips::publish::PublishError;
use seeon_ml_worker::clips::recorder::{
    Admit, Boundary, CAP_SECONDS, EXTENSION_SECONDS, MAX_PENDING_ALERTS,
    NATIVE_COMPLETION_GRACE_SECONDS, PlaneRefusal, RecordPlane, Recorder, RecorderError, State,
};
use seeon_ml_worker::clips::reserve::SaveOutcome;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::msg::RecordReceipt;
use seeon_ml_worker::seam::Clock;

const SOURCE_ID: u32 = 4;
const BINDING: MediaBinding = MediaBinding {
    token: 1,
    generation: 1,
    epoch: 1,
};

struct ManualClock {
    now: Mutex<Duration>,
}

impl ManualClock {
    fn new(now: Duration) -> Self {
        Self {
            now: Mutex::new(now),
        }
    }

    fn advance(&self, by: Duration) {
        *self.now.lock().expect("clock") += by;
    }
}

impl Clock for ManualClock {
    fn monotonic(&self) -> Duration {
        *self.now.lock().expect("clock")
    }

    fn wall(&self) -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH + self.monotonic()
    }

    fn pause(&self, limit: Duration) {
        self.advance(limit);
    }
}

struct PlaneCounts {
    starts: Cell<u32>,
    stops: Cell<u32>,
    next_request: Cell<u64>,
    session_id: Cell<u32>,
    session_valid: Cell<u32>,
    binding: Cell<MediaBinding>,
    refuse: Cell<bool>,
    start_delay: Cell<Duration>,
    last_started: Cell<Option<RecordTicket>>,
    refuse_stop: Cell<bool>,
}

/// Counts every start the recorder asks for. Refuses while `refuse` is set so
/// pending can be staged without a recording.
struct CountingPlane {
    counts: Rc<PlaneCounts>,
    clock: Arc<ManualClock>,
}

impl CountingPlane {
    fn open(counts: Rc<PlaneCounts>, clock: Arc<ManualClock>) -> Self {
        Self { counts, clock }
    }
}

impl RecordPlane for CountingPlane {
    fn start(
        &mut self,
        lookback_seconds: u32,
        forward_seconds: u32,
    ) -> Result<RecordTicket, PlaneRefusal> {
        assert_eq!(lookback_seconds, 15, "native lookback stays unchanged");
        assert_eq!(forward_seconds, 120, "native forward stays unchanged");
        self.counts.starts.set(self.counts.starts.get() + 1);
        self.clock.advance(self.counts.start_delay.get());
        if self.counts.refuse.get() {
            return Err(PlaneRefusal::Busy);
        }
        let request_id = self.counts.next_request.get();
        self.counts.next_request.set(request_id + 1);
        let ticket = RecordTicket {
            binding: self.counts.binding.get(),
            request_id,
            source_id: SOURCE_ID,
            session_id: self.counts.session_id.get(),
            session_valid: self.counts.session_valid.get(),
            coalesced: 0,
        };
        self.counts.last_started.set(Some(ticket));
        Ok(ticket)
    }

    fn stop(&mut self, _ticket: &RecordTicket) -> Result<(), PlaneRefusal> {
        self.counts.stops.set(self.counts.stops.get() + 1);
        if self.counts.refuse_stop.get() {
            Err(PlaneRefusal::Busy)
        } else {
            Ok(())
        }
    }
}

struct Harness {
    recorder: Recorder<CountingPlane>,
    clock: Arc<ManualClock>,
    counts: Rc<PlaneCounts>,
}

impl Harness {
    fn new() -> Self {
        let clock = Arc::new(ManualClock::new(Duration::from_secs(1_000)));
        let counts = Rc::new(PlaneCounts {
            starts: Cell::new(0),
            stops: Cell::new(0),
            next_request: Cell::new(1),
            session_id: Cell::new(0),
            session_valid: Cell::new(0),
            binding: Cell::new(BINDING),
            refuse: Cell::new(false),
            start_delay: Cell::new(Duration::ZERO),
            last_started: Cell::new(None),
            refuse_stop: Cell::new(false),
        });
        Self {
            recorder: Recorder::new(
                SOURCE_ID,
                CountingPlane::open(Rc::clone(&counts), Arc::clone(&clock)),
                Arc::clone(&clock) as Arc<dyn Clock>,
            ),
            clock,
            counts,
        }
    }

    fn refusing() -> Self {
        let clock = Arc::new(ManualClock::new(Duration::from_secs(1_000)));
        let counts = Rc::new(PlaneCounts {
            starts: Cell::new(0),
            stops: Cell::new(0),
            next_request: Cell::new(1),
            session_id: Cell::new(0),
            session_valid: Cell::new(0),
            binding: Cell::new(BINDING),
            refuse: Cell::new(true),
            start_delay: Cell::new(Duration::ZERO),
            last_started: Cell::new(None),
            refuse_stop: Cell::new(false),
        });
        Self {
            recorder: Recorder::new(
                SOURCE_ID,
                CountingPlane::open(Rc::clone(&counts), Arc::clone(&clock)),
                Arc::clone(&clock) as Arc<dyn Clock>,
            ),
            clock,
            counts,
        }
    }
}

fn at(text: &str) -> Utc {
    Utc::parse(text).expect("RFC 3339 timestamp")
}

fn receipt(ticket: RecordTicket) -> RecordReceipt {
    RecordReceipt {
        ticket: RecordTicket {
            session_id: 7,
            session_valid: 1,
            ..ticket
        },
        result: MediaResult::Ok,
        error: 0,
        duration_ms: 30_000,
        width: 1280,
        height: 720,
        contains_video: true,
        contains_audio: false,
        directory: "/tmp/sealed".into(),
        filename: "clip.mp4".into(),
    }
}

fn unchanged(recorder: &Recorder<CountingPlane>) -> (State, usize, Boundary, u64) {
    (
        recorder.state(),
        recorder.pending(),
        recorder.boundary(),
        recorder.counters().sequence,
    )
}

fn started(recorder: &mut Recorder<CountingPlane>, event_ref: &str, when: &str) -> RecordTicket {
    match recorder.admit(event_ref, at(when)) {
        Ok(Admit::Started(ticket)) => ticket,
        other => panic!("expected a started recording, got {other:?}"),
    }
}

fn finalize_failed_without_publication(
    _: &seeon_ml_worker::clips::recorder::ClipSealed,
) -> Result<SaveOutcome, PublishError> {
    // The callback reports finalization failure without claiming a publication.
    Ok(SaveOutcome::FinalizeFailed(None))
}

fn assert_overdue<T: std::fmt::Debug>(outcome: Result<T, RecorderError>, admitted: RecordTicket) {
    match outcome {
        Err(RecorderError::ReceiptOverdue { ticket }) => assert_eq!(ticket, admitted),
        other => panic!("expected receipt failure for {admitted:?}, got {other:?}"),
    }
}

#[test]
fn normal_mode_start_extend_stop_and_errors_are_unchanged() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let first = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(recorder.state(), State::Recording);
    assert_eq!(recorder.pending(), 0);
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000e0b2",
            at("2026-08-20T17:21:00Z")
        ),
        Ok(Admit::Extended)
    ));
    assert_eq!(recorder.counters().extended, 1);
    assert_eq!(recorder.counters().sequence, 2);
    assert_eq!(counts.starts.get(), 1);
    assert!(matches!(
        recorder.admit("  ", at("2026-08-20T17:21:01Z")),
        Err(RecorderError::BlankEventRef)
    ));

    clock.advance(Duration::from_secs(u64::from(EXTENSION_SECONDS)));
    assert_eq!(recorder.tick().expect("early stop"), State::Stopping);
    assert_eq!(counts.stops.get(), 1);
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000e0c3",
            at("2026-08-20T17:21:50Z")
        ),
        Ok(Admit::Queued)
    ));
    assert_eq!(recorder.boundary(), Boundary::ExtensionRaced);
    assert_eq!(recorder.pending(), 1);
    assert_eq!(recorder.counters().raced, 1);

    let mut saved = 0_u32;
    let mut seen = Vec::new();
    let outcome = recorder
        .on_receipt(&receipt(first), |sealed| {
            saved += 1;
            seen = sealed.contributors.clone();
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("save");
    assert!(matches!(outcome, SaveOutcome::FinalizeFailed(None)));
    assert_eq!(saved, 1);
    assert_eq!(seen.len(), 2);
    assert_eq!(recorder.state(), State::Recording);
    assert_eq!(recorder.pending(), 0);
    assert_eq!(counts.starts.get(), 2, "queued alert starts after seal");

    let wrong = RecordTicket {
        source_id: SOURCE_ID + 1,
        ..first
    };
    assert!(matches!(
        recorder.on_receipt(&receipt(wrong), finalize_failed_without_publication),
        Err(RecorderError::WrongSource {
            expected: SOURCE_ID,
            received
        }) if received == SOURCE_ID + 1
    ));
    assert!(matches!(
        recorder.take_unstarted(),
        Err(RecorderError::NotQuiesced)
    ));
}

#[test]
fn refused_start_stays_pending_and_take_unstarted_does_not_steal_it() {
    let Harness {
        mut recorder,
        counts,
        ..
    } = Harness::refusing();
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000e0a1",
            at("2026-08-20T17:20:58Z")
        ),
        Ok(Admit::Refused(PlaneRefusal::Busy))
    ));
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(recorder.state(), State::Idle);
    assert_eq!(recorder.pending(), 1);
    assert_eq!(recorder.counters().refused, 1);
    assert!(matches!(
        recorder.take_unstarted(),
        Err(RecorderError::NotQuiesced)
    ));
    assert_eq!(recorder.pending(), 1);
    assert!(!recorder.is_quiesced());
    assert_eq!(recorder.tick().expect("retry refused start"), State::Idle);
    assert_eq!(counts.starts.get(), 2, "idle tick retries a refused start");
    assert_eq!(recorder.pending(), 1);
}

#[test]
fn quiesce_blocks_admit_tick_and_receipt_including_save_failure() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let active = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58.197192Z",
    );
    assert_eq!(counts.starts.get(), 1);
    recorder.quiesce();
    recorder.quiesce();
    assert_eq!(
        recorder.state(),
        State::Recording,
        "quiesce must not pretend the native recording stopped"
    );
    assert_eq!(counts.stops.get(), 0);

    clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)) + Duration::from_secs(1));
    assert_eq!(recorder.tick().expect("quiesced cap"), State::Recording);
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0);
    assert_eq!(recorder.boundary(), Boundary::None);

    let queued_at = at("2026-08-20T17:21:10.000001Z");
    assert!(matches!(
        recorder.admit("00000000-0000-4000-8000-00000000e0b2", queued_at),
        Ok(Admit::Queued)
    ));
    assert_eq!(recorder.pending(), 1);
    assert_eq!(recorder.counters().extended, 0);
    assert_eq!(recorder.counters().raced, 0);
    assert_eq!(recorder.counters().sequence, 2);
    assert_eq!(counts.starts.get(), 1);

    let mut saved = 0_u32;
    let mut contributors = Vec::new();
    let failed = recorder.on_receipt(&receipt(active), |sealed| {
        saved += 1;
        contributors = sealed.contributors.clone();
        assert_eq!(sealed.ticket, receipt(active).ticket);
        assert_eq!(sealed.duration_ms, 30_000);
        assert!(sealed.contains_video);
        Err(PublishError::Io(std::io::Error::other("volume full")))
    });
    assert!(matches!(failed, Err(RecorderError::Save(_))));
    assert_eq!(saved, 1);
    assert_eq!(
        contributors,
        vec![Contributor {
            event_ref: "00000000-0000-4000-8000-00000000e0a1".to_owned(),
            detected_at: at("2026-08-20T17:20:58.197192Z"),
        }]
    );
    assert_eq!(recorder.state(), State::Idle);
    assert_eq!(
        recorder.pending(),
        1,
        "queued alert must survive save failure"
    );
    assert_eq!(counts.starts.get(), 1, "save failure must not restart");

    let late_at = at("2026-08-20T17:22:00Z");
    assert!(matches!(
        recorder.admit("00000000-0000-4000-8000-00000000e0c3", late_at),
        Ok(Admit::Queued)
    ));
    assert_eq!(recorder.state(), State::Idle);
    assert_eq!(recorder.pending(), 2);
    assert_eq!(recorder.tick().expect("quiesced idle"), State::Idle);
    assert_eq!(counts.starts.get(), 1);

    let drained = recorder.take_unstarted().expect("drain after quiesce");
    assert_eq!(
        drained,
        vec![
            Contributor {
                event_ref: "00000000-0000-4000-8000-00000000e0b2".to_owned(),
                detected_at: queued_at,
            },
            Contributor {
                event_ref: "00000000-0000-4000-8000-00000000e0c3".to_owned(),
                detected_at: late_at,
            },
        ]
    );
    assert_eq!(recorder.pending(), 0);
    assert!(recorder.take_unstarted().expect("second drain").is_empty());
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0);
}

#[test]
fn active_contributors_save_once_and_are_not_drained_as_unstarted() {
    let Harness {
        mut recorder,
        counts,
        ..
    } = Harness::new();
    let active = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    recorder.quiesce();
    let mut saves = 0_u32;
    recorder
        .on_receipt(&receipt(active), |sealed| {
            saves += 1;
            assert_eq!(sealed.contributors.len(), 1);
            assert_eq!(
                sealed.contributors[0].event_ref,
                "00000000-0000-4000-8000-00000000e0a1"
            );
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("active save");
    assert_eq!(saves, 1);
    assert_eq!(counts.starts.get(), 1);
    let drained = recorder.take_unstarted().expect("nothing unstarted");
    assert!(drained.is_empty());
    assert!(matches!(
        recorder.on_receipt(&receipt(active), |_| unreachable!("duplicate receipt must not save")),
        Err(RecorderError::DuplicateRequest(request)) if request == active.request_id
    ));
}

#[test]
fn pending_full_still_applies_after_quiesce() {
    let Harness {
        mut recorder,
        counts,
        ..
    } = Harness::refusing();
    recorder.quiesce();
    for index in 0..MAX_PENDING_ALERTS {
        let event_ref = format!("00000000-0000-4000-8000-{index:012x}");
        assert!(matches!(
            recorder.admit(&event_ref, at("2026-08-20T17:20:58Z")),
            Ok(Admit::Queued)
        ));
    }
    assert_eq!(counts.starts.get(), 0);
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000ffff",
            at("2026-08-20T17:20:59Z")
        ),
        Err(RecorderError::PendingFull)
    ));
    assert_eq!(recorder.pending(), MAX_PENDING_ALERTS);
    let drained = recorder.take_unstarted().expect("bounded drain");
    assert_eq!(drained.len(), MAX_PENDING_ALERTS);
    assert_eq!(drained[0].event_ref, "00000000-0000-4000-8000-000000000000");
}

#[test]
fn reused_sdk_session_seals_each_new_request() {
    let Harness { mut recorder, .. } = Harness::new();
    let first = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    let mut seen = Vec::new();
    recorder
        .on_receipt(&receipt(first), |sealed| {
            seen.push(sealed.ticket);
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("first request");
    let second = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0b2",
        "2026-08-20T17:21:10Z",
    );
    assert_eq!(second.session_id, first.session_id);
    assert_ne!(second.request_id, first.request_id);
    recorder
        .on_receipt(&receipt(second), |sealed| {
            assert_eq!(sealed.ticket.request_id, second.request_id);
            assert_eq!(sealed.ticket.session_id, 7);
            assert_eq!(sealed.ticket.session_valid, 1);
            seen.push(sealed.ticket);
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("reused session, new request");
    assert_eq!(seen.len(), 2);
}

#[test]
fn async_session_assignment_keeps_receipt_session() {
    let Harness { mut recorder, .. } = Harness::new();
    let active = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    assert_eq!(active.session_valid, 0);
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket.request_id, active.request_id);
            assert_eq!(sealed.ticket.session_id, 7);
            assert_eq!(sealed.ticket.session_valid, 1);
            assert_ne!(sealed.ticket.session_id, active.session_id);
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("asynchronously assigned session");
}

#[test]
fn duplicate_after_idle_and_new_binding_are_distinct() {
    let Harness {
        mut recorder,
        counts,
        ..
    } = Harness::new();
    let first = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    recorder
        .on_receipt(&receipt(first), |_| Ok(SaveOutcome::FinalizeFailed(None)))
        .expect("first seal");
    assert_eq!(recorder.state(), State::Idle);
    let idle = unchanged(&recorder);
    assert!(matches!(
        recorder.on_receipt(&receipt(first), |_| unreachable!("duplicate after idle must not save")),
        Err(RecorderError::DuplicateRequest(request)) if request == first.request_id
    ));
    assert_eq!(unchanged(&recorder), idle);
    let second = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0b2",
        "2026-08-20T17:21:10Z",
    );
    let active = unchanged(&recorder);
    assert!(matches!(
        recorder.on_receipt(&receipt(first), |_| unreachable!("old request must not save")),
        Err(RecorderError::DuplicateRequest(request)) if request == first.request_id
    ));
    assert_eq!(unchanged(&recorder), active);
    recorder
        .on_receipt(&receipt(second), |_| Ok(SaveOutcome::FinalizeFailed(None)))
        .expect("second request");
    counts.next_request.set(first.request_id);
    counts.binding.set(MediaBinding {
        generation: BINDING.generation + 4,
        ..BINDING
    });
    let rebound = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0c3",
        "2026-08-20T17:21:20Z",
    );
    assert_eq!(rebound.request_id, first.request_id);
    assert_ne!(rebound.binding, first.binding);
    recorder
        .on_receipt(&receipt(rebound), |sealed| {
            assert_eq!(sealed.ticket.binding, rebound.binding);
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("same request under a new binding is not a duplicate");
}

#[test]
fn known_session_must_be_valid_and_match_receipt() {
    let Harness {
        mut recorder,
        counts,
        ..
    } = Harness::new();
    counts.session_id.set(7);
    counts.session_valid.set(1);
    let active = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    let before = unchanged(&recorder);
    let contradicted = RecordReceipt {
        ticket: RecordTicket {
            session_id: 8,
            session_valid: 1,
            ..active
        },
        ..receipt(active)
    };
    assert!(matches!(
        recorder.on_receipt(&contradicted, |_| unreachable!(
            "contradiction must not save"
        )),
        Err(RecorderError::SessionContradiction {
            admitted: 7,
            received: 8
        })
    ));
    assert_eq!(unchanged(&recorder), before);
    for invalid in [
        RecordTicket {
            session_id: 0,
            session_valid: 0,
            ..active
        },
        RecordTicket {
            session_id: u32::MAX,
            session_valid: 1,
            ..active
        },
    ] {
        let receipt = RecordReceipt {
            ticket: invalid,
            ..receipt(active)
        };
        assert!(matches!(
            recorder.on_receipt(&receipt, |_| unreachable!("invalid receipt must not save")),
            Err(RecorderError::InvalidReceiptTicket)
        ));
        assert_eq!(unchanged(&recorder), before);
    }
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket, active);
            assert_eq!(sealed.contributors.len(), 1);
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("matching known session");
}

#[test]
fn receipt_refusals_preserve_pending_state_and_contributors() {
    let Harness {
        mut recorder,
        clock,
        ..
    } = Harness::new();
    let active = started(
        &mut recorder,
        "00000000-0000-4000-8000-00000000e0a1",
        "2026-08-20T17:20:58Z",
    );
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000e0b2",
            at("2026-08-20T17:21:00Z")
        ),
        Ok(Admit::Extended)
    ));
    clock.advance(Duration::from_secs(u64::from(EXTENSION_SECONDS)));
    assert_eq!(recorder.tick().expect("early stop"), State::Stopping);
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000e0c3",
            at("2026-08-20T17:21:50Z")
        ),
        Ok(Admit::Queued)
    ));
    let before = unchanged(&recorder);
    let mut refused = |ticket: RecordTicket| {
        let attempted = RecordReceipt {
            ticket,
            ..receipt(active)
        };
        let failed = recorder.on_receipt(&attempted, |_| unreachable!("refusal must not save"));
        assert!(failed.is_err());
        assert_eq!(unchanged(&recorder), before);
    };
    refused(RecordTicket {
        request_id: active.request_id + 9,
        ..active
    });
    refused(RecordTicket {
        source_id: SOURCE_ID + 3,
        ..active
    });
    refused(RecordTicket {
        binding: MediaBinding {
            generation: BINDING.generation + 1,
            ..BINDING
        },
        ..active
    });
    refused(RecordTicket {
        binding: MediaBinding {
            epoch: BINDING.epoch + 1,
            ..BINDING
        },
        ..active
    });
    refused(RecordTicket {
        binding: MediaBinding {
            token: BINDING.token + 1,
            ..BINDING
        },
        ..active
    });
    refused(RecordTicket {
        session_valid: 0,
        ..active
    });
    recorder.quiesce();
    recorder
        .on_receipt(
            &RecordReceipt {
                ticket: RecordTicket {
                    session_valid: 0,
                    session_id: 0,
                    ..active
                },
                result: MediaResult::Stale,
                contains_video: false,
                duration_ms: 0,
                ..receipt(active)
            },
            |sealed| {
                assert_eq!(sealed.ticket.session_valid, 0);
                assert_eq!(sealed.contributors.len(), 2);
                Ok(SaveOutcome::FinalizeFailed(None))
            },
        )
        .expect("cancelled receipt may omit a session");
    assert_eq!(recorder.pending(), 1);
}

#[test]
fn active_contributors_are_bounded_before_refusal_mutates_the_recording() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let expected: Vec<String> = (0..MAX_PENDING_ALERTS)
        .map(|index| format!("00000000-0000-4000-8000-{index:012x}"))
        .collect();
    let active = started(&mut recorder, &expected[0], "2026-08-20T17:20:58Z");
    for event_ref in expected.iter().skip(1) {
        assert!(matches!(
            recorder.admit(event_ref, at("2026-08-20T17:20:58Z")),
            Ok(Admit::Extended)
        ));
    }
    assert_eq!(recorder.counters().sequence, MAX_PENDING_ALERTS as u64);
    assert_eq!(
        recorder.counters().extended,
        (MAX_PENDING_ALERTS - 1) as u64
    );
    clock.advance(Duration::from_secs(u64::from(EXTENSION_SECONDS - 1)));
    let before = unchanged(&recorder);
    let counters = recorder.counters();
    for index in MAX_PENDING_ALERTS..MAX_PENDING_ALERTS + 2 {
        let event_ref = format!("00000000-0000-4000-8000-{index:012x}");
        assert!(matches!(
            recorder.admit(&event_ref, at("2026-08-20T17:21:42Z")),
            Err(RecorderError::ContributorsFull)
        ));
        assert_eq!(unchanged(&recorder), before);
        assert_eq!(recorder.counters(), counters);
        assert_eq!(counts.starts.get(), 1);
        assert_eq!(counts.stops.get(), 0);
    }
    clock.advance(Duration::from_secs(1));
    assert_eq!(
        recorder.tick().expect("original extension due"),
        State::Stopping
    );
    assert_eq!(
        counts.stops.get(),
        1,
        "refusal must not renew the extension"
    );
    recorder.quiesce();
    assert!(
        recorder
            .take_unstarted()
            .expect("only active work")
            .is_empty()
    );
    clock.advance(Duration::from_secs(u64::from(
        CAP_SECONDS + NATIVE_COMPLETION_GRACE_SECONDS - EXTENSION_SECONDS,
    )));
    assert_overdue(recorder.tick(), active);
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket, receipt(active).ticket);
            let accepted: Vec<String> = sealed
                .contributors
                .iter()
                .map(|contributor| contributor.event_ref.clone())
                .collect();
            assert_eq!(
                accepted, expected,
                "no accepted contributor may be discarded"
            );
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("all bounded active contributors save");
    assert_overdue(recorder.tick(), active);
    assert_eq!(counts.starts.get(), 1);
}

#[test]
fn refused_pending_batch_is_bounded_and_becomes_bounded_active_work() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::refusing();
    let expected: Vec<String> = (0..MAX_PENDING_ALERTS)
        .map(|index| format!("00000000-0000-4000-8000-{index:012x}"))
        .collect();
    for event_ref in &expected {
        assert!(matches!(
            recorder.admit(event_ref, at("2026-08-20T17:20:58Z")),
            Ok(Admit::Refused(PlaneRefusal::Busy))
        ));
    }
    let before = unchanged(&recorder);
    let counters = recorder.counters();
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000ffff",
            at("2026-08-20T17:21:00Z")
        ),
        Err(RecorderError::PendingFull)
    ));
    assert_eq!(unchanged(&recorder), before);
    assert_eq!(recorder.counters(), counters);
    assert_eq!(counts.starts.get(), MAX_PENDING_ALERTS as u32);

    clock.advance(Duration::from_secs(u64::from(
        CAP_SECONDS + NATIVE_COMPLETION_GRACE_SECONDS + 1,
    )));
    assert_eq!(
        recorder.tick().expect("no successful acknowledgement yet"),
        State::Idle
    );
    counts.refuse.set(false);
    assert_eq!(recorder.tick().expect("retry accepted"), State::Recording);
    let active = counts
        .last_started
        .get()
        .expect("plane acknowledged ticket");
    assert_eq!(recorder.pending(), 0);
    assert_eq!(counts.starts.get(), MAX_PENDING_ALERTS as u32 + 2);
    let before = unchanged(&recorder);
    let counters = recorder.counters();
    assert!(matches!(
        recorder.admit(
            "00000000-0000-4000-8000-00000000ffff",
            at("2026-08-20T17:23:30Z")
        ),
        Err(RecorderError::ContributorsFull)
    ));
    assert_eq!(unchanged(&recorder), before);
    assert_eq!(recorder.counters(), counters);
    recorder.quiesce();
    assert!(
        recorder
            .take_unstarted()
            .expect("batch has started")
            .is_empty()
    );
    recorder
        .on_receipt(&receipt(active), |sealed| {
            let accepted: Vec<String> = sealed
                .contributors
                .iter()
                .map(|contributor| contributor.event_ref.clone())
                .collect();
            assert_eq!(accepted, expected);
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("entire admitted batch saves");
}

#[test]
fn admit_before_tick_extends_only_just_before_the_content_cap() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let active = started(&mut recorder, "first", "2026-08-20T17:20:58Z");
    clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)) - Duration::from_nanos(1));
    assert!(matches!(
        recorder.admit("just-before", at("2026-08-20T17:22:57.999999Z")),
        Ok(Admit::Extended)
    ));
    clock.advance(Duration::from_nanos(1));
    assert!(matches!(
        recorder.admit("exact-cap", at("2026-08-20T17:22:58Z")),
        Ok(Admit::Queued)
    ));
    clock.advance(Duration::from_nanos(1));
    assert!(matches!(
        recorder.admit("after-cap", at("2026-08-20T17:22:58.000001Z")),
        Ok(Admit::Queued)
    ));
    assert_eq!(recorder.state(), State::Recording);
    assert_eq!(recorder.pending(), 2);
    assert_eq!(recorder.counters().extended, 1);
    assert_eq!(recorder.counters().raced, 2);
    assert_eq!(recorder.counters().sequence, 4);
    assert_eq!(
        recorder.tick().expect("content cap is not receipt expiry"),
        State::Recording
    );
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0, "the native owner seals the cap");
    recorder.quiesce();
    let unstarted = recorder.take_unstarted().expect("post-cap alerts only");
    assert_eq!(
        unstarted
            .iter()
            .map(|alert| alert.event_ref.as_str())
            .collect::<Vec<_>>(),
        ["exact-cap", "after-cap"]
    );
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket, receipt(active).ticket);
            assert_eq!(
                sealed
                    .contributors
                    .iter()
                    .map(|alert| alert.event_ref.as_str())
                    .collect::<Vec<_>>(),
                ["first", "just-before"]
            );
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("only pre-cap contributors save with the admitted request");
    assert_eq!(counts.starts.get(), 1);
}

#[test]
fn post_cap_pending_is_bounded_and_admission_alone_latches_receipt_expiry() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let active = started(&mut recorder, "active", "2026-08-20T17:20:58Z");
    clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)));
    for index in 0..MAX_PENDING_ALERTS {
        let event_ref = format!("00000000-0000-4000-8000-{index:012x}");
        assert!(matches!(
            recorder.admit(&event_ref, at("2026-08-20T17:22:58Z")),
            Ok(Admit::Queued)
        ));
    }
    let before = unchanged(&recorder);
    let counters = recorder.counters();
    assert!(matches!(
        recorder.admit("refused-at-cap", at("2026-08-20T17:22:58Z")),
        Err(RecorderError::PendingFull)
    ));
    assert_eq!(unchanged(&recorder), before);
    assert_eq!(recorder.counters(), counters);
    clock.advance(Duration::from_secs(u64::from(
        NATIVE_COMPLETION_GRACE_SECONDS,
    )));
    assert_overdue(
        recorder.admit("refused-at-delivery-deadline", at("2026-08-20T17:23:28Z")),
        active,
    );
    assert_eq!(unchanged(&recorder), before);
    assert_eq!(recorder.counters(), counters);
    assert_overdue(recorder.tick(), active);
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0);
    recorder.quiesce();
    assert_eq!(
        recorder
            .take_unstarted()
            .expect("bounded pending only")
            .len(),
        MAX_PENDING_ALERTS
    );
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.contributors.len(), 1);
            assert_eq!(sealed.contributors[0].event_ref, "active");
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("expiry did not discard active work");
    assert_overdue(recorder.tick(), active);
}

#[test]
fn receipt_delivery_deadline_is_exact_and_sticky_in_recording_and_stopping() {
    for early_stop in [false, true] {
        let Harness {
            mut recorder,
            clock,
            counts,
        } = Harness::new();
        let active = started(&mut recorder, "active", "2026-08-20T17:20:58Z");
        let expected_state = if early_stop {
            clock.advance(Duration::from_secs(u64::from(EXTENSION_SECONDS)));
            assert_eq!(recorder.tick().expect("early stop"), State::Stopping);
            clock.advance(Duration::from_secs(u64::from(
                CAP_SECONDS - EXTENSION_SECONDS,
            )));
            State::Stopping
        } else {
            clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)));
            State::Recording
        };
        assert_eq!(
            recorder.tick().expect("120 seconds is not receipt expiry"),
            expected_state
        );
        assert!(matches!(
            recorder.admit("pending", at("2026-08-20T17:22:58Z")),
            Ok(Admit::Queued)
        ));
        clock.advance(
            Duration::from_secs(u64::from(NATIVE_COMPLETION_GRACE_SECONDS))
                - Duration::from_nanos(1),
        );
        assert_eq!(
            recorder.tick().expect("just before delivery deadline"),
            expected_state
        );
        let before = unchanged(&recorder);
        let counters = recorder.counters();
        clock.advance(Duration::from_nanos(1));
        assert_overdue(recorder.tick(), active);
        assert_eq!(
            unchanged(&recorder),
            before,
            "expiry is not native completion"
        );
        clock.advance(Duration::from_nanos(1));
        assert_overdue(recorder.tick(), active);
        for index in 0..MAX_PENDING_ALERTS + 1 {
            assert_overdue(
                recorder.admit(&format!("refused-{index}"), at("2026-08-20T17:23:28Z")),
                active,
            );
            assert_eq!(unchanged(&recorder), before);
            assert_eq!(recorder.counters(), counters);
        }
        assert_eq!(counts.starts.get(), 1);
        assert_eq!(counts.stops.get(), if early_stop { 1 } else { 0 });
    }
}

#[test]
fn delayed_matching_receipt_before_deadline_saves_and_starts_pending() {
    for early_stop in [false, true] {
        let Harness {
            mut recorder,
            clock,
            counts,
        } = Harness::new();
        let active = started(&mut recorder, "active", "2026-08-20T17:20:58Z");
        if early_stop {
            clock.advance(Duration::from_secs(u64::from(EXTENSION_SECONDS)));
            assert_eq!(recorder.tick().expect("early stop"), State::Stopping);
            clock.advance(Duration::from_secs(u64::from(
                CAP_SECONDS - EXTENSION_SECONDS,
            )));
        } else {
            clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)));
            assert_eq!(recorder.tick().expect("content cap"), State::Recording);
        }
        assert!(matches!(
            recorder.admit("pending", at("2026-08-20T17:22:58Z")),
            Ok(Admit::Queued)
        ));
        clock.advance(
            Duration::from_secs(u64::from(NATIVE_COMPLETION_GRACE_SECONDS))
                - Duration::from_nanos(1),
        );
        let mut saves = 0;
        recorder
            .on_receipt(&receipt(active), |sealed| {
                saves += 1;
                assert_eq!(sealed.ticket, receipt(active).ticket);
                assert_eq!(sealed.contributors.len(), 1);
                assert_eq!(sealed.contributors[0].event_ref, "active");
                assert_eq!(sealed.duration_ms, receipt(active).duration_ms);
                Ok(SaveOutcome::FinalizeFailed(None))
            })
            .expect("delayed receipt is still within its fixed window");
        assert_eq!(saves, 1);
        assert_eq!(counts.starts.get(), 2);
        assert_eq!(recorder.pending(), 0);
        let next = counts
            .last_started
            .get()
            .expect("next acknowledged request");
        assert_ne!(next.request_id, active.request_id);
        clock.advance(Duration::from_nanos(2));
        let before = unchanged(&recorder);
        assert!(matches!(
            recorder.on_receipt(&receipt(active), |_| unreachable!("duplicate must not save")),
            Err(RecorderError::DuplicateRequest(request)) if request == active.request_id
        ));
        let foreign = RecordTicket {
            source_id: SOURCE_ID + 1,
            ..next
        };
        assert!(matches!(
            recorder.on_receipt(&receipt(foreign), |_| unreachable!(
                "foreign receipt must not save"
            )),
            Err(RecorderError::WrongSource { .. })
        ));
        assert_eq!(unchanged(&recorder), before);
        assert_eq!(
            recorder
                .tick()
                .expect("rejections do not poison the new request"),
            State::Recording
        );
        recorder.quiesce();
        recorder
            .on_receipt(&receipt(next), |sealed| {
                assert_eq!(sealed.contributors.len(), 1);
                assert_eq!(sealed.contributors[0].event_ref, "pending");
                Ok(SaveOutcome::FinalizeFailed(None))
            })
            .expect("healthy next request saves");
    }
}

#[test]
fn rejected_receipts_neither_renew_delivery_deadline_nor_latch_failure() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let previous = started(&mut recorder, "previous", "2026-08-20T17:20:57Z");
    recorder
        .on_receipt(&receipt(previous), finalize_failed_without_publication)
        .expect("previous request sealed");
    let active = started(&mut recorder, "active", "2026-08-20T17:20:58Z");
    clock.advance(
        Duration::from_secs(u64::from(CAP_SECONDS + NATIVE_COMPLETION_GRACE_SECONDS))
            - Duration::from_nanos(1),
    );
    let before = unchanged(&recorder);
    let counters = recorder.counters();
    assert!(matches!(
        recorder.on_receipt(&receipt(previous), |_| unreachable!("duplicate must not save")),
        Err(RecorderError::DuplicateRequest(request)) if request == previous.request_id
    ));
    assert_eq!(unchanged(&recorder), before);
    for wrong in [
        RecordTicket {
            request_id: active.request_id + 9,
            ..active
        },
        RecordTicket {
            source_id: SOURCE_ID + 1,
            ..active
        },
        RecordTicket {
            binding: MediaBinding {
                epoch: BINDING.epoch + 1,
                ..BINDING
            },
            ..active
        },
    ] {
        assert!(
            recorder
                .on_receipt(&receipt(wrong), |_| unreachable!("rejection must not save"))
                .is_err()
        );
        assert_eq!(unchanged(&recorder), before);
        assert_eq!(recorder.counters(), counters);
    }
    assert_eq!(
        recorder
            .tick()
            .expect("invalid receipts did not fail healthy work"),
        State::Recording
    );
    let before = unchanged(&recorder);
    clock.advance(Duration::from_nanos(1));
    let wrong_source = RecordTicket {
        source_id: SOURCE_ID + 1,
        ..active
    };
    assert!(matches!(
        recorder.on_receipt(&receipt(wrong_source), |_| unreachable!(
            "identity rejects before mutation"
        )),
        Err(RecorderError::WrongSource { .. })
    ));
    assert_eq!(unchanged(&recorder), before);
    assert_overdue(recorder.tick(), active);
    assert_eq!(counts.starts.get(), 2);
    assert_eq!(counts.stops.get(), 0);
}

#[test]
fn latched_failure_survives_late_save_without_starting_pending() {
    for save_fails in [false, true] {
        let Harness {
            mut recorder,
            clock,
            counts,
        } = Harness::new();
        let active = started(&mut recorder, "first", "2026-08-20T17:20:58Z");
        assert!(matches!(
            recorder.admit("extended", at("2026-08-20T17:20:59Z")),
            Ok(Admit::Extended)
        ));
        clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)));
        assert!(matches!(
            recorder.admit("pending", at("2026-08-20T17:22:58Z")),
            Ok(Admit::Queued)
        ));
        clock.advance(Duration::from_secs(u64::from(
            NATIVE_COMPLETION_GRACE_SECONDS,
        )));
        assert_overdue(recorder.tick(), active);
        clock.advance(Duration::from_secs(1));
        let mut saves = 0;
        let saved = recorder.on_receipt(&receipt(active), |sealed| {
            saves += 1;
            assert_eq!(sealed.ticket, receipt(active).ticket);
            assert_eq!(sealed.result, MediaResult::Ok);
            assert!(sealed.contains_video);
            assert_eq!(sealed.duration_ms, receipt(active).duration_ms);
            assert_eq!(
                sealed
                    .contributors
                    .iter()
                    .map(|alert| alert.event_ref.as_str())
                    .collect::<Vec<_>>(),
                ["first", "extended"]
            );
            if save_fails {
                Err(PublishError::Io(std::io::Error::other("volume full")))
            } else {
                Ok(SaveOutcome::FinalizeFailed(None))
            }
        });
        if save_fails {
            assert!(matches!(saved, Err(RecorderError::Save(_))));
        } else {
            assert!(matches!(saved, Ok(SaveOutcome::FinalizeFailed(None))));
        }
        assert_eq!(saves, 1);
        assert_eq!(recorder.state(), State::Idle);
        assert_eq!(
            recorder.pending(),
            1,
            "accepted pending work survives the late save"
        );
        assert_eq!(counts.starts.get(), 1, "failure suppresses start_pending");
        let before = unchanged(&recorder);
        let counters = recorder.counters();
        assert_overdue(recorder.tick(), active);
        assert!(matches!(
            recorder.on_receipt(&receipt(active), |_| unreachable!("late receipt saves only once")),
            Err(RecorderError::DuplicateRequest(request)) if request == active.request_id
        ));
        assert_overdue(
            recorder.admit("refused", at("2026-08-20T17:23:28Z")),
            active,
        );
        assert_eq!(unchanged(&recorder), before);
        assert_eq!(recorder.counters(), counters);
        recorder.quiesce();
        let unstarted = recorder
            .take_unstarted()
            .expect("pending ownership preserved");
        assert_eq!(unstarted.len(), 1);
        assert_eq!(unstarted[0].event_ref, "pending");
        assert_overdue(recorder.tick(), active);
        assert_eq!(counts.starts.get(), 1);
        assert_eq!(counts.stops.get(), 0);
    }
}

#[test]
fn start_acknowledgement_anchors_both_immutable_deadlines() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let before_start = clock.monotonic();
    let acknowledgement_delay = Duration::from_secs(7);
    counts.start_delay.set(acknowledgement_delay);
    let active = started(&mut recorder, "first", "2026-08-20T17:20:58Z");
    assert_eq!(clock.monotonic(), before_start + acknowledgement_delay);
    clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)) - acknowledgement_delay);
    assert!(matches!(
        recorder.admit("before-acknowledged-cap", at("2026-08-20T17:22:58Z")),
        Ok(Admit::Extended)
    ));
    clock.advance(acknowledgement_delay);
    assert!(matches!(
        recorder.admit("at-acknowledged-cap", at("2026-08-20T17:23:05Z")),
        Ok(Admit::Queued)
    ));
    assert_eq!(
        recorder.tick().expect("acknowledged content cap"),
        State::Recording
    );
    clock.advance(
        Duration::from_secs(u64::from(NATIVE_COMPLETION_GRACE_SECONDS)) - Duration::from_nanos(1),
    );
    assert_eq!(
        recorder
            .tick()
            .expect("acknowledged delivery deadline has not arrived"),
        State::Recording
    );
    clock.advance(Duration::from_nanos(1));
    assert_overdue(recorder.tick(), active);
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0);
}

#[test]
fn quiesced_expiry_keeps_active_contributors_out_of_unstarted_work() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let active = started(&mut recorder, "first", "2026-08-20T17:20:58Z");
    assert!(matches!(
        recorder.admit("extended", at("2026-08-20T17:20:59Z")),
        Ok(Admit::Extended)
    ));
    recorder.quiesce();
    assert!(matches!(
        recorder.admit("unstarted", at("2026-08-20T17:21:00Z")),
        Ok(Admit::Queued)
    ));
    let before = unchanged(&recorder);
    let counters = recorder.counters();
    clock.advance(Duration::from_secs(u64::from(
        CAP_SECONDS + NATIVE_COMPLETION_GRACE_SECONDS,
    )));
    assert_overdue(recorder.tick(), active);
    assert_eq!(unchanged(&recorder), before);
    assert_overdue(
        recorder.admit("refused", at("2026-08-20T17:23:28Z")),
        active,
    );
    assert_eq!(recorder.counters(), counters);
    assert_eq!(recorder.state(), State::Recording);
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0);
    let unstarted = recorder
        .take_unstarted()
        .expect("only never-started work drains");
    assert_eq!(unstarted.len(), 1);
    assert_eq!(unstarted[0].event_ref, "unstarted");
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket, receipt(active).ticket);
            assert_eq!(
                sealed
                    .contributors
                    .iter()
                    .map(|alert| alert.event_ref.as_str())
                    .collect::<Vec<_>>(),
                ["first", "extended"]
            );
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("quiesced native work still saves on its real matching receipt");
    assert!(
        recorder
            .take_unstarted()
            .expect("no active work became unstarted")
            .is_empty()
    );
    assert_overdue(recorder.tick(), active);
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 0);
}

#[test]
fn matching_receipt_drained_at_delivery_deadline_does_not_false_fail() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    let active = started(&mut recorder, "active", "2026-08-20T17:20:58Z");
    clock.advance(Duration::from_secs(u64::from(CAP_SECONDS)));
    assert!(matches!(
        recorder.admit("pending", at("2026-08-20T17:22:58Z")),
        Ok(Admit::Queued)
    ));
    clock.advance(Duration::from_secs(u64::from(
        NATIVE_COMPLETION_GRACE_SECONDS,
    )));
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket, receipt(active).ticket);
            assert_eq!(sealed.contributors.len(), 1);
            assert_eq!(sealed.contributors[0].event_ref, "active");
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("drain an actual matching receipt before evaluating expiry");
    assert_eq!(counts.starts.get(), 2);
    assert_eq!(recorder.pending(), 0);
    assert_eq!(
        recorder
            .tick()
            .expect("actual receipt prevents false expiry"),
        State::Recording
    );
}

#[test]
fn refused_early_stop_preserves_native_identity_and_immutable_deadlines() {
    let Harness {
        mut recorder,
        clock,
        counts,
    } = Harness::new();
    counts.refuse_stop.set(true);
    let active = started(&mut recorder, "first", "2026-08-20T17:20:58Z");
    clock.advance(Duration::from_secs(u64::from(EXTENSION_SECONDS)));
    assert_eq!(
        recorder.tick().expect("unsupported early stop"),
        State::Recording
    );
    assert_eq!(recorder.boundary(), Boundary::ExtensionBounded);
    assert_eq!(counts.stops.get(), 1);
    assert!(matches!(
        recorder.admit("after-stop-refusal", at("2026-08-20T17:21:43Z")),
        Ok(Admit::Extended)
    ));
    clock.advance(
        Duration::from_secs(u64::from(CAP_SECONDS - EXTENSION_SECONDS)) - Duration::from_nanos(1),
    );
    assert!(matches!(
        recorder.admit("just-before-cap", at("2026-08-20T17:22:57.999999Z")),
        Ok(Admit::Extended)
    ));
    clock.advance(Duration::from_nanos(1));
    assert!(matches!(
        recorder.admit("at-cap", at("2026-08-20T17:22:58Z")),
        Ok(Admit::Queued)
    ));
    assert_eq!(
        recorder.tick().expect("cap still belongs to native"),
        State::Recording
    );
    clock.advance(Duration::from_secs(u64::from(
        NATIVE_COMPLETION_GRACE_SECONDS,
    )));
    assert_overdue(recorder.tick(), active);
    recorder.quiesce();
    let unstarted = recorder.take_unstarted().expect("cap alert never started");
    assert_eq!(unstarted.len(), 1);
    assert_eq!(unstarted[0].event_ref, "at-cap");
    recorder
        .on_receipt(&receipt(active), |sealed| {
            assert_eq!(sealed.ticket, receipt(active).ticket);
            assert_eq!(sealed.contributors.len(), 3);
            assert_eq!(sealed.contributors[0].event_ref, "first");
            assert_eq!(sealed.contributors[1].event_ref, "after-stop-refusal");
            assert_eq!(sealed.contributors[2].event_ref, "just-before-cap");
            Ok(SaveOutcome::FinalizeFailed(None))
        })
        .expect("stop refusal and expiry did not discard accepted contributors");
    assert_overdue(recorder.tick(), active);
    assert_eq!(counts.starts.get(), 1);
    assert_eq!(counts.stops.get(), 1);
}
