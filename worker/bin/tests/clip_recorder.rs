//! Recorder admission after shutdown quiescence.
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
    Admit, Boundary, CAP_SECONDS, EXTENSION_SECONDS, MAX_PENDING_ALERTS, PlaneRefusal, RecordPlane,
    Recorder, RecorderError, State,
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
}

/// Counts every start the recorder asks for. Refuses while `refuse` is set so
/// pending can be staged without a recording.
struct CountingPlane {
    counts: Rc<PlaneCounts>,
    refuse: Cell<bool>,
}

impl CountingPlane {
    fn open(counts: Rc<PlaneCounts>, refuse: bool) -> Self {
        Self {
            counts,
            refuse: Cell::new(refuse),
        }
    }
}

impl RecordPlane for CountingPlane {
    fn start(
        &mut self,
        _lookback_seconds: u32,
        _forward_seconds: u32,
    ) -> Result<RecordTicket, PlaneRefusal> {
        self.counts.starts.set(self.counts.starts.get() + 1);
        if self.refuse.get() {
            return Err(PlaneRefusal::Busy);
        }
        let request_id = self.counts.next_request.get();
        self.counts.next_request.set(request_id + 1);
        Ok(RecordTicket {
            binding: self.counts.binding.get(),
            request_id,
            source_id: SOURCE_ID,
            session_id: self.counts.session_id.get(),
            session_valid: self.counts.session_valid.get(),
            coalesced: 0,
        })
    }

    fn stop(&mut self, _ticket: &RecordTicket) -> Result<(), PlaneRefusal> {
        self.counts.stops.set(self.counts.stops.get() + 1);
        Ok(())
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
        });
        Self {
            recorder: Recorder::new(
                SOURCE_ID,
                CountingPlane::open(Rc::clone(&counts), false),
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
        });
        Self {
            recorder: Recorder::new(
                SOURCE_ID,
                CountingPlane::open(Rc::clone(&counts), true),
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
    assert_eq!(recorder.tick(), State::Stopping);
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
    recorder.tick();
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
    assert_eq!(recorder.tick(), State::Recording);
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
    assert_eq!(recorder.tick(), State::Idle);
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
    assert_eq!(recorder.tick(), State::Stopping);
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
