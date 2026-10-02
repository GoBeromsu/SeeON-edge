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
}

/// Counts every start the recorder asks for. Refuses while `refuse` is set so
/// pending can be staged without a recording.
struct CountingPlane {
    counts: Rc<PlaneCounts>,
    next_session: Cell<u32>,
    refuse: Cell<bool>,
}

impl CountingPlane {
    fn open(counts: Rc<PlaneCounts>, refuse: bool) -> Self {
        Self {
            counts,
            next_session: Cell::new(1),
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
        let session_id = self.next_session.get();
        self.next_session.set(session_id + 1);
        Ok(RecordTicket {
            binding: BINDING,
            request_id: u64::from(session_id),
            source_id: SOURCE_ID,
            session_id,
            session_valid: 1,
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
        ticket,
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
        Err(RecorderError::WrongCamera {
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
        assert_eq!(sealed.ticket, active);
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
        Err(RecorderError::DuplicateSealed(session)) if session == active.session_id
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
