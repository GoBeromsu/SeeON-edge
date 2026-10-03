//! Private ownership assertions; counted native calls are not SDK evidence.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use seeon_deepstream_native::MediaBinding;

use super::*;

struct CountedClock {
    now: Mutex<Duration>,
    reads: AtomicU64,
}

impl CountedClock {
    fn advance(&self, by: Duration) {
        *self.now.lock().expect("clock") += by;
    }
}

impl Clock for CountedClock {
    fn monotonic(&self) -> Duration {
        self.reads.fetch_add(1, Ordering::Relaxed);
        *self.now.lock().expect("clock")
    }

    fn wall(&self) -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH + self.monotonic()
    }

    fn pause(&self, limit: Duration) {
        self.advance(limit);
    }
}

struct CountedPlane {
    starts: Rc<Cell<usize>>,
    ticket: RecordTicket,
}

impl RecordPlane for CountedPlane {
    fn start(&mut self, _: u32, _: u32) -> Result<RecordTicket, PlaneRefusal> {
        self.starts.set(self.starts.get() + 1);
        Ok(self.ticket)
    }

    fn stop(&mut self, _: &RecordTicket) -> Result<(), PlaneRefusal> {
        panic!("receipt ownership must not stop native work");
    }
}

fn harness() -> (Recorder<CountedPlane>, Arc<CountedClock>, Rc<Cell<usize>>) {
    let clock = Arc::new(CountedClock {
        now: Mutex::new(Duration::from_secs(1_000)),
        reads: AtomicU64::new(0),
    });
    let starts = Rc::new(Cell::new(0));
    let ticket = RecordTicket {
        binding: MediaBinding {
            token: 73,
            generation: 7,
            epoch: 11,
        },
        request_id: 42,
        source_id: 4,
        session_id: 0,
        session_valid: 0,
        coalesced: 0,
    };
    let recorder = Recorder::new(
        ticket.source_id,
        CountedPlane {
            starts: Rc::clone(&starts),
            ticket,
        },
        Arc::clone(&clock) as Arc<dyn Clock>,
    );
    (recorder, clock, starts)
}

fn at() -> Utc {
    Utc::parse("2026-08-20T17:20:58Z").expect("timestamp")
}

fn recording_deadlines(recorder: &Recorder<CountedPlane>) -> [Duration; 3] {
    [
        recorder.hard_deadline,
        recorder.receipt_deadline,
        recorder.stop_due,
    ]
}

fn receipt(ticket: RecordTicket, result: MediaResult) -> RecordReceipt {
    RecordReceipt {
        ticket: RecordTicket {
            session_id: if result == MediaResult::Ok {
                7
            } else {
                NO_SESSION
            },
            session_valid: u32::from(result == MediaResult::Ok),
            coalesced: 1,
            ..ticket
        },
        result,
        error: 0,
        duration_ms: 30_123,
        width: 1280,
        height: 720,
        contains_video: result == MediaResult::Ok,
        contains_audio: false,
        directory: "/tmp/sealed-owner".into(),
        filename: "original.mp4".into(),
    }
}

#[test]
fn failed_seal_keeps_exact_owner_without_reading_clocks_or_restarting() {
    for result in [MediaResult::Ok, MediaResult::Fatal] {
        let (mut recorder, clock, starts) = harness();
        let Admit::Started(admitted) = recorder.admit("z", at()).expect("start") else {
            panic!("expected start");
        };
        recorder.admit("b", at()).expect("extend");
        recorder
            .admit("a", at().plus_millis(-1_000))
            .expect("earlier contributor");
        clock.advance(seconds(CAP_SECONDS));
        recorder.admit("pending", at()).expect("queue at cap");
        let original = receipt(admitted, result);
        let expected = ClipSealed {
            ticket: original.ticket,
            result,
            contains_video: original.contains_video,
            duration_ms: original.duration_ms,
            boundary: Boundary::ExtensionBounded,
            contributors: [("a", -1_000), ("b", 0), ("z", 0)]
                .into_iter()
                .map(|(event_ref, offset)| Contributor {
                    event_ref: event_ref.into(),
                    detected_at: at().plus_millis(offset),
                })
                .collect(),
            path: original.directory.join(&original.filename),
        };
        let deadlines = recording_deadlines(&recorder);
        let reads = clock.reads.load(Ordering::Relaxed);
        let mut saves = 0;
        assert!(matches!(
            recorder.on_receipt(&original, |sealed| {
                saves += 1;
                assert_eq!(sealed, &expected);
                Ok(SaveOutcome::FinalizeFailed(None))
            }),
            Err(RecorderError::Unpublished { ticket }) if ticket == original.ticket
        ));
        assert_eq!(recorder.failed_seal.as_ref(), Some(&expected));
        assert_eq!(recorder.sealed.back(), Some(&original.ticket));
        assert!(recorder.ticket.is_none());
        assert!(recorder.contributors.is_empty());
        let counters = recorder.counters;
        clock.advance(seconds(NATIVE_COMPLETION_GRACE_SECONDS + 1));
        for blank in [false, true] {
            let event_ref = if blank { " " } else { "refused" };
            assert!(matches!(
                recorder.admit_with_receipt_observation_cutoff(event_ref, at(), Duration::MAX),
                Err(RecorderError::Unpublished { ticket }) if ticket == original.ticket
            ));
            assert!(matches!(
                recorder.tick_with_receipt_observation_cutoff(Duration::MAX),
                Err(RecorderError::Unpublished { ticket }) if ticket == original.ticket
            ));
        }
        assert!(matches!(recorder.start_pending(), Admit::Queued));
        recorder.quiesce();
        recorder.quiesce();
        let pending = recorder
            .take_unstarted()
            .expect("only unstarted work drains");
        assert_eq!(
            pending,
            vec![Contributor {
                event_ref: "pending".into(),
                detected_at: at()
            }]
        );
        assert!(matches!(
            recorder.on_receipt(&original, |_| panic!("duplicate must not save")),
            Err(RecorderError::DuplicateRequest(42))
        ));
        assert_eq!(saves, 1);
        assert_eq!(starts.get(), 1);
        assert_eq!(recorder.state, State::Finalizing);
        assert_eq!(recorder.failed_seal.as_ref(), Some(&expected));
        assert_eq!(recorder.counters, counters);
        assert_eq!(recorder.receipt_overdue, None);
        assert_eq!(recording_deadlines(&recorder), deadlines);
        assert_eq!(clock.reads.load(Ordering::Relaxed), reads);
    }
}

#[test]
fn fresh_content_time_and_earlier_overdue_identity_survive_failed_publication() {
    let (mut recorder, clock, starts) = harness();
    let Admit::Started(admitted) = recorder.admit("active", at()).expect("start") else {
        panic!("expected start");
    };
    let deadlines = recording_deadlines(&recorder);
    let cutoff = recorder.hard_deadline - Duration::from_nanos(1);
    clock.advance(seconds(CAP_SECONDS));
    assert!(matches!(
        recorder.admit_with_receipt_observation_cutoff("pending", at(), cutoff),
        Ok(Admit::Queued)
    ));
    assert_eq!(recorder.counters.extended, 0, "cutoff cannot renew content");
    clock.advance(seconds(NATIVE_COMPLETION_GRACE_SECONDS));
    assert_eq!(
        recorder
            .tick_with_receipt_observation_cutoff(cutoff)
            .expect("pre-drain expiry cutoff"),
        State::Recording
    );
    assert!(
        matches!(recorder.tick(), Err(RecorderError::ReceiptOverdue { ticket }) if ticket == admitted)
    );
    let original = receipt(admitted, MediaResult::Ok);
    assert_ne!(original.ticket, admitted, "receipt assigns native session");
    let mut saves = 0;
    assert!(matches!(
        recorder.on_receipt(&original, |_| {
            saves += 1;
            Ok(SaveOutcome::FinalizeFailed(None))
        }),
        Err(RecorderError::ReceiptOverdue { ticket }) if ticket == admitted
    ));
    let reads = clock.reads.load(Ordering::Relaxed);
    assert!(
        matches!(recorder.admit("refused", at()), Err(RecorderError::ReceiptOverdue { ticket }) if ticket == admitted)
    );
    assert!(
        matches!(recorder.tick(), Err(RecorderError::ReceiptOverdue { ticket }) if ticket == admitted)
    );
    assert_eq!(recorder.receipt_overdue, Some(admitted));
    assert_eq!(
        recorder.failed_seal.as_ref().expect("retained seal").ticket,
        original.ticket
    );
    assert_eq!(recorder.state, State::Finalizing);
    assert_eq!(recorder.pending.len(), 1);
    assert_eq!(starts.get(), 1);
    assert_eq!(saves, 1);
    assert_eq!(recording_deadlines(&recorder), deadlines);
    assert_eq!(clock.reads.load(Ordering::Relaxed), reads);
}
