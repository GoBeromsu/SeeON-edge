//! Unit-only recording replies and receipts over the real publication channels.
//! No SDK owner, encoded media, or production incident is exercised.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use seeon_deepstream_native::{MediaBinding, MediaPoll, MediaResult, RecordTicket};
use seeon_worker::episode::BusinessEvent;

use super::{LiveSink, PendingEvent, apply_sink};
use crate::clips::recorder::{FORWARD_SECONDS, LOOKBACK_SECONDS, State};
use crate::media::Command;
use crate::msg::RecordReceipt;
use crate::records::lanes::Lanes;
use crate::run::execution::{output, publication};
use crate::run::pump::{CameraPolicy, PolicyPump};
use crate::seam::{Clock, IdSource, RandomIds};

pub(super) const BOOT: &str = "00000000-0000-4000-8000-000000000148";
pub(super) const FIRST: &str = "00000000-0000-4000-8000-0000000000b1";
pub(super) const NEXT: &str = "00000000-0000-4000-8000-0000000000b2";
pub(super) const SUFFIX: &str = "00000000-0000-4000-8000-0000000000b3";
pub(super) const WAITING: &str = "00000000-0000-4000-8000-0000000000b4";
const SOURCE: u32 = 19;
pub(super) const BINDING: MediaBinding = MediaBinding {
    token: 11,
    generation: 3,
    epoch: 5,
};

#[derive(Default)]
pub(super) struct TestClock(Mutex<Duration>, Mutex<Option<u64>>);

impl TestClock {
    pub(super) fn at(&self, seconds: u64) {
        *self.0.lock().unwrap() = Duration::from_secs(seconds);
    }
    pub(super) fn jump_on_wall(&self, seconds: u64) {
        *self.1.lock().unwrap() = Some(seconds);
    }
}

impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        *self.0.lock().unwrap()
    }
    fn wall(&self) -> SystemTime {
        if let Some(seconds) = self.1.lock().unwrap().take() {
            self.at(seconds);
        }
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000) + self.monotonic()
    }
    fn pause(&self, limit: Duration) {
        *self.0.lock().unwrap() += limit;
    }
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) struct Fixture {
    pub(super) session: output::Session,
    pub(super) clock: Arc<TestClock>,
    pub(super) receipts: SyncSender<RecordReceipt>,
    pub(super) lanes: Arc<Lanes>,
    pub(super) commands: Option<Receiver<Command>>,
    _root: Scratch,
}

impl Fixture {
    pub(super) fn new() -> Self {
        Self::open(&[SOURCE])
    }
    pub(super) fn two() -> Self {
        Self::open(&[SOURCE, 23])
    }
    fn open(sources: &[u32]) -> Self {
        let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
        std::fs::create_dir(&root.0).unwrap();
        let clock = Arc::new(TestClock::default());
        let cameras: Vec<_> = sources
            .iter()
            .map(|&source| {
                (
                    source,
                    binding(source),
                    format!("camera-{source}"),
                    format!("facility-{source}"),
                )
            })
            .collect();
        let (commands, command_rx, receipts, records) = publication::channels();
        let publications = publication::open(
            publication::PublicationConfig {
                boot_id: BOOT,
                state_dir: &root.0.join("state"),
                record_dir: &root.0.join("records"),
                store_root: &root.0.join("store"),
                cameras: &cameras,
                config_version: 1,
                manifest_sha: None,
            },
            commands,
            records,
            clock.clone(),
        )
        .unwrap();
        let pump = PolicyPump::new(
            sources
                .iter()
                .map(|&source_id| CameraPolicy {
                    source_id,
                    stage: None,
                })
                .collect(),
            None,
            clock.clone(),
        )
        .unwrap();
        let mut session = output::tests::session(pump, publications, BOOT.to_owned());
        let lanes = Arc::new(Lanes::new(8).unwrap());
        session.records = Some(Arc::clone(&lanes));
        Self {
            session,
            clock,
            receipts,
            lanes,
            commands: Some(command_rx),
            _root: root,
        }
    }

    pub(super) fn start(&mut self, sequence: u64, identity: &str, request: u64) -> RecordTicket {
        self.start_on(SOURCE, sequence, identity, request)
    }
    pub(super) fn start_on(
        &mut self,
        source: u32,
        sequence: u64,
        identity: &str,
        request: u64,
    ) -> RecordTicket {
        let mut held = super::super::sink();
        held.triggered.push(event_on(source, sequence, identity));
        let ticket = self.apply_with_start(&mut held, request);
        assert!(held.triggered.is_empty());
        ticket
    }

    pub(super) fn apply_with_start(&mut self, held: &mut LiveSink, request: u64) -> RecordTicket {
        let ticket = RecordTicket {
            binding: held.triggered[0].frame.binding,
            request_id: request,
            source_id: held.triggered[0].frame.source_id,
            session_id: u32::MAX,
            session_valid: 0,
            coalesced: 0,
        };
        let inbox = self.commands.take().unwrap();
        // One bounded unit reply, not a native admission or lifetime proof.
        let worker = std::thread::spawn(move || {
            let Command::RecordStart {
                source_id,
                binding,
                lookback_seconds,
                forward_seconds,
                reply,
            } = inbox.recv_timeout(Duration::from_secs(5)).unwrap()
            else {
                panic!("expected one unit RecordStart");
            };
            assert_eq!((source_id, binding), (ticket.source_id, ticket.binding));
            assert_eq!(
                (lookback_seconds, forward_seconds),
                (LOOKBACK_SECONDS, FORWARD_SECONDS)
            );
            reply.send(Ok(MediaPoll::Ready(ticket))).unwrap();
            inbox
        });
        let result = apply_sink(&mut self.session, self.clock.as_ref(), held);
        self.commands = Some(worker.join().unwrap());
        result.unwrap();
        let index = self
            .session
            .publications
            .stager_index(&format!("camera-{}", ticket.source_id))
            .unwrap();
        assert_eq!(
            self.session.publications.recorders[index].state(),
            State::Recording
        );
        ticket
    }

    pub(super) fn receipt(&self, admitted: RecordTicket, result: MediaResult) -> RecordReceipt {
        RecordReceipt {
            ticket: RecordTicket {
                session_id: 47,
                session_valid: 1,
                ..admitted
            },
            result,
            error: if result == MediaResult::Ok { 0 } else { -1 },
            // Unit-constructed positive duration exercises metadata, not media validity.
            duration_ms: 30_000,
            width: 0,
            height: 0,
            contains_video: false,
            contains_audio: false,
            directory: self.session.publications.record_dir.clone(),
            filename: "unit-constructed-no-media.mp4".into(),
        }
    }
}

pub(super) fn event(sequence: u64, identity: &str) -> PendingEvent {
    event_on(SOURCE, sequence, identity)
}
fn binding(source: u32) -> MediaBinding {
    MediaBinding {
        token: u64::from(source) - 8,
        ..BINDING
    }
}
pub(super) fn event_on(source: u32, sequence: u64, identity: &str) -> PendingEvent {
    let mut frame = super::super::packet(source, sequence, &[1]).frame;
    frame.binding = binding(source);
    PendingEvent::held_admitted(
        frame,
        BusinessEvent {
            domain: "fall".into(),
            event_type: "fall".into(),
            identity: identity.into(),
            camera_id: format!("camera-{source}"),
            facility_id: format!("facility-{source}"),
            time_sec: 1.0,
            probability: Some(0.5),
            person_id: Some(7),
            bed_id: None,
        },
    )
}
