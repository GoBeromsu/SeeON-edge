use super::super::Publications;
use crate::clips::recorder::Admit;
use crate::media::Command;
use crate::records::builder::{Frame, Stream};
use crate::run::events::StagedEvent;
use crate::seam::Clock;
use seeon_deepstream_native::{MediaBinding, MediaPoll, RecordTicket};
use seeon_worker::episode::BusinessEvent;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(super) const BOOT_ID: &str = "00000000-0000-4000-8000-000000000099";
pub(super) const CAMERA_ID: &str = "00000000-0000-4000-8000-000000000003";
pub(super) const FACILITY_ID: &str = "00000000-0000-4000-8000-000000000004";
pub(super) const INVALID_MEDIA: &[u8] = b"malformed encoded recording, not successful media";

pub(super) struct Scratch(pub(super) PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("owned CPU publication fixture cleanup");
    }
}

pub(super) struct TestClock(pub(super) AtomicU64);
impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        Duration::ZERO
    }
    fn wall(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_micros(self.0.load(Ordering::SeqCst))
    }
    fn pause(&self, _: Duration) {
        panic!("CPU publication must not wait through the clock");
    }
}

pub(super) fn staged_event(
    output: &mut Publications,
    identity: &str,
) -> (BusinessEvent, StagedEvent) {
    let source = BusinessEvent {
        domain: "fall".into(),
        event_type: "FALL_DETECTED".into(),
        identity: identity.into(),
        camera_id: CAMERA_ID.into(),
        facility_id: FACILITY_ID.into(),
        time_sec: 12.0,
        probability: Some(0.9),
        person_id: Some(7),
        bed_id: None,
    };
    let cutoff = output.clock.monotonic();
    let event = output.admit(0, &source, cutoff).unwrap().unwrap().event;
    assert_ne!(
        event.identity, source.identity,
        "incident admission mints the UUID"
    );
    let mut prepared = output
        .prepare(
            0,
            &event,
            &Stream {
                camera_id: CAMERA_ID.into(),
                worker_boot_id: BOOT_ID.into(),
                source_generation: 3,
                stream_epoch: 5,
            },
            Frame {
                frame_seq: 90,
                source_pts_ns: Some(12_000_000_000),
            },
            None,
        )
        .unwrap();
    let staged = output.stage(0, &mut prepared, &mut |_| {}).unwrap();
    assert!(staged.admission.accepted);
    assert_eq!(staged.event_ref, event.identity);
    (event, staged)
}

pub(super) fn start(
    output: &mut Publications,
    inbox: &mut Receiver<Command>,
    event: &BusinessEvent,
    staged: &StagedEvent,
    request_id: u64,
) -> RecordTicket {
    std::thread::scope(|scope| {
        let media = scope.spawn(move || {
            let Command::RecordStart {
                source_id,
                binding,
                reply,
                ..
            } = inbox.recv_timeout(Duration::from_secs(5)).unwrap()
            else {
                panic!("expected the recorder's RecordStart command");
            };
            assert_eq!(source_id, 3);
            assert_eq!(
                binding,
                MediaBinding {
                    token: 10,
                    generation: 3,
                    epoch: 5
                }
            );
            let ticket = RecordTicket {
                source_id,
                binding,
                request_id,
                session_id: 40 + u32::try_from(request_id).unwrap(),
                session_valid: 1,
                coalesced: 0,
            };
            reply.send(Ok(MediaPoll::Ready(ticket))).unwrap();
            ticket
        });
        let cutoff = output.clock.monotonic();
        let admitted = output.admit_recording(0, event, staged, cutoff).unwrap();
        let ticket = media.join().unwrap();
        assert_eq!(admitted, Admit::Started(ticket));
        ticket
    })
}
