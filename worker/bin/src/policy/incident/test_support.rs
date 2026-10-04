use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use seeon_worker::episode::{BusinessEvent, EpisodeAuthority, EpisodeProposal};

use super::super::identity::Limits;
use super::{AdmittedIncident, CooldownKey, IncidentAuditSnapshot, IncidentManager, state};
use crate::seam::{Clock, IdSource};

pub(super) struct TestClock {
    pub(super) wall_seconds: AtomicI64,
    pub(super) monotonic_samples: AtomicUsize,
}

impl Clock for TestClock {
    fn monotonic(&self) -> Duration {
        self.monotonic_samples.fetch_add(1, Ordering::SeqCst);
        Duration::ZERO
    }

    fn wall(&self) -> SystemTime {
        let seconds = self.wall_seconds.load(Ordering::SeqCst);
        let duration = Duration::from_secs(seconds.unsigned_abs());
        if seconds < 0 {
            UNIX_EPOCH - duration
        } else {
            UNIX_EPOCH + duration
        }
    }

    fn pause(&self, _: Duration) {}
}

#[derive(Default)]
pub(super) struct TestIds {
    pub(super) attempts: AtomicUsize,
    pub(super) fail: AtomicBool,
    pub(super) invalid: AtomicBool,
}

impl IdSource for TestIds {
    fn uuid4(&self) -> io::Result<String> {
        let number = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail.load(Ordering::SeqCst) {
            return Err(io::Error::other("test UUID allocation refusal"));
        }
        if self.invalid.load(Ordering::SeqCst) {
            return Ok("not-a-uuid".into());
        }
        Ok(format!("00000000-0000-4000-8000-{number:012x}"))
    }
}

pub(super) fn setup(max_bytes: usize) -> (IncidentManager, Arc<TestClock>, Arc<TestIds>) {
    let clock = Arc::new(TestClock {
        wall_seconds: AtomicI64::new(100),
        monotonic_samples: AtomicUsize::new(0),
    });
    let ids = Arc::new(TestIds::default());
    let limits = Limits {
        max_bytes,
        ..Limits::default()
    };
    let manager = IncidentManager::new(clock.clone(), ids.clone(), limits).unwrap();
    (manager, clock, ids)
}

/// Actual episode owner output, not a delivery-shaped fabricated identity.
pub(super) fn event() -> BusinessEvent {
    let mut authority = EpisodeAuthority::new("boot", "epoch", 7, 1, 4).unwrap();
    let proposal = EpisodeProposal {
        camera_id: "camera".into(),
        facility_id: "facility".into(),
        event_type: "fall".into(),
        track_id: 7,
        bed_id: None,
        frame_index: 100,
        time_sec: 10.0,
        qualifying: true,
        confirmed_recovery: false,
        probability: Some(0.8),
        domain: None,
        generation: 11,
        confirmation_votes: 1,
        confirmation_window: 4,
    };
    authority
        .propose(&proposal)
        .unwrap()
        .expect("genuine onset")
}

pub(super) fn charge(source: &BusinessEvent, now: Duration) -> usize {
    state::encoded_charge(&CooldownKey::from(source), now).unwrap()
}

pub(super) fn attempts(ids: &TestIds) -> usize {
    ids.attempts.load(Ordering::SeqCst)
}

pub(super) fn assert_preserved(source: &BusinessEvent, admission: &AdmittedIncident) {
    let mut restored = admission.event.clone();
    restored.identity = source.identity.clone();
    assert_eq!(&restored, source);
    let expected = IncidentAuditSnapshot {
        edge_event_id: admission.event.identity.clone(),
        source_identity: source.identity.clone(),
        cooldown_key: CooldownKey::from(source),
        domain: source.domain.clone(),
        event_type: source.event_type.clone(),
        camera: source.camera_id.clone(),
        facility: source.facility_id.clone(),
        time_sec: source.time_sec,
        probability: source.probability,
        person_id: source.person_id,
        bed_id: source.bed_id,
    };
    assert_eq!(admission.audit, expected);
}

pub(super) fn assert_uuid4(text: &str) {
    let bytes = text.as_bytes();
    assert_eq!(bytes.len(), 36);
    for (index, byte) in bytes.iter().enumerate() {
        if [8, 13, 18, 23].contains(&index) {
            assert_eq!(*byte, b'-');
        } else {
            assert!(byte.is_ascii_digit() || (b'a'..=b'f').contains(byte));
        }
    }
    assert_eq!(bytes[14], b'4');
    assert!((b'8'..=b'b').contains(&bytes[19]));
}
