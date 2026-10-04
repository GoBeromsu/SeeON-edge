use super::super::RecordingPublisher;
use crate::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use crate::clips::manifest::{ClipMetadata, Contributor};
use crate::clips::publish::{MANIFEST_FILE, PublishError, TERMINAL_MARKER};
use crate::clips::recorder::{Boundary, ClipSealed};
use crate::clips::reserve::{ReservePool, SaveOutcome};
use crate::clips::sealed::{SealedClip, SealedContributor, SealedEvent, SealedSidecars};
use crate::clips::store::ClipStore;
use crate::clips::time::Utc;
use crate::delivery::DeliveryQueue;
use crate::seam::{IdSource, RandomIds};
use seeon_deepstream_native::{MediaResult, RecordTicket};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(super) const CLIP_ID: &str = "boot-camera-session";
pub(super) const EVENT_REF: &str = "00000000-0000-4000-8000-000000000001";
pub(super) const INVALID_MEDIA: &[u8] = b"invalid encoded media";

pub(super) struct Scratch(pub(super) PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("owned recording fixture cleanup");
    }
}

pub(super) struct SaveFixture {
    pub(super) root: Scratch,
    pub(super) store: ClipStore,
    pub(super) reserve: ReservePool,
    pub(super) queue: DeliveryQueue,
    pub(super) sidecars: SealedSidecars,
    events: BTreeMap<String, SealedEvent>,
    pub(super) sealed: ClipSealed,
    pub(super) now: Utc,
}

impl SaveFixture {
    pub(super) fn new(duration_ms: u64) -> Self {
        let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
        std::fs::create_dir(&root.0).unwrap();
        let store = ClipStore::new(root.0.join("store"));
        let reserve = ReservePool::arm(&store, 1).unwrap();
        let queue = DeliveryQueue::open(&root.0.join("queue"), true).unwrap();
        let sidecars = SealedSidecars::new(root.0.join(crate::clips::sealed::SIDECAR_DIR));
        let source = root.0.join("invalid-media.mp4");
        std::fs::write(&source, INVALID_MEDIA).unwrap();
        let event = SealedEvent {
            domain: "fall".into(),
            event_type: "FALL_DETECTED".into(),
            identity: EVENT_REF.into(),
            camera_id: "camera-real".into(),
            facility_id: "facility-real".into(),
            time_sec: 12.0,
            probability: None,
        };
        let detected_at = Utc::parse("2026-08-20T17:20:58.197192Z").unwrap();
        Self {
            root,
            store,
            reserve,
            queue,
            sidecars,
            events: BTreeMap::from([(EVENT_REF.to_owned(), event)]),
            sealed: ClipSealed {
                ticket: RecordTicket::default(),
                result: MediaResult::Ok,
                contains_video: true,
                duration_ms,
                boundary: Boundary::ExtensionRaced,
                contributors: vec![Contributor {
                    event_ref: EVENT_REF.into(),
                    detected_at,
                }],
                path: source,
            },
            now: detected_at.plus_millis(60_000),
        }
    }

    pub(super) fn save(&mut self) -> Result<SaveOutcome, PublishError> {
        RecordingPublisher {
            reserve: &mut self.reserve,
            store: &self.store,
            queue: &self.queue,
            sidecars: &self.sidecars,
            events: &self.events,
        }
        .save(CLIP_ID, &self.sealed, self.now)
    }

    fn observation(&self) -> SealedClip {
        SealedClip {
            clip_id: CLIP_ID.into(),
            path: self.sealed.path.to_str().unwrap().into(),
            duration_ms: i64::try_from(self.sealed.duration_ms).unwrap(),
            boundary: "extension_raced".into(),
            contributors: vec![SealedContributor {
                event_ref: EVENT_REF.into(),
                detected_at: self.sealed.contributors[0].detected_at.iso_micros(),
            }],
        }
    }

    pub(super) fn persist_observation(&self) -> PathBuf {
        self.sidecars
            .persist(&self.observation(), &self.events)
            .unwrap()
    }

    pub(super) fn metadata(&self) -> ClipMetadata {
        let event = &self.events[EVENT_REF];
        let events = BTreeMap::from([(
            EVENT_REF.into(),
            ContributorEvent {
                camera_id: event.camera_id.clone(),
                facility_id: event.facility_id.clone(),
                domain: event.domain.clone(),
                event_type: event.event_type.clone(),
            },
        )]);
        flow_metadata(
            CLIP_ID,
            &events,
            self.sealed.extension(),
            FLOW_ENCODER,
            self.now,
        )
        .unwrap()
    }

    pub(super) fn assert_retained(&self) {
        assert_eq!(std::fs::read(&self.sealed.path).unwrap(), INVALID_MEDIA);
        let pending = self.sidecars.pending("camera-real").unwrap();
        assert!(pending.malformed.is_empty());
        assert_eq!(pending.recoveries.len(), 1);
        assert_eq!(pending.recoveries[0].sealed, self.observation());
        assert_eq!(pending.recoveries[0].events, self.events);
    }

    pub(super) fn assert_no_terminal(&self) {
        for name in [MANIFEST_FILE, TERMINAL_MARKER] {
            assert_absent(&self.store.clip_dir(CLIP_ID).join(name));
        }
    }
}

pub(super) fn assert_absent(path: &Path) {
    assert_eq!(
        path.symlink_metadata().unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
}
