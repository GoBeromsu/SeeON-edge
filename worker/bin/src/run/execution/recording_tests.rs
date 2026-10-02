use super::*;
use crate::clips::manifest::Contributor;
use crate::clips::recorder::Boundary;
use crate::seam::{IdSource, RandomIds};
use seeon_deepstream_native::RecordTicket;
use std::path::PathBuf;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("owned recording fixture cleanup");
    }
}

#[test]
fn media_receipt_accepts_regular_owned_file_but_not_traversal_or_symlink() {
    let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
    std::fs::create_dir(&root.0).unwrap();
    let media = root.0.join("owned.mp4");
    std::fs::write(&media, b"path admission only, not encoded media").unwrap();
    let mut receipt = crate::msg::RecordReceipt {
        ticket: RecordTicket::default(),
        result: MediaResult::Ok,
        error: 0,
        duration_ms: 30_000,
        width: 1280,
        height: 720,
        contains_video: true,
        contains_audio: false,
        directory: root.0.clone(),
        filename: "owned.mp4".into(),
    };
    assert!(admits_path(&root.0, &receipt));
    receipt.filename = "../owned.mp4".into();
    assert!(!admits_path(&root.0, &receipt));
    std::os::unix::fs::symlink(&media, root.0.join("link.mp4")).unwrap();
    receipt.filename = "link.mp4".into();
    assert!(!admits_path(&root.0, &receipt));
}

#[test]
fn failed_codec_probe_retains_original_media_and_durable_attribution() {
    let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
    std::fs::create_dir(&root.0).unwrap();
    let store = ClipStore::new(root.0.join("store"));
    let mut reserve = ReservePool::arm(&store, 1).unwrap();
    let queue = DeliveryQueue::open(&root.0.join("queue"), true).unwrap();
    let sidecars = SealedSidecars::new(root.0.join(crate::clips::sealed::SIDECAR_DIR));
    let source = root.0.join("invalid-media.mp4");
    // Invalid encoded bytes deliberately exercise failure, not inference or encoding success.
    std::fs::write(&source, b"invalid encoded media").unwrap();
    let event_ref = "00000000-0000-4000-8000-000000000001";
    let event = SealedEvent {
        domain: "fall".into(),
        event_type: "FALL_DETECTED".into(),
        identity: event_ref.into(),
        camera_id: "camera-real".into(),
        facility_id: "facility-real".into(),
        time_sec: 12.0,
        probability: None,
    };
    let events = BTreeMap::from([(event_ref.to_owned(), event.clone())]);
    let detected_at = Utc::parse("2026-08-20T17:20:58.197192Z").unwrap();
    let sealed = ClipSealed {
        ticket: RecordTicket::default(),
        result: MediaResult::Ok,
        contains_video: true,
        duration_ms: 30_000,
        boundary: Boundary::ExtensionRaced,
        contributors: vec![Contributor {
            event_ref: event_ref.into(),
            detected_at,
        }],
        path: source.clone(),
    };
    let result = RecordingPublisher {
        reserve: &mut reserve,
        store: &store,
        queue: &queue,
        sidecars: &sidecars,
        events: &events,
    }
    .save("boot-camera-session", &sealed, detected_at);
    assert!(result.is_err());
    assert_eq!(std::fs::read(&source).unwrap(), b"invalid encoded media");
    assert!(
        !store
            .clip_dir("boot-camera-session")
            .join("manifest.json")
            .exists()
    );
    let pending = sidecars.pending("camera-real").unwrap();
    assert!(pending.malformed.is_empty());
    assert_eq!(pending.recoveries.len(), 1);
    let recovered = &pending.recoveries[0];
    assert_eq!(recovered.events[event_ref], event);
    assert_eq!(recovered.sealed.clip_id, "boot-camera-session");
    assert_eq!(recovered.sealed.path, source.to_str().unwrap());
    assert_eq!(recovered.sealed.boundary, "extension_raced");
    assert_eq!(
        recovered.sealed.contributors[0].detected_at,
        detected_at.iso_micros()
    );
    let staged = store
        .reserve("camera-real", "boot-camera-session")
        .unwrap()
        .artifact_path();
    std::fs::rename(&source, &staged).unwrap();
    let replay = super::super::publication::replay_before_media(
        &root.0,
        &["camera-real".into()],
        &store,
        &queue,
        &crate::seam::SystemClock::new(),
    )
    .unwrap();
    assert_eq!(replay.reports[0].failed, 1);
    assert_eq!(replay.reports[0].missing_media, 0);
    assert_eq!(std::fs::read(staged).unwrap(), b"invalid encoded media");
    assert_eq!(sidecars.pending("camera-real").unwrap().recoveries.len(), 1);
}
