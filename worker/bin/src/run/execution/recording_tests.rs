use super::*;
use crate::clips::publish::TERMINAL_MARKER;
use crate::clips::reserve::FINALIZE_FAILED;
use crate::seam::{IdSource, RandomIds};
use seeon_deepstream_native::RecordTicket;
use support::{CLIP_ID, EVENT_REF, INVALID_MEDIA, SaveFixture, Scratch, assert_absent};

#[path = "recording_failure_tests.rs"]
mod failure_tests;
#[path = "recording_test_support.rs"]
mod support;

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
fn failed_codec_probe_publishes_unavailable_and_retires_durable_attribution() {
    let mut fixture = SaveFixture::new(30_000);
    assert!(
        std::process::Command::new(Tools::default().ffprobe)
            .arg("-version")
            .output()
            .expect("actual ffprobe must run")
            .status
            .success()
    );
    assert!(measured_codec(&fixture.sealed.path).is_err());
    let slots = fixture.reserve.available();
    let saved = fixture.save();
    let Ok(SaveOutcome::Saved(published)) = saved else {
        panic!("codec refusal must complete FINALIZE_FAILED publication: {saved:?}");
    };
    assert!(published.video_path.is_none());
    assert!(published.admitted && !published.resumed);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&published.manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["state"], "UNAVAILABLE");
    assert_eq!(manifest["reason_code"], FINALIZE_FAILED);
    assert_eq!(manifest["video_available"], false);
    assert_eq!(manifest["event_refs"], serde_json::json!([EVENT_REF]));
    assert_eq!(manifest["finalized_at"], fixture.now.iso_millis());
    assert!(manifest["path"].is_null());
    for key in ["codec", "sha256", "size_bytes", "mime_type", "duration_ms"] {
        assert!(
            manifest.get(key).is_none(),
            "unavailable has no {key} claim"
        );
    }
    let entry: serde_json::Value = serde_json::from_slice(
        &crate::delivery::DeliveryEntry::from(published.entry)
            .to_bytes()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(entry["local_state"], "UNAVAILABLE");
    assert_eq!(entry["unavailable_reason"], FINALIZE_FAILED);
    assert_eq!(fixture.queue.entries().unwrap(), vec![entry]);
    assert!(
        fixture
            .store
            .clip_dir(CLIP_ID)
            .join(TERMINAL_MARKER)
            .is_file()
    );
    assert_absent(&fixture.sidecars.directory().join(format!("{CLIP_ID}.json")));
    assert_eq!(
        fixture
            .sidecars
            .pending("camera-real")
            .unwrap()
            .recoveries
            .len(),
        0
    );
    assert_eq!(std::fs::read(&fixture.sealed.path).unwrap(), INVALID_MEDIA);
    assert_eq!(fixture.reserve.available(), slots);
}

#[test]
fn zero_duration_ready_retains_observation_before_metadata_refusal() {
    let mut fixture = SaveFixture::new(0);
    assert!(matches!(
        fixture.save(),
        Err(PublishError::Reservation("sealed metadata"))
    ));
    assert!(fixture.queue.entries().unwrap().is_empty());
    fixture.assert_retained();
    fixture.assert_no_terminal();
    let staged = fixture
        .store
        .reserve("camera-real", CLIP_ID)
        .unwrap()
        .artifact_path();
    std::fs::rename(&fixture.sealed.path, &staged).unwrap();
    let replay = super::super::publication::replay_before_media(
        &fixture.root.0,
        &["camera-real".into()],
        &fixture.store,
        &fixture.queue,
        &crate::seam::SystemClock::new(),
    )
    .unwrap();
    assert_eq!(replay.reports[0].failed, 1);
    assert_eq!(replay.reports[0].missing_media, 0);
    assert_eq!(std::fs::read(staged).unwrap(), INVALID_MEDIA);
    assert_eq!(
        fixture
            .sidecars
            .pending("camera-real")
            .unwrap()
            .recoveries
            .len(),
        1
    );
}
