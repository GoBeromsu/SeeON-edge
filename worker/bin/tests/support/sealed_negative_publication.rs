use super::*;

use seeon_ml_worker::clips::publish::MANIFEST_FILE;
use seeon_ml_worker::clips::reserve::{ENCODER_FAILED, NO_FRAMES};
use seeon_ml_worker::clips::sealed::{ReplayOutcome, ReplayReport};
use seeon_ml_worker::clips::store::ClipStore;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::DeliveryQueue;
use seeon_ml_worker::run::clip_output::publish_recovery;

/// A persisted negative sidecar beside a fresh clip store and queue.
fn published_bench(sealed: &SealedUnavailable) -> (SealedSidecars, ClipStore, DeliveryQueue) {
    let sidecars = sidecars(&sealed.clip_id);
    let queue_dir = work_of(&sidecars).join("store/delivery-queue");
    fs::create_dir_all(&queue_dir).expect("queue dir");
    let queue = DeliveryQueue::open(&queue_dir, true).expect("queue");
    persist(&sidecars, sealed).expect("persist");
    let store = ClipStore::new(work_of(&sidecars).join("store"));
    (sidecars, store, queue)
}

fn now() -> Utc {
    Utc::parse("2026-08-17T10:00:00.000000+00:00").expect("now")
}

#[test]
fn negative_missing_media_is_refused_and_the_sidecar_retained() {
    let sidecars = sidecars("missing-media");
    let path = persist(&sidecars, &negative("neg-missing", 4, 1, false)).expect("persist");
    let bytes = fs::read(&path).expect("bytes");
    let recovery = only_recovery(&sidecars);
    let missing = |_: &Recovery| ReplayOutcome::<(), ()>::MissingMedia;
    refused(
        sidecars.replay_one(&recovery, missing),
        "NegativeMissingMedia",
    );
    refused(sidecars.replay(CAMERA, missing), "NegativeMissingMedia");
    assert_eq!(fs::read(&path).expect("retained"), bytes);
    let report = sidecars.replay(CAMERA, |_| ReplayOutcome::<(), ()>::Published(()));
    let published = ReplayReport {
        published: 1,
        ..ReplayReport::default()
    };
    assert_eq!(report.expect("published"), published);
    assert!(!path.exists(), "publication retires negative evidence");
}

#[test]
fn fresh_unavailable_publication_maps_native_reason_without_media_or_codec() {
    let cases = [
        (0_i32, false, NO_FRAMES),
        (-3, false, ENCODER_FAILED),
        (5, true, ENCODER_FAILED),
    ];
    for (result, video, reason) in cases {
        let clip_id = format!("neg-publish-{}", result.to_string().replace('-', "m"));
        let (sidecars, store, queue) = published_bench(&negative(&clip_id, 2_000, result, video));
        let recovery = only_recovery(&sidecars);
        let bytes = fs::read(&recovery.sidecar_path).expect("bytes");
        let published = publish_recovery(&recovery, &store, &queue, "", now()).expect("publish");
        assert!(published.video_path.is_none() && !published.resumed);
        let manifest: Value = serde_json::from_slice(&published.manifest_bytes).expect("JSON");
        assert_eq!(manifest["state"], json!("UNAVAILABLE"));
        assert_eq!(manifest["reason_code"], json!(reason));
        let entries = queue.entries().expect("entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["unavailable_reason"], json!(reason));
        assert_eq!(entries[0]["local_state"], json!("UNAVAILABLE"));
        assert_eq!(
            fs::read(&recovery.sidecar_path).expect("caller retires"),
            bytes
        );
        let again = publish_recovery(&recovery, &store, &queue, "", now()).expect("resume");
        assert!(again.resumed);
        assert_eq!(again.manifest_bytes, published.manifest_bytes);
        assert_eq!(queue.entries().expect("entries again"), entries);
    }
}

#[test]
fn zero_or_unrepresentable_duration_refuses_publication_and_retains_the_sidecar() {
    let cases = [
        (0, "NonPositiveDuration"),
        (u64::MAX, "Reservation(\"duration\")"),
    ];
    for (duration, variant) in cases {
        let clip_id = format!("neg-duration-{duration}");
        let (sidecars, store, queue) = published_bench(&negative(&clip_id, duration, 0, false));
        let recovery = only_recovery(&sidecars);
        let bytes = fs::read(&recovery.sidecar_path).expect("bytes");
        let report = sidecars.replay(CAMERA, |recovery| {
            let error = publish_recovery(recovery, &store, &queue, "", now()).expect_err("refused");
            assert!(
                format!("{error:?}").contains(variant),
                "{variant}: {error:?}"
            );
            ReplayOutcome::<(), _>::Failed(error)
        });
        let failed = ReplayReport {
            failed: 1,
            ..ReplayReport::default()
        };
        assert_eq!(report.expect("replay"), failed);
        assert!(queue.entries().expect("entries").is_empty());
        assert!(!store.clip_dir(&clip_id).join(MANIFEST_FILE).exists());
        assert_eq!(fs::read(&recovery.sidecar_path).expect("retained"), bytes);
    }
}
