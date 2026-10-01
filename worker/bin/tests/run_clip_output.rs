//! Ready sealed-replay composition against a real clip store and delivery queue.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use seeon_ml_worker::clips::publish::{MANIFEST_FILE, MEDIA_FILE, PublishError};
use seeon_ml_worker::clips::sealed::{Recovery, SealedClip, SealedContributor, SealedEvent};
use seeon_ml_worker::clips::store::ClipStore;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::DeliveryQueue;
use seeon_ml_worker::run::clip_output::{ClipOutputError, publish_recovery};

const MEDIA: &[u8] = b"synthetic sealed recovery media";
const CODEC: &str = "h264";
const CAMERA: &str = "cam-1";
const FACILITY: &str = "fac-1";
const CLIP: &str = "clip-1";
const REF: &str = "evt-1";
const DETECTED: &str = "2026-01-02T03:04:05.123456Z";

fn work(name: &str) -> PathBuf {
    let parent = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&parent).expect("test parent");
    let path = parent.join(format!("run_clip_output-{}-{name}", std::process::id()));
    fs::create_dir(&path).expect("unique work dir");
    path
}

struct Bench {
    root: PathBuf,
    source: PathBuf,
    sidecar: PathBuf,
    store: ClipStore,
    queue: DeliveryQueue,
}

impl Drop for Bench {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

impl Bench {
    fn new(name: &str) -> Self {
        let root = work(name);
        let source = root.join("source.mp4");
        fs::write(&source, MEDIA).expect("source media");
        let sidecar = root.join("flow-sealed").join(format!("{CLIP}.json"));
        fs::create_dir_all(sidecar.parent().expect("sidecar parent")).expect("sidecar dir");
        fs::write(&sidecar, b"{\"kept\":true}\n").expect("sidecar");
        let store_root = root.join("store");
        fs::create_dir_all(&store_root).expect("store root");
        let queue = DeliveryQueue::open(&root.join("delivery-queue"), true).expect("queue");
        Self {
            root,
            source,
            sidecar,
            store: ClipStore::new(store_root),
            queue,
        }
    }

    fn recovery(&self) -> Recovery {
        recovery_at(&self.source, &self.sidecar, CAMERA, FACILITY, REF)
    }
}

fn recovery_at(
    source: &Path,
    sidecar: &Path,
    camera: &str,
    facility: &str,
    event_ref: &str,
) -> Recovery {
    let mut events = BTreeMap::new();
    events.insert(
        event_ref.to_owned(),
        SealedEvent {
            domain: "fall".to_owned(),
            event_type: "fall_detected".to_owned(),
            identity: event_ref.to_owned(),
            camera_id: camera.to_owned(),
            facility_id: facility.to_owned(),
            time_sec: 12.5,
            probability: 0.9,
        },
    );
    Recovery {
        sealed: SealedClip {
            clip_id: CLIP.to_owned(),
            path: source.display().to_string(),
            duration_ms: 4_000,
            boundary: "end".to_owned(),
            contributors: vec![SealedContributor {
                event_ref: event_ref.to_owned(),
                detected_at: DETECTED.to_owned(),
            }],
        },
        events,
        camera_id: CAMERA.to_owned(),
        sidecar_path: sidecar.to_path_buf(),
    }
}

fn now() -> Utc {
    Utc::parse("2026-01-02T03:05:00.000000Z").expect("clock")
}

fn later() -> Utc {
    Utc::parse("2026-01-02T04:00:00.000000Z").expect("later clock")
}

fn publish(
    bench: &Bench,
    recovery: &Recovery,
    clock: Utc,
) -> Result<seeon_ml_worker::clips::publish::Published, ClipOutputError> {
    publish_recovery(recovery, &bench.store, &bench.queue, CODEC, clock)
}

#[test]
fn publication_before_clip_end_keeps_the_end_as_finalized_at() {
    let bench = Bench::new("early-clock");
    let recovery = bench.recovery();
    let early = Utc::parse("2026-01-02T03:03:40.000000Z").expect("before clip end");
    let published = publish(&bench, &recovery, early).expect("publish");
    assert_eq!(
        published.entry.fields().finalized_at.as_deref(),
        Some("2026-01-02T03:03:54.123Z")
    );
    sidecar_kept(&bench);
}

fn manifest_bytes(bench: &Bench) -> Vec<u8> {
    fs::read(bench.store.clip_dir(CLIP).join(MANIFEST_FILE)).expect("manifest")
}

fn media_bytes(bench: &Bench) -> Vec<u8> {
    fs::read(bench.store.clip_dir(CLIP).join(MEDIA_FILE)).expect("final media")
}

fn sidecar_kept(bench: &Bench) {
    assert_eq!(
        fs::read(&bench.sidecar).expect("sidecar"),
        b"{\"kept\":true}\n"
    );
}

#[test]
fn successful_replay_publishes_actual_bytes_and_one_queue_entry() {
    let bench = Bench::new("success");
    let recovery = bench.recovery();
    let published = publish(&bench, &recovery, now()).expect("publish");
    assert!(!published.resumed);
    assert!(published.admitted);
    assert_eq!(media_bytes(&bench), MEDIA);
    assert_eq!(published.manifest_bytes, manifest_bytes(&bench));
    assert_eq!(published.entry.fields().codec.as_deref(), Some(CODEC));
    assert_eq!(published.entry.fields().camera_id, CAMERA);
    assert_eq!(published.entry.fields().facility_id, FACILITY);
    assert_eq!(published.entry.fields().duration_ms, Some(4_000));
    assert_eq!(
        published.entry.fields().size_bytes,
        Some(i64::try_from(MEDIA.len()).expect("size"))
    );
    let queued = bench.queue.entries().expect("entries");
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0]["codec"].as_str(), Some(CODEC));
    assert_eq!(queued[0]["camera_id"].as_str(), Some(CAMERA));
    assert_eq!(
        queued[0]["sha256"].as_str(),
        published.entry.fields().sha256.as_deref()
    );
    assert!(!bench.source.exists(), "source moved into the store");
    sidecar_kept(&bench);
}

#[test]
fn repeated_publication_after_the_clock_advances_resumes_without_a_duplicate() {
    let bench = Bench::new("resume");
    let recovery = bench.recovery();
    let first = publish(&bench, &recovery, now()).expect("first");
    let second = publish(&bench, &recovery, later()).expect("second");
    assert!(second.resumed);
    assert!(!second.admitted);
    assert_eq!(second.manifest_bytes, first.manifest_bytes);
    assert_eq!(bench.queue.entries().expect("entries").len(), 1);
    assert_eq!(media_bytes(&bench), MEDIA);
    sidecar_kept(&bench);
}

#[test]
fn staged_media_without_the_source_still_publishes() {
    let bench = Bench::new("staged");
    let recovery = bench.recovery();
    let reservation = bench.store.reserve(CAMERA, CLIP).expect("reserve");
    fs::write(reservation.artifact_path(), MEDIA).expect("staged artifact");
    fs::remove_file(&bench.source).expect("source gone");
    let published = publish(&bench, &recovery, now()).expect("publish staged");
    assert!(!published.resumed);
    assert_eq!(media_bytes(&bench), MEDIA);
    assert!(!reservation.artifact_path().exists());
    sidecar_kept(&bench);
}

#[test]
fn a_tampered_manifest_cannot_be_blessed_or_overwritten() {
    let bench = Bench::new("tamper");
    let recovery = bench.recovery();
    publish(&bench, &recovery, now()).expect("original");
    let manifest = bench.store.clip_dir(CLIP).join(MANIFEST_FILE);
    let original = fs::read(&manifest).expect("original manifest");
    let swapped = original
        .iter()
        .map(|byte| if *byte == b'1' { b'9' } else { *byte })
        .collect::<Vec<_>>();
    assert_ne!(swapped, original);
    fs::write(&manifest, &swapped).expect("tamper identity bytes");
    let refused = publish(&bench, &recovery, later());
    assert!(matches!(
        refused,
        Err(ClipOutputError::Publish(PublishError::Conflict))
            | Err(ClipOutputError::ManifestUnreadable)
    ));
    assert_eq!(fs::read(&manifest).expect("manifest kept"), swapped);
    assert_eq!(bench.queue.entries().expect("entries").len(), 1);
    assert_eq!(media_bytes(&bench), MEDIA);
    sidecar_kept(&bench);
}

#[test]
fn missing_contributor_cross_camera_and_facility_are_refused() {
    let bench = Bench::new("identity");
    let mut missing = bench.recovery();
    missing.events.clear();
    assert!(matches!(
        publish(&bench, &missing, now()),
        Err(ClipOutputError::Metadata(_))
    ));

    let mut crossed = bench.recovery();
    crossed.events.get_mut(REF).expect("event").camera_id = "cam-2".to_owned();
    assert!(matches!(
        publish(&bench, &crossed, now()),
        Err(ClipOutputError::CameraMismatch)
    ));

    let mut facility = bench.recovery();
    facility.events.insert(
        "evt-2".to_owned(),
        SealedEvent {
            domain: "fall".to_owned(),
            event_type: "fall_detected".to_owned(),
            identity: "evt-2".to_owned(),
            camera_id: CAMERA.to_owned(),
            facility_id: "fac-2".to_owned(),
            time_sec: 13.0,
            probability: 0.8,
        },
    );
    facility.sealed.contributors.push(SealedContributor {
        event_ref: "evt-2".to_owned(),
        detected_at: "2026-01-02T03:04:06.000000Z".to_owned(),
    });
    assert!(matches!(
        publish(&bench, &facility, now()),
        Err(ClipOutputError::Metadata(_))
    ));
    assert!(!bench.store.clip_dir(CLIP).join(MANIFEST_FILE).exists());
    sidecar_kept(&bench);
}

#[test]
fn missing_source_staged_and_final_media_fails_without_removing_the_sidecar() {
    let bench = Bench::new("missing");
    fs::remove_file(&bench.source).expect("no source");
    let recovery = bench.recovery();
    assert!(matches!(
        publish(&bench, &recovery, now()),
        Err(ClipOutputError::Publish(PublishError::MissingMedia))
    ));
    assert!(!bench.store.clip_dir(CLIP).join(MANIFEST_FILE).exists());
    sidecar_kept(&bench);
}
