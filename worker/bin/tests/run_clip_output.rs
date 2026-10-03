//! Sealed terminal reconciliation against a real clip store and delivery queue.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use seeon_ml_worker::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use seeon_ml_worker::clips::manifest::{ClipMetadata, Contributor, Extension, MAX_MANIFEST_BYTES};
use seeon_ml_worker::clips::publish::{
    MANIFEST_FILE, MEDIA_FILE, PublishError, Published, Publisher, TERMINAL_MARKER,
};
use seeon_ml_worker::clips::reserve::FINALIZE_FAILED;
use seeon_ml_worker::clips::sealed::{Recovery, SealedClip, SealedContributor, SealedEvent};
use seeon_ml_worker::clips::store::ClipStore;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::DeliveryQueue;
use seeon_ml_worker::json::{Json, Serialiser};
use seeon_ml_worker::run::clip_output::{ClipOutputError, publish_recovery, resume_terminal};

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
            probability: Some(0.9),
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

fn terminal_metadata(recovery: &Recovery) -> ClipMetadata {
    let events = recovery
        .events
        .iter()
        .map(|(event_ref, event)| {
            (
                event_ref.clone(),
                ContributorEvent {
                    camera_id: event.camera_id.clone(),
                    facility_id: event.facility_id.clone(),
                    domain: event.domain.clone(),
                    event_type: event.event_type.clone(),
                },
            )
        })
        .collect();
    let extension = Extension {
        boundary: recovery.sealed.boundary.clone(),
        duration_ms: recovery.sealed.duration_ms,
        contributors: recovery
            .sealed
            .contributors
            .iter()
            .map(|contributor| Contributor {
                event_ref: contributor.event_ref.clone(),
                detected_at: Utc::parse(&contributor.detected_at).expect("detected_at"),
            })
            .collect(),
    };
    flow_metadata(
        &recovery.sealed.clip_id,
        &events,
        extension,
        FLOW_ENCODER,
        now(),
    )
    .expect("prior terminal metadata")
}

fn finalize_failed(bench: &Bench, recovery: &Recovery) -> Published {
    let reservation = bench.store.reserve(CAMERA, CLIP).expect("reserve");
    Publisher::new(&bench.queue)
        .publish_unavailable(
            &reservation,
            &terminal_metadata(recovery),
            FINALIZE_FAILED,
            None,
        )
        .expect("actual FINALIZE_FAILED terminal")
}

fn canonical_bytes(value: &serde_json::Value) -> Vec<u8> {
    let body = Serialiser::ModelSelection
        .canonical(&Json::from(value))
        .expect("canonical test bytes");
    format!("{body}\n").into_bytes()
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
            probability: Some(0.8),
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

#[test]
fn an_absent_terminal_is_none_even_without_media() {
    let bench = Bench::new("terminal-absent");
    fs::remove_file(&bench.source).expect("no source");
    assert!(
        resume_terminal(&bench.recovery(), &bench.store, &bench.queue)
            .expect("absent manifest")
            .is_none()
    );
    assert!(bench.queue.entries().expect("entries").is_empty());
    assert!(!bench.store.clip_dir(CLIP).join(MANIFEST_FILE).exists());
    sidecar_kept(&bench);
}

#[test]
fn finalize_failed_without_any_media_keeps_its_old_timestamp_and_entry() {
    let bench = Bench::new("terminal-without-media");
    let recovery = bench.recovery();
    let first = finalize_failed(&bench, &recovery);
    fs::remove_file(&bench.source).expect("original gone");
    let marker = bench.store.clip_dir(CLIP).join(TERMINAL_MARKER);
    let marker_bytes = fs::read(&marker).expect("marker");
    let queued = bench.queue.entries().expect("entries");
    assert!(!bench.store.staging_dir(CLIP).join("artifact.mp4").exists());
    assert!(!bench.store.clip_dir(CLIP).join(MEDIA_FILE).exists());

    let resumed = publish_recovery(&recovery, &bench.store, &bench.queue, "", later())
        .expect("terminal needs neither caller codec nor media");

    assert!(resumed.resumed && !resumed.admitted);
    assert_eq!(resumed.manifest_bytes, first.manifest_bytes);
    assert_eq!(manifest_bytes(&bench), first.manifest_bytes);
    assert_eq!(resumed.entry, first.entry);
    assert_eq!(
        resumed.entry.fields().finalized_at.as_deref(),
        Some("2026-01-02T03:05:00.000Z")
    );
    assert_eq!(
        resumed.entry.fields().unavailable_reason.as_deref(),
        Some(FINALIZE_FAILED)
    );
    assert!(resumed.entry.fields().codec.is_none());
    assert!(resumed.video_path.is_none());
    assert_eq!(fs::read(marker).expect("marker unchanged"), marker_bytes);
    assert_eq!(bench.queue.entries().expect("entries"), queued);
    sidecar_kept(&bench);
}

#[test]
fn finalize_failed_replays_the_same_entry_after_ack_before_marker() {
    let bench = Bench::new("terminal-ack-before-marker");
    let recovery = bench.recovery();
    let reservation = bench.store.reserve(CAMERA, CLIP).expect("reserve");
    fs::create_dir_all(&reservation.final_dir).expect("final dir");
    let blocked = reservation
        .final_dir
        .join(format!(".{TERMINAL_MARKER}.tmp"));
    fs::create_dir(&blocked).expect("block marker after queue admission");
    let failed = Publisher::new(&bench.queue).publish_unavailable(
        &reservation,
        &terminal_metadata(&recovery),
        FINALIZE_FAILED,
        None,
    );
    assert!(matches!(failed, Err(PublishError::Io(_))), "{failed:?}");
    let original = manifest_bytes(&bench);
    let queued = bench.queue.entries().expect("entry before crash");
    assert_eq!(queued.len(), 1);
    let entry_id = queued[0]["entry_id"].as_str().expect("entry id");
    let queue_path = bench.queue.directory().join(format!("{entry_id}.json"));
    let queue_bytes = fs::read(&queue_path).expect("entry bytes");
    assert!(
        bench
            .queue
            .acknowledge_backend(entry_id, 204)
            .expect("ACK before marker")
    );
    assert!(bench.queue.entries().expect("ACK removed entry").is_empty());
    assert!(!reservation.final_dir.join(TERMINAL_MARKER).exists());
    fs::remove_file(&bench.source).expect("original gone");
    fs::remove_dir(&blocked).expect("allow marker repair");

    let resumed = resume_terminal(&recovery, &bench.store, &bench.queue)
        .expect("resume terminal without media")
        .expect("existing terminal");

    assert!(resumed.resumed && resumed.admitted);
    assert_eq!(resumed.entry.entry_id(), entry_id);
    assert_eq!(manifest_bytes(&bench), original);
    assert_eq!(bench.queue.entries().expect("stable readmission"), queued);
    assert_eq!(
        fs::read(queue_path).expect("stable entry bytes"),
        queue_bytes
    );
    assert!(reservation.final_dir.join(TERMINAL_MARKER).is_file());
    assert!(!reservation.staging_dir.exists());
    sidecar_kept(&bench);
}

#[test]
fn canonical_conflicting_or_unknown_terminals_keep_their_bytes_and_sidecar() {
    let conflicts = [
        ("camera_id", serde_json::json!("cam-2")),
        ("event_ref", serde_json::json!("evt-2")),
        ("reason_code", serde_json::json!("NO_FRAMES")),
        ("reason_code", serde_json::json!("ENCODER_FAILED")),
        ("reason_code", serde_json::json!("UNKNOWN")),
        ("state", serde_json::json!("UNKNOWN")),
        ("source_error_reason", serde_json::json!("native error")),
        ("source_error_reason", serde_json::Value::Null),
        ("manifest_schema_version", serde_json::json!(3)),
        ("state_version", serde_json::json!(1)),
        ("finalized", serde_json::json!(false)),
        ("finalized_at", serde_json::json!(12)),
        (
            "finalized_at",
            serde_json::json!("2026-01-02T03:00:00.000Z"),
        ),
        ("unknown_field", serde_json::json!(true)),
    ];
    for (index, (field, value)) in conflicts.into_iter().enumerate() {
        let bench = Bench::new(&format!("terminal-conflict-{index}"));
        let recovery = bench.recovery();
        finalize_failed(&bench, &recovery);
        fs::remove_file(&bench.source).expect("no original media");
        let marker = bench.store.clip_dir(CLIP).join(TERMINAL_MARKER);
        let marker_bytes = fs::read(&marker).expect("original marker");
        let queued = bench.queue.entries().expect("original entries");
        let mut value_model: serde_json::Value =
            serde_json::from_slice(&manifest_bytes(&bench)).expect("prior manifest");
        value_model[field] = value;
        let conflicting = canonical_bytes(&value_model);
        fs::write(bench.store.clip_dir(CLIP).join(MANIFEST_FILE), &conflicting)
            .expect("canonical conflict");

        let refused = publish_recovery(&recovery, &bench.store, &bench.queue, "", later());

        assert!(
            matches!(
                refused,
                Err(ClipOutputError::Publish(PublishError::Conflict))
            ),
            "{field}: {refused:?}"
        );
        assert_eq!(manifest_bytes(&bench), conflicting);
        assert_eq!(fs::read(marker).expect("marker kept"), marker_bytes);
        assert_eq!(bench.queue.entries().expect("entries kept"), queued);
        sidecar_kept(&bench);
    }
}

#[test]
fn a_timestamp_alone_is_not_a_terminal_identity() {
    let bench = Bench::new("terminal-timestamp-only");
    let recovery = bench.recovery();
    finalize_failed(&bench, &recovery);
    fs::remove_file(&bench.source).expect("no original media");
    let bytes = canonical_bytes(&serde_json::json!({"finalized_at": now().iso_millis()}));
    fs::write(bench.store.clip_dir(CLIP).join(MANIFEST_FILE), &bytes).expect("timestamp only");
    assert!(matches!(
        resume_terminal(&recovery, &bench.store, &bench.queue),
        Err(ClipOutputError::Publish(PublishError::Conflict))
    ));
    assert_eq!(manifest_bytes(&bench), bytes);
    sidecar_kept(&bench);
}

#[test]
fn malformed_noncanonical_and_over_bound_terminals_are_not_absent() {
    for shape in [
        "empty",
        "json",
        "utf8",
        "no-newline",
        "extra-newline",
        "whitespace",
        "duplicate",
        "over-bound",
        "not-object",
        "invalid-stamp",
    ] {
        let bench = Bench::new(&format!("terminal-unreadable-{shape}"));
        let recovery = bench.recovery();
        finalize_failed(&bench, &recovery);
        fs::remove_file(&bench.source).expect("no original media");
        let original = manifest_bytes(&bench);
        let bytes = match shape {
            "empty" => Vec::new(),
            "json" => b"{broken}\n".to_vec(),
            "utf8" => vec![0xff, b'\n'],
            "no-newline" => original[..original.len() - 1].to_vec(),
            "extra-newline" => [original.as_slice(), b"\n"].concat(),
            "whitespace" => [b" ".as_slice(), original.as_slice()].concat(),
            "duplicate" => b"{\"finalized_at\":\"x\",\"finalized_at\":\"x\"}\n".to_vec(),
            "over-bound" => vec![b'x'; MAX_MANIFEST_BYTES + 1],
            "not-object" => b"[]\n".to_vec(),
            "invalid-stamp" => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&original).expect("manifest");
                value["finalized_at"] = serde_json::json!("not a UTC instant");
                canonical_bytes(&value)
            }
            _ => unreachable!(),
        };
        fs::write(bench.store.clip_dir(CLIP).join(MANIFEST_FILE), &bytes).expect("invalid bytes");

        let refused = resume_terminal(&recovery, &bench.store, &bench.queue);

        assert!(
            matches!(refused, Err(ClipOutputError::ManifestUnreadable)),
            "{shape}: {refused:?}"
        );
        assert_eq!(manifest_bytes(&bench), bytes);
        assert_eq!(bench.queue.entries().expect("entries").len(), 1);
        sidecar_kept(&bench);
    }
}

#[test]
fn existing_ready_uses_the_canonical_codec_not_the_later_callers_codec() {
    let bench = Bench::new("ready-canonical-codec");
    let recovery = bench.recovery();
    let first = publish(&bench, &recovery, now()).expect("first READY");
    for codec in ["", "h265"] {
        let resumed = publish_recovery(&recovery, &bench.store, &bench.queue, codec, later())
            .expect("canonical codec and actual verified bytes");
        assert!(resumed.resumed && !resumed.admitted);
        assert_eq!(resumed.entry, first.entry);
        assert_eq!(resumed.manifest_bytes, first.manifest_bytes);
    }
    assert_eq!(media_bytes(&bench), MEDIA);
    assert_eq!(bench.queue.entries().expect("entries").len(), 1);
    sidecar_kept(&bench);
}

#[test]
fn existing_ready_refuses_changed_missing_or_duration_conflicting_media() {
    for change in [
        "hash",
        "size",
        "missing-final",
        "manifest-duration",
        "sealed-duration",
    ] {
        let bench = Bench::new(&format!("ready-unverified-{change}"));
        let mut recovery = bench.recovery();
        let first = publish(&bench, &recovery, now()).expect("first READY");
        let marker = bench.store.clip_dir(CLIP).join(TERMINAL_MARKER);
        fs::remove_file(&marker).expect("marker lost");
        assert!(
            bench
                .queue
                .acknowledge_backend(first.entry.entry_id(), 204)
                .expect("ACK")
        );
        let video = bench.store.clip_dir(CLIP).join(MEDIA_FILE);
        let mut expected_manifest = first.manifest_bytes;
        match change {
            "hash" => {
                let mut altered = MEDIA.to_vec();
                altered[0] ^= 1;
                fs::write(&video, altered).expect("same size, wrong hash");
            }
            "size" => fs::write(&video, [MEDIA, b"extra"].concat()).expect("wrong size"),
            "missing-final" => {
                fs::remove_file(&video).expect("final media gone");
                fs::write(&bench.source, MEDIA).expect("original cannot replace a prior final");
                let reservation = bench.store.reserve(CAMERA, CLIP).expect("reserve");
                fs::write(reservation.artifact_path(), MEDIA).expect("staged cannot replace final");
            }
            "manifest-duration" => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&expected_manifest).expect("manifest");
                value["duration_ms"] = serde_json::json!(4_001);
                expected_manifest = canonical_bytes(&value);
                fs::write(&first.manifest_path, &expected_manifest).expect("duration conflict");
            }
            "sealed-duration" => recovery.sealed.duration_ms += 1,
            _ => unreachable!(),
        }
        let expected_media = if change == "missing-final" {
            None
        } else {
            Some(fs::read(&video).expect("media before refusal"))
        };

        let refused = publish_recovery(&recovery, &bench.store, &bench.queue, "", later());

        if change == "missing-final" {
            assert!(
                matches!(
                    refused,
                    Err(ClipOutputError::Publish(PublishError::MissingMedia))
                ),
                "{refused:?}"
            );
            assert_eq!(fs::read(&bench.source).expect("original kept"), MEDIA);
            assert_eq!(
                fs::read(bench.store.staging_dir(CLIP).join("artifact.mp4")).expect("staged kept"),
                MEDIA
            );
        } else {
            assert!(
                matches!(
                    refused,
                    Err(ClipOutputError::Publish(PublishError::Conflict))
                ),
                "{change}: {refused:?}"
            );
        }
        assert_eq!(manifest_bytes(&bench), expected_manifest);
        if let Some(expected_media) = expected_media {
            assert_eq!(fs::read(&video).expect("media retained"), expected_media);
        } else {
            assert!(!video.exists(), "no replacement final may be fabricated");
        }
        assert!(
            !marker.exists(),
            "unverified READY must not repair the marker"
        );
        assert!(bench.queue.entries().expect("no readmission").is_empty());
        sidecar_kept(&bench);
    }
}

#[test]
fn a_conflicting_marker_prevents_terminal_cleanup() {
    let bench = Bench::new("terminal-marker-conflict");
    let recovery = bench.recovery();
    let first = finalize_failed(&bench, &recovery);
    fs::remove_file(&bench.source).expect("original gone");
    let reservation = bench.store.reserve(CAMERA, CLIP).expect("reserve");
    fs::write(reservation.artifact_path(), MEDIA).expect("retained staged media");
    fs::write(reservation.final_dir.join(MEDIA_FILE), MEDIA).expect("retained final media");
    let marker = reservation.final_dir.join(TERMINAL_MARKER);
    fs::write(&marker, b"conflicting marker\n").expect("marker conflict");
    assert!(matches!(
        resume_terminal(&recovery, &bench.store, &bench.queue),
        Err(ClipOutputError::Publish(PublishError::Conflict))
    ));
    assert_eq!(manifest_bytes(&bench), first.manifest_bytes);
    assert_eq!(
        fs::read(marker).expect("marker kept"),
        b"conflicting marker\n"
    );
    assert_eq!(
        fs::read(reservation.artifact_path()).expect("staged kept"),
        MEDIA
    );
    assert_eq!(media_bytes(&bench), MEDIA);
    assert_eq!(bench.queue.entries().expect("entries").len(), 1);
    sidecar_kept(&bench);
}
