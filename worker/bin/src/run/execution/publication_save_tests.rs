//! CPU composition only: real incident/staging/recorder/publication and ffprobe,
//! with a synthetic media command channel. Not native, PostgreSQL, or T37 E2E.

use super::*;
use crate::clips::publish::{MANIFEST_FILE, MEDIA_FILE, TERMINAL_MARKER};
use crate::clips::rendition::Tools;
use crate::clips::reserve::FINALIZE_FAILED;
use crate::seam::IdSource;
use seeon_deepstream_native::MediaResult;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use support::{
    BOOT_ID, CAMERA_ID, FACILITY_ID, INVALID_MEDIA, Scratch, TestClock, staged_event, start,
};

#[path = "publication_save_support.rs"]
mod support;

#[test]
fn cpu_composition_failed_codec_probe_publishes_unavailable_and_keeps_admission_live() {
    failed_codec_probe_continuation(None, |output, _| output.reserve.available());
}

pub(super) fn failed_codec_probe_continuation(
    store_root: Option<&Path>,
    prepare_store: impl FnOnce(&Publications, &str) -> usize,
) {
    let root = Scratch(std::env::temp_dir().join(RandomIds.uuid4().unwrap()));
    std::fs::create_dir(&root.0).unwrap();
    let store_root = store_root
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.0.join("store"));
    let wall = Arc::new(TestClock(AtomicU64::new(
        u64::try_from(Utc::parse("2026-08-20T17:20:58.197192Z").unwrap().micros()).unwrap(),
    )));
    let clock: Arc<dyn Clock> = wall.clone();
    let (commands, mut inbox, receipts, records) = channels();
    let cameras = [(
        3,
        MediaBinding {
            token: 10,
            generation: 3,
            epoch: 5,
        },
        CAMERA_ID.to_owned(),
        FACILITY_ID.to_owned(),
    )];
    let mut output = open(
        PublicationConfig {
            boot_id: BOOT_ID,
            state_dir: &root.0.join("state"),
            record_dir: &root.0.join("records"),
            store_root: &store_root,
            cameras: &cameras,
            config_version: 8,
            manifest_sha: None,
        },
        commands,
        records,
        Arc::clone(&clock),
    )
    .unwrap();
    assert_eq!(
        output.reserve.available(),
        ReservePool::slots_for(cameras.len())
    );
    let (first, first_staged) = staged_event(&mut output, "fall-episode:first");
    let ticket = start(&mut output, &mut inbox, &first, &first_staged, 1);
    wall.0.fetch_add(2_000_000, Ordering::SeqCst);
    let (second, second_staged) = staged_event(&mut output, "fall-episode:extended");
    assert_eq!(
        output
            .admit_recording(0, &second, &second_staged, clock.monotonic())
            .unwrap(),
        Admit::Extended
    );
    let events_before = output.queue.entries().unwrap();
    let attribution_before = output.events.clone();
    assert_eq!(events_before.len(), 2);
    let source = output.record_dir.join("invalid-media.mp4");
    std::fs::write(&source, INVALID_MEDIA).unwrap();
    assert!(source.symlink_metadata().unwrap().is_file());
    assert!(
        std::process::Command::new(Tools::default().ffprobe)
            .arg("-version")
            .output()
            .expect("actual ffprobe must run")
            .status
            .success()
    );
    assert!(super::super::recording::measured_codec(&source).is_err());
    let clip_id = format!("{BOOT_ID}-{}-{}", ticket.source_id, ticket.request_id);
    let expected_slots = prepare_store(&output, &clip_id);
    wall.0.fetch_add(60_000_000, Ordering::SeqCst);
    let finalized_at = Utc::from_system(clock.wall());
    receipts
        .send(RecordReceipt {
            ticket,
            result: MediaResult::Ok,
            error: 0,
            duration_ms: 30_000,
            width: 1280,
            height: 720,
            contains_video: true,
            contains_audio: false,
            directory: output.record_dir.clone(),
            filename: "invalid-media.mp4".into(),
        })
        .unwrap();

    let drained = output.drain_records(clock.as_ref());

    if !matches!(drained, Ok(true)) {
        assert_eq!(output.queue.entries().unwrap(), events_before);
        assert_eq!(output.events, attribution_before);
        assert_eq!(std::fs::read(&source).unwrap(), INVALID_MEDIA);
        let pending = output.sidecars.pending(CAMERA_ID).unwrap();
        assert!(pending.malformed.is_empty());
        let [retained] = pending.recoveries.as_slice() else {
            panic!("uncompleted publication must retain its sealed sidecar: {pending:?}");
        };
        assert_eq!(retained.sealed.clip_id, clip_id);
        assert_eq!(retained.sealed.path, source.to_str().unwrap());
        assert_eq!(retained.events, attribution_before);
    }
    assert!(
        matches!(drained, Ok(true)),
        "completed negative must drain successfully: {drained:?}"
    );
    assert_eq!(output.reserve.available(), expected_slots);
    let directory = output.store.clip_dir(&clip_id);
    let bytes = std::fs::read(directory.join(MANIFEST_FILE)).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(manifest["clip_id"], clip_id);
    assert_eq!(manifest["camera_id"], CAMERA_ID);
    assert!(manifest.get("facility_id").is_none());
    assert_eq!(manifest["domain"], first.domain);
    assert_eq!(manifest["event_type"], first.event_type);
    assert_eq!(manifest["state"], "UNAVAILABLE");
    assert_eq!(manifest["reason_code"], FINALIZE_FAILED);
    assert_eq!(manifest["video_available"], false);
    assert_eq!(manifest["detected_at"], first_staged.detected_at);
    assert_eq!(manifest["finalized_at"], finalized_at.iso_millis());
    assert_eq!(
        manifest["event_refs"],
        serde_json::json!([first.identity, second.identity])
    );
    assert_eq!(manifest["extension"]["boundary"], "extension_bounded");
    assert_eq!(manifest["extension"]["duration_s"], 30.0);
    assert_eq!(
        manifest["extension"]["contributors"],
        serde_json::json!([
            {"event_ref": first.identity, "detected_at": first_staged.detected_at},
            {"event_ref": second.identity, "detected_at": second_staged.detected_at}
        ])
    );
    assert!(manifest["path"].is_null());
    for key in ["codec", "sha256", "size_bytes", "mime_type", "duration_ms"] {
        assert!(
            manifest.get(key).is_none(),
            "unavailable has no {key} claim"
        );
    }
    let entries = output.queue.entries().unwrap();
    assert_eq!(entries.len(), 3);
    for event in events_before {
        assert!(
            entries.contains(&event),
            "event delivery must survive clip failure"
        );
    }
    let clips: Vec<_> = entries
        .iter()
        .filter(|entry| entry["clip_id"] == clip_id)
        .collect();
    assert_eq!(clips.len(), 1);
    let clip = clips[0];
    assert_eq!(clip["clip_id"], clip_id);
    assert_eq!(clip["camera_id"], CAMERA_ID);
    assert_eq!(clip["facility_id"], FACILITY_ID);
    assert_eq!(clip["local_state"], "UNAVAILABLE");
    assert_eq!(clip["unavailable_reason"], FINALIZE_FAILED);
    assert_eq!(clip["event_ids"], manifest["event_refs"]);
    assert_eq!(clip["finalized_at"], finalized_at.iso_millis());
    for key in [
        "media_reference",
        "codec",
        "sha256",
        "size_bytes",
        "mime_type",
        "duration_ms",
    ] {
        assert!(clip.get(key).is_some_and(serde_json::Value::is_null));
    }
    let marker: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join(TERMINAL_MARKER)).unwrap()).unwrap();
    assert_eq!(marker["clip_id"], clip_id);
    assert_eq!(marker["local_state"], "UNAVAILABLE");
    assert_eq!(marker["entry_id"], clip["entry_id"]);
    assert_eq!(
        marker["manifest_sha256"],
        crate::clips::durable::sha256_hex(&bytes)
    );
    assert_eq!(std::fs::read(source).unwrap(), INVALID_MEDIA);
    assert!(!directory.join(MEDIA_FILE).exists());
    let pending = output.sidecars.pending(CAMERA_ID).unwrap();
    assert!(pending.recoveries.is_empty() && pending.malformed.is_empty());
    assert_eq!(
        output
            .sidecars
            .directory()
            .join(format!("{clip_id}.json"))
            .symlink_metadata()
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::NotFound
    );
    assert!(output.recordings_complete());
    assert!(output.events.is_empty());
    assert_eq!(output.recorders[0].state(), State::Idle);
    assert!(!output.recorders[0].is_quiesced());
    let (next, next_staged) = staged_event(&mut output, "fall-episode:next");
    let next_ticket = start(&mut output, &mut inbox, &next, &next_staged, 2);
    assert_ne!(ticket, next_ticket);
    assert_eq!(output.recorders[0].state(), State::Recording);
    assert_eq!(output.recorders[0].pending(), 0);
    assert_eq!(
        output.events.keys().collect::<Vec<_>>(),
        vec![&next.identity]
    );
}
