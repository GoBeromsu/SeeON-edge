//! READY manifest and queue tied to the captured native receipt.

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use seeon_deepstream_native::RecordTicket;
use serde_json::Value;

use super::super::Publications;
use super::assertions::{Admitted, GrownFile};
use super::support::BOOT_ID;
use crate::clips::durable;
use crate::clips::publish::{MANIFEST_FILE, MEDIA_FILE};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PublishedState {
    pub media: Vec<u8>,
    pub manifest: Vec<u8>,
    pub queue_entry: Value,
    pub sha256: String,
}

pub(super) fn assert_published(
    publications: &Publications,
    record_dir: &Path,
    admitted: &Admitted,
    ticket: &RecordTicket,
    duration_ms: u64,
    sealed: &GrownFile,
) {
    assert_clip(
        publications,
        record_dir,
        admitted,
        ticket,
        duration_ms,
        sealed,
    );
    let clip_id = format!("{BOOT_ID}-{}-{}", ticket.source_id, ticket.request_id);
    assert_eq!(sole_clip(publications), clip_id);
    assert_eq!(
        clip_entries(publications).len(),
        1,
        "exactly one clip delivery entry"
    );
    assert!(
        !mp4_left(record_dir),
        "native MP4 remained in the record directory"
    );
}

pub(super) fn assert_clip(
    publications: &Publications,
    record_dir: &Path,
    admitted: &Admitted,
    ticket: &RecordTicket,
    duration_ms: u64,
    sealed: &GrownFile,
) -> PublishedState {
    let clip_id = format!("{BOOT_ID}-{}-{}", ticket.source_id, ticket.request_id);
    let clip_dir = publications.store.clip_dir(&clip_id);
    let media_path = clip_dir.join(MEDIA_FILE);
    let published = fs::read(&media_path).expect("published MP4");
    assert!(published.len() >= 16 && &published[4..8] == b"ftyp");
    let meta = fs::symlink_metadata(&media_path).expect("published metadata");
    assert!(meta.is_file());
    assert_eq!(
        meta.len(),
        sealed.len,
        "published size is not the sealed size"
    );
    if same_device(record_dir, publications.store.root()) {
        assert_eq!((meta.dev(), meta.ino()), (sealed.dev, sealed.ino));
    }
    let (sha, size) = durable::sha256_file(&media_path).expect("published hash");
    assert_eq!(size, u64::try_from(published.len()).expect("size"));
    assert_eq!(size, sealed.len);
    let queue_entry = assert_queue(
        publications,
        &sha,
        size,
        &clip_id,
        duration_ms,
        admitted.event_ref,
        admitted.camera_id,
    );
    let manifest = assert_manifest(&clip_dir, &sha, size, &clip_id, duration_ms, admitted);
    PublishedState {
        media: published,
        manifest,
        queue_entry,
        sha256: sha,
    }
}

pub(super) fn assert_exact_clips(publications: &Publications, tickets: &[RecordTicket]) {
    let expected: BTreeSet<_> = tickets
        .iter()
        .map(|ticket| format!("{BOOT_ID}-{}-{}", ticket.source_id, ticket.request_id))
        .collect();
    assert_eq!(expected.len(), tickets.len());
    let actual: BTreeSet<_> = fs::read_dir(publications.store.root().join("clips"))
        .expect("clips")
        .map(|entry| {
            entry
                .expect("clip entry")
                .file_name()
                .into_string()
                .expect("clip name")
        })
        .filter(|name| !name.starts_with('.'))
        .collect();
    assert_eq!(actual, expected, "published clip directories differ");
    let clips = clip_entries(publications);
    let queued: BTreeSet<_> = clips
        .iter()
        .map(|entry| {
            entry
                .get("clip_id")
                .and_then(Value::as_str)
                .expect("clip id")
                .to_owned()
        })
        .collect();
    assert_eq!(clips.len(), tickets.len(), "unexpected CLIP queue entries");
    assert_eq!(
        queued, expected,
        "queue clip IDs differ from published clips"
    );
}

pub(super) fn assert_no_mp4(record_dir: &Path) {
    assert!(
        !mp4_left(record_dir),
        "native MP4 remained in the record directory"
    );
}

fn clip_entries(publications: &Publications) -> Vec<Value> {
    publications
        .queue
        .entries()
        .expect("delivery queue")
        .into_iter()
        .filter(|entry| entry.get("kind").and_then(Value::as_str) == Some("CLIP"))
        .collect()
}

fn assert_queue(
    publications: &Publications,
    sha: &str,
    size: u64,
    clip_id: &str,
    duration_ms: u64,
    event_ref: &str,
    camera_id: &str,
) -> Value {
    let clips = clip_entries(publications);
    let matching: Vec<_> = clips
        .iter()
        .filter(|entry| entry.get("clip_id").and_then(Value::as_str) == Some(clip_id))
        .collect();
    assert_eq!(matching.len(), 1, "exactly one queue entry for {clip_id}");
    let clip = matching[0];
    assert_eq!(clip.get("clip_id").and_then(Value::as_str), Some(clip_id));
    assert_eq!(
        clip.get("media_reference").and_then(Value::as_str),
        Some(reference(clip_id).as_str())
    );
    assert_eq!(
        clip.get("local_state").and_then(Value::as_str),
        Some("VERIFIED")
    );
    assert_eq!(clip.get("sha256").and_then(Value::as_str), Some(sha));
    assert_eq!(clip.get("size_bytes").and_then(Value::as_u64), Some(size));
    assert_eq!(
        clip.get("duration_ms").and_then(Value::as_i64),
        Some(i64::try_from(duration_ms).expect("duration"))
    );
    let ids = clip
        .get("event_ids")
        .and_then(Value::as_array)
        .expect("event_ids");
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0].as_str(), Some(event_ref));
    assert_eq!(
        clip.get("camera_id").and_then(Value::as_str),
        Some(camera_id)
    );
    clip.clone()
}

fn assert_manifest(
    clip_dir: &Path,
    sha: &str,
    size: u64,
    clip_id: &str,
    duration_ms: u64,
    admitted: &Admitted,
) -> Vec<u8> {
    let bytes = fs::read(clip_dir.join(MANIFEST_FILE)).expect("manifest");
    let manifest: Value = serde_json::from_slice(&bytes).expect("manifest json");
    assert_eq!(manifest.get("state").and_then(Value::as_str), Some("READY"));
    assert_eq!(
        manifest.get("clip_id").and_then(Value::as_str),
        Some(clip_id)
    );
    assert_eq!(
        manifest.get("path").and_then(Value::as_str),
        Some(reference(clip_id).as_str())
    );
    assert_eq!(manifest.get("sha256").and_then(Value::as_str), Some(sha));
    assert_eq!(
        manifest.get("size_bytes").and_then(Value::as_u64),
        Some(size)
    );
    assert_eq!(
        manifest.get("duration_ms").and_then(Value::as_i64),
        Some(i64::try_from(duration_ms).expect("duration"))
    );
    let detected = admitted.detected_at.iso_micros();
    assert_eq!(
        manifest.get("detected_at").and_then(Value::as_str),
        Some(detected.as_str())
    );
    assert_eq!(
        manifest.get("event_ref").and_then(Value::as_str),
        Some(admitted.event_ref)
    );
    assert_eq!(
        manifest.get("camera_id").and_then(Value::as_str),
        Some(admitted.camera_id)
    );
    assert_eq!(
        manifest
            .pointer("/extension/contributors/0/detected_at")
            .and_then(Value::as_str),
        Some(detected.as_str())
    );
    bytes
}

fn sole_clip(publications: &Publications) -> String {
    let mut found = fs::read_dir(publications.store.root().join("clips"))
        .expect("clips")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| !name.starts_with('.'))
        })
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1, "exactly one published clip directory");
    let path = found.pop().expect("published clip");
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("clip name");
    assert_eq!(path, publications.store.clip_dir(name));
    name.to_owned()
}

fn reference(clip_id: &str) -> String {
    format!("clips/{clip_id}/{MEDIA_FILE}")
}

fn same_device(record_dir: &Path, store_root: &Path) -> bool {
    match (fs::metadata(record_dir), fs::metadata(store_root)) {
        (Ok(record), Ok(store)) => record.dev() == store.dev(),
        _ => false,
    }
}

fn mp4_left(record_dir: &Path) -> bool {
    fs::read_dir(record_dir)
        .expect("record directory")
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "mp4")
        })
}
