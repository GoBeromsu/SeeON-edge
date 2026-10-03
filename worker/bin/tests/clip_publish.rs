//! T24: clip publication against the clip-manifest goldens.
//!
//! Every case publishes into a fresh clip store and delivery queue under
//! `CARGO_TARGET_TMPDIR`. The manifest bytes, the queued `ClipEntry`, the
//! store layout and the relay PUT body are compared with values recorded from
//! the Python writer (`d/clip-manifests/*`, `r/clip-put-*`). Media files are
//! opaque bytes here: publication moves them and takes the measured facts from
//! the caller. The backend lane also parses every published manifest with the
//! backend oracle `parse_manifest_bytes` through `$SEEON_TEST_PYTHON`.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt, symlink};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use seeon_ml_worker::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use seeon_ml_worker::clips::manifest::{
    ClipMetadata, Contributor, Extension, MAX_MANIFEST_BYTES, MediaFacts,
};
use seeon_ml_worker::clips::publish::{
    MANIFEST_FILE, MEDIA_FILE, PublishError, Published, Publisher, TERMINAL_MARKER,
};
use seeon_ml_worker::clips::store::{ClipStore, Reservation};
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::sender::clip_request;
use seeon_ml_worker::delivery::{
    AdmissionFault, ClipEntry, ClipFields, DeliveryQueue, LOCK_FILE_NAME,
};

const MEDIA: &[u8] = b"opaque clip media bytes";
const CASES: [&str; 5] = [
    "ready",
    "unavailable",
    "corrupt-missing",
    "corrupt-existing",
    "flow-publish",
];

fn wire() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/worker-wire")
}

fn read_json(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).expect("fixture is JSON")
}

fn golden(name: &str) -> Value {
    read_json(&wire().join(format!("d/clip-manifests/{name}.json")))
}

fn golden_manifest(case: &Value) -> Vec<u8> {
    let file = case["manifest"]["payload_file"]
        .as_str()
        .expect("payload_file");
    fs::read(wire().join(file)).expect("golden manifest bytes")
}

fn text(value: &Value, key: &str) -> String {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is text"))
        .to_owned()
}

fn optional(value: &Value, key: &str) -> Option<String> {
    value[key].as_str().map(str::to_owned)
}

fn strings(value: &Value) -> Vec<String> {
    let items = value.as_array().expect("array");
    items
        .iter()
        .map(|item| item.as_str().expect("text").to_owned())
        .collect()
}

fn utc(value: &Value) -> Utc {
    Utc::parse(value.as_str().expect("timestamp text")).expect("RFC 3339 timestamp")
}

/// One test's private clip store and delivery queue; `<work>` in the goldens.
struct Bench {
    work: PathBuf,
    store: ClipStore,
    queue: DeliveryQueue,
}

impl Bench {
    fn new(test: &str, case: &str) -> Self {
        let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("clip_publish-{test}"));
        if work.exists() {
            fs::remove_dir_all(&work).expect("clear previous run");
        }
        let root = work.join(case).join("store");
        fs::create_dir_all(&root).expect("store root");
        let queue = DeliveryQueue::open(&root.join("delivery-queue"), true).expect("queue");
        Self {
            work,
            store: ClipStore::new(root),
            queue,
        }
    }

    fn place(&self, text: &str) -> PathBuf {
        PathBuf::from(text.replacen("<work>", self.work.to_str().expect("utf-8 path"), 1))
    }

    fn reserve(&self, camera_id: &str, clip_id: &str, with_media: bool) -> Reservation {
        let reservation = self.store.reserve(camera_id, clip_id).expect("reserve");
        if with_media {
            fs::write(reservation.artifact_path(), MEDIA).expect("artifact");
        }
        reservation
    }

    fn queued(&self) -> Vec<Value> {
        self.queue.entries().expect("queue entries")
    }
}

/// Metadata exactly as the Python writer received it.
fn metadata(inputs: &Value) -> ClipMetadata {
    let meta = &inputs["metadata"];
    assert!(
        meta["extension"].is_null(),
        "non-flow cases carry no extension"
    );
    let duration_s = meta["duration_s"].as_f64().expect("duration_s");
    ClipMetadata {
        clip_id: text(&inputs["reservation"], "clip_id"),
        camera_id: text(meta, "camera_id"),
        facility_id: text(meta, "facility_id"),
        domain: text(meta, "domain"),
        event_type: text(meta, "event_type"),
        event_refs: strings(&meta["event_refs"]),
        detected_at: utc(&meta["detected_at"]),
        started_at: utc(&meta["started_at"]),
        clip_start_at: utc(&meta["clip_start_at"]),
        clip_end_at: utc(&meta["clip_end_at"]),
        finalized_at: utc(&meta["finalized_at"]),
        duration_ms: (duration_s * 1000.0).round() as i64,
        encoder: text(meta, "encoder"),
        truncation_reasons: strings(&meta["truncation_reasons"]),
        extension: None,
    }
}

fn flow_meta(inputs: &Value) -> ClipMetadata {
    let sealed = &inputs["sealed"];
    let contributors = sealed["contributors"].as_array().expect("contributors");
    let events: BTreeMap<String, ContributorEvent> = inputs["events"]
        .as_object()
        .expect("events")
        .iter()
        .map(|(event_ref, event)| {
            let event = ContributorEvent {
                camera_id: text(event, "camera_id"),
                facility_id: text(event, "facility_id"),
                domain: text(event, "domain"),
                event_type: text(event, "event_type"),
            };
            (event_ref.clone(), event)
        })
        .collect();
    let extension = Extension {
        boundary: text(sealed, "boundary"),
        contributors: contributors
            .iter()
            .map(|c| Contributor {
                event_ref: text(c, "event_ref"),
                detected_at: utc(&c["detected_at"]),
            })
            .collect(),
        duration_ms: sealed["duration_ms"].as_i64().expect("duration_ms"),
    };
    let meta = flow_metadata(
        &text(sealed, "clip_id"),
        &events,
        extension,
        FLOW_ENCODER,
        utc(&inputs["now"]),
    )
    .expect("flow metadata");
    // Independent oracle: the earliest contributor by detected_at fixes the
    // event fields (Python `_publish` sorts before it reads `events`).
    let mut by_time: Vec<&Value> = contributors.iter().collect();
    by_time.sort_by_key(|c| utc(&c["detected_at"]));
    let earliest_ref = text(by_time[0], "event_ref");
    let earliest = &inputs["events"][earliest_ref.as_str()];
    assert_eq!(meta.event_refs[0], earliest_ref);
    assert_eq!(meta.camera_id, text(earliest, "camera_id"));
    assert_eq!(meta.facility_id, text(earliest, "facility_id"));
    assert_eq!(meta.domain, text(earliest, "domain"));
    assert_eq!(meta.event_type, text(earliest, "event_type"));
    meta
}

fn media(case: &Value) -> MediaFacts {
    let fields = &case["media_facts_informational"]["fields"];
    MediaFacts {
        sha256: text(fields, "sha256"),
        size_bytes: fields["size_bytes"].as_i64().expect("size_bytes"),
        codec: text(fields, "codec"),
        duration_ms: fields["duration_ms"].as_i64().expect("duration_ms"),
    }
}

/// The single `ClipEntry` the Python writer admitted.
fn port_entry(case: &Value) -> Value {
    let calls = case["ports"]["delivery_queue"]
        .as_array()
        .expect("delivery_queue calls");
    let admits: Vec<&Value> = calls
        .iter()
        .filter(|call| call["call"] == "try_admit")
        .collect();
    assert_eq!(admits.len(), 1, "the writer admits exactly once");
    assert_eq!(admits[0]["entry_class"], "ClipEntry");
    admits[0]["entry"].clone()
}

fn entry_from(value: &Value) -> ClipEntry {
    ClipEntry::new(ClipFields {
        clip_id: text(value, "clip_id"),
        event_ids: strings(&value["event_ids"]),
        camera_id: text(value, "camera_id"),
        facility_id: text(value, "facility_id"),
        local_state: text(value, "local_state"),
        state_version: value["state_version"].as_i64().expect("state_version"),
        media_reference: optional(value, "media_reference"),
        sha256: optional(value, "sha256"),
        size_bytes: value["size_bytes"].as_i64(),
        mime_type: optional(value, "mime_type"),
        codec: optional(value, "codec"),
        duration_ms: value["duration_ms"].as_i64(),
        clip_start_at: optional(value, "clip_start_at"),
        clip_end_at: optional(value, "clip_end_at"),
        finalized_at: optional(value, "finalized_at"),
        unavailable_reason: optional(value, "unavailable_reason"),
        entry_id: text(value, "entry_id"),
    })
    .expect("golden entry is a valid ClipEntry")
}

/// Publishes one golden case the way its recorded writer did.
fn publish_case(bench: &Bench, name: &str, case: &Value) -> Result<Published, PublishError> {
    let inputs = &case["inputs"];
    let publisher = Publisher::new(&bench.queue);
    match name {
        "flow-publish" | "flow-resume" => {
            let meta = flow_meta(inputs);
            let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, true);
            publisher.publish_ready(&reservation, &meta, &media(case))
        }
        _ => {
            let meta = metadata(inputs);
            let with_media = !inputs["artifact_path"].is_null();
            let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, with_media);
            let golden_reservation = &inputs["reservation"];
            assert_eq!(
                reservation.final_dir,
                bench.place(&text(golden_reservation, "final_dir"))
            );
            assert_eq!(
                reservation.staging_dir,
                bench.place(&text(golden_reservation, "staging_dir"))
            );
            match name {
                "ready" => publisher.publish_ready(&reservation, &meta, &media(case)),
                "unavailable" => publisher.publish_unavailable(
                    &reservation,
                    &meta,
                    &text(inputs, "reason_code"),
                    inputs["metadata"]["source_error_reason"].as_str(),
                ),
                // No media survives, so the facts offered must be ignored.
                "corrupt-missing" => {
                    let offered = media(&golden("ready"));
                    publisher.publish_existing_corrupt(&reservation, &meta, &offered)
                }
                "corrupt-existing" => {
                    let first = publisher.publish_ready(&reservation, &meta, &media(case))?;
                    let again =
                        publisher.publish_existing_corrupt(&reservation, &meta, &media(case))?;
                    assert_eq!(again.manifest_bytes, first.manifest_bytes);
                    assert!(
                        again.resumed && !again.admitted,
                        "the prior publication stands"
                    );
                    Ok(again)
                }
                other => panic!("unknown case {other}"),
            }
        }
    }
}

fn retained_media(bench: &Bench, case: &Value) -> Reservation {
    let meta = metadata(&case["inputs"]);
    let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, true);
    fs::create_dir_all(&reservation.final_dir).expect("final directory");
    fs::write(reservation.final_dir.join(MEDIA_FILE), MEDIA).expect("retained final media");
    reservation
}

fn assert_retained_media(reservation: &Reservation) {
    assert_eq!(
        fs::read(reservation.artifact_path()).expect("staged media"),
        MEDIA
    );
    assert_eq!(
        fs::read(reservation.final_dir.join(MEDIA_FILE)).expect("final media"),
        MEDIA
    );
}

fn publish_reserved_unavailable(
    bench: &Bench,
    reservation: &Reservation,
    case: &Value,
) -> Result<Published, PublishError> {
    let inputs = &case["inputs"];
    Publisher::new(&bench.queue).publish_unavailable(
        reservation,
        &metadata(inputs),
        &text(inputs, "reason_code"),
        inputs["metadata"]["source_error_reason"].as_str(),
    )
}

/// Store entries other than the delivery queue and Python's audit directory.
fn listing(root: &Path) -> Vec<(String, String)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
        for item in fs::read_dir(dir).expect("read_dir") {
            let path = item.expect("dir entry").path();
            let relative = path
                .strip_prefix(root)
                .expect("under root")
                .to_str()
                .expect("utf-8");
            if relative == "delivery-queue" {
                continue;
            }
            let is_dir = path.is_dir();
            out.push((
                if is_dir { "dir" } else { "file" }.to_owned(),
                relative.to_owned(),
            ));
            if is_dir {
                walk(root, &path, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

fn golden_listing(case: &Value) -> Vec<(String, String)> {
    let items = case["clip_store_listing"].as_array().expect("listing");
    let mut out: Vec<(String, String)> = items
        .iter()
        .map(|item| (text(item, "kind"), text(item, "path")))
        .filter(|(_, path)| !path.contains("/terminal-outcomes"))
        .collect();
    out.sort();
    out
}

/// Manifest bytes, entry, queue and layout for one case.
fn check_case(test: &str, name: &str) {
    let case = golden(name);
    let bench = Bench::new(test, name);
    let published = publish_case(&bench, name, &case).expect("publication succeeds");
    let golden_bytes = golden_manifest(&case);

    assert_eq!(
        published.manifest_bytes, golden_bytes,
        "{name}: manifest bytes"
    );
    assert_eq!(
        fs::read(&published.manifest_path).expect("manifest"),
        golden_bytes
    );
    assert_eq!(
        golden_bytes.len() as u64,
        case["manifest"]["size_bytes"].as_u64().expect("size")
    );
    let expected = port_entry(&case);
    assert_eq!(published.entry, entry_from(&expected), "{name}: ClipEntry");

    let mut queued = bench.queued();
    assert_eq!(queued.len(), 1, "{name}: exactly one queued entry");
    let kind = queued[0].as_object_mut().expect("object").remove("kind");
    assert_eq!(kind, Some(Value::from("CLIP")));
    assert_eq!(queued[0], expected, "{name}: queued entry");

    let ready = case["worker"]["state"] == "READY";
    let clip_dir = published
        .manifest_path
        .parent()
        .expect("clip dir")
        .to_path_buf();
    assert_eq!(
        published.video_path,
        ready.then(|| clip_dir.join(MEDIA_FILE))
    );
    if ready {
        assert_eq!(
            fs::read(clip_dir.join(MEDIA_FILE)).expect("clip media"),
            MEDIA
        );
    }
    assert!(
        clip_dir.join(TERMINAL_MARKER).is_file(),
        "{name}: terminal marker"
    );
    assert_eq!(
        listing(bench.store.root()),
        golden_listing(&case),
        "{name}: store layout"
    );
}

#[test]
fn ready_publishes_golden_manifest_entry_and_layout() {
    check_case("ready", "ready");
}

#[test]
fn unavailable_publishes_golden_manifest_with_its_reason() {
    check_case("unavailable", "unavailable");
}

#[test]
fn corrupt_without_media_publishes_unavailable_corrupt() {
    check_case("corrupt-missing", "corrupt-missing");
}

#[test]
fn corrupt_with_media_keeps_the_ready_publication() {
    check_case("corrupt-existing", "corrupt-existing");
}

#[test]
fn flow_orders_contributors_and_publishes_golden_manifest() {
    check_case("flow-publish", "flow-publish");
}

#[test]
fn flow_resume_rewrites_nothing_and_admits_once() {
    let case = golden("flow-resume");
    assert_eq!(case["resume"]["manifest_bytes_unchanged"], true);
    let bench = Bench::new("flow-resume", "flow");
    let first = publish_case(&bench, "flow-resume", &case).expect("first publication");
    let second = publish_case(&bench, "flow-resume", &case).expect("resumed publication");
    assert!(!first.resumed && first.admitted);
    assert!(second.resumed && !second.admitted, "resume admits nothing");
    assert_eq!(second.manifest_bytes, golden_manifest(&case));
    assert_eq!(
        fs::read(&second.manifest_path).expect("manifest"),
        golden_manifest(&case)
    );
    assert_eq!(second.entry, first.entry);
    assert_eq!(bench.queued().len(), 1);
}

#[test]
fn crash_before_marker_readmits_without_a_second_entry() {
    let case = golden("ready");
    let bench = Bench::new("crash-before-marker", "ready");
    let first = publish_case(&bench, "ready", &case).expect("first publication");
    let clip_dir = first
        .manifest_path
        .parent()
        .expect("clip dir")
        .to_path_buf();
    fs::remove_file(clip_dir.join(TERMINAL_MARKER)).expect("simulate a crash before the marker");
    let meta = metadata(&case["inputs"]);
    let reservation = bench
        .store
        .reserve(&meta.camera_id, &meta.clip_id)
        .expect("re-reserve");
    let again = Publisher::new(&bench.queue)
        .publish_ready(&reservation, &meta, &media(&case))
        .expect("resumed publication");
    assert!(
        again.resumed && again.admitted,
        "a missing marker re-admits"
    );
    assert!(clip_dir.join(TERMINAL_MARKER).is_file());
    let queued = bench.queued();
    assert_eq!(queued.len(), 1, "re-admission is idempotent");
    assert_eq!(queued[0]["entry_id"], Value::from(first.entry.entry_id()));
}

#[test]
fn unavailable_queue_failure_retains_staged_and_final_media() {
    let case = golden("unavailable");
    let bench = Bench::new("unavailable-queue-failure", "unavailable");
    let reservation = retained_media(&bench, &case);
    let lock = bench.queue.directory().join(LOCK_FILE_NAME);
    fs::remove_file(&lock).expect("remove queue lock");
    fs::create_dir(&lock).expect("block queue admission");

    let failed = publish_reserved_unavailable(&bench, &reservation, &case);

    assert!(matches!(failed, Err(PublishError::Queue(_))), "{failed:?}");
    assert_retained_media(&reservation);
    assert_eq!(
        fs::read(reservation.final_dir.join(MANIFEST_FILE)).expect("manifest"),
        golden_manifest(&case)
    );
    assert!(!reservation.final_dir.join(TERMINAL_MARKER).exists());
    fs::remove_dir(&lock).expect("unblock queue admission");
    assert!(bench.queued().is_empty());

    let resumed = publish_reserved_unavailable(&bench, &reservation, &case).expect("resume");
    assert!(resumed.resumed && resumed.admitted);
    assert_eq!(resumed.entry, entry_from(&port_entry(&case)));
    assert!(reservation.final_dir.join(TERMINAL_MARKER).is_file());
    assert!(!reservation.final_dir.join(MEDIA_FILE).exists());
    assert!(!reservation.staging_dir.exists());
}

#[test]
fn unavailable_queue_refusal_retains_staged_and_final_media() {
    let case = golden("unavailable");
    let bench = Bench::new("unavailable-queue-refusal", "unavailable");
    let reservation = retained_media(&bench, &case);
    let mut conflicting = port_entry(&case);
    conflicting["unavailable_reason"] = Value::from("OTHER_REASON");
    let entry = entry_from(&conflicting);
    assert!(
        bench
            .queue
            .try_admit(&entry.into())
            .expect("conflicting entry")
            .accepted
    );
    let queued = bench.queued();

    let failed = publish_reserved_unavailable(&bench, &reservation, &case);

    assert!(
        matches!(
            failed,
            Err(PublishError::Refused(Some(AdmissionFault::Conflict)))
        ),
        "{failed:?}"
    );
    assert_retained_media(&reservation);
    assert!(!reservation.final_dir.join(TERMINAL_MARKER).exists());
    assert_eq!(bench.queued(), queued);
}

#[test]
fn unavailable_marker_failure_retains_media_until_identical_resume() {
    let case = golden("unavailable");
    for acknowledged in [false, true] {
        let bench = Bench::new(
            &format!("unavailable-marker-failure-{acknowledged}"),
            "unavailable",
        );
        let reservation = retained_media(&bench, &case);
        let manifest = reservation.final_dir.join(MANIFEST_FILE);
        let marker = reservation.final_dir.join(TERMINAL_MARKER);
        let blocked = reservation
            .final_dir
            .join(format!(".{TERMINAL_MARKER}.tmp"));
        fs::create_dir(&blocked).expect("block marker write after queue admission");

        let failed = publish_reserved_unavailable(&bench, &reservation, &case);

        assert!(matches!(failed, Err(PublishError::Io(_))), "{failed:?}");
        assert_retained_media(&reservation);
        assert_eq!(
            fs::read(&manifest).expect("manifest"),
            golden_manifest(&case)
        );
        assert!(!marker.exists());
        let queued = bench.queued();
        assert_eq!(queued.len(), 1);
        let expected = entry_from(&port_entry(&case));
        assert_eq!(queued[0]["entry_id"], expected.entry_id());
        let manifest_inode = fs::metadata(&manifest).expect("manifest metadata").ino();
        let queue_path = bench
            .queue
            .directory()
            .join(format!("{}.json", expected.entry_id()));
        let queue_bytes = fs::read(&queue_path).expect("queued bytes");
        let queue_inode = fs::metadata(&queue_path).expect("queue metadata").ino();
        if acknowledged {
            assert!(
                bench
                    .queue
                    .acknowledge_backend(expected.entry_id(), 204)
                    .expect("ACK before marker")
            );
            assert!(bench.queued().is_empty());
        }
        fs::remove_dir(&blocked).expect("allow marker write");

        let resumed = publish_reserved_unavailable(&bench, &reservation, &case).expect("resume");

        assert!(resumed.resumed && resumed.admitted);
        assert_eq!(resumed.entry, expected);
        assert_eq!(resumed.manifest_bytes, golden_manifest(&case));
        assert_eq!(
            fs::metadata(&manifest).expect("manifest metadata").ino(),
            manifest_inode
        );
        assert_eq!(bench.queued(), queued, "the same stable entry is admitted");
        assert_eq!(fs::read(&queue_path).expect("queued bytes"), queue_bytes);
        if !acknowledged {
            assert_eq!(
                fs::metadata(&queue_path).expect("queue metadata").ino(),
                queue_inode
            );
        }
        assert_eq!(read_json(&marker)["entry_id"], resumed.entry.entry_id());
        assert_eq!(
            read_json(&marker)["manifest_sha256"],
            case["manifest"]["sha256"]
        );
        assert!(!reservation.final_dir.join(MEDIA_FILE).exists());
        assert!(!reservation.staging_dir.exists());
    }
}

#[test]
fn unavailable_completed_resume_keeps_exact_marker_and_backend_ack() {
    let case = golden("unavailable");
    let bench = Bench::new("unavailable-marked-resume", "unavailable");
    let first = publish_case(&bench, "unavailable", &case).expect("first publication");
    let reservation = retained_media(&bench, &case);
    let marker = reservation.final_dir.join(TERMINAL_MARKER);
    let marker_bytes = fs::read(&marker).expect("marker");
    let marker_inode = fs::metadata(&marker).expect("marker metadata").ino();
    let manifest_inode = fs::metadata(&first.manifest_path)
        .expect("manifest metadata")
        .ino();
    assert!(
        bench
            .queue
            .acknowledge_backend(first.entry.entry_id(), 204)
            .expect("backend ACK")
    );

    let resumed = publish_reserved_unavailable(&bench, &reservation, &case).expect("resume");

    assert!(resumed.resumed && !resumed.admitted);
    assert_eq!(resumed.entry, first.entry);
    assert_eq!(fs::read(&marker).expect("marker"), marker_bytes);
    assert_eq!(
        fs::metadata(&marker).expect("marker metadata").ino(),
        marker_inode
    );
    assert_eq!(
        fs::metadata(&first.manifest_path)
            .expect("manifest metadata")
            .ino(),
        manifest_inode
    );
    assert!(
        bench.queued().is_empty(),
        "a valid marker keeps the backend ACK"
    );
    assert!(!reservation.final_dir.join(MEDIA_FILE).exists());
    assert!(!reservation.staging_dir.exists());
}

#[test]
fn unavailable_conflicting_or_malformed_marker_refuses_cleanup() {
    let case = golden("unavailable");
    for corruption in [
        "clip_id",
        "entry_id",
        "manifest_sha256",
        "local_state",
        "empty",
        "malformed",
        "extra-byte",
        "oversized",
        "directory",
        "symlink",
    ] {
        let bench = Bench::new(&format!("unavailable-marker-{corruption}"), "unavailable");
        let first = publish_case(&bench, "unavailable", &case).expect("first publication");
        let reservation = retained_media(&bench, &case);
        let marker = reservation.final_dir.join(TERMINAL_MARKER);
        let expected = fs::read(&marker).expect("marker");
        assert!(
            bench
                .queue
                .acknowledge_backend(first.entry.entry_id(), 204)
                .expect("backend ACK")
        );
        match corruption {
            "directory" => {
                fs::remove_file(&marker).expect("remove marker");
                fs::create_dir(&marker).expect("directory in place of marker");
            }
            "symlink" => {
                let target = bench.work.join("marker-target");
                fs::write(&target, &expected).expect("symlink target");
                fs::remove_file(&marker).expect("remove marker");
                symlink(&target, &marker).expect("symlink in place of marker");
            }
            _ => {
                let conflicting = match corruption {
                    "empty" => Vec::new(),
                    "malformed" => vec![b'{'; expected.len()],
                    "extra-byte" => [expected.as_slice(), b"\n"].concat(),
                    "oversized" => vec![b'x'; MAX_MANIFEST_BYTES + 1],
                    field => {
                        let mut value: Value =
                            serde_json::from_slice(&expected).expect("marker JSON");
                        value[field] = Value::from(
                            "x".repeat(value[field].as_str().expect("marker field").len()),
                        );
                        let mut bytes = serde_json::to_vec(&value).expect("marker JSON");
                        bytes.push(b'\n');
                        assert_eq!(bytes.len(), expected.len(), "same-sized conflicting marker");
                        bytes
                    }
                };
                fs::write(&marker, conflicting).expect("contradictory marker");
            }
        }
        let before = fs::symlink_metadata(&marker).expect("marker metadata");
        let bytes_before = before
            .is_file()
            .then(|| fs::read(&marker).expect("marker bytes"));

        let refused = publish_reserved_unavailable(&bench, &reservation, &case);

        assert!(
            matches!(refused, Err(PublishError::Conflict)),
            "{corruption}: {refused:?}"
        );
        assert_retained_media(&reservation);
        assert_eq!(
            fs::read(&first.manifest_path).expect("manifest"),
            first.manifest_bytes
        );
        let after = fs::symlink_metadata(&marker).expect("marker metadata");
        assert_eq!(
            after.ino(),
            before.ino(),
            "the contradictory marker is not replaced"
        );
        assert_eq!(after.file_type(), before.file_type());
        if let Some(bytes) = bytes_before {
            assert_eq!(fs::read(&marker).expect("marker bytes"), bytes);
        }
        assert!(bench.queued().is_empty());
    }
}

#[test]
fn ready_failures_keep_staged_or_final_media() {
    let case = golden("ready");
    for boundary in ["manifest", "queue", "marker", "conflicting-marker"] {
        let bench = Bench::new(&format!("ready-failure-{boundary}"), "ready");
        let meta = metadata(&case["inputs"]);
        let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, true);
        fs::create_dir(&reservation.final_dir).expect("final directory");
        let staged_copy = reservation.staging_dir.join("sealed-copy.mp4");
        fs::write(&staged_copy, MEDIA).expect("retained staged copy");
        let blocked = match boundary {
            "manifest" => reservation.final_dir.join(format!(".{MANIFEST_FILE}.tmp")),
            "queue" => {
                let lock = bench.queue.directory().join(LOCK_FILE_NAME);
                fs::remove_file(&lock).expect("remove queue lock");
                lock
            }
            "marker" => reservation
                .final_dir
                .join(format!(".{TERMINAL_MARKER}.tmp")),
            "conflicting-marker" => reservation.final_dir.join(TERMINAL_MARKER),
            _ => unreachable!(),
        };
        if boundary == "conflicting-marker" {
            fs::write(&blocked, b"contradictory marker").expect("conflicting marker");
            fs::write(reservation.final_dir.join(MEDIA_FILE), MEDIA).expect("retained final copy");
        } else {
            fs::create_dir(&blocked).expect("block publication");
        }

        let failed = Publisher::new(&bench.queue).publish_ready(&reservation, &meta, &media(&case));

        match boundary {
            "queue" => assert!(matches!(failed, Err(PublishError::Queue(_))), "{failed:?}"),
            "conflicting-marker" => {
                assert!(matches!(failed, Err(PublishError::Conflict)), "{failed:?}")
            }
            _ => assert!(matches!(failed, Err(PublishError::Io(_))), "{failed:?}"),
        }
        assert_eq!(
            fs::read(reservation.final_dir.join(MEDIA_FILE)).expect("final media"),
            MEDIA
        );
        assert_eq!(fs::read(&staged_copy).expect("staged copy"), MEDIA);
        if boundary == "conflicting-marker" {
            assert_eq!(
                fs::read(reservation.artifact_path()).expect("staged artifact"),
                MEDIA
            );
            assert_eq!(fs::read(&blocked).expect("marker"), b"contradictory marker");
        } else {
            assert!(
                !reservation.artifact_path().exists(),
                "the artifact survives at its final path"
            );
            assert!(!reservation.final_dir.join(TERMINAL_MARKER).exists());
            fs::remove_dir(&blocked).expect("remove publication obstacle");
        }
        let accepted_count = if boundary == "marker" { 1 } else { 0 };
        assert_eq!(bench.queued().len(), accepted_count);
    }
}

#[test]
fn unavailable_cleanup_errors_are_returned_after_terminal_publication() {
    let case = golden("unavailable");
    for cleanup in ["media", "staging"] {
        let bench = Bench::new(&format!("unavailable-cleanup-{cleanup}"), "unavailable");
        let reservation = retained_media(&bench, &case);
        let video = reservation.final_dir.join(MEDIA_FILE);
        if cleanup == "media" {
            fs::remove_file(&video).expect("remove final media");
            fs::create_dir(&video).expect("block final-media unlink");
        } else {
            fs::remove_file(reservation.artifact_path()).expect("remove artifact");
            fs::remove_dir(&reservation.staging_dir).expect("remove staging directory");
            fs::write(&reservation.staging_dir, MEDIA).expect("block staging-tree removal");
        }

        let failed = publish_reserved_unavailable(&bench, &reservation, &case);

        assert!(matches!(failed, Err(PublishError::Io(_))), "{failed:?}");
        assert_eq!(
            fs::read(reservation.final_dir.join(MANIFEST_FILE)).expect("manifest"),
            golden_manifest(&case)
        );
        assert!(reservation.final_dir.join(TERMINAL_MARKER).is_file());
        assert_eq!(bench.queued().len(), 1);
        if cleanup == "media" {
            assert!(video.is_dir());
            assert_eq!(
                fs::read(reservation.artifact_path()).expect("staged artifact"),
                MEDIA
            );
            fs::remove_dir(&video).expect("allow final-media cleanup");
        } else {
            assert!(
                !video.exists(),
                "media cleanup followed terminal publication"
            );
            assert_eq!(
                fs::read(&reservation.staging_dir).expect("staging bytes"),
                MEDIA
            );
            fs::remove_file(&reservation.staging_dir).expect("allow staging cleanup");
        }
        let resumed =
            publish_reserved_unavailable(&bench, &reservation, &case).expect("resume cleanup");
        assert!(resumed.resumed && !resumed.admitted);
        assert!(!video.exists());
        assert!(!reservation.staging_dir.exists());
    }
}

#[test]
fn conflicting_outcome_is_refused_and_keeps_the_manifest() {
    let case = golden("ready");
    let bench = Bench::new("conflict", "ready");
    let first = publish_case(&bench, "ready", &case).expect("ready publication");
    let meta = metadata(&case["inputs"]);
    let reservation = bench
        .store
        .reserve(&meta.camera_id, &meta.clip_id)
        .expect("re-reserve");
    let refused = Publisher::new(&bench.queue).publish_unavailable(
        &reservation,
        &meta,
        "ENCODER_FAILED",
        None,
    );
    assert!(
        matches!(refused, Err(PublishError::Conflict)),
        "got {refused:?}"
    );
    assert_eq!(
        fs::read(&first.manifest_path).expect("manifest"),
        first.manifest_bytes
    );
    assert_eq!(bench.queued().len(), 1);
}

#[test]
fn reservation_mismatch_is_refused_before_any_write() {
    let case = golden("ready");
    let bench = Bench::new("mismatch", "ready");
    let mut meta = metadata(&case["inputs"]);
    let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, true);
    meta.camera_id = "cmsnw6rjc01vhlh01oswn99zz".to_owned();
    let refused = Publisher::new(&bench.queue).publish_ready(&reservation, &meta, &media(&case));
    assert!(
        matches!(refused, Err(PublishError::Reservation(_))),
        "got {refused:?}"
    );
    assert!(!reservation.final_dir.join(MANIFEST_FILE).exists());
    assert!(reservation.artifact_path().is_file(), "media stays staged");
    assert!(bench.queued().is_empty());
}

#[test]
fn ready_without_media_is_refused_before_any_write() {
    let case = golden("ready");
    let bench = Bench::new("no-media", "ready");
    let meta = metadata(&case["inputs"]);
    let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, false);
    let refused = Publisher::new(&bench.queue).publish_ready(&reservation, &meta, &media(&case));
    assert!(
        matches!(refused, Err(PublishError::MissingMedia)),
        "got {refused:?}"
    );
    assert!(!reservation.final_dir.join(MANIFEST_FILE).exists());
    assert!(bench.queued().is_empty());
}

#[test]
fn corrupt_entry_builds_the_recorded_put_body() {
    let recorded = read_json(&wire().join("r/clip-put-unavailable-corrupt.json"));
    let case = golden("corrupt-missing");
    let bench = Bench::new("put-corrupt", "corrupt-missing");
    let mut meta = metadata(&case["inputs"]);
    meta.event_refs = strings(&recorded["event_refs"]);
    let reservation = bench.reserve(&meta.camera_id, &meta.clip_id, false);
    Publisher::new(&bench.queue)
        .publish_corrupt(&reservation, &meta)
        .expect("corrupt publication");
    let queued = bench.queued();
    let request = clip_request(&queued[0]).expect("PUT request");
    let body: Value = serde_json::from_slice(&request.body).expect("body is JSON");
    assert_eq!(body, recorded);
    assert_eq!(
        request.path(),
        format!("api/v1/relay/clips/{}", meta.clip_id)
    );
    assert_eq!(
        request.state_version,
        recorded["state_version"].as_i64().expect("version")
    );
}

#[test]
fn ready_entry_builds_a_ready_put_body() {
    let recorded = read_json(&wire().join("r/clip-put-ready.json"));
    let case = golden("ready");
    let bench = Bench::new("put-ready", "ready");
    let published = publish_case(&bench, "ready", &case).expect("ready publication");
    let request = clip_request(&bench.queued()[0]).expect("PUT request");
    let body: Value = serde_json::from_slice(&request.body).expect("body is JSON");
    let keys = |value: &Value| {
        value
            .as_object()
            .expect("object")
            .keys()
            .cloned()
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(&body), keys(&recorded));
    assert_eq!(body["state"], recorded["state"]);
    let facts = media(&case);
    assert_eq!(body["sha256"], Value::from(facts.sha256));
    assert_eq!(body["size_bytes"], Value::from(facts.size_bytes));
    assert_eq!(
        body["event_refs"],
        Value::from(published.entry.fields().event_ids.clone())
    );
}

const BACKEND_PARSE: &str = "
import json, sys
from backend.app.features.clips.manifest import parse_manifest_bytes
out = []
for path in sys.argv[1:]:
    with open(path, 'rb') as handle:
        manifest = parse_manifest_bytes(handle.read())
    if manifest is None:
        out.append({'verdict': 'rejected'})
    else:
        out.append({'as_response': manifest.as_response(),
                    'event_refs': list(manifest.event_refs), 'verdict': 'parsed'})
print(json.dumps(out))
";

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn backend_parses_every_published_manifest_to_the_golden_view() {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is required");
    let mut paths = Vec::new();
    let mut expected = Vec::new();
    for name in CASES {
        let case = golden(name);
        let bench = Bench::new(&format!("backend-{name}"), name);
        paths.push(
            publish_case(&bench, name, &case)
                .expect("publication")
                .manifest_path,
        );
        expected.push(case["backend"].clone());
    }
    let output = Command::new(python)
        .arg("-c")
        .arg(BACKEND_PARSE)
        .args(&paths)
        .output()
        .expect("python runs");
    assert!(
        output.status.success(),
        "backend parser exited with {:?}",
        output.status
    );
    let parsed: Value = serde_json::from_slice(&output.stdout).expect("parser output is JSON");
    assert_eq!(parsed, Value::from(expected));
}
