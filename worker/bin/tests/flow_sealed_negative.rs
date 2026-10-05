//! Unavailable flow-sealed observations on a real temp filesystem: round trip,
//! strict reader refusals, immutable publish-once, retention, and fresh
//! UNAVAILABLE publication through `clip_output`.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use seeon_ml_worker::clips::sealed::{
    Recovery, SealedClip, SealedContributor, SealedError, SealedEvent, SealedObservation,
    SealedSidecars, SealedUnavailable,
};

const CAMERA: &str = "cam-negative";

#[path = "support/sealed_negative_publication.rs"]
mod sealed_negative_publication;

fn sidecars(test: &str) -> SealedSidecars {
    let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("flow_negative-{test}"));
    if work.exists() {
        fs::remove_dir_all(&work).expect("clear previous run");
    }
    fs::create_dir_all(&work).expect("work");
    SealedSidecars::new(work.join("state/flow-sealed"))
}

/// `<work>` of a bench made by `sidecars`.
fn work_of(sidecars: &SealedSidecars) -> &Path {
    sidecars
        .directory()
        .parent()
        .and_then(Path::parent)
        .expect("work")
}

fn event(identity: &str, camera_id: &str) -> SealedEvent {
    SealedEvent {
        domain: "fall".to_owned(),
        event_type: "fall_detected".to_owned(),
        identity: identity.to_owned(),
        camera_id: camera_id.to_owned(),
        facility_id: "facility-1".to_owned(),
        time_sec: 12.5,
        probability: Some(0.5),
    }
}

fn events() -> BTreeMap<String, SealedEvent> {
    BTreeMap::from([("evt-1".to_owned(), event("evt-1", CAMERA))])
}

fn contributor(event_ref: &str) -> SealedContributor {
    let detected_at = "2026-08-17T09:00:00.000000+00:00".to_owned();
    SealedContributor {
        event_ref: event_ref.to_owned(),
        detected_at,
    }
}

fn negative(clip_id: &str, duration_ms: u64, result: i32, video: bool) -> SealedUnavailable {
    SealedUnavailable {
        clip_id: clip_id.to_owned(),
        path: None,
        duration_ms,
        boundary: "none".to_owned(),
        contributors: vec![contributor("evt-1")],
        native_result: result,
        contains_video: video,
    }
}

fn persist(sidecars: &SealedSidecars, sealed: &SealedUnavailable) -> Result<PathBuf, SealedError> {
    sidecars.persist(&SealedObservation::Unavailable(sealed.clone()), &events())
}

/// Asserts a refusal by its typed variant name.
fn refused<T: Debug, E: Debug>(result: Result<T, E>, variant: &str) {
    let text = format!("{result:?}");
    assert!(
        text.starts_with("Err(") && text.contains(variant),
        "{variant}: {text}"
    );
}

fn only_recovery(sidecars: &SealedSidecars) -> Recovery {
    let pending = sidecars.pending(CAMERA).expect("pending");
    assert!(pending.malformed.is_empty(), "{:?}", pending.malformed);
    assert_eq!(pending.recoveries.len(), 1);
    pending.recoveries.into_iter().next().expect("recovery")
}

#[test]
fn unavailable_round_trips_full_u64_zero_and_unknown_native_results() {
    let mut located = negative("neg-located", 0, 0, false);
    located.path = Some("/srv/seeon/flow-out/neg-located.mp4".to_owned());
    let max = negative("neg-max", u64::MAX, i32::MIN, false);
    for sealed in [max, located, negative("neg-unknown", 7, 9_999, true)] {
        let sidecars = sidecars(&sealed.clip_id);
        let bytes = fs::read(persist(&sidecars, &sealed).expect("persist")).expect("bytes");
        let document: Value = serde_json::from_slice(&bytes).expect("JSON");
        let keys: Vec<&String> = document.as_object().expect("object").keys().collect();
        assert_eq!(keys, ["unavailable"]);
        let inner = &document["unavailable"];
        assert_eq!(inner["duration_ms"], json!(sealed.duration_ms));
        assert_eq!(inner["native_result"], json!(sealed.native_result));
        let path = inner.get("path").and_then(Value::as_str);
        assert_eq!(
            path,
            sealed.path.as_deref(),
            "absent path is omitted, not null"
        );
        let recovery = only_recovery(&sidecars);
        assert_eq!(
            recovery.sealed,
            SealedObservation::Unavailable(sealed.clone())
        );
        assert_eq!(
            recovery.sealed.duration_ms(),
            i128::from(sealed.duration_ms)
        );
        assert_eq!(recovery.sealed.path(), sealed.path.as_deref());
        assert_eq!(
            (recovery.camera_id, recovery.events),
            (CAMERA.to_owned(), events())
        );
    }
}

#[test]
fn persist_refuses_invalid_unavailable_observations_without_writing() {
    let mut empty_path = negative("neg-empty-path", 1, 0, false);
    empty_path.path = Some(String::new());
    let mut two_cameras = negative("neg-two-cameras", 1, 1, false);
    two_cameras.contributors.push(contributor("evt-2"));
    let mut mixed = events();
    mixed.insert("evt-2".to_owned(), event("evt-2", "cam-other"));
    let renamed = BTreeMap::from([("evt-1".to_owned(), event("evt-other", CAMERA))]);
    let cases = [
        (
            negative("neg-zero-video", 1, 0, true),
            events(),
            "InvalidObservation",
        ),
        (empty_path, events(), "InvalidObservation"),
        (two_cameras, mixed, "InvalidAttribution"),
        (
            negative("neg-renamed", 1, 1, false),
            renamed,
            "InvalidAttribution",
        ),
        (
            negative("neg-unknown-ref", 1, 1, false),
            BTreeMap::new(),
            "UnknownRef",
        ),
        (negative(".hidden", 1, 1, false), events(), "InvalidClipId"),
    ];
    let sidecars = sidecars("refusals");
    for (sealed, events, variant) in cases {
        let observation = SealedObservation::Unavailable(sealed);
        refused(sidecars.persist(&observation, &events), variant);
        assert!(!sidecars.directory().exists(), "{variant}: nothing written");
    }
}

#[test]
fn reader_keeps_ambiguous_or_mistyped_negative_sidecars_as_malformed() {
    let sidecars = sidecars("reader");
    let base_path = persist(&sidecars, &negative("neg-base", 3, 0, false)).expect("persist");
    let base: Value = serde_json::from_slice(&fs::read(base_path).expect("base")).expect("JSON");
    let mut renamed_events = base["unavailable"]["events"].clone();
    renamed_events[0]["identity"] = json!("evt-x");
    let fields = [
        ("unknown-field", "reason_code", json!("NO_FRAMES")),
        ("null-path", "path", Value::Null),
        ("empty-path", "path", json!("")),
        ("string-bool", "contains_video", json!("false")),
        ("int-bool", "contains_video", json!(0)),
        ("zero-video", "contains_video", json!(true)),
        ("wide-native", "native_result", json!(2_147_483_648_i64)),
        ("float-native", "native_result", json!(1.0)),
        ("negative-duration", "duration_ms", json!(-1)),
        ("null-camera", "camera_id", Value::Null),
        ("other-camera", "camera_id", json!("cam-other")),
        ("no-contributors", "contributors", json!([])),
        ("event-identity", "events", renamed_events),
    ];
    let mut forged = vec![("neg-renamed".to_owned(), base.clone())];
    for (name, key, value) in fields {
        let mut document = base.clone();
        document["unavailable"]["clip_id"] = json!(format!("neg-{name}"));
        document["unavailable"][key] = value;
        forged.push((format!("neg-{name}"), document));
    }
    let mut mixed = base.clone();
    mixed["unavailable"]["clip_id"] = json!("neg-mixed");
    mixed["clip_id"] = json!("neg-mixed");
    let mut flattened = base["unavailable"].clone();
    flattened["clip_id"] = json!("neg-flattened");
    flattened["path"] = json!("/srv/seeon/flow-out/neg-flattened.mp4");
    forged.extend([
        ("neg-mixed".to_owned(), mixed),
        ("neg-flattened".to_owned(), flattened),
    ]);
    let mut kept: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    for (clip_id, document) in forged {
        let path = sidecars.directory().join(format!("{clip_id}.json"));
        fs::write(&path, document.to_string()).expect("forged");
        kept.push((path, document.to_string().into_bytes()));
    }
    kept.sort();
    let pending = sidecars.pending(CAMERA).expect("pending");
    let paths: Vec<PathBuf> = kept.iter().map(|(path, _)| path.clone()).collect();
    assert_eq!(pending.malformed, paths);
    assert_eq!(pending.recoveries.len(), 1);
    assert_eq!(pending.recoveries[0].sealed.clip_id(), "neg-base");
    for (path, bytes) in &kept {
        assert_eq!(&fs::read(path).expect("kept"), bytes, "{}", path.display());
    }
}

#[test]
fn immutable_publication_accepts_exact_retries_and_preserves_contradictions() {
    let sidecars = sidecars("immutable");
    let sealed = negative("neg-once", 5, 3, false);
    let path = persist(&sidecars, &sealed).expect("first");
    let bytes = fs::read(&path).expect("bytes");
    assert_eq!(persist(&sidecars, &sealed).expect("retry"), path);
    assert_eq!(
        fs::read_dir(sidecars.directory()).expect("dir").count(),
        1,
        "no staging"
    );
    let ready = SealedObservation::Ready(SealedClip {
        clip_id: "neg-once".to_owned(),
        path: "/srv/seeon/flow-out/neg-once.mp4".to_owned(),
        duration_ms: 5,
        boundary: "none".to_owned(),
        contributors: vec![contributor("evt-1")],
    });
    refused(sidecars.persist(&ready, &events()), "ImmutableConflict");
    let changed = negative("neg-once", 6, 3, false);
    refused(persist(&sidecars, &changed), "ImmutableConflict");
    assert_eq!(fs::read(&path).expect("kept"), bytes);
    fs::write(&path, b"{").expect("malformed");
    refused(persist(&sidecars, &sealed), "ImmutableConflict");
    assert_eq!(fs::read(&path).expect("malformed kept"), b"{");

    let outside = work_of(&sidecars).join("outside.json");
    fs::write(&outside, &bytes).expect("outside");
    let link = sidecars.directory().join("neg-link.json");
    symlink(&outside, &link).expect("symlink");
    fs::create_dir(sidecars.directory().join("neg-dir.json")).expect("directory target");
    for clip_id in ["neg-link", "neg-dir"] {
        let target = negative(clip_id, 5, 3, false);
        refused(persist(&sidecars, &target), "ImmutableUnreadable");
    }
    assert!(
        link.symlink_metadata()
            .expect("link")
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read(&outside).expect("outside kept"), bytes);
    assert!(sidecars.directory().join("neg-dir.json").is_dir());
}
