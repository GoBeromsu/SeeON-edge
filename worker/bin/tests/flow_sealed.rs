//! T35: flow-sealed sidecars (G16) against the `d/flow-sealed` goldens.
//!
//! Sidecars are written into a fresh `<work>/state/flow-sealed` under
//! `CARGO_TARGET_TMPDIR`; their bytes, the directory layout and the
//! `pending_for_camera` view are compared with values recorded from the
//! Python writer. Replay runs with a recording closure, or with the real T24
//! publisher where a published clip is the observable. The backend lane also
//! reads the Rust-written sidecars with the Python reader.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Map, Value};

use seeon_ml_worker::clips::durable::sha256_file;
use seeon_ml_worker::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use seeon_ml_worker::clips::manifest::{Contributor, Extension, MediaFacts};
use seeon_ml_worker::clips::publish::{PublishError, Published, Publisher};
use seeon_ml_worker::clips::sealed::{
    Recovery, ReplayReport, SealedClip, SealedContributor, SealedError, SealedEvent, SealedSidecars,
};
use seeon_ml_worker::clips::store::ClipStore;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::delivery::DeliveryQueue;

const A: &str = "sealed-a-non-ascii";
const B: &str = "sealed-b-out-of-order";

fn wire() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/worker-wire")
}

fn cases() -> Value {
    let bytes = fs::read(wire().join("d/flow-sealed/cases.json")).expect("cases.json");
    serde_json::from_slice(&bytes).expect("cases.json is JSON")
}

fn call<'a>(cases: &'a Value, case: &str) -> &'a Value {
    let calls = cases["calls"].as_array().expect("calls");
    calls
        .iter()
        .find(|call| call["case"] == case)
        .unwrap_or_else(|| panic!("no call {case}"))
}

fn text(value: &Value, key: &str) -> String {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is text"))
        .to_owned()
}

fn sealed_of(call: &Value) -> SealedClip {
    let sealed = &call["sealed"];
    let contributors = sealed["contributors"].as_array().expect("contributors");
    SealedClip {
        clip_id: text(sealed, "clip_id"),
        path: text(sealed, "path"),
        duration_ms: sealed["duration_ms"].as_i64().expect("duration_ms"),
        boundary: text(sealed, "boundary"),
        contributors: contributors
            .iter()
            .map(|c| SealedContributor {
                event_ref: text(c, "event_ref"),
                detected_at: text(c, "detected_at"),
            })
            .collect(),
    }
}

/// `BusinessEvent` inputs; an integer identity is written as its text.
fn events_of(call: &Value) -> BTreeMap<String, SealedEvent> {
    let events = call["events"].as_object().expect("events");
    events
        .iter()
        .map(|(event_ref, event)| {
            let identity = match &event["identity"] {
                Value::String(identity) => identity.clone(),
                Value::Number(identity) => identity.to_string(),
                other => panic!("identity {other}"),
            };
            let sealed = SealedEvent {
                domain: text(event, "domain"),
                event_type: text(event, "event_type"),
                identity,
                camera_id: text(event, "camera_id"),
                facility_id: text(event, "facility_id"),
                time_sec: event["time_sec"].as_f64().expect("time_sec"),
                probability: match &event["probability"] {
                    Value::Null => None,
                    Value::Number(number) => Some(number.as_f64().expect("finite probability")),
                    other => panic!("probability {other}"),
                },
            };
            (event_ref.clone(), sealed)
        })
        .collect()
}

fn golden_sidecar(clip_id: &str) -> Vec<u8> {
    fs::read(wire().join(format!("d/flow-sealed/state/{clip_id}.json"))).expect("golden sidecar")
}

fn mode(path: &Path) -> String {
    let mode = fs::metadata(path).expect("metadata").permissions().mode();
    format!("0o{:o}", mode & 0o777)
}

/// One test's `<work>`: the sidecar directory plus a clip store and queue.
struct Bench {
    work: PathBuf,
    sidecars: SealedSidecars,
}

impl Bench {
    fn new(test: &str) -> Self {
        let work = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("flow_sealed-{test}"));
        if work.exists() {
            fs::remove_dir_all(&work).expect("clear previous run");
        }
        fs::create_dir_all(&work).expect("work");
        let sidecars = SealedSidecars::new(work.join("state/flow-sealed"));
        Self { work, sidecars }
    }

    fn place(&self, text: &str) -> String {
        text.replacen("<work>", self.work.to_str().expect("utf-8 path"), 1)
    }

    fn persist(&self, cases: &Value, case: &str) -> PathBuf {
        let call = call(cases, case);
        self.sidecars
            .persist(&sealed_of(call), &events_of(call))
            .expect("persist")
    }

    /// Persists `case` with its media path moved under `<work>/media`; the
    /// media file is created when `with_media`.
    fn persist_with_media(&self, cases: &Value, case: &str, with_media: bool) -> PathBuf {
        let call = call(cases, case);
        let mut sealed = sealed_of(call);
        let media = self
            .work
            .join("media")
            .join(format!("{}.mp4", sealed.clip_id));
        if with_media {
            fs::create_dir_all(media.parent().expect("parent")).expect("media dir");
            fs::write(&media, format!("media of {}", sealed.clip_id)).expect("media");
        }
        sealed.path = media.to_str().expect("utf-8").to_owned();
        self.sidecars
            .persist(&sealed, &events_of(call))
            .expect("persist")
    }

    fn sidecar_names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(self.sidecars.directory())
            .expect("sidecar directory")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .into_string()
                    .expect("utf-8")
            })
            .collect();
        names.sort();
        names
    }

    fn camera(&self, cases: &Value) -> String {
        text(&cases["pending_for_camera"], "camera_id")
    }
}

/// The golden `pending_for_camera` item for a recovery (fields the sidecar
/// carries; the reader's absent optional fields are the golden's nulls).
fn view(recovery: &Recovery) -> Value {
    let sealed = &recovery.sealed;
    let contributors: Vec<Value> = sealed
        .contributors
        .iter()
        .map(|c| serde_json::json!({"detected_at": c.detected_at, "event_ref": c.event_ref}))
        .collect();
    let events: Map<String, Value> = recovery
        .events
        .iter()
        .map(|(event_ref, e)| {
            let event = serde_json::json!({
                "domain": e.domain, "event_type": e.event_type, "identity": e.identity,
                "camera_id": e.camera_id, "facility_id": e.facility_id,
                "time_sec": e.time_sec, "probability": e.probability,
            });
            (event_ref.clone(), event)
        })
        .collect();
    serde_json::json!({
        "camera_id": recovery.camera_id,
        "events": events,
        "sealed": {
            "boundary": sealed.boundary, "clip_id": sealed.clip_id,
            "contributors": contributors, "duration_ms": sealed.duration_ms, "path": sealed.path,
        },
        "sidecar_path": recovery.sidecar_path.to_str().expect("utf-8"),
    })
}

fn without_nulls(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), without_nulls(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(without_nulls).collect()),
        other => other.clone(),
    }
}

fn golden_pending(bench: &Bench, cases: &Value) -> Vec<Value> {
    let result = cases["pending_for_camera"]["result"]
        .as_array()
        .expect("result");
    result
        .iter()
        .map(|item| {
            let mut item = without_nulls(item);
            let path = bench.place(&text(&item, "sidecar_path"));
            item["sidecar_path"] = Value::from(path);
            item
        })
        .collect()
}

#[test]
fn persist_writes_golden_bytes_in_a_private_directory() {
    let cases = cases();
    let bench = Bench::new("persist");
    let camera = bench.camera(&cases);
    assert_eq!(
        cases["pending_for_camera_before_any_persist"],
        Value::Array(vec![])
    );
    let before = bench
        .sidecars
        .pending(&camera)
        .expect("pending before persist");
    assert!(before.recoveries.is_empty() && before.malformed.is_empty());

    let refused = call(&cases, "unknown-contributor");
    assert_eq!(refused["result"]["verdict"], "refused");
    let outcome = bench
        .sidecars
        .persist(&sealed_of(refused), &events_of(refused));
    assert!(
        matches!(outcome, Err(SealedError::UnknownRef)),
        "{outcome:?}"
    );
    assert_eq!(refused["directory_after"]["exists"], false);
    assert!(
        !bench.sidecars.directory().exists(),
        "refusal creates nothing"
    );

    for case in ["out-of-order", "non-ascii"] {
        let golden = call(&cases, case);
        let returned = bench.persist(&cases, case);
        let result = &golden["result"];
        assert_eq!(
            returned.to_str().expect("utf-8"),
            bench.place(&text(result, "returned_path"))
        );
        let clip_id = text(&golden["sealed"], "clip_id");
        assert_eq!(
            fs::read(&returned).expect("sidecar"),
            golden_sidecar(&clip_id)
        );

        let after = &golden["directory_after"];
        assert_eq!(mode(bench.sidecars.directory()), text(after, "mode"));
        let entries = after["entries"].as_array().expect("entries");
        let names: Vec<String> = entries.iter().map(|e| text(e, "name")).collect();
        assert_eq!(
            bench.sidecar_names(),
            names,
            "{case}: no temporary file remains"
        );
        for entry in entries {
            let path = bench.sidecars.directory().join(text(entry, "name"));
            assert_eq!(mode(&path), text(entry, "mode"));
            assert_eq!(
                Some(fs::metadata(&path).expect("size").len()),
                entry["size_bytes"].as_u64()
            );
        }
    }
}

#[test]
fn pending_reads_sidecars_in_file_name_order_matching_the_golden_view() {
    let cases = cases();
    let bench = Bench::new("pending");
    bench.persist(&cases, "out-of-order");
    bench.persist(&cases, "non-ascii");
    let pending = bench
        .sidecars
        .pending(&bench.camera(&cases))
        .expect("pending");
    assert!(pending.malformed.is_empty());
    let views: Vec<Value> = pending.recoveries.iter().map(view).collect();
    assert_eq!(views, golden_pending(&bench, &cases));

    let other = bench.sidecars.pending("another-camera").expect("pending");
    assert!(other.recoveries.is_empty() && other.malformed.is_empty());
}

#[test]
fn replay_publishes_in_file_name_order_and_a_second_replay_does_nothing() {
    let cases = cases();
    let bench = Bench::new("replay");
    let camera = bench.camera(&cases);
    bench.persist_with_media(&cases, "out-of-order", true);
    bench.persist_with_media(&cases, "non-ascii", true);
    let seen = RefCell::new(Vec::new());
    let report = bench
        .sidecars
        .replay(&camera, |recovery| {
            assert!(
                recovery.sidecar_path.is_file(),
                "retired only after the PUT"
            );
            seen.borrow_mut().push(recovery.sealed.clip_id.clone());
            Ok::<(), ()>(())
        })
        .expect("replay");
    assert_eq!(*seen.borrow(), [A, B]);
    let expected = ReplayReport {
        published: 2,
        ..ReplayReport::default()
    };
    assert_eq!(report, expected);
    assert!(bench.sidecar_names().is_empty(), "both sidecars retired");

    let again = bench
        .sidecars
        .replay(&camera, |recovery| -> Result<(), ()> {
            panic!("second replay issued a PUT for {}", recovery.sealed.clip_id)
        })
        .expect("second replay");
    assert_eq!(again, ReplayReport::default());
}

fn now() -> Utc {
    Utc::parse("2026-08-17T10:00:00.000000+00:00").expect("now")
}

/// The flow publication a recovery stands for, through the T24 publisher.
fn publish_recovery(
    store: &ClipStore,
    queue: &DeliveryQueue,
    recovery: &Recovery,
) -> Result<Published, PublishError> {
    let sealed = &recovery.sealed;
    let events: BTreeMap<String, ContributorEvent> = recovery
        .events
        .iter()
        .map(|(event_ref, event)| {
            let event = ContributorEvent {
                camera_id: event.camera_id.clone(),
                facility_id: event.facility_id.clone(),
                domain: event.domain.clone(),
                event_type: event.event_type.clone(),
            };
            (event_ref.clone(), event)
        })
        .collect();
    let extension = Extension {
        boundary: sealed.boundary.clone(),
        contributors: sealed
            .contributors
            .iter()
            .map(|c| Contributor {
                event_ref: c.event_ref.clone(),
                detected_at: Utc::parse(&c.detected_at).expect("detected_at"),
            })
            .collect(),
        duration_ms: sealed.duration_ms,
    };
    let meta = flow_metadata(&sealed.clip_id, &events, extension, FLOW_ENCODER, now())
        .expect("flow metadata");
    // Independent oracle: the earliest contributor by detected_at fixes the
    // event fields (Python `_publish` sorts before it reads `events`).
    let mut by_time: Vec<_> = sealed.contributors.iter().collect();
    by_time.sort_by_key(|c| Utc::parse(&c.detected_at).expect("detected_at"));
    let earliest = &recovery.events[&by_time[0].event_ref];
    assert_eq!(meta.event_refs[0], by_time[0].event_ref);
    assert_eq!(meta.camera_id, earliest.camera_id);
    assert_eq!(meta.facility_id, earliest.facility_id);
    assert_eq!(meta.domain, earliest.domain);
    assert_eq!(meta.event_type, earliest.event_type);
    let reservation = store.reserve(&meta.camera_id, &meta.clip_id)?;
    fs::copy(&sealed.path, reservation.artifact_path()).map_err(PublishError::Io)?;
    let (sha256, size) = sha256_file(&reservation.artifact_path()).map_err(PublishError::Io)?;
    let media = MediaFacts {
        sha256,
        size_bytes: i64::try_from(size).expect("size"),
        codec: "h264".to_owned(),
        duration_ms: sealed.duration_ms,
    };
    Publisher::new(queue).publish_ready(&reservation, &meta, &media)
}

#[derive(Clone, Copy, Debug)]
enum Kill {
    AfterPersist,
    BeforePut,
    AfterPutBeforeRetire,
}

#[test]
fn kill_between_persist_and_put_publishes_exactly_one_clip() {
    let cases = cases();
    for kill in [
        Kill::AfterPersist,
        Kill::BeforePut,
        Kill::AfterPutBeforeRetire,
    ] {
        let bench = Bench::new(&format!("kill-{kill:?}"));
        let camera = bench.camera(&cases);
        let store = ClipStore::new(bench.work.join("store"));
        let queue_dir = bench.work.join("store/delivery-queue");
        fs::create_dir_all(&queue_dir).expect("queue dir");
        bench.persist_with_media(&cases, "non-ascii", true);
        if !matches!(kill, Kill::AfterPersist) {
            let queue = DeliveryQueue::open(&queue_dir, true).expect("queue");
            let killed = catch_unwind(AssertUnwindSafe(|| {
                bench
                    .sidecars
                    .replay(&camera, |recovery| -> Result<(), PublishError> {
                        if let Kill::AfterPutBeforeRetire = kill {
                            publish_recovery(&store, &queue, recovery)?;
                            panic!("killed before the retire");
                        }
                        panic!("killed before the PUT")
                    })
            }));
            assert!(killed.is_err(), "{kill:?}: the worker died");
        }

        let queue = DeliveryQueue::open(&queue_dir, true).expect("reopened queue");
        let sidecars = SealedSidecars::new(bench.sidecars.directory());
        let report = sidecars
            .replay(&camera, |recovery| {
                publish_recovery(&store, &queue, recovery)
            })
            .expect("replay after restart");
        assert_eq!(report.published, 1, "{kill:?}: {report:?}");
        let clips: Vec<Value> = queue
            .entries()
            .expect("entries")
            .into_iter()
            .filter(|entry| entry["clip_id"] == A)
            .collect();
        assert_eq!(clips.len(), 1, "{kill:?}: exactly one clip");
        assert!(
            bench.sidecar_names().is_empty(),
            "{kill:?}: sidecar retired"
        );
    }
}

#[test]
fn missing_media_removes_the_sidecar_and_counts_it() {
    let cases = cases();
    let bench = Bench::new("missing-media");
    bench.persist_with_media(&cases, "non-ascii", false);
    let report = bench
        .sidecars
        .replay(&bench.camera(&cases), |recovery| -> Result<(), ()> {
            panic!("no PUT without media: {}", recovery.sealed.clip_id)
        })
        .expect("replay");
    let expected = ReplayReport {
        missing_media: 1,
        ..ReplayReport::default()
    };
    assert_eq!(report, expected);
    assert!(bench.sidecar_names().is_empty(), "sidecar removed");
}

#[test]
fn failed_put_keeps_the_sidecar_and_continues() {
    let cases = cases();
    let bench = Bench::new("failed-put");
    let camera = bench.camera(&cases);
    let kept = bench.persist_with_media(&cases, "non-ascii", true);
    bench.persist_with_media(&cases, "out-of-order", true);
    let bytes = fs::read(&kept).expect("sidecar a");
    let report = bench
        .sidecars
        .replay(&camera, |recovery| {
            if recovery.sealed.clip_id == A {
                Err("relay refused")
            } else {
                Ok(())
            }
        })
        .expect("replay");
    let expected = ReplayReport {
        published: 1,
        failed: 1,
        ..ReplayReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(bench.sidecar_names(), [format!("{A}.json")]);
    assert_eq!(fs::read(&kept).expect("sidecar a"), bytes, "kept unchanged");

    let seen = RefCell::new(Vec::new());
    bench
        .sidecars
        .replay(&camera, |recovery| {
            seen.borrow_mut().push(recovery.sealed.clip_id.clone());
            Ok::<(), ()>(())
        })
        .expect("later replay");
    assert_eq!(*seen.borrow(), [A]);
}

#[test]
fn malformed_sidecars_are_kept_unchanged_and_counted() {
    let cases = cases();
    let bench = Bench::new("malformed");
    let camera = bench.camera(&cases);
    bench.persist_with_media(&cases, "out-of-order", true);
    let directory = bench.sidecars.directory().to_path_buf();
    let truncated = directory.join("0-truncated.json");
    let golden = golden_sidecar(A);
    fs::write(&truncated, &golden[..golden.len() / 2]).expect("truncated");
    let mut missing_key: Value = serde_json::from_slice(&golden).expect("golden sidecar");
    missing_key
        .as_object_mut()
        .expect("object")
        .remove("boundary");
    let missing_key_path = directory.join("1-missing-key.json");
    fs::write(&missing_key_path, missing_key.to_string()).expect("missing key");
    let malformed = [
        (
            truncated,
            fs::read(directory.join("0-truncated.json")).expect("bytes"),
        ),
        (
            missing_key_path.clone(),
            fs::read(&missing_key_path).expect("bytes"),
        ),
    ];

    let pending = bench.sidecars.pending(&camera).expect("pending");
    let paths: Vec<PathBuf> = malformed.iter().map(|(path, _)| path.clone()).collect();
    assert_eq!(pending.malformed, paths);

    let report = bench
        .sidecars
        .replay(&camera, |_| Ok::<(), ()>(()))
        .expect("replay");
    let expected = ReplayReport {
        published: 1,
        malformed: 2,
        ..ReplayReport::default()
    };
    assert_eq!(report, expected);
    for (path, bytes) in &malformed {
        assert_eq!(&fs::read(path).expect("kept"), bytes, "{}", path.display());
    }
}

fn event_with(identity: &str, probability: Option<f64>) -> SealedEvent {
    SealedEvent {
        domain: "fall".to_owned(),
        event_type: "fall_detected".to_owned(),
        identity: identity.to_owned(),
        camera_id: "cam-optional".to_owned(),
        facility_id: "facility-1".to_owned(),
        time_sec: 12.5,
        probability,
    }
}

fn clip_with(clip_id: &str, refs: &[&str]) -> SealedClip {
    SealedClip {
        clip_id: clip_id.to_owned(),
        path: format!("/srv/seeon/flow-out/{clip_id}.mp4"),
        duration_ms: 4_000,
        boundary: "end".to_owned(),
        contributors: refs
            .iter()
            .enumerate()
            .map(|(index, event_ref)| SealedContributor {
                event_ref: (*event_ref).to_owned(),
                detected_at: format!("2026-08-17T09:00:0{index}.000000+00:00"),
            })
            .collect(),
    }
}

fn persisted_events(bench: &Bench, clip_id: &str) -> BTreeMap<String, SealedEvent> {
    let pending = bench.sidecars.pending("cam-optional").expect("pending");
    assert!(pending.malformed.is_empty());
    let recovery = pending
        .recoveries
        .iter()
        .find(|recovery| recovery.sealed.clip_id == clip_id)
        .unwrap_or_else(|| panic!("{clip_id} was not recovered"));
    recovery.events.clone()
}

#[test]
fn absent_probability_persists_as_null_and_zero_stays_present() {
    let bench = Bench::new("optional-probability");
    let absent = event_with("evt-absent", None);
    let zero = event_with("evt-zero", Some(0.0));
    let present = event_with("evt-present", Some(0.875));
    let clip = clip_with(
        "sealed-optional-probability",
        &["evt-absent", "evt-zero", "evt-present"],
    );
    let events = BTreeMap::from([
        ("evt-absent".to_owned(), absent.clone()),
        ("evt-zero".to_owned(), zero.clone()),
        ("evt-present".to_owned(), present.clone()),
    ]);
    let path = bench.sidecars.persist(&clip, &events).expect("persist");
    let bytes = fs::read(&path).expect("sidecar bytes");
    let document: Value = serde_json::from_slice(&bytes).expect("sidecar JSON");
    let written: Vec<&Value> = document["events"]
        .as_array()
        .expect("events")
        .iter()
        .collect();
    assert_eq!(written[0]["probability"], Value::Null);
    assert_eq!(written[1]["probability"], serde_json::json!(0.0));
    assert_eq!(written[2]["probability"], serde_json::json!(0.875));
    assert_ne!(written[0]["probability"], written[1]["probability"]);

    let recovered = persisted_events(&bench, "sealed-optional-probability");
    assert_eq!(recovered["evt-absent"].probability, None);
    assert_eq!(recovered["evt-zero"].probability, Some(0.0));
    assert_eq!(recovered["evt-present"].probability, Some(0.875));
    assert_eq!(recovered["evt-absent"], absent);
    assert_eq!(recovered["evt-zero"], zero);
}

#[test]
fn nonfinite_probability_is_refused_and_invalid_shapes_stay_malformed() {
    let bench = Bench::new("invalid-probability");
    let clip = clip_with("sealed-nonfinite", &["evt-nan"]);
    let events = BTreeMap::from([("evt-nan".to_owned(), event_with("evt-nan", Some(f64::NAN)))]);
    let refused = bench.sidecars.persist(&clip, &events);
    assert!(
        matches!(refused, Err(SealedError::NonFinite)),
        "{refused:?}"
    );
    assert!(
        !bench
            .sidecars
            .directory()
            .join("sealed-nonfinite.json")
            .exists(),
        "non-finite persist writes nothing"
    );

    let legal = clip_with("sealed-legal-zero", &["evt-zero"]);
    let legal_events = BTreeMap::from([("evt-zero".to_owned(), event_with("evt-zero", Some(0.0)))]);
    let legal_path = bench
        .sidecars
        .persist(&legal, &legal_events)
        .expect("legal persist");
    let legal_bytes = fs::read(&legal_path).expect("legal bytes");
    let document: Value = serde_json::from_slice(&legal_bytes).expect("legal JSON");
    let shapes = [
        (Value::Bool(false), "bool"),
        (serde_json::json!({"value": 0.0}), "object"),
        (Value::String("0.0".to_owned()), "string"),
        (serde_json::json!(["0.0"]), "array"),
    ];
    for (shape, name) in shapes {
        let mut forged = document.clone();
        forged["clip_id"] = Value::from(format!("sealed-{name}"));
        forged["events"][0]["identity"] = Value::from(format!("evt-{name}"));
        forged["events"][0]["probability"] = shape;
        forged["contributors"][0]["event_ref"] = Value::from(format!("evt-{name}"));
        let forged_path = bench
            .sidecars
            .directory()
            .join(format!("sealed-{name}.json"));
        fs::write(&forged_path, forged.to_string()).expect("forged sidecar");
    }
    let mut missing = document.clone();
    missing["clip_id"] = Value::from("sealed-missing");
    missing["events"][0]["identity"] = Value::from("evt-missing");
    missing["contributors"][0]["event_ref"] = Value::from("evt-missing");
    missing["events"][0]
        .as_object_mut()
        .expect("event")
        .remove("probability");
    fs::write(
        bench.sidecars.directory().join("sealed-missing.json"),
        missing.to_string(),
    )
    .expect("missing probability");

    let pending = bench.sidecars.pending("cam-optional").expect("pending");
    assert_eq!(pending.recoveries.len(), 1);
    assert_eq!(
        pending.recoveries[0].events["evt-zero"].probability,
        Some(0.0)
    );
    let mut malformed: Vec<String> = pending
        .malformed
        .iter()
        .map(|path| {
            path.file_name()
                .expect("name")
                .to_str()
                .expect("utf-8")
                .to_owned()
        })
        .collect();
    malformed.sort();
    assert_eq!(
        malformed,
        [
            "sealed-array.json",
            "sealed-bool.json",
            "sealed-missing.json",
            "sealed-object.json",
            "sealed-string.json",
        ]
    );
    assert_eq!(fs::read(&legal_path).expect("legal kept"), legal_bytes);
}

/// Python `_event_payload` through `FlowSealedSidecars.persist`. The current
/// Python `BusinessEvent.probability` annotation is `float`, so the oracle
/// compares a legal zero; null is the Rust wire for an absent value.
const PROBABILITY_ORACLE: &str = r#"
import json, pathlib, sys
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.smart_record_actor import ClipContributor, ClipSealed
from worker.types import BusinessEvent
spec = json.loads(sys.argv[1])
events = {
    item["identity"]: BusinessEvent(
        domain=item["domain"], event_type=item["event_type"], identity=item["identity"],
        camera_id=item["camera_id"], facility_id=item["facility_id"],
        time_sec=item["time_sec"], probability=item["probability"],
    )
    for item in spec["events"]
}
sealed = ClipSealed(
    clip_id=spec["clip_id"], path=spec["path"], duration_ms=spec["duration_ms"],
    contributors=tuple(ClipContributor(**item) for item in spec["contributors"]),
    boundary=spec["boundary"],
)
print(FlowSealedSidecars(pathlib.Path(sys.argv[2])).persist(sealed, events).read_text(encoding="utf-8"), end="")
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with the worker package on PYTHONPATH"]
fn python_writes_absent_and_zero_probability_with_the_same_sidecar_bytes() {
    let python = std::env::var("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON");
    let bench = Bench::new("python-probability");
    let clip = clip_with("sealed-python-probability", &["evt-absent", "evt-zero"]);
    let events = BTreeMap::from([
        ("evt-absent".to_owned(), event_with("evt-absent", None)),
        ("evt-zero".to_owned(), event_with("evt-zero", Some(0.0))),
    ]);
    let rust_path = bench
        .sidecars
        .persist(&clip, &events)
        .expect("rust persist");
    let rust_bytes = fs::read(&rust_path).expect("rust bytes");
    fs::remove_file(&rust_path).expect("clear rust sidecar");

    let spec = serde_json::json!({
        "clip_id": clip.clip_id,
        "path": clip.path,
        "duration_ms": clip.duration_ms,
        "boundary": clip.boundary,
        "contributors": clip.contributors.iter().map(|contributor| serde_json::json!({
            "event_ref": contributor.event_ref,
            "detected_at": contributor.detected_at,
        })).collect::<Vec<_>>(),
        "events": events.values().map(|event| serde_json::json!({
            "domain": event.domain,
            "event_type": event.event_type,
            "identity": event.identity,
            "camera_id": event.camera_id,
            "facility_id": event.facility_id,
            "time_sec": event.time_sec,
            "probability": event.probability,
        })).collect::<Vec<_>>(),
    });
    let output = Command::new(python)
        .args(["-c", PROBABILITY_ORACLE])
        .arg(spec.to_string())
        .arg(bench.sidecars.directory())
        .output()
        .expect("python");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, rust_bytes);
}

/// Python's `FlowSealedSidecars.pending_for_camera` over the Rust-written
/// directory, rendered to plain JSON.
const READER: &str = r#"
import dataclasses, datetime, enum, json, pathlib, sys
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
def plain(o):
    if dataclasses.is_dataclass(o):
        return {f.name: plain(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, pathlib.PurePath):
        return str(o)
    if isinstance(o, enum.Enum):
        return o.value
    if isinstance(o, datetime.datetime):
        return o.isoformat(timespec="microseconds")
    return o
reader = FlowSealedSidecars(pathlib.Path(sys.argv[1]))
print(json.dumps(plain(reader.pending_for_camera(sys.argv[2]))))
"#;

#[test]
#[ignore = "requires SEEON_TEST_PYTHON with the worker package on PYTHONPATH"]
fn python_reads_the_rust_sidecars() {
    let python = std::env::var("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON");
    let cases = cases();
    let bench = Bench::new("python");
    bench.persist(&cases, "out-of-order");
    bench.persist(&cases, "non-ascii");
    let output = Command::new(python)
        .args(["-c", READER])
        .arg(bench.sidecars.directory())
        .arg(bench.camera(&cases))
        .output()
        .expect("python");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let read: Value = serde_json::from_slice(&output.stdout).expect("reader JSON");
    let mut golden = cases["pending_for_camera"]["result"].clone();
    for item in golden.as_array_mut().expect("result") {
        let path = bench.place(&text(item, "sidecar_path"));
        item["sidecar_path"] = Value::from(path);
    }
    assert_eq!(read, golden);
}
