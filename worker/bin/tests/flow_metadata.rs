//! Review rows 1-3 and 6: `flow_metadata` against Python
//! `FlowClipPublisher._publish` (`worker/pipeline/output/evidence/
//! flow_clip_publication.py:75-127`).
//!
//! The CPU lane checks values derived by hand from that Python source: the
//! earliest contributor's event fixes camera, facility, domain and event type;
//! refusals follow Python's order; `finalized_at` is clamped up to the clip
//! end. The backend lane runs the same synthetic cases through the Python
//! publisher, `finalize_ready_manifest` and `manifest_payload` under
//! `$SEEON_TEST_PYTHON` and compares the manifest bytes Python writes with the
//! bytes Rust writes. The bed-exit golden is that Python output, never Rust's.

use std::collections::BTreeMap;
use std::process::Command;

use serde_json::{Value, json};

use seeon_ml_worker::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use seeon_ml_worker::clips::manifest::{
    ClipMetadata, Contributor, Extension, ManifestError, MediaFacts, Terminal, manifest_bytes,
};
use seeon_ml_worker::clips::time::Utc;

const CAMERA: &str = "cam-flow-1";
const FACILITY: &str = "facility-1";
const CLIP: &str = "cam-flow-1-20260820T095945250731Z-00e500000001";
const BED_1: &str = "00000000-0000-4000-8000-0000000bed01";
const BED_2: &str = "00000000-0000-4000-8000-0000000bed02";
const FALL: &str = "00000000-0000-4000-8000-00000000fa11";
const SHA256: &str = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0";

fn event(camera: &str, facility: &str, domain: &str, event_type: &str) -> Value {
    json!({"camera_id": camera, "facility_id": facility, "domain": domain, "event_type": event_type})
}

/// Two bed exits (`worker/domains/bed_exit/detector.py:318,325`) and a later
/// fall; the contributors arrive out of order, the fall first.
fn bed_exit_out_of_order() -> Value {
    let bed = event(CAMERA, FACILITY, "bed_exit", "bed-exit");
    json!({
        "boundary": "extension_bounded", "clip_id": CLIP, "duration_ms": 4000,
        "now": "2026-08-20T10:01:00+00:00",
        "events": {BED_1: bed.clone(), BED_2: bed, FALL: event(CAMERA, FACILITY, "fall", "fall")},
        "contributors": [
            {"event_ref": FALL, "detected_at": "2026-08-20T10:00:03.000731Z"},
            {"event_ref": BED_1, "detected_at": "2026-08-20T10:00:00.250731Z"},
            {"event_ref": BED_2, "detected_at": "2026-08-20T10:00:01.750731Z"},
        ],
    })
}

fn with(mut case: Value, pointer: &str, value: Value) -> Value {
    *case.pointer_mut(pointer).expect("case field") = value;
    case
}

/// The clock reads 10:00:04, before the clip ends at 10:00:05.250731.
fn clock_before_end() -> Value {
    let case = with(bed_exit_out_of_order(), "/duration_ms", json!(20_000));
    with(case, "/now", json!("2026-08-20T10:00:04+00:00"))
}

fn spans_cameras() -> Value {
    let pointer = format!("/events/{BED_2}/camera_id");
    with(bed_exit_out_of_order(), &pointer, json!("cam-flow-2"))
}

fn spans_facilities() -> Value {
    let pointer = format!("/events/{FALL}/facility_id");
    with(bed_exit_out_of_order(), &pointer, json!("facility-2"))
}

fn zero_duration() -> Value {
    with(bed_exit_out_of_order(), "/duration_ms", json!(0))
}

fn missing_event() -> Value {
    let mut case = bed_exit_out_of_order();
    case["events"]
        .as_object_mut()
        .expect("events")
        .remove(BED_2);
    case
}

fn text(value: &Value, key: &str) -> String {
    value[key].as_str().expect("text field").to_owned()
}

fn at(text: &str) -> Utc {
    Utc::parse(text).expect("timestamp")
}

fn run(case: &Value) -> Result<ClipMetadata, ManifestError> {
    let events: BTreeMap<String, ContributorEvent> = case["events"]
        .as_object()
        .expect("events")
        .iter()
        .map(|(event_ref, event)| {
            let fields = ContributorEvent {
                camera_id: text(event, "camera_id"),
                facility_id: text(event, "facility_id"),
                domain: text(event, "domain"),
                event_type: text(event, "event_type"),
            };
            (event_ref.clone(), fields)
        })
        .collect();
    let contributors = case["contributors"]
        .as_array()
        .expect("contributors")
        .iter()
        .map(|item| Contributor {
            event_ref: text(item, "event_ref"),
            detected_at: at(&text(item, "detected_at")),
        })
        .collect();
    let extension = Extension {
        boundary: text(case, "boundary"),
        contributors,
        duration_ms: case["duration_ms"].as_i64().expect("duration_ms"),
    };
    flow_metadata(
        CLIP,
        &events,
        extension,
        FLOW_ENCODER,
        at(&text(case, "now")),
    )
}

#[test]
fn earliest_contributor_fixes_the_bed_exit_type_and_domain() {
    let meta = run(&bed_exit_out_of_order()).expect("flow metadata");
    assert_eq!(meta.event_type, "bed-exit");
    assert_eq!(meta.domain, "bed_exit");
    assert_eq!(
        (meta.camera_id.as_str(), meta.facility_id.as_str()),
        (CAMERA, FACILITY)
    );
    assert_eq!(meta.event_refs, [BED_1, BED_2, FALL]);
    assert_eq!(meta.detected_at, at("2026-08-20T10:00:00.250731Z"));
    assert_eq!(meta.clip_start_at, at("2026-08-20T09:59:45.250731Z"));
    assert_eq!(meta.started_at, meta.clip_start_at);
    assert_eq!(meta.clip_end_at, at("2026-08-20T09:59:49.250731Z"));
    assert_eq!(meta.finalized_at, at("2026-08-20T10:01:00Z"));
    let extension = meta.extension.expect("extension");
    let order: Vec<&str> = extension
        .contributors
        .iter()
        .map(|c| c.event_ref.as_str())
        .collect();
    assert_eq!(order, [BED_1, BED_2, FALL]);
}

#[test]
fn finalized_at_is_clamped_up_to_the_clip_end() {
    let meta = run(&clock_before_end()).expect("flow metadata");
    assert_eq!(meta.clip_end_at, at("2026-08-20T10:00:05.250731Z"));
    assert_eq!(meta.finalized_at, meta.clip_end_at);
}

#[test]
fn contributors_on_two_cameras_are_refused() {
    assert_eq!(
        run(&spans_cameras()).err(),
        Some(ManifestError::SpansCameras)
    );
}

#[test]
fn contributors_in_two_facilities_are_refused() {
    assert_eq!(
        run(&spans_facilities()).err(),
        Some(ManifestError::SpansFacilities)
    );
}

#[test]
fn a_zero_duration_is_refused() {
    assert_eq!(
        run(&zero_duration()).err(),
        Some(ManifestError::NonPositiveDuration)
    );
}

#[test]
fn a_contributor_without_its_event_is_refused() {
    assert_eq!(
        run(&missing_event()).err(),
        Some(ManifestError::MissingEvent)
    );
}

const PYTHON_PUBLISH: &str = "
import json, sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from worker.pipeline.output.evidence import evidence_manifest
from worker.pipeline.output.evidence.clip_manifest_payload import manifest_payload
from worker.pipeline.output.evidence.evidence_media import MediaFacts
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublisher
from worker.pipeline.output.evidence.smart_record_actor import ClipContributor, ClipSealed
from worker.types import BusinessEvent

class Allocator:
    def reserve_existing(self, camera_id, clip_id):
        return SimpleNamespace(camera_id=camera_id, clip_id=clip_id)

class Capture:
    def publish_adopted_ready(self, reservation, source_path, metadata):
        return metadata

media = json.loads(sys.argv[1])
evidence_manifest.inspect_finalized_media = lambda path, ffprobe_bin='ffprobe': MediaFacts(
    sha256=media['sha256'], size_bytes=media['size_bytes'],
    duration_ms=media['duration_ms'], video_codec=media['codec'])
out = []
for case in json.loads(sys.argv[2]):
    events = {ref: BusinessEvent(identity=1, time_sec=0.0, probability=0.9, **fields)
              for ref, fields in case['events'].items()}
    sealed = ClipSealed(
        clip_id=case['clip_id'], path='synthetic.mp4', duration_ms=case['duration_ms'],
        contributors=tuple(ClipContributor(**item) for item in case['contributors']),
        boundary=case['boundary'])
    now = datetime.fromisoformat(case['now'])
    publisher = FlowClipPublisher(allocator=Allocator(), publisher=Capture(), now=lambda: now)
    try:
        meta = publisher._publish(sealed, events)
    except Exception as exc:
        out.append({'raised': type(exc).__name__})
        continue
    manifest = evidence_manifest.finalize_ready_manifest(
        video_path=Path('clip.mp4'), clip_id=case['clip_id'], camera_id=meta.camera_id,
        event_refs=meta.event_refs, clip_start_at=meta.clip_start_at,
        clip_end_at=meta.clip_end_at, finalized_at=meta.finalized_at)
    payload = manifest_payload(manifest, meta, path=f\"clips/{case['clip_id']}/clip.mp4\",
                               video_available=True)
    out.append({'facility_id': meta.facility_id,
                'manifest': json.dumps(payload, separators=(',', ':'), sort_keys=True) + '\\n'})
print(json.dumps(out))
";

fn python_exception(error: ManifestError) -> &'static str {
    match error {
        ManifestError::MissingEvent => "KeyError",
        _ => "ValueError",
    }
}

#[test]
#[ignore = "requires SEEON_TEST_PYTHON"]
fn python_publisher_writes_the_same_flow_manifest_bytes() {
    let python = std::env::var_os("SEEON_TEST_PYTHON").expect("SEEON_TEST_PYTHON is required");
    let media = MediaFacts {
        sha256: SHA256.to_owned(),
        size_bytes: 48_213,
        codec: "h264".to_owned(),
        duration_ms: 4_000,
    };
    let media_json = json!({"sha256": media.sha256, "size_bytes": media.size_bytes,
        "duration_ms": media.duration_ms, "codec": media.codec});
    let cases = [
        bed_exit_out_of_order(),
        clock_before_end(),
        spans_cameras(),
        spans_facilities(),
        zero_duration(),
        missing_event(),
    ];
    let output = Command::new(python)
        .arg("-c")
        .arg(PYTHON_PUBLISH)
        .arg(media_json.to_string())
        .arg(Value::from(cases.to_vec()).to_string())
        .output()
        .expect("python runs");
    assert!(
        output.status.success(),
        "python exited with {:?}",
        output.status
    );
    let golden: Vec<Value> = serde_json::from_slice(&output.stdout).expect("python output is JSON");
    assert_eq!(golden.len(), cases.len());
    let terminal = Terminal::Ready(media);
    for (case, expected) in cases.iter().zip(&golden) {
        match run(case) {
            Ok(meta) => {
                let bytes = manifest_bytes(&meta, &terminal).expect("manifest bytes");
                assert_eq!(
                    String::from_utf8(bytes).expect("utf-8"),
                    text(expected, "manifest")
                );
                assert_eq!(meta.facility_id, text(expected, "facility_id"));
            }
            Err(error) => assert_eq!(python_exception(error), text(expected, "raised")),
        }
    }
    assert_eq!(
        text(&golden[0], "manifest")
            .matches("\"event_type\":\"bed-exit\"")
            .count(),
        1
    );
}
