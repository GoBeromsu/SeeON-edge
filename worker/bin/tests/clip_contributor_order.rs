use std::collections::BTreeMap;
use std::sync::Arc;

use seeon_deepstream_native::{MediaBinding, MediaResult, RecordTicket};
use seeon_ml_worker::clips::entry::{ContributorEvent, FLOW_ENCODER, flow_metadata};
use seeon_ml_worker::clips::manifest::{MediaFacts, Terminal, manifest_bytes};
use seeon_ml_worker::clips::recorder::{Admit, PlaneRefusal, RecordPlane, Recorder, RecorderError};
use seeon_ml_worker::clips::reserve::SaveOutcome;
use seeon_ml_worker::clips::time::Utc;
use seeon_ml_worker::msg::RecordReceipt;
use seeon_ml_worker::seam::SystemClock;

const SOURCE_ID: u32 = 4;
const TICKET: RecordTicket = RecordTicket {
    binding: MediaBinding {
        token: 1,
        generation: 1,
        epoch: 1,
    },
    request_id: 1,
    source_id: SOURCE_ID,
    session_id: 7,
    session_valid: 1,
    coalesced: 0,
};

struct TestPlane;

impl RecordPlane for TestPlane {
    fn start(
        &mut self,
        _lookback_seconds: u32,
        _forward_seconds: u32,
    ) -> Result<RecordTicket, PlaneRefusal> {
        Ok(TICKET)
    }

    fn stop(&mut self, _ticket: &RecordTicket) -> Result<(), PlaneRefusal> {
        Ok(())
    }
}

fn at(text: &str) -> Utc {
    Utc::parse(text).expect("timestamp")
}

fn receipt() -> RecordReceipt {
    RecordReceipt {
        ticket: TICKET,
        result: MediaResult::Ok,
        error: 0,
        duration_ms: 30_000,
        width: 1280,
        height: 720,
        contains_video: true,
        contains_audio: false,
        directory: "/tmp/sealed".into(),
        filename: "clip.mp4".into(),
    }
}

fn assert_order(admissions: &[(&str, &str)], expected_order: &[&str], primary: &str) {
    let mut recorder = Recorder::new(SOURCE_ID, TestPlane, Arc::new(SystemClock::default()));
    for (index, (event_ref, detected_at)) in admissions.iter().enumerate() {
        let admitted = recorder.admit(event_ref, at(detected_at));
        if index == 0 {
            assert!(matches!(admitted, Ok(Admit::Started(ticket)) if ticket == TICKET));
        } else {
            assert!(matches!(admitted, Ok(Admit::Extended)));
        }
    }

    let mut sealed = None;
    let outcome = recorder.on_receipt(&receipt(), |clip| {
        sealed = Some(clip.clone());
        Ok(SaveOutcome::FinalizeFailed(None))
    });
    assert!(matches!(outcome, Err(RecorderError::Unpublished { ticket }) if ticket == TICKET));
    let sealed = sealed.expect("receipt reaches the real recorder seal");
    let sealed_order: Vec<&str> = sealed
        .contributors
        .iter()
        .map(|contributor| contributor.event_ref.as_str())
        .collect();
    assert_eq!(sealed_order, expected_order.to_vec());

    let events: BTreeMap<String, ContributorEvent> = admissions
        .iter()
        .map(|(event_ref, _)| {
            (
                (*event_ref).to_owned(),
                ContributorEvent {
                    camera_id: "camera-1".to_owned(),
                    facility_id: "facility-1".to_owned(),
                    domain: format!("domain-{event_ref}"),
                    event_type: format!("type-{event_ref}"),
                },
            )
        })
        .collect();
    let metadata = flow_metadata(
        "clip-1",
        &events,
        sealed.extension(),
        FLOW_ENCODER,
        at("2026-01-01T00:01:00Z"),
    )
    .expect("flow metadata");
    assert_eq!(metadata.domain, format!("domain-{primary}"));
    assert_eq!(metadata.event_type, format!("type-{primary}"));
    let primary_detected_at = admissions
        .iter()
        .find(|item| item.0 == primary)
        .expect("primary event input")
        .1;
    assert_eq!(metadata.detected_at, at(primary_detected_at));
    let metadata_order: Vec<&str> = metadata.event_refs.iter().map(String::as_str).collect();
    assert_eq!(metadata_order, expected_order.to_vec());
    let extension_order: Vec<&str> = metadata
        .extension
        .as_ref()
        .expect("extension")
        .contributors
        .iter()
        .map(|contributor| contributor.event_ref.as_str())
        .collect();
    assert_eq!(extension_order, expected_order.to_vec());

    let terminal = Terminal::Ready(MediaFacts {
        sha256: "0".repeat(64),
        size_bytes: 1,
        codec: "h264".to_owned(),
        duration_ms: 30_000,
    });
    let bytes = manifest_bytes(&metadata, &terminal).expect("canonical manifest bytes");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes).expect("manifest JSON");
    assert_eq!(manifest["event_ref"], primary);
    assert_eq!(manifest["event_refs"], serde_json::json!(expected_order));
    assert_eq!(manifest["domain"], format!("domain-{primary}"));
    assert_eq!(manifest["event_type"], format!("type-{primary}"));
    let serialized_order: Vec<&str> = manifest["extension"]["contributors"]
        .as_array()
        .expect("serialized contributors")
        .iter()
        .map(|item| item["event_ref"].as_str().expect("event ref"))
        .collect();
    assert_eq!(serialized_order, expected_order);
}

#[test]
fn equal_timestamps_preserve_admission_order_through_metadata() {
    assert_order(
        &[
            ("event-b", "2026-01-01T00:00:00Z"),
            ("event-a", "2026-01-01T00:00:00Z"),
        ],
        &["event-b", "event-a"],
        "event-b",
    );
}

#[test]
fn distinct_timestamps_remain_chronological_through_metadata() {
    assert_order(
        &[
            ("event-late", "2026-01-01T00:00:20Z"),
            ("event-early", "2026-01-01T00:00:00Z"),
            ("event-middle", "2026-01-01T00:00:10Z"),
        ],
        &["event-early", "event-middle", "event-late"],
        "event-early",
    );
}
