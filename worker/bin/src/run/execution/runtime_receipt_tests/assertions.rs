//! Observable unit publication, identity, and retained-state assertions.

use std::sync::mpsc::TryRecvError;

use seeon_deepstream_native::RecordTicket;
use serde_json::{Value, json};

use super::super::super::RuntimeError;
use super::fixture::{BINDING, BOOT, Fixture};
use crate::clips::recorder::{Boundary, Counters, RecorderError, State};
use crate::exit::Exit;
use crate::records::RecordKind;
use crate::run::execution::publication::PublicationError;

const CAMERA: &str = "camera-19";

pub(super) fn overdue(result: Result<(), RuntimeError>, admitted: RecordTicket) {
    let error = result.unwrap_err();
    assert_eq!(error.exit(), Exit::Runtime);
    match error {
        RuntimeError::Publication(PublicationError::Recorder(RecorderError::ReceiptOverdue {
            ticket,
        })) => assert_eq!(ticket, admitted),
        other => panic!("expected original-ticket receipt failure, got {other:?}"),
    }
}

impl Fixture {
    pub(super) fn snapshot(&self) -> (State, Boundary, usize, Counters) {
        let recorder = &self.session.publications.recorders[0];
        (
            recorder.state(),
            recorder.boundary(),
            recorder.pending(),
            recorder.counters(),
        )
    }

    pub(super) fn no_commands(&self) {
        assert!(
            matches!(
                self.commands.as_ref().unwrap().try_recv(),
                Err(TryRecvError::Empty)
            ),
            "unit receipt processing must not issue extra media commands"
        );
    }

    pub(super) fn delivery(&self, sequence: u64, identity: &str) {
        let drained = self.lanes.drain_for(CAMERA, BOOT, 8).unwrap().unwrap();
        assert!(drained.gaps.is_empty());
        assert_eq!(drained.records.len(), 1);
        let body = drained.records[0].body();
        assert_eq!(body.record_kind, RecordKind::EventDelivery);
        assert_eq!(body.camera_id, CAMERA);
        assert_eq!(body.worker_boot_id, BOOT);
        assert_eq!(
            (body.source_generation, body.stream_epoch),
            (BINDING.generation, BINDING.epoch)
        );
        assert_eq!(body.frame_seq, Some(sequence));
        assert_eq!(body.source_pts_ns, Some((sequence * 100_000_000) as i64));
        assert_eq!(body.causal_unit_id, identity);
        assert_eq!(body.outcome, "admitted");
    }
}

pub(super) fn clip_id(ticket: RecordTicket) -> String {
    format!("{BOOT}-{}-{}", ticket.source_id, ticket.request_id)
}
pub(super) fn unavailable(fixture: &Fixture, admitted: RecordTicket, identity: &str, reason: &str) {
    let id = clip_id(admitted);
    let directory = fixture.session.publications.store.clip_dir(&id);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let camera = format!("camera-{}", admitted.source_id);
    assert_eq!(manifest["clip_id"], id);
    assert_eq!(manifest["camera_id"], camera);
    assert_eq!(manifest["event_refs"], json!([identity]));
    assert_eq!(
        manifest["extension"]["contributors"][0]["event_ref"],
        identity
    );
    assert_eq!(manifest["state"], "UNAVAILABLE");
    assert_eq!(manifest["reason_code"], reason);
    assert_eq!(manifest["video_available"], false);
    assert!(manifest["path"].is_null());
    assert!(!directory.join(crate::clips::publish::MEDIA_FILE).exists());
    let entries = fixture.session.publications.queue.entries().unwrap();
    let entry = entries.iter().find(|entry| entry["clip_id"] == id).unwrap();
    assert_eq!(entry["kind"], "CLIP");
    assert_eq!(entry["camera_id"], camera);
    assert_eq!(
        entry["facility_id"],
        format!("facility-{}", admitted.source_id)
    );
    assert_eq!(entry["event_ids"], json!([identity]));
    assert_eq!(entry["local_state"], "UNAVAILABLE");
    assert_eq!(entry["unavailable_reason"], reason);
    assert!(entry["media_reference"].is_null());
    for field in ["codec", "sha256", "size_bytes", "mime_type", "duration_ms"] {
        assert!(
            manifest.get(field).is_none(),
            "no measured media fact: {field}"
        );
        assert!(entry[field].is_null(), "no measured media fact: {field}");
    }
    assert!(
        !fixture
            .session
            .publications
            .store
            .clip_dir(&format!("{BOOT}-{}-47", admitted.source_id))
            .exists()
    );
}
