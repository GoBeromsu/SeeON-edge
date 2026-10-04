//! Real fall receipts retained by LiveSink and handed to the canonical lanes.

use seeon_deepstream_native::FrameIdentity;
use seeon_worker::trace::DecisionTraceSnapshot;

use crate::records::builder::{AuthorityRole, DecisionSource, Stream};
use crate::records::{Record, RecordKind};

use super::{LiveSink, RuntimeError};
use support::{Fixture, consume, input, next_due, stage};

#[path = "runtime_decision_identity_tests.rs"]
mod identity_tests;
#[path = "runtime_incident_tests.rs"]
mod incident_tests;
#[path = "runtime_decision_retention_tests.rs"]
mod retention_tests;
#[path = "runtime_decision_test_support.rs"]
mod support;

const BOOT: &str = "00000000-0000-4000-8000-000000000661";

fn wire(record: &Record) -> serde_json::Value {
    serde_json::from_str(&crate::records::id::canonical(&record.to_json()).unwrap()).unwrap()
}

fn of_kind(records: &[Record], kind: RecordKind) -> Vec<&Record> {
    records
        .iter()
        .filter(|record| record.body().record_kind == kind)
        .collect()
}

fn assert_snapshot(
    fixture: &Fixture,
    record: &Record,
    frame: FrameIdentity,
    snapshot: &DecisionTraceSnapshot,
    generation: Option<u64>,
) {
    let stream = Stream {
        camera_id: format!("camera-{}", frame.source_id),
        worker_boot_id: BOOT.into(),
        source_generation: frame.binding.generation,
        stream_epoch: frame.binding.epoch,
    };
    let trace = crate::run::decision::trace_id(
        snapshot,
        "fall.v2",
        &fixture.policy_ids[(frame.source_id - 19) as usize],
    )
    .unwrap();
    let expected = crate::run::records::decision_record(
        &stream,
        crate::records::builder::Frame {
            frame_seq: frame.sequence,
            source_pts_ns: Some(frame.pts_ns as i64),
        },
        record.body().observed_at_ns,
        snapshot,
        &DecisionSource {
            generation,
            module_qualified_id: Some("fall.v2".into()),
            authority_role: AuthorityRole::Authoritative,
            decision_trace_id: Some(trace),
        },
    )
    .unwrap();
    assert_eq!(record.body().payload, expected.body().payload);
    assert_eq!(record.body().outcome, expected.body().outcome);
    assert_eq!(record.body().causal_unit_id, expected.body().causal_unit_id);
    assert_eq!(record.body().frame_seq, Some(frame.sequence));
    assert_eq!(record.body().source_pts_ns, Some(frame.pts_ns as i64));
    assert_eq!(record.body().camera_id, stream.camera_id);
    assert_eq!(record.body().worker_boot_id, BOOT);
    assert_eq!(record.body().source_generation, frame.binding.generation);
    assert_eq!(record.body().stream_epoch, frame.binding.epoch);
}
