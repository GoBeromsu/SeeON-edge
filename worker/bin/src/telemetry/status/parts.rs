//! The runtime-status sub-objects: `clip_recorder`, `worker` and
//! `delivery_queue` (Python `telemetry/wire.py`), plus the integer helpers.

use crate::delivery::CapacitySnapshot;
use crate::json::Json;

use super::super::gpu::text;
use super::{ClipRecorderStatus, WorkerStatus};

pub(super) fn count(value: u64) -> Json {
    Json::Int(i128::from(value))
}

pub(super) fn optional_count(value: Option<u64>) -> Json {
    value.map_or(Json::Null, count)
}

pub(super) fn size(value: usize) -> Json {
    Json::Int(value as i128)
}

pub(super) fn recorder(status: &ClipRecorderStatus) -> Json {
    Json::Object(vec![
        ("available".into(), Json::Bool(status.available)),
        (
            "dropped_frames".into(),
            optional_count(status.dropped_frames),
        ),
        (
            "dropped_events".into(),
            optional_count(status.dropped_events),
        ),
        ("failed_writes".into(), optional_count(status.failed_writes)),
        (
            "finalized_clips".into(),
            optional_count(status.finalized_clips),
        ),
        (
            "video_unavailable_clips".into(),
            optional_count(status.video_unavailable_clips),
        ),
        ("active_clips".into(), optional_count(status.active_clips)),
        ("encoder".into(), text(status.encoder.as_deref())),
    ])
}

pub(super) fn worker(status: &WorkerStatus) -> Json {
    Json::Object(vec![
        ("alive".into(), Json::Bool(status.alive)),
        (
            "pid".into(),
            status.pid.map_or(Json::Null, |pid| count(pid.into())),
        ),
        (
            "started_at_sec".into(),
            status.started_at_sec.map_or(Json::Null, Json::Float),
        ),
        (
            "profile_boot_error".into(),
            text(status.profile_boot_error.as_deref()),
        ),
    ])
}

/// Python `_delivery_queue_payload`: `by_kind` keyed by the wire kind.
pub fn delivery_queue(snapshot: &CapacitySnapshot) -> Json {
    let by_kind = snapshot
        .by_kind
        .iter()
        .map(|(kind, entries)| (kind.as_str().to_owned(), size(*entries)))
        .collect();
    Json::Object(vec![
        ("accepted_count".into(), size(snapshot.accepted_count)),
        ("accepted_bytes".into(), count(snapshot.accepted_bytes)),
        (
            "max_accepted_entries".into(),
            size(snapshot.max_accepted_entries),
        ),
        (
            "max_accepted_bytes".into(),
            count(snapshot.max_accepted_bytes),
        ),
        ("by_kind".into(), Json::Object(by_kind)),
        (
            "dead_lettered_count".into(),
            size(snapshot.dead_lettered_count),
        ),
        (
            "dead_lettered_bytes".into(),
            count(snapshot.dead_lettered_bytes),
        ),
        (
            "oldest_event_accepted_at".into(),
            text(snapshot.oldest_event_accepted_at.as_deref()),
        ),
    ])
}
