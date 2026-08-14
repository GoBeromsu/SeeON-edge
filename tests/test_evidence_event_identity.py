from __future__ import annotations

import json
from pathlib import Path

from worker.pipeline.output.evidence.event_identity import EventIdentityStore
from worker.pipeline.output.evidence.event_payload import (
    MutableWorkerEventPayload,
    WorkerEventPayload,
)

_RUNTIME_MANIFEST_SHA256 = "a" * 64
_DECISION_TRACE_ID = "b" * 64


def test_enrich_preserves_nested_audit_clip_redaction_and_wire_values(
    tmp_path: Path,
) -> None:
    event: WorkerEventPayload = {
        "event_id": "upstream-event-7",
        "idempotency_key": "source-7",
        "edge_event_id": "replace-me",
        "event_type": "fall",
        "probability": 0.93,
        "detected_at": "replace-me",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "clip_id": "clip-7",
        "evidence": {"domain": "fall", "redacted": True},
        "audit": {
            "runtime_manifest_sha256": _RUNTIME_MANIFEST_SHA256,
            "decision_trace_id": _DECISION_TRACE_ID,
            "redaction": "operator-only",
        },
        "snapshot_jpeg": b"jpeg-bytes",
        "snapshot": {"path": "snapshots/redacted.jpg", "redacted": True},
    }
    store = EventIdentityStore(tmp_path / "event-identities.jsonl")

    enriched: MutableWorkerEventPayload = store.enrich(event, "facility-1", "camera-1")
    repeated = store.enrich(event, "facility-1", "camera-1")

    assert enriched["edge_event_id"] == repeated["edge_event_id"]
    assert enriched["detected_at"] == repeated["detected_at"]
    assert enriched["edge_event_id"] != "replace-me"
    assert enriched["detected_at"] != "replace-me"
    assert enriched["event_id"] == "upstream-event-7"
    assert enriched["idempotency_key"] == "source-7"
    assert enriched["clip_id"] == "clip-7"
    assert enriched["audit"] == {
        "runtime_manifest_sha256": _RUNTIME_MANIFEST_SHA256,
        "decision_trace_id": _DECISION_TRACE_ID,
        "redaction": "operator-only",
    }
    assert enriched["evidence"] == {"domain": "fall", "redacted": True}
    assert enriched["snapshot"] == {
        "path": "snapshots/redacted.jpg",
        "redacted": True,
    }
    assert enriched["snapshot_jpeg"] == b"jpeg-bytes"
    assert enriched["audit"] is not event["audit"]
    assert enriched["evidence"] is not event["evidence"]
    assert enriched["snapshot"] is not event["snapshot"]

    wire_payload = dict(enriched)
    del wire_payload["snapshot_jpeg"]
    assert json.loads(json.dumps(wire_payload)) == wire_payload
