from __future__ import annotations

import json
from pathlib import Path

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from worker.runtime.provenance.models import AppliedRuntimeManifest
from worker.runtime.provenance.store import (
    AppliedRuntimeManifestStore,
    ProvenanceRetentionPolicy,
)

_MANIFEST = "a" * 64


def _manifest(*camera_ids: str) -> AppliedRuntimeManifest:
    canonical = json.dumps(
        {
            "cameras": [{"camera_id": camera_id} for camera_id in camera_ids],
            "manifest_schema_version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return AppliedRuntimeManifest(1, canonical, _MANIFEST)


def test_delivery_queue_refuses_conflicting_event_basis(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path / "delivery")
    first = EventEntry(
        edge_event_id="event-a",
        event_type="fall",
        detected_at="2026-08-13T00:00:01Z",
        camera_id="camera-a",
        facility_id="facility-a",
        decision_trace=b'{"reason":"fall-onset"}',
        values=b'{"fall_probability":0.9}',
    )
    assert queue.try_admit(first).accepted
    conflicting = EventEntry(
        edge_event_id="event-a",
        event_type="fall",
        detected_at="2026-08-13T00:00:01Z",
        camera_id="camera-a",
        facility_id="facility-a",
        decision_trace=b'{"reason":"fall-onset"}',
        values=b'{"fall_probability":0.8}',
    )
    result = queue.try_admit(conflicting)
    assert not result.accepted
    assert result.fault is not None


def test_provenance_history_is_retained_locally_within_bound(tmp_path: Path) -> None:
    store = AppliedRuntimeManifestStore(
        tmp_path / "edge.sqlite3",
        ProvenanceRetentionPolicy(max_boots=2, max_boots_per_camera=1),
    )
    for index in range(3):
        store.persist(
            _manifest("camera-a"),
            boot_instance_id=f"boot-{index}",
            applied_at=f"2026-08-13T00:00:0{index}Z",
        )

    records = tuple((tmp_path / "runtime-provenance").glob("*.json"))
    assert len(records) == 2
    assert not (tmp_path / "delivery-queue").exists()
