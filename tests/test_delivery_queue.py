from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shared.events import envelope_limits
from shared.events.delivery_queue import (
    MAX_ACCEPTED_BYTES,
    MAX_ACCEPTED_ENTRIES,
    AdmissionFault,
    DeliveryQueue,
    EventEntry,
    SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
)


def _event(number: int) -> EventEntry:
    return EventEntry(
        edge_event_id=f"event-{number}",
        event_type="fall",
        detected_at="2026-08-21T00:00:00Z",
        camera_id="camera",
        facility_id="facility",
        decision_trace=b"trace",
        values=b"values",
    )


def _attachment() -> SnapshotAttachmentEntry:
    return SnapshotAttachmentEntry(
        "event-1",
        "snapshot-1",
        "a" * 64,
        "published/snapshot.jpg",
        10,
        "image/jpeg",
    )


def _disposition() -> SnapshotDispositionEntry:
    return SnapshotDispositionEntry("event-1", "snapshot-1", "unavailable", "camera offline")


def test_temp_left_by_kill_before_publish_is_not_an_admitted_entry(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    orphan = tmp_path / ".interrupted.tmp"
    orphan.write_bytes(b'{"half":true}')
    reopened = DeliveryQueue(tmp_path)
    assert reopened.accepted_count == 0
    assert not orphan.exists()
    assert reopened.try_admit(_event(1)).accepted
    published = next(tmp_path.glob("*.json"))
    assert json.loads(published.read_bytes())["edge_event_id"] == "event-1"
    assert queue.accepted_count == 1


def test_entry_and_byte_capacity_refuse_with_typed_fault_without_eviction(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    for number in range(MAX_ACCEPTED_ENTRIES):
        assert queue.try_admit(_event(number)).accepted
    refused = queue.try_admit(_event(MAX_ACCEPTED_ENTRIES))
    assert refused == type(refused)(False, AdmissionFault.ENTRY_CAPACITY)
    assert queue.accepted_count == MAX_ACCEPTED_ENTRIES

    byte_queue = DeliveryQueue(tmp_path / "bytes")
    (tmp_path / "bytes" / "already-full.json").write_bytes(b"x")
    os.truncate(tmp_path / "bytes" / "already-full.json", MAX_ACCEPTED_BYTES)
    byte_refused = byte_queue.try_admit(_event(1))
    assert byte_refused.fault is AdmissionFault.BYTE_CAPACITY
    assert byte_queue.accepted_count == 1


def test_concurrent_mixed_admission_never_over_admits(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)

    def admit(number: int) -> bool:
        if number % 3 == 0:
            entry = _event(number)
        elif number % 3 == 1:
            entry = SnapshotAttachmentEntry(
                f"event-{number}",
                f"snapshot-{number}",
                "b" * 64,
                "ref",
                number,
                "image/jpeg",
            )
        else:
            entry = SnapshotDispositionEntry(
                f"event-{number}",
                f"snapshot-{number}",
                "unavailable",
                "not captured",
            )
        return queue.try_admit(entry).accepted

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(admit, range(MAX_ACCEPTED_ENTRIES + 128)))
    assert sum(results) == MAX_ACCEPTED_ENTRIES
    assert queue.accepted_count == MAX_ACCEPTED_ENTRIES


def test_startup_reconstructs_count_and_bytes_from_published_files(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    for entry in (_event(1), _attachment(), _disposition()):
        assert queue.try_admit(entry).accepted
    expected_paths = tuple(tmp_path.glob("*.json"))
    assert DeliveryQueue(tmp_path).accepted_count == len(expected_paths)
    assert DeliveryQueue(tmp_path).accepted_bytes == sum(
        path.stat().st_size for path in expected_paths
    )


def test_acknowledgement_removes_only_the_named_entry(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    event, attachment, disposition = _event(1), _attachment(), _disposition()
    for entry in (event, attachment, disposition):
        assert queue.try_admit(entry).accepted
    assert queue.acknowledge(event.entry_id)
    remaining = {record["entry_id"] for record in queue.entries()}
    assert remaining == {attachment.entry_id, disposition.entry_id}
    assert queue.acknowledge_backend(attachment.entry_id, 409)
    assert {record["entry_id"] for record in queue.entries()} == {disposition.entry_id}
    assert not queue.acknowledge_backend(disposition.entry_id, 503)


def test_unacknowledged_entry_is_never_evicted(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    event = _event(1)
    assert queue.try_admit(event).accepted
    for number in range(2, 100):
        assert queue.try_admit(_event(number)).accepted
    assert event.entry_id in {record["entry_id"] for record in queue.entries()}


def test_maximum_envelope_size_tracks_named_constant(monkeypatch) -> None:
    original = envelope_limits.DECISION_TRACE_BYTES_MAX
    before = envelope_limits.maximum_serialized_envelope_bytes()
    monkeypatch.setattr(envelope_limits, "DECISION_TRACE_BYTES_MAX", original + 3)
    assert envelope_limits.maximum_serialized_envelope_bytes() == before + 4
