from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import EventReceipt
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager


@dataclass
class AcceptingTransport:
    events: list[str]
    attachments: list[dict[str, object]]
    dispositions: list[dict[str, object]]

    def send_event(self, _payload: str, event_id: str) -> EventReceipt:
        self.events.append(event_id)
        return EventReceipt("accepted", event_id, "backend-event")

    def send_snapshot_attachment(self, payload: dict[str, object]) -> None:
        self.attachments.append(payload)

    def send_snapshot_disposition(self, payload: dict[str, object]) -> None:
        self.dispositions.append(payload)


@pytest.mark.parametrize(
    "crash_point",
    [
        "event_admission",
        "snapshot_stage",
        "snapshot_publish",
        "attachment_admission",
        "acknowledgement",
    ],
)
def test_restart_preserves_event_and_classifies_missing_snapshot_once(
    tmp_path: Path, crash_point: str
) -> None:
    queue_directory = tmp_path / "delivery-queue"
    stager = DurableEvidenceStager(
        queue_directory,
        camera_id="camera-a",
        facility_id="facility-a",
        config_version=1,
    )
    event = {
        "edge_event_id": "event-a",
        "event_type": "fall",
        "detected_at": "2026-08-21T00:00:00Z",
    }
    stager.stage(event)
    assert DeliveryQueue(queue_directory).accepted_count == 1
    if crash_point == "event_admission":
        _assert_replayable_event(queue_directory)
        return

    snapshot = {
        "snapshot_id": "snapshot-a",
        "path": "snapshots/snapshot-a.jpg",
        "sha256": "a" * 64,
        "size_bytes": 7,
        "mime_type": "image/jpeg",
    }
    if crash_point == "snapshot_stage":
        stager.record_snapshot_disposition("event-a", "snapshot-a", "MISSING", "capture failed")
    elif crash_point == "snapshot_publish":
        stager.record_snapshot_disposition(
            "event-a", "snapshot-a", "MISSING", "publish interrupted"
        )
    else:
        stager.attach_snapshot("event-a", snapshot)
        stager.attach_snapshot("event-a", snapshot)

    entries = tuple(DeliveryQueue(queue_directory).entries())
    assert sum(entry["kind"] == "EVENT" for entry in entries) == 1
    assert sum(entry["kind"] == "SNAPSHOT_ATTACHMENT" for entry in entries) <= 1
    assert any(entry["kind"] == "SNAPSHOT_DISPOSITION" for entry in entries) or any(
        entry["kind"] == "SNAPSHOT_ATTACHMENT" for entry in entries
    )

    transport = AcceptingTransport([], [], [])
    sender = EvidenceSender(
        queue_directory,
        SenderConfig("http://relay.test", "token", "camera-a"),
        transport=transport,
    )
    sender.run_once()
    if crash_point == "acknowledgement":
        assert DeliveryQueue(queue_directory).accepted_count >= 1
    assert transport.events == ["event-a"]
    remaining = tuple(DeliveryQueue(queue_directory).entries())
    assert all(entry["kind"] != "EVENT" for entry in remaining)
    while DeliveryQueue(queue_directory).accepted_count:
        sender.run_once()
    assert len(transport.attachments) <= 1
    assert len(transport.dispositions) <= 1


def _assert_replayable_event(queue_directory: Path) -> None:
    entries = tuple(DeliveryQueue(queue_directory).entries())
    events = [entry for entry in entries if entry["kind"] == "EVENT"]
    assert len(events) == 1
