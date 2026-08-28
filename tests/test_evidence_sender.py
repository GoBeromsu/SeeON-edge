from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shared.events.delivery_queue import (
    DeliveryQueue,
    EventEntry,
    SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
)
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep


@dataclass
class Transport:
    calls: list[str] = field(default_factory=list)
    event_result: EventReceipt | DeliveryFailure | None = None
    attachment_result: DeliveryFailure | None = None
    disposition_result: DeliveryFailure | None = None

    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt | DeliveryFailure:
        self.calls.append(f"event:{edge_event_id}")
        assert json.loads(payload_json)["edge_event_id"] == edge_event_id
        return self.event_result or EventReceipt("accepted", edge_event_id, "backend-event")

    def send_snapshot_attachment(self, payload: dict[str, object]) -> DeliveryFailure | None:
        self.calls.append(f"attachment:{payload['snapshot_id']}")
        return self.attachment_result

    def send_snapshot_disposition(self, payload: dict[str, object]) -> DeliveryFailure | None:
        self.calls.append(f"disposition:{payload['snapshot_id']}")
        return self.disposition_result


def _sender(directory: Path, transport: Transport) -> EvidenceSender:
    return EvidenceSender(
        directory, SenderConfig("http://relay.test", "token", "camera-a"), transport=transport
    )


def _event() -> EventEntry:
    return EventEntry(
        edge_event_id="event-a",
        event_type="fall",
        detected_at="2026-08-22T00:00:00Z",
        camera_id="camera-a",
        facility_id="facility-a",
        decision_trace=b"{}",
        values=b'{"edge_event_id":"event-a"}',
    )


def _attachment() -> SnapshotAttachmentEntry:
    return SnapshotAttachmentEntry(
        "event-a", "snapshot-a", "a" * 64, "snapshots/a.jpg", 7, "image/jpeg"
    )


def _disposition() -> SnapshotDispositionEntry:
    return SnapshotDispositionEntry("event-a", "snapshot-missing", "MISSING", "capture failed")


def test_event_is_sent_before_optional_snapshot_entries(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_attachment()).accepted
    assert queue.try_admit(_event()).accepted
    transport = Transport()
    assert _sender(tmp_path, transport).run_once() is SenderStep.EVENT_ACKED
    assert transport.calls == ["event:event-a"]
    assert next(iter(DeliveryQueue(tmp_path).entries()))["kind"] == "SNAPSHOT_ATTACHMENT"


def test_retry_keeps_the_unacknowledged_event_durable(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    transport = Transport(event_result=DeliveryFailure(DeliveryDisposition.RETRY, "TEMPORARY"))
    assert _sender(tmp_path, transport).run_once() is SenderStep.RETRY_SCHEDULED
    assert [entry["entry_id"] for entry in DeliveryQueue(tmp_path).entries()] == ["event-event-a"]


def test_event_receipt_acknowledges_only_the_matching_event(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    assert queue.try_admit(_attachment()).accepted
    transport = Transport(event_result=EventReceipt("accepted", "event-a", "backend-event"))
    assert _sender(tmp_path, transport).run_once() is SenderStep.EVENT_ACKED
    entries = tuple(DeliveryQueue(tmp_path).entries())
    assert [entry["kind"] for entry in entries] == ["SNAPSHOT_ATTACHMENT"]


def test_attachment_conflict_is_retained_without_delivery_proof(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_attachment()).accepted
    transport = Transport(
        attachment_result=DeliveryFailure(DeliveryDisposition.PERMANENT, "CONFLICT", 409)
    )

    assert _sender(tmp_path, transport).run_once() is SenderStep.RETRY_SCHEDULED
    assert [entry["kind"] for entry in DeliveryQueue(tmp_path).entries()] == [
        "SNAPSHOT_ATTACHMENT"
    ]


def test_attachment_acknowledgement_does_not_remove_event_or_disposition(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_attachment()).accepted
    assert queue.try_admit(_disposition()).accepted
    transport = Transport()
    assert _sender(tmp_path, transport).run_once() is SenderStep.CLIP_ACKED
    assert [entry["kind"] for entry in DeliveryQueue(tmp_path).entries()] == [
        "SNAPSHOT_DISPOSITION"
    ]


def test_attachment_identity_is_idempotent_and_delivered_once(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    first = queue.try_admit(_attachment())
    replay = queue.try_admit(_attachment())
    assert first.accepted and replay.accepted and replay.already_admitted
    transport = Transport()
    assert _sender(tmp_path, transport).run_once() is SenderStep.CLIP_ACKED
    assert transport.calls == ["attachment:snapshot-a"]


def test_terminal_disposition_is_delivered_without_event_mutation(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    assert queue.try_admit(_disposition()).accepted
    transport = Transport()
    assert _sender(tmp_path, transport).run_once() is SenderStep.EVENT_ACKED
    assert _sender(tmp_path, transport).run_once() is SenderStep.CLIP_ACKED
    assert transport.calls == ["event:event-a", "disposition:snapshot-missing"]
