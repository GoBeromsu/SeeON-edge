from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

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
    assert [entry["kind"] for entry in DeliveryQueue(tmp_path).entries()] == ["SNAPSHOT_ATTACHMENT"]


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


class _CollectingSink:
    def __init__(self) -> None:
        self.records: list[object] = []

    def try_emit(self, record: object) -> bool:
        self.records.append(record)
        return True


def test_sink_without_observing_boot_id_refuses_at_construction(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="observing boot id"):
        EvidenceSender(
            tmp_path,
            SenderConfig("http://relay.test", "token", "camera-a"),
            transport=Transport(),
            execution_records=_CollectingSink(),
        )


def test_delivered_event_emits_process_scoped_acceptance_record(tmp_path: Path) -> None:
    from shared.events.execution_records import PROCESS_SCOPE

    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    sender = EvidenceSender(
        tmp_path,
        SenderConfig("http://relay.test", "token", "camera-a"),
        transport=Transport(event_result=EventReceipt("accepted_local", "event-a", "")),
        execution_records=sink,
        observing_boot_id="boot-observer",
    )
    assert sender.run_once() is SenderStep.EVENT_ACKED
    (record,) = sink.records
    assert record.record_kind == "backend.acceptance"  # type: ignore[attr-defined]
    assert record.worker_boot_id == "boot-observer"  # type: ignore[attr-defined]
    assert record.source_generation == PROCESS_SCOPE  # type: ignore[attr-defined]
    assert record.stream_epoch == PROCESS_SCOPE  # type: ignore[attr-defined]
    assert record.causal_unit_id == "event-a"  # type: ignore[attr-defined]
    assert record.outcome == "accepted_local"  # type: ignore[attr-defined]
    payload = record.payload  # type: ignore[attr-defined]
    assert payload["accepted_local"] is True
    assert payload["hub_accepted"] is False
    assert payload["origin_boot_id"] is None


def _sender_with_sink(
    directory: Path, transport: Transport, sink: _CollectingSink
) -> EvidenceSender:
    return EvidenceSender(
        directory,
        SenderConfig("http://relay.test", "token", "camera-a"),
        transport=transport,
        execution_records=sink,
        observing_boot_id="boot-observer",
    )


def _only_delivery(sink: _CollectingSink) -> object:
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.record_kind == "event.delivery"  # type: ignore[attr-defined]
    return record


def test_happy_path_emits_acceptance_and_zero_delivery_records(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    sender = _sender_with_sink(tmp_path, Transport(), sink)
    assert sender.run_once() is SenderStep.EVENT_ACKED
    kinds = [record.record_kind for record in sink.records]  # type: ignore[attr-defined]
    assert kinds == ["backend.acceptance"]


def test_transient_retry_emits_retry_transient_without_consuming_attempt(
    tmp_path: Path,
) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    transport = Transport(event_result=DeliveryFailure(DeliveryDisposition.RETRY, "TEMPORARY", 503))
    sender = _sender_with_sink(tmp_path, transport, sink)
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "retry-transient"  # type: ignore[attr-defined]
    assert record.payload["attempt"] == 0  # type: ignore[attr-defined]
    assert record.payload["failure_class"] == "RETRY"  # type: ignore[attr-defined]
    assert sender._attempts.get("event-event-a", 0) == 0  # noqa: SLF001
    assert [entry["entry_id"] for entry in DeliveryQueue(tmp_path).entries()] == ["event-event-a"]


def test_send_exception_emits_retry_counted(tmp_path: Path) -> None:
    class _Raising(Transport):
        def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt:
            del payload_json, edge_event_id
            raise RuntimeError("entry payload is corrupt")

    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    sender = _sender_with_sink(tmp_path, _Raising(), sink)
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "retry-counted"  # type: ignore[attr-defined]
    assert record.payload["attempt"] == 1  # type: ignore[attr-defined]
    assert record.payload["failure_class"] == "exception"  # type: ignore[attr-defined]


def test_permanent_422_retained_emits_refused_retained(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    transport = Transport(
        event_result=DeliveryFailure(DeliveryDisposition.PERMANENT, "UNPROCESSABLE", 422)
    )
    sender = _sender_with_sink(tmp_path, transport, sink)
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "refused-retained"  # type: ignore[attr-defined]
    assert record.payload["retained"] is True  # type: ignore[attr-defined]
    assert record.payload["status_code"] == 422  # type: ignore[attr-defined]
    assert record.payload["dead_letter_dir"] == f"{tmp_path.name}-dead-letter"  # type: ignore[attr-defined]
    assert not tuple(DeliveryQueue(tmp_path).entries())


def test_permanent_422_retention_full_emits_refused_retention_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.events import delivery_queue as module

    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 0)
    sink = _CollectingSink()
    transport = Transport(
        event_result=DeliveryFailure(DeliveryDisposition.PERMANENT, "UNPROCESSABLE", 422)
    )
    sender = _sender_with_sink(tmp_path, transport, sink)
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "refused-retention-full"  # type: ignore[attr-defined]
    assert record.payload["retained"] is False  # type: ignore[attr-defined]
    assert [entry["entry_id"] for entry in DeliveryQueue(tmp_path).entries()] == ["event-event-a"]


def test_exhausted_after_max_attempts_emits_exhausted_retained(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    sender = _sender_with_sink(
        tmp_path,
        Transport(event_result=DeliveryFailure(DeliveryDisposition.PERMANENT, "HTTP_500")),
        sink,
    )
    sender._attempts["event-event-a"] = 10  # noqa: SLF001
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "exhausted-retained"  # type: ignore[attr-defined]
    assert record.payload["attempt"] == 10  # type: ignore[attr-defined]
    assert record.payload["retained"] is True  # type: ignore[attr-defined]
    assert not tuple(DeliveryQueue(tmp_path).entries())


def test_edge_event_id_mismatch_emits_retry_counted(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    sender = _sender_with_sink(
        tmp_path,
        Transport(event_result=EventReceipt("accepted", "other-event", "backend-event")),
        sink,
    )
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "retry-counted"  # type: ignore[attr-defined]
    assert record.payload["failure_class"] == "edge_event_id_mismatch"  # type: ignore[attr-defined]
    assert record.payload["attempt"] == 1  # type: ignore[attr-defined]


def test_acknowledge_failure_emits_ack_removal_deferred(tmp_path: Path) -> None:
    import errno
    from unittest.mock import patch

    import shared.events.delivery_queue as queue_module

    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_event()).accepted
    sink = _CollectingSink()
    sender = _sender_with_sink(tmp_path, Transport(), sink)
    real_unlink = queue_module.Path.unlink

    def _failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.parent == tmp_path:
            raise OSError(errno.EIO, "io error")
        real_unlink(self, *args, **kwargs)

    with patch.object(queue_module.Path, "unlink", _failing_unlink):
        assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "ack-removal-deferred"  # type: ignore[attr-defined]
    assert record.payload["failure_class"] == "acknowledge"  # type: ignore[attr-defined]
