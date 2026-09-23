"""Generation-6 adversarial cases for the observability-14 delivery-leg delta."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from observability_stack_fixtures import serve_backend, wait_until
from test_observability_end_to_end import (
    _BUDGET_BYTES,
    _CAMERA,
    _QUERY_FROM_NS,
    _QUERY_TO_NS,
    _RELAY_TOKEN,
    _drive_frames,
    _exporter,
    _identity,
    _NoClipPublisher,
    _pump,
    _RecordingPlane,
    _triggered_decisions,
)
from test_worker_flow_evidence_binding import _binding, _event, _Plane, _trigger

from shared.events.delivery_queue import (
    AdmissionFault,
    AdmissionResult,
    DeliveryQueue,
    EventEntry,
)
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.output.evidence import evidence_sender as sender_module
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.pipeline.output.evidence.smart_record_actor import SmartRecordActor
from worker.runtime.flow.evidence import FlowEvidenceBinding

_OBSERVING_BOOT = "boot-1"


class _CollectingSink:
    def __init__(self) -> None:
        self.records: list[object] = []

    def try_emit(self, record: object) -> bool:
        self.records.append(record)
        return True


class _BoomSink:
    def try_emit(self, record: object) -> bool:
        del record
        raise RuntimeError("sink exploded")


class _NoneStager:
    def stage(self, event: dict[str, object]) -> None:
        del event

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        del edge_event_id, clip_id


class _FalseStager:
    def stage(self, event: dict[str, object]) -> AdmissionResult:
        del event
        return AdmissionResult(False, AdmissionFault.ENTRY_CAPACITY)

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        del edge_event_id, clip_id


class Transport:
    def __init__(
        self,
        event_result: EventReceipt | DeliveryFailure | None = None,
    ) -> None:
        self.event_result = event_result

    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt | DeliveryFailure:
        del payload_json
        return self.event_result or EventReceipt("accepted", edge_event_id, "backend-event")

    def send_snapshot_attachment(self, payload: dict[str, object]) -> None:
        del payload

    def send_snapshot_disposition(self, payload: dict[str, object]) -> None:
        del payload

    def send_clip(self, claim: object) -> None:
        del claim
        raise AssertionError("no clip delivery expected")


class _RaisingTransport(Transport):
    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt:
        del payload_json, edge_event_id
        raise RuntimeError("entry payload is corrupt")


class _TwoTransientThenLocal:
    def __init__(self) -> None:
        self.calls = 0

    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt | DeliveryFailure:
        del payload_json
        self.calls += 1
        if self.calls <= 2:
            return DeliveryFailure(DeliveryDisposition.RETRY, "TEMPORARY", 503)
        return EventReceipt("accepted_local", edge_event_id, "")

    def send_snapshot_attachment(self, payload: dict[str, object]) -> None:
        del payload

    def send_snapshot_disposition(self, payload: dict[str, object]) -> None:
        del payload

    def send_clip(self, claim: object) -> None:
        del claim
        raise AssertionError("no clip delivery expected")


def _queued_event() -> EventEntry:
    return EventEntry(
        edge_event_id="event-a",
        event_type="fall",
        detected_at="2026-08-22T00:00:00Z",
        camera_id="camera-a",
        facility_id="facility-a",
        decision_trace=b"{}",
        values=b'{"edge_event_id":"event-a"}',
    )


def _sender_with_sink(directory: Path, transport: Transport, sink: object) -> EvidenceSender:
    return EvidenceSender(
        directory,
        SenderConfig("http://relay.test", "token", "camera-a"),
        transport=transport,
        execution_records=sink,  # type: ignore[arg-type]
        observing_boot_id="boot-observer",
    )


def _only_delivery(sink: _CollectingSink) -> object:
    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.record_kind == "event.delivery"  # type: ignore[attr-defined]
    return record


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _entry_ids(directory: Path) -> list[str]:
    return [str(entry["entry_id"]) for entry in DeliveryQueue(directory).entries()]


def test_g6_1_none_stage_result_is_refused_unproven_and_not_admitted(tmp_path: Path) -> None:
    plane, now = _Plane(), [0.0]
    _actor, binding, _stager, _ = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC)], tmp_path)
    binding.stager = _NoneStager()  # type: ignore[assignment]
    sink = _CollectingSink()
    binding.execution_records = sink
    with pytest.raises(RuntimeError, match="unproven-admission"):
        binding.emit_for_frame(_event("one"), _trigger())
    (record,) = sink.records
    assert record.record_kind == "event.delivery"  # type: ignore[attr-defined]
    assert record.outcome == "refused"  # type: ignore[attr-defined]
    reason = record.payload["reason"]  # type: ignore[attr-defined]
    assert isinstance(reason, str)
    assert reason.startswith("unproven-admission")
    assert plane.starts == []


def test_g6_2_false_admission_result_refuses_with_fault(tmp_path: Path) -> None:
    plane, now = _Plane(), [0.0]
    _actor, binding, _stager, _ = _binding(plane, now, [datetime(2026, 1, 1, tzinfo=UTC)], tmp_path)
    binding.stager = _FalseStager()  # type: ignore[assignment]
    sink = _CollectingSink()
    binding.execution_records = sink
    with pytest.raises(RuntimeError, match="entry_capacity"):
        binding.emit_for_frame(_event("one"), _trigger())
    (record,) = sink.records
    assert record.record_kind == "event.delivery"  # type: ignore[attr-defined]
    assert record.outcome == "refused"  # type: ignore[attr-defined]
    assert record.payload["reason"] == "entry_capacity"  # type: ignore[attr-defined]
    assert plane.starts == []


def test_g6_3_transport_raise_is_retry_counted_and_stays_queued(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_queued_event()).accepted
    sink = _CollectingSink()
    sender = _sender_with_sink(tmp_path, _RaisingTransport(), sink)
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "retry-counted"  # type: ignore[attr-defined]
    assert record.payload["attempt"] == 1  # type: ignore[attr-defined]
    assert record.payload["failure_class"] == "exception"  # type: ignore[attr-defined]
    assert _entry_ids(tmp_path) == ["event-event-a"]


def test_g6_4_exhausted_retention_full_keeps_entry_queued_and_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.events import delivery_queue as module

    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_queued_event()).accepted
    monkeypatch.setattr(module, "MAX_DEAD_LETTERED_ENTRIES", 0)
    sink = _CollectingSink()
    sender = _sender_with_sink(tmp_path, Transport(), sink)
    sender._attempts["event-event-a"] = sender_module._MAX_ENTRY_ATTEMPTS  # noqa: SLF001
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    record = _only_delivery(sink)
    assert record.outcome == "exhausted-retention-full"  # type: ignore[attr-defined]
    assert record.payload["retained"] is False  # type: ignore[attr-defined]
    assert _entry_ids(tmp_path) == ["event-event-a"]
    assert "event-event-a" in sender._deferred  # noqa: SLF001


def test_g6_5_mismatched_receipt_is_retry_counted_without_acceptance(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_queued_event()).accepted
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
    kinds = [item.record_kind for item in sink.records]  # type: ignore[attr-defined]
    assert "backend.acceptance" not in kinds
    assert _entry_ids(tmp_path) == ["event-event-a"]


def test_g6_6_sink_raise_does_not_change_sender_step_or_queue(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    control_dir = tmp_path / "control"
    boom_dir = tmp_path / "boom"
    control_dir.mkdir()
    boom_dir.mkdir()
    assert DeliveryQueue(control_dir).try_admit(_queued_event()).accepted
    assert DeliveryQueue(boom_dir).try_admit(_queued_event()).accepted
    control_sink = _CollectingSink()
    control = _sender_with_sink(control_dir, Transport(), control_sink)
    boom = _sender_with_sink(boom_dir, Transport(), _BoomSink())
    with caplog.at_level("WARNING", logger="worker.pipeline.diagnostics.record_builder"):
        boom_step = boom.run_once()
    control_step = control.run_once()
    assert boom_step is control_step is SenderStep.EVENT_ACKED
    assert _entry_ids(boom_dir) == _entry_ids(control_dir) == []
    warnings = [
        item
        for item in caplog.records
        if item.levelname == "WARNING" and item.name == "worker.pipeline.diagnostics.record_builder"
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "camera_id=camera-a" in message
    assert "record_kind=backend.acceptance" in message


def test_g6_7_query_returns_delivery_outcomes_then_local_acceptance(tmp_path: Path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=64)
    emitted: list[object] = []
    queue_dir = tmp_path / "delivery-queue"
    stager = DurableEvidenceStager(queue_dir, camera_id=_CAMERA, facility_id="facility-a")
    plane = _RecordingPlane()
    sealed: list[FlowEvidenceBinding] = []
    actor = SmartRecordActor(
        camera_id=_CAMERA,
        media_plane=plane,
        clock=lambda: 0.0,
        sink=lambda clip: sealed[0].on_sealed(clip),
        lookback_sec=10,
        clip_id_factory=lambda: "primary-clip",
    )
    binding = FlowEvidenceBinding(
        actor=actor,
        stager=stager,
        publisher=_NoClipPublisher(),
        sidecars=FlowSealedSidecars(tmp_path / "sidecars"),
        camera_id=_CAMERA,
        execution_records=lanes,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    sealed.append(binding)
    pump = _pump(
        lanes,
        identity=_identity(),
        emitted=emitted,
        fall_transition=0.9,
        event_sink=binding,
    )
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            _drive_frames(pump, 3, publish=True, consume=True)

            def _admitted() -> bool:
                return any(
                    row["record_kind"] == "event.delivery" and row["outcome"] == "admitted"
                    for row in _query(backend)["records"]
                )

            wait_until(_admitted, timeout=5.0, what="admitted event.delivery on Backend")
            triggered = _triggered_decisions(_query(backend))
            assert triggered, "the immediate classifier must trigger a fall in this fixture"
            assert emitted, "a triggered decision must emit an alert"
            (event,) = emitted
            edge_event_id = str(event.identity)  # type: ignore[attr-defined]
            transport = _TwoTransientThenLocal()
            sender = EvidenceSender(
                queue_dir,
                SenderConfig("http://relay.test", "token", _CAMERA),
                transport=transport,
                execution_records=lanes,
                observing_boot_id=_OBSERVING_BOOT,
            )
            assert sender.run_once() is SenderStep.RETRY_SCHEDULED
            assert sender.run_once() is SenderStep.RETRY_SCHEDULED
            assert sender.run_once() is SenderStep.EVENT_ACKED

            def _complete() -> bool:
                body = _query(backend)
                deliveries = [
                    row
                    for row in body["records"]
                    if row["record_kind"] == "event.delivery"
                    and row["causal_unit_id"] == edge_event_id
                ]
                acceptances = [
                    row
                    for row in body["records"]
                    if row["record_kind"] == "backend.acceptance"
                    and row["causal_unit_id"] == edge_event_id
                ]
                return len(deliveries) == 3 and len(acceptances) == 1

            wait_until(
                _complete,
                timeout=5.0,
                what="three event.delivery outcomes and one backend.acceptance",
            )
            body = _query(backend)
            deliveries = [
                row
                for row in body["records"]
                if row["record_kind"] == "event.delivery" and row["causal_unit_id"] == edge_event_id
            ]
            deliveries.sort(key=lambda row: int(row["producer_sequence"]))
            assert [row["outcome"] for row in deliveries] == [
                "admitted",
                "retry-transient",
                "retry-transient",
            ]
            assert [int(row["producer_sequence"]) for row in deliveries] == [0, 1, 2]
            assert all(row["causal_unit_id"] == edge_event_id for row in deliveries)
            acceptances = [
                row
                for row in body["records"]
                if row["record_kind"] == "backend.acceptance"
                and row["causal_unit_id"] == edge_event_id
            ]
            (acceptance,) = acceptances
            assert acceptance["payload"]["accepted_local"] is True
            assert acceptance["payload"]["hub_accepted"] is False
            assert acceptance["causal_unit_id"] == edge_event_id
        finally:
            if exporter is not None:
                exporter.stop()
