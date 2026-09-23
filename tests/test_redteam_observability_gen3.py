"""Generation-3 adversarial cases for the observability-11 delta."""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

import pytest
from observability_stack_fixtures import serve_backend, wait_until
from test_execution_record_wiring import _metadata, _pump
from test_flow_policy_pump_preview import _domain_decider, _fall_input
from test_observability_end_to_end import (
    _BUDGET_BYTES,
    _PROVENANCE,
    _QUERY_FROM_NS,
    _QUERY_TO_NS,
    _RELAY_TOKEN,
    _exporter,
)

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.execution_records import WireBatch, WireBatchReceipt, WireRecord
from shared.events.execution_records_client import ExecutionRecordsClient
from worker.pipeline.decision import unwrap_decider
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import LANE_OVERFLOW_CAUSE, ExecutionRecordLanes
from worker.runtime.flow.policy_pump import NativePolicyPump

_CAMERA = "cam-1"
_BOOT_OVERFLOW = "boot-1"
_BOOT_OTHER = "boot-2"
_FLUSH_MS = 20
_UNWRAP_JOIN_SEC = 1.0
_WARMUP_FRAMES = 5


def _lane_record(
    *,
    seq: int = 0,
    observed: int = 1_000,
    boot: str = _BOOT_OVERFLOW,
) -> WireRecord:
    return WireRecord(
        record_kind="sdk.frame",
        camera_id=_CAMERA,
        worker_boot_id=boot,
        source_generation=0,
        stream_epoch=1,
        producer="sdk",
        producer_sequence=seq,
        observed_at_ns=observed,
        time_quality="monotonic",
        causal_unit_id=f"{_CAMERA}:{boot}:1:frame:{seq}",
        outcome="accepted",
        payload={"boot": boot, "n": seq},
    )


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _drive_pump(pump: NativePolicyPump, count: int, *, start: int = 0) -> None:
    child = pump._child
    assert isinstance(child, UUID)
    for seq in range(start, start + count):
        pump._process(_metadata(child=child, seq=seq, pts=100 + seq * 66_666_667))


def _interleave_two_boots(lanes: ExecutionRecordLanes) -> None:
    assert lanes.try_emit(_lane_record(boot=_BOOT_OVERFLOW, seq=0, observed=1_000)) is True
    assert lanes.try_emit(_lane_record(boot=_BOOT_OTHER, seq=0, observed=3_000)) is True
    assert lanes.try_emit(_lane_record(boot=_BOOT_OVERFLOW, seq=1, observed=1_500)) is True
    assert lanes.try_emit(_lane_record(boot=_BOOT_OTHER, seq=1, observed=4_000)) is True
    assert lanes.try_emit(_lane_record(boot=_BOOT_OVERFLOW, seq=2, observed=2_000)) is False


def _leave_only_overflow(lanes: ExecutionRecordLanes) -> None:
    """Drop queued records after overflow so the lane holds loss only.

    ``try_emit`` cannot overflow an empty deque (capacity >= 1 admits first).
    Clearing the admitted record after a real overflow is the public-loss
    state ``cameras_with_work`` / the exporter must still flush.
    """
    with lanes._condition:  # noqa: SLF001
        found = False
        for lane in lanes._lanes.values():  # noqa: SLF001
            if lane.overflow:
                lane.records.clear()
                found = True
        assert found, "expected a lane-overflow to remain after clearing records"


class _AlwaysWarmupClassifier:
    def __init__(self) -> None:
        self.current_call_missing_score_reasons: dict[int, str] = {}

    def update(self, rows: object, live_track_ids: tuple[int, ...]) -> dict[int, object]:
        del rows
        self.current_call_missing_score_reasons = dict.fromkeys(
            live_track_ids, "classifier-warmup"
        )
        return {}

    def probabilities_for(self, track_id: int) -> object | None:
        del track_id
        return None


class _UpdateOnlyClassifier:
    def update(self, rows: object, live_track_ids: tuple[int, ...]) -> dict[int, object]:
        del rows, live_track_ids
        return {}


class _SelfRef:
    def __init__(self) -> None:
        self.decider = self


class _FailThenReal:
    def __init__(self, inner: ExecutionRecordsClient, *, fail_times: int = 1) -> None:
        self._inner = inner
        self._fail_times = fail_times
        self.failure_count = 0
        self.success_count = 0
        self.posted: list[WireBatch] = []

    def post_batch(self, batch: WireBatch) -> WireBatchReceipt | DeliveryFailure:
        if self._fail_times > 0:
            self._fail_times -= 1
            self.failure_count += 1
            return DeliveryFailure(DeliveryDisposition.RETRY, "HTTP_503", status_code=503)
        self.success_count += 1
        self.posted.append(batch)
        return self._inner.post_batch(batch)


def test_g3_1_interleaved_boots_isolate_overflow_and_restart_sequences(tmp_path) -> None:
    isolated = ExecutionRecordLanes(lane_capacity=2)
    _interleave_two_boots(isolated)
    other = isolated.drain_for(_CAMERA, _BOOT_OTHER, limit=8)
    overflowed = isolated.drain_for(_CAMERA, _BOOT_OVERFLOW, limit=8)
    assert other is not None
    assert overflowed is not None
    assert {row.worker_boot_id for row in other.records} == {_BOOT_OTHER}
    assert {row.worker_boot_id for row in overflowed.records} == {_BOOT_OVERFLOW}
    assert [row.producer_sequence for row in other.records] == [0, 1]
    assert [row.producer_sequence for row in overflowed.records] == [0, 1]
    assert other.gaps == ()
    assert overflowed.gaps
    assert all(gap.cause == LANE_OVERFLOW_CAUSE for gap in overflowed.gaps)
    assert all(gap.from_sequence == 2 and gap.to_sequence == 2 for gap in overflowed.gaps)

    lanes = ExecutionRecordLanes(lane_capacity=2)
    _interleave_two_boots(lanes)
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(
                lanes, backend.base_url, backend.relay_token, batch_max=32, flush_ms=_FLUSH_MS
            )
            exporter.start()

            def _both_boots_and_overflow() -> bool:
                body = _query(backend)
                boots = {row["payload"]["boot"] for row in body.get("records", ())}
                overflow = [
                    row
                    for row in body.get("coverage", ())
                    if row["coverage_kind"] == "MISSING_NOT_RECORDED"
                    and row["cause"] == LANE_OVERFLOW_CAUSE
                ]
                return {_BOOT_OVERFLOW, _BOOT_OTHER} <= boots and len(overflow) == 1

            wait_until(
                _both_boots_and_overflow,
                timeout=5.0,
                what="both boots exported with overflow bound to one boot",
            )
            body = _query(backend)
            by_boot: dict[str, list[int]] = {}
            for row in body["records"]:
                boot = str(row["payload"]["boot"])
                by_boot.setdefault(boot, []).append(int(row["producer_sequence"]))
            assert by_boot[_BOOT_OVERFLOW] == [0, 1]
            assert by_boot[_BOOT_OTHER] == [0, 1]
            overflow = [
                row
                for row in body["coverage"]
                if row["coverage_kind"] == "MISSING_NOT_RECORDED"
                and row["cause"] == LANE_OVERFLOW_CAUSE
            ]
            assert len(overflow) == 1
            row = overflow[0]
            assert int(row["from_sequence"]) == 2
            assert int(row["to_sequence"]) == 2
            assert int(row["from_ns"]) == 2_000
            assert int(row["to_ns"]) == 2_000
        finally:
            if exporter is not None:
                exporter.stop()


def test_g3_2_overflow_only_lane_exports_within_flush_ms(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=1)
    assert lanes.try_emit(_lane_record(seq=0, observed=1_000)) is True
    assert lanes.try_emit(_lane_record(seq=1, observed=2_000)) is False
    _leave_only_overflow(lanes)
    assert lanes.queued() == 0
    assert (_CAMERA, _BOOT_OVERFLOW) in lanes.cameras_with_work()

    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(
                lanes, backend.base_url, backend.relay_token, batch_max=8, flush_ms=_FLUSH_MS
            )
            exporter.start()

            def _overflow_coverage() -> bool:
                body = _query(backend)
                return any(
                    row["coverage_kind"] == "MISSING_NOT_RECORDED"
                    and row["cause"] == LANE_OVERFLOW_CAUSE
                    and row["from_sequence"] is not None
                    for row in body.get("coverage", ())
                )

            wait_until(
                _overflow_coverage,
                timeout=(_FLUSH_MS / 1000.0) + 2.0,
                what="overflow-only gap on Backend within FLUSH_MS bound",
            )
            body = _query(backend)
            assert body["records"] == []
            overflow = [
                row
                for row in body["coverage"]
                if row["coverage_kind"] == "MISSING_NOT_RECORDED"
                and row["cause"] == LANE_OVERFLOW_CAUSE
            ]
            assert overflow
        finally:
            if exporter is not None:
                exporter.stop()


def test_g3_3_export_failed_gap_delivers_without_new_record(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    assert lanes.try_emit(_lane_record(seq=0, observed=1_000)) is True
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            inner = ExecutionRecordsClient(backend.base_url, backend.relay_token)
            client = _FailThenReal(inner)
            exporter = ExecutionRecordExporter(
                lanes=lanes,
                client=client,  # type: ignore[arg-type]
                provenance=_PROVENANCE,
                batch_max=8,
                flush_ms=_FLUSH_MS,
            )
            exporter.start()

            def _export_failed_coverage() -> bool:
                body = _query(backend)
                return any(
                    row["coverage_kind"] == "MISSING_NOT_RECORDED"
                    and row["cause"] == "export-failed"
                    for row in body.get("coverage", ())
                )

            wait_until(
                _export_failed_coverage,
                timeout=5.0,
                what="export-failed gap delivered with no new valid record",
            )
            assert client.failure_count == 1
            assert client.success_count >= 1
            assert exporter.failures()
            body = _query(backend)
            assert body["records"] == []
            failed = [
                row
                for row in body["coverage"]
                if row["coverage_kind"] == "MISSING_NOT_RECORDED"
                and row["cause"] == "export-failed"
            ]
            assert failed
            assert all(int(row["from_sequence"]) == 0 for row in failed)
        finally:
            if exporter is not None:
                exporter.stop()


def test_g3_4_classifier_without_missing_score_reasons_is_typeerror() -> None:
    decider = _domain_decider(_UpdateOnlyClassifier())
    with pytest.raises(TypeError, match="current_call_missing_score_reasons"):
        decider.update(_fall_input(time_sec=1.0, frame_index=1))


def test_g3_5_always_warmup_emits_decisions_without_model_scores() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=256)
    pump = _pump(lanes, identity=None, fall_transition=0.9)
    pump._decision.deciders[0].classifier = _AlwaysWarmupClassifier()
    _drive_pump(pump, _WARMUP_FRAMES)
    drained = lanes.drain_for(_CAMERA, "boot-1", limit=256)
    assert drained is not None
    scores = [row for row in drained.records if row.record_kind == "model.score"]
    decisions = [row for row in drained.records if row.record_kind == "policy.decision"]
    assert scores == []
    assert len(decisions) == _WARMUP_FRAMES
    for row in decisions:
        missing = row.payload["missing_values"]
        assert "classifier-warmup" in missing.values()


def test_g3_6_unwrap_decider_self_ref_terminates() -> None:
    wrapped = _SelfRef()
    result: list[object] = []
    errors: list[BaseException] = []

    def _call() -> None:
        try:
            result.append(unwrap_decider(wrapped))
        except BaseException as exc:  # noqa: BLE001 - hang guard must surface any failure
            errors.append(exc)

    thread = threading.Thread(target=_call, name="unwrap-self-ref", daemon=True)
    thread.start()
    thread.join(timeout=_UNWRAP_JOIN_SEC)
    assert not thread.is_alive(), "unwrap_decider hung on a self-referential wrapper"
    assert errors == []
    assert result
    assert result[0] is wrapped
