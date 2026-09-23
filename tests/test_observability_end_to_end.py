"""Hermetic end-to-end: pump -> lanes -> exporter -> Backend query."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from observability_stack_fixtures import serve_backend, wait_until
from test_execution_record_wiring import _metadata, _pump

from shared.events.execution_records import WireProvenance
from shared.events.execution_records_client import ExecutionRecordsClient
from worker.domains.registry import FALL_MODULE_QUALIFIED_ID
from worker.pipeline.diagnostics.exporter import ExecutionRecordExporter
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.runtime.flow.execution_record_emit import emit_policy_consume
from worker.runtime.flow.policy_pump import DecisionIdentity
from worker.types.metadata import MetadataCounters, MetadataFrame

_CAMERA = "cam-1"
_RELAY_TOKEN = "obs-relay-token"
_BUDGET_BYTES = 2**20
_PTS_STEP_NS = 66_666_667
_QUERY_FROM_NS = 0
_QUERY_TO_NS = (1 << 62) - 1
_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="model-1",
    calibration_digest="cal-1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg-1",
    policy_identity="fall.policy:2",
)


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        module_qualified_id=FALL_MODULE_QUALIFIED_ID,
        effective_policy_id="a" * 64,
    )


def _frame(pump: object, seq: int) -> MetadataFrame:
    child = pump._child  # noqa: SLF001
    assert isinstance(child, UUID)
    return _metadata(child=child, seq=seq, pts=100 + seq * _PTS_STEP_NS)


def _drive_frames(
    pump: object,
    count: int,
    *,
    publish: bool,
    consume: bool,
) -> None:
    slot = pump._slot  # noqa: SLF001
    for seq in range(count):
        metadata = _frame(pump, seq)
        if publish:
            assert slot.publish(metadata) is True
        pump._process(metadata)  # noqa: SLF001
        if consume:
            emit_policy_consume(
                pump._execution_records,  # noqa: SLF001
                metadata,
                before=MetadataCounters(),
                after=slot.counters(),
                processed_count=seq + 1,
            )


def _exporter(
    lanes: ExecutionRecordLanes,
    base_url: str,
    relay_token: str,
    *,
    batch_max: int = 16,
    flush_ms: int = 20,
) -> ExecutionRecordExporter:
    return ExecutionRecordExporter(
        lanes=lanes,
        client=ExecutionRecordsClient(base_url, relay_token),
        provenance=_PROVENANCE,
        batch_max=batch_max,
        flush_ms=flush_ms,
    )


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _triggered_decisions(body: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for record in body["records"]:
        if record["record_kind"] != "policy.decision":
            continue
        payload = record["payload"]
        if isinstance(payload, dict) and payload.get("triggered") is True:
            found.append(record)
    return found


def _availability_kind_at(body: dict[str, Any], timestamp_ns: int) -> str | None:
    for row in body["availability"]:
        if row["from_ns"] <= timestamp_ns <= row["to_ns"]:
            return str(row["kind"])
    return None


def test_alert_joins_record_with_four_kinds_provenance_and_availability(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=64)
    emitted: list[object] = []
    pump = _pump(lanes, identity=_identity(), emitted=emitted, fall_transition=0.9)
    exporter: ExecutionRecordExporter | None = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            _drive_frames(pump, 3, publish=True, consume=True)

            def _triggered() -> bool:
                return bool(_triggered_decisions(_query(backend)))

            wait_until(_triggered, timeout=5.0, what="triggered policy.decision on Backend")
            body = _query(backend)
            triggered = _triggered_decisions(body)
            assert triggered, "the immediate classifier must trigger a fall in this fixture"
            (record,) = triggered
            assert emitted, "a triggered decision must emit an alert"
            (event,) = emitted
            audit = event.audit  # type: ignore[attr-defined]
            assert audit is not None
            assert audit["decision_trace_id"] == record["payload"]["decision_trace_id"]

            kinds = {row["record_kind"] for row in body["records"]}
            assert "sdk.frame" in kinds
            assert "model.score" in kinds
            assert "policy.decision" in kinds
            assert "policy.consume" in kinds

            provenance_ids = {row["provenance_id"] for row in body["records"]}
            assert len(provenance_ids) == 1
            (provenance_id,) = provenance_ids
            assert provenance_id

            observed = [int(row["observed_at_ns"]) for row in body["records"]]
            first_observed = min(observed)
            last_observed = max(observed)
            assert _availability_kind_at(body, first_observed) == "AVAILABLE"
            assert _availability_kind_at(body, last_observed) == "AVAILABLE"
            tail = [
                row
                for row in body["availability"]
                if int(row["from_ns"]) > last_observed and row["kind"] == "UNKNOWN"
            ]
            assert tail, "the tail after last_observed must be UNKNOWN"

            queryable = body["queryable_range"]
            assert queryable["min_observed_at_ns"] == first_observed
            assert queryable["max_observed_at_ns"] == last_observed
        finally:
            if exporter is not None:
                exporter.stop()


def test_lane_overflow_is_reported_as_missing_not_recorded(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=2)
    pump = _pump(lanes, identity=_identity(), emitted=[], fall_transition=0.9)
    # Mixed producers on purpose: sdk.frame carries PTS-derived ns while
    # policy.decision / model.score carry process-monotonic ns, so dropped
    # items arrive with non-monotonic observed_at_ns. The lane must still
    # synthesize a valid per-producer gap (min/max ns) rather than kill the
    # exporter thread on a contract error.
    _drive_frames(pump, 6, publish=True, consume=True)
    exporter: ExecutionRecordExporter | None = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token, batch_max=32)
            exporter.start()

            def _overflow_row() -> dict[str, Any] | None:
                body = _query(backend)
                for row in body["coverage"]:
                    if (
                        row["coverage_kind"] == "MISSING_NOT_RECORDED"
                        and row["cause"] == "lane-overflow"
                        and row["from_sequence"] is not None
                        and row["to_sequence"] is not None
                    ):
                        return row
                return None

            wait_until(
                lambda: _overflow_row() is not None,
                timeout=5.0,
                what="lane-overflow MISSING_NOT_RECORDED coverage",
            )
            body = _query(backend)
            row = _overflow_row()
            assert row is not None
            assert row["exact"] is True
            assert int(row["to_sequence"]) >= int(row["from_sequence"])
            missing = [
                item
                for item in body["availability"]
                if item["kind"] == "MISSING_NOT_RECORDED"
                and not (
                    int(item["to_ns"]) < int(row["from_ns"])
                    or int(item["from_ns"]) > int(row["to_ns"])
                )
            ]
            assert missing, "loss must appear as MISSING_NOT_RECORDED availability, never hidden"
        finally:
            if exporter is not None:
                exporter.stop()


def test_restart_reopens_the_same_execution_records(tmp_path) -> None:
    lanes = ExecutionRecordLanes(lane_capacity=64)
    pump = _pump(lanes, identity=_identity(), emitted=[], fall_transition=0.9)
    exporter: ExecutionRecordExporter | None = None
    record_ids: list[str] = []
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            _drive_frames(pump, 3, publish=True, consume=True)
            wait_until(
                lambda: len(_query(backend)["records"]) >= 1,
                timeout=5.0,
                what="exported execution records before restart",
            )
            record_ids = [row["record_id"] for row in _query(backend)["records"]]
            assert record_ids
        finally:
            if exporter is not None:
                exporter.stop()

    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as restarted:
        body = _query(restarted)
        assert [row["record_id"] for row in body["records"]] == record_ids
