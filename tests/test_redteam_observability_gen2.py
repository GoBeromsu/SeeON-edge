"""Generation-2 adversarial cases for the observability-10 delta."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from observability_stack_fixtures import serve_backend, wait_until
from test_execution_record_wiring import _metadata, _pump
from test_observability_end_to_end import (
    _BUDGET_BYTES,
    _QUERY_FROM_NS,
    _QUERY_TO_NS,
    _RELAY_TOKEN,
    _exporter,
    _identity,
)

from backend.app.features.diagnostics import records as backend_records
from shared.events import execution_records as shared_execution_records
from shared.events.execution_records import WireRecord
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall.classifier import (
    FALL_STRIDE_FRAMES,
    FALL_WINDOW_FRAMES,
    FallWindowClassifier,
)
from worker.interfaces.fall_model import FallProbabilities
from worker.pipeline.decision import unwrap_decider
from worker.pipeline.diagnostics.emit_policy import policy_decision_record
from worker.pipeline.diagnostics.lanes import RECORD_INVALID_CAUSE, ExecutionRecordLanes
from worker.pipeline.diagnostics.record_builder import fall_causal_unit_id
from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.runtime.worker import _WindowGatedDecider
from worker.types.business_event import BusinessEvent
from worker.types.trace import DecisionTraceSnapshot

pytest_plugins = (
    "tests_support.postgres_sandbox",
    "tests_support.postgres_diagnostics_sandbox",
)

_CAMERA = "cam-1"
_REAL_ZERO_UNIT = fall_causal_unit_id("cam-1", "boot-1", 3, 0, 0)
_NO_TRACK_UNIT = fall_causal_unit_id("cam-1", "boot-1", 3, None, 0)
_NO_GENERATION_UNIT = fall_causal_unit_id("cam-1", "boot-1", 3, 0, None)
_E2E_NO_TRACK_UNIT = fall_causal_unit_id("cam-1", "boot-1", 3, None, None)


def _snapshot(*, track_id: int | None) -> DecisionTraceSnapshot:
    return DecisionTraceSnapshot(
        reason="score-missing",
        previous_state="not-evaluated",
        current_state="not-evaluated",
        triggered=False,
        track_id=track_id,
        bed_id=None,
    )


def _policy_record(snapshot: DecisionTraceSnapshot, *, generation: int | None) -> WireRecord:
    record = policy_decision_record(
        snapshot,
        camera_id="cam-1",
        worker_boot_id="boot-1",
        source_generation=1,
        stream_epoch=3,
        frame_seq=7,
        source_pts_ns=None,
        generation=generation,
        module_qualified_id="fall.v2",
        authority_role="authoritative",
        observed_at_ns=1_000,
    )
    assert record is not None
    return record


def _lane_record(*, seq: int = 0, observed: int = 1_000) -> WireRecord:
    return WireRecord(
        record_kind="sdk.frame",
        camera_id="cam-1",
        worker_boot_id="boot-1",
        source_generation=0,
        stream_epoch=1,
        producer="sdk",
        producer_sequence=seq,
        observed_at_ns=observed,
        time_quality="monotonic",
        causal_unit_id="cam-1:boot-1:1:frame:0",
        outcome="accepted",
        payload={"n": seq},
    )


def _invalid_sequenced_record() -> WireRecord:
    template = _lane_record()
    broken = object.__new__(WireRecord)
    for name in WireRecord.__slots__:
        object.__setattr__(broken, name, getattr(template, name))
    object.__setattr__(broken, "time_quality", "not-a-quality")
    return broken


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _drive_pump(pump: NativePolicyPump, count: int, *, start: int = 0) -> None:
    child = pump._child
    assert isinstance(child, UUID)
    for seq in range(start, start + count):
        pump._process(_metadata(child=child, seq=seq, pts=100 + seq * 66_666_667))


def _gated_pump(lanes: ExecutionRecordLanes) -> NativePolicyPump:
    pump = _pump(lanes, identity=_identity(), fall_transition=0.9)
    inner = pump._decision.deciders[0]
    gated = _WindowGatedDecider(
        decider=inner,
        window=DetectionWindow(start="00:00", end="00:00", tz="UTC"),
        clock=lambda: datetime.now(UTC),
    )
    object.__setattr__(pump._decision, "deciders", (gated,))
    return pump


class _Nested:
    def __init__(self, inner: object) -> None:
        self.decider = inner


class _Plain:
    def update(self, input_value: object) -> tuple[BusinessEvent, ...]:
        del input_value
        return ()


class _BoomSink:
    def try_emit(self, record: object) -> bool:
        del record
        raise RuntimeError("sink exploded")


class _StubFallModel:
    def predict(self, features: object) -> FallProbabilities:
        del features
        return FallProbabilities(0.1, 0.9, 0.0)


def test_g2_1_missing_track_and_generation_units_never_alias_zero_through_query(
    tmp_path, postgres_product_sandbox, postgres_audit_runtime, postgres_lifespan_diagnostics_schema
) -> None:
    no_track = _policy_record(_snapshot(track_id=None), generation=0)
    no_generation = _policy_record(_snapshot(track_id=0), generation=None)
    real_zero = _policy_record(_snapshot(track_id=0), generation=0)

    assert no_track.causal_unit_id == _NO_TRACK_UNIT
    assert no_generation.causal_unit_id == _NO_GENERATION_UNIT
    assert real_zero.causal_unit_id == _REAL_ZERO_UNIT
    assert len({_NO_TRACK_UNIT, _NO_GENERATION_UNIT, _REAL_ZERO_UNIT, _E2E_NO_TRACK_UNIT}) == 4
    assert no_track.payload["track_id"] is None
    assert no_generation.payload["track_id"] == 0

    lanes = ExecutionRecordLanes(lane_capacity=64)
    assert lanes.try_emit(real_zero) is True
    pump = _gated_pump(lanes)
    exporter = None
    with serve_backend(
        tmp_path,
        budget_bytes=_BUDGET_BYTES,
        relay_token=_RELAY_TOKEN,
        sandbox=postgres_product_sandbox,
        audit_runtime=postgres_audit_runtime,
        diagnostics_schema=postgres_lifespan_diagnostics_schema,
    ) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            _drive_pump(pump, 1)

            def _both_units_visible() -> bool:
                body = _query(backend)
                unit_ids = {row["causal_unit_id"] for row in body.get("units", ())}
                return _E2E_NO_TRACK_UNIT in unit_ids and _REAL_ZERO_UNIT in unit_ids

            wait_until(
                _both_units_visible,
                timeout=5.0,
                what="no-track unit listed separately from :0:0 on Backend",
            )
            body = _query(backend)
            unit_ids = {row["causal_unit_id"] for row in body["units"]}
            assert _E2E_NO_TRACK_UNIT in unit_ids
            assert _REAL_ZERO_UNIT in unit_ids
            assert _E2E_NO_TRACK_UNIT != _REAL_ZERO_UNIT
        finally:
            if exporter is not None:
                exporter.stop()


def test_g2_2_stride_not_due_emits_one_score_then_missing_reason() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=256)
    pump = _pump(lanes, identity=None, fall_transition=0.9)
    pump._decision.deciders[0].classifier = FallWindowClassifier(_StubFallModel())
    scoring_calls = FALL_WINDOW_FRAMES
    _drive_pump(pump, scoring_calls + 1)
    drained = lanes.drain_for("cam-1", "boot-1", limit=256)
    assert drained is not None
    scores = [row for row in drained.records if row.record_kind == "model.score"]
    decisions = [row for row in drained.records if row.record_kind == "policy.decision"]
    score_seqs = [row.frame_seq for row in scores]
    n_seq = scoring_calls - 1
    n1_seq = scoring_calls
    assert score_seqs == [n_seq]
    assert n1_seq not in score_seqs
    n1 = [row for row in decisions if row.frame_seq == n1_seq]
    assert n1
    for row in n1:
        missing = row.payload["missing_values"]
        assert "classifier-stride-not-due" in missing.values()
    assert scoring_calls % FALL_STRIDE_FRAMES == 0


def test_g2_3_try_emit_swallows_sink_raise_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    control_events: list[object] = []
    boom_events: list[object] = []
    control = _pump(None, identity=None, emitted=control_events, fall_transition=0.9)
    boom = _pump(_BoomSink(), identity=None, emitted=boom_events, fall_transition=0.9)  # type: ignore[arg-type]
    with caplog.at_level("WARNING", logger="worker.pipeline.diagnostics.record_builder"):
        boom._process(_metadata(child=boom._child))
    control._process(_metadata(child=control._child))
    messages = [item.getMessage() for item in caplog.records]
    logged = any("camera_id=cam-1" in message and "record_kind=" in message for message in messages)
    assert logged, messages
    assert any("model.score" in message or "policy.decision" in message for message in messages)
    assert boom._decision.last_trace_snapshots == control._decision.last_trace_snapshots
    assert boom_events == control_events


def test_g2_4_record_invalid_gap_keeps_sequence_without_orphan() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=8)
    assert lanes.try_emit(_lane_record()) is True
    assert lanes.try_emit(_invalid_sequenced_record()) is False
    first = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert first is not None
    assert [row.producer_sequence for row in first.records] == [0]
    assert len(first.gaps) == 1
    gap = first.gaps[0]
    assert gap.cause == RECORD_INVALID_CAUSE
    assert gap.from_sequence == 1
    assert gap.to_sequence == 1
    assert gap.record_count == 1

    assert lanes.try_emit(_lane_record()) is True
    second = lanes.drain_for("cam-1", "boot-1", limit=8)
    assert second is not None
    assert [row.producer_sequence for row in second.records] == [2]
    assert all(item.from_sequence != 1 for item in second.gaps)


def test_g2_5_unwrap_decider_nested_plain_and_undecorated() -> None:
    inner = _Plain()
    nested = _Nested(_Nested(inner))
    assert unwrap_decider(nested) is inner
    assert unwrap_decider(inner) is inner
    bare = object()
    assert unwrap_decider(bare) is bare


def test_g2_6_backend_canonical_json_is_shared_identity() -> None:
    assert backend_records.canonical_json is shared_execution_records.canonical_json
    payload = {"b": 2, "a": 1}
    assert backend_records.canonical_json(payload) == shared_execution_records.canonical_json(
        payload
    )
