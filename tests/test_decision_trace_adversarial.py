from __future__ import annotations

from datetime import UTC, datetime

import pytest

import worker.runtime.worker as worker_module
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.detection_window import DetectionWindow
from worker.types import (
    BusinessEvent,
    DecisionInput,
    DecisionTraceSnapshot,
)
from worker.types.trace import canonical_trace_number, decision_trace_id


def _result(
    *,
    source_time: float | None = 1.0,
) -> DecisionInput:
    person = BoundingBox(0, 0, 2, 3, 0.9)
    bed = BoundingBox(0, 0, 4, 4, 0.8)
    observation = FrameObservation(
        detections=((person,), ()),
        regions=((bed,), ()),
        track_ids=(5,),
    )
    return DecisionInput(
        observation=observation,
        frame_width=4,
        frame_height=4,
        live_track_ids=(5,),
        time_sec=source_time,
        frame_index=7,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )


def _snapshot(
    *,
    reason: str = "below-threshold",
    previous_state: str = "clear",
    current_state: str = "clear",
    triggered: bool = False,
    values: dict[str, int | float] | None = None,
    missing_values: dict[str, str] | None = None,
) -> DecisionTraceSnapshot:
    return DecisionTraceSnapshot(
        reason=reason,
        previous_state=previous_state,
        current_state=current_state,
        triggered=triggered,
        track_id=5,
        bed_id=None,
        values={} if values is None else values,
        missing_values={} if missing_values is None else missing_values,
    )


class _SnapshotDecider:
    def __init__(self) -> None:
        self.calls = 0
        self.last_trace_snapshots = (
            _snapshot(
                reason="fall-onset",
                current_state="fall",
                triggered=True,
                values={
                    "fall_probability": 0.9,
                    "operating_threshold": 0.7,
                    "window_frames": 1,
                },
            ),
        )

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        self.calls += 1
        return ()


def test_closed_window_emits_current_not_evaluated_trace_instead_of_stale_trigger() -> None:
    now = [datetime(2026, 1, 1, 23, 0, tzinfo=UTC)]
    inner = _SnapshotDecider()
    gated = worker_module._WindowGatedDecider(  # noqa: SLF001
        inner,
        DetectionWindow(start="21:00", end="06:00", tz="UTC"),
        clock=lambda: now[0],
    )
    decision_input = _result()

    assert gated.update(decision_input) == ()
    assert gated.last_trace_snapshots[0].triggered
    now[0] = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    assert gated.update(decision_input) == ()
    assert inner.calls == 1
    current = gated.last_trace_snapshots
    assert current == (
        DecisionTraceSnapshot(
            reason="outside-detection-window",
            previous_state="not-evaluated",
            current_state="not-evaluated",
            triggered=False,
            track_id=None,
            bed_id=None,
            missing_values={"decision_state": "outside-detection-window"},
        ),
    )
    assert not current[0].triggered


def test_canonical_trace_numbers_preserve_int_float_types_and_normalize_negative_zero() -> None:
    assert canonical_trace_number(1) == 1
    assert type(canonical_trace_number(1)) is int
    assert canonical_trace_number(1.0) == 1.0
    assert type(canonical_trace_number(1.0)) is float
    assert canonical_trace_number(-0.0) == 0.0
    assert str(canonical_trace_number(-0.0)) == "0.0"
    assert canonical_trace_number(0.123456789) == 0.123457


def test_snapshot_values_are_canonical_before_content_identity() -> None:
    positive_zero = _snapshot(
        values={
            "fall_probability": 0.123456789,
            "operating_threshold": 0.7,
            "window_frames": 1,
            "grace_frames_before": 0.0,
        }
    )
    negative_zero = _snapshot(
        values={
            "fall_probability": 0.123456791,
            "operating_threshold": 0.7000000001,
            "window_frames": 1,
            "grace_frames_before": -0.0,
        }
    )

    first = decision_trace_id(
        positive_zero, module_qualified_id="fall.v1", effective_policy_id="c" * 64
    )
    second = decision_trace_id(
        negative_zero, module_qualified_id="fall.v1", effective_policy_id="c" * 64
    )

    assert positive_zero.values == negative_zero.values
    assert first == second


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_snapshot_rejects_every_non_finite_numeric_value(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _snapshot(values={"fall_probability": value})


def _unsafe_trace_texts() -> tuple[str, ...]:
    return (
        "password=" + "camera-secret",
        "https" + "://example.invalid/private",
        "/" + "private/trace.txt",
        "line\nfeed",
        "a" * 256,
    )


@pytest.mark.parametrize("unsafe", _unsafe_trace_texts())
@pytest.mark.parametrize("field", ["reason", "previous_state", "current_state"])
def test_snapshot_rejects_private_or_opaque_free_text_fields(field: str, unsafe: str) -> None:
    fields: dict[str, object] = {
        "reason": "below-threshold",
        "previous_state": "clear",
        "current_state": "clear",
        "triggered": False,
        "track_id": None,
        "bed_id": None,
    }
    fields[field] = unsafe

    with pytest.raises(ValueError, match="decision trace"):
        DecisionTraceSnapshot(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("unsafe", _unsafe_trace_texts())
def test_snapshot_rejects_private_value_names_and_missing_reasons(unsafe: str) -> None:
    with pytest.raises(ValueError, match="decision trace"):
        _snapshot(values={unsafe: 1})
    with pytest.raises(ValueError, match="decision trace"):
        _snapshot(missing_values={"fall_probability": unsafe})
