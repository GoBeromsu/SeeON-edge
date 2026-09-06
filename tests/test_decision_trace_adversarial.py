from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

import worker.runtime.worker as worker_module
from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.runner import pose_result
from worker.domains.detection_window import DetectionWindow
from worker.pipeline.trace import TraceCapture, TraceIdentity
from worker.types import (
    BusinessEvent,
    DecisionInput,
    DecisionTraceSnapshot,
    FramePacket,
    ModuleResult,
)
from worker.types.trace import canonical_trace_number

_RUNTIME_SHA256 = "a" * 64
_COMPONENT_SHA256 = "b" * 64
_POLICY_SHA256 = "c" * 64


@dataclass(frozen=True)
class _TraceResult:
    module_results: tuple[object, ...]
    observation: FrameObservation
    decision_input: DecisionInput


def _packet(*, camera_id: str = "camera-a", pts: float | None = 1.0) -> FramePacket:
    return FramePacket(
        camera_id=camera_id,
        frame=Frame(7, 1.0, np.zeros((4, 4, 3), dtype=np.uint8)),
        pts=pts,
        seq=11,
        width=4,
        height=4,
        decode_time_ms=0.0,
        worker_boot_id="boot-a",
        stream_epoch=3,
    )


def _result(
    *,
    person_confidence: float = 0.9,
    bed_confidence: float = 0.8,
    source_time: float | None = 1.0,
    observed_components: tuple[str, ...] = (),
) -> _TraceResult:
    person = BoundingBox(0, 0, 2, 3, person_confidence)
    bed = BoundingBox(0, 0, 4, 4, bed_confidence)
    observation = FrameObservation(
        detections=((person,), ()),
        regions=((bed,), ()),
        track_ids=(5,),
    )
    decision_input = DecisionInput(
        observation=observation,
        frame_width=4,
        frame_height=4,
        live_track_ids=(5,),
        time_sec=source_time,
        frame_index=7,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )
    module_results = tuple(
        ModuleResult(name, pose_result((), ()), 0.0, output_adapter=name)
        for name in observed_components
    )
    return _TraceResult(module_results, observation, decision_input)


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


def _capture(
    provider: object,
    *,
    components: tuple[str, ...] = ("fall-classifier",),
) -> TraceCapture:
    qualified = tuple(f"{component}.sha256.{_COMPONENT_SHA256}" for component in components)
    return TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="fall.v1",
                component_qualified_ids=qualified,
                policy_qualified_id="fall.policy.v1",
                effective_policy_id=_POLICY_SHA256,
                runtime_manifest_sha256=_RUNTIME_SHA256,
                snapshot_provider=lambda: provider,
            ),
        )
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
    decision_input = _result().decision_input

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
    frame = _capture(current).build(_packet(), _result(), ())
    assert frame.decisions[0].snapshot is current[0]
    assert not frame.decisions[0].snapshot.triggered


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

    first = _capture(positive_zero).build(_packet(), _result(), ()).decisions[0]
    second = _capture(negative_zero).build(_packet(), _result(), ()).decisions[0]

    assert first.snapshot.values == second.snapshot.values
    assert first.trace_id == second.trace_id


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_snapshot_rejects_every_non_finite_numeric_value(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _snapshot(values={"fall_probability": value})


@pytest.mark.parametrize("field", ["pts", "source_time", "person", "bed"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_analysis_rejects_non_finite_values_and_confidences(field: str, value: float) -> None:
    packet = _packet(pts=value if field == "pts" else 1.0)
    result = _result(
        source_time=value if field == "source_time" else 1.0,
        person_confidence=value if field == "person" else 0.9,
        bed_confidence=value if field == "bed" else 0.8,
    )

    with pytest.raises(ValueError, match="finite"):
        _capture(_snapshot()).build(packet, result, ())


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


def test_analysis_preserves_camera_identity_byte_exactly() -> None:
    camera_id = "camera-e\u0301-\ud55c\uae00"

    frame = _capture(_snapshot()).build(_packet(camera_id=camera_id), _result(), ())

    assert frame.analysis.frame_key[1] == camera_id
    assert frame.analysis.frame_key[1].encode("utf-8") == camera_id.encode("utf-8")


def test_component_states_distinguish_observed_and_executed_components() -> None:
    frame = _capture(
        _snapshot(),
        components=("pose", "fall-classifier", "fall-latch"),
    ).build(_packet(), _result(observed_components=("pose",)), ())

    assert tuple(component.observation_state for component in frame.analysis.components) == (
        "observed",
        "executed",
        "executed",
    )


def test_not_evaluated_decision_marks_camera_local_components_not_applicable() -> None:
    frame = _capture(
        DecisionTraceSnapshot(
            reason="outside-detection-window",
            previous_state="not-evaluated",
            current_state="not-evaluated",
            triggered=False,
            track_id=None,
            bed_id=None,
            missing_values={"decision_state": "outside-detection-window"},
        ),
        components=("pose", "fall-classifier", "fall-latch"),
    ).build(_packet(), _result(observed_components=("pose",)), ())

    assert tuple(component.observation_state for component in frame.analysis.components) == (
        "observed",
        "not-applicable",
        "not-applicable",
    )
