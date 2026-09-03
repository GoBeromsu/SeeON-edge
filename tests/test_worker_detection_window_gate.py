"""Common per-domain detection-window gate (issue #24).

``WorkerRuntime._build_decider`` wraps a domain's real ``Decider`` in
``_WindowGatedDecider`` whenever that domain has a resolved detection window
-- except "bed_exit", which keeps gating internally (``BedExitMonitor``
tracks per-frame containment/latch state regardless of the window and only
gates final event *emission*; wrapping it here too would freeze that internal
state while the window is closed). This file characterizes:

- ``_WindowGatedDecider`` itself: skips ``update()`` (and the wrapped
  decider's internal state) entirely outside its window, passes through
  unchanged inside it.
- "fall" (representative of any windowed, non-bed_exit domain): runs 24/7
  when unconfigured, gated once ``domains.detection_windows["fall"]`` is set.
- "bed_exit": never wrapped by this gate, regardless of a configured window.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import final

import numpy as np
from numpy.typing import NDArray

import worker.runtime.worker as worker_module
from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.domains.bed_exit import BedExitMonitor
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallV2Probabilities
from worker.runtime.config import CameraRuntimeConfig, WorkerConfig
from worker.runtime.worker import WorkerRuntime
from worker.types import BusinessEvent, DecisionInput


@final
class _RecordingDecider:
    def __init__(self, events: tuple[BusinessEvent, ...] = ()) -> None:
        self.calls = 0
        self._events = events

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        self.calls += 1
        return self._events


def _input() -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=1,
        frame_height=1,
        live_track_ids=(),
        time_sec=0.0,
        frame_index=0,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_window_gated_decider_skips_update_and_wrapped_state_outside_window() -> None:
    inner = _RecordingDecider()
    window = DetectionWindow(start="21:00", end="06:00", tz="UTC")
    gated = worker_module._WindowGatedDecider(  # noqa: SLF001
        inner, window, clock=lambda: datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    )

    assert gated.update(_input()) == ()
    assert inner.calls == 0


def test_window_gated_decider_passes_through_inside_window() -> None:
    expected = (
        BusinessEvent(
            domain="fall",
            event_type="fall",
            identity="0",
            camera_id="camera-1",
            facility_id="facility-1",
            time_sec=0.0,
            probability=1.0,
            person_id=0,
        ),
    )
    inner = _RecordingDecider(expected)
    window = DetectionWindow(start="21:00", end="06:00", tz="UTC")
    gated = worker_module._WindowGatedDecider(  # noqa: SLF001
        inner, window, clock=lambda: datetime(2026, 1, 1, 23, 0, tzinfo=UTC)
    )

    assert gated.update(_input()) == expected
    assert inner.calls == 1


@final
class _FakeFallModel:
    def __init__(self) -> None:
        self.operating_threshold = 0.5

    def predict(self, _features: NDArray[np.float32]) -> FallV2Probabilities:
        return FallV2Probabilities(0.0, 0.99, 0.1)


@final
class _UnusedServingClient:
    def create(self, task: str, **_options: object) -> object:
        raise AssertionError(f"serving client should not be used to build a decider ({task})")


def _config(detection_windows: dict[str, object] | None = None) -> WorkerConfig:
    domains: dict[str, object] = {"enabled": ["fall", "bed_exit"]}
    if detection_windows is not None:
        domains["detection_windows"] = detection_windows
    return WorkerConfig.model_validate(
        {
            "relay": {"url": "http://relay.test", "token": "relay-token"},
            "domains": domains,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://example.test/camera-1",
                }
            ],
        }
    )


def _runtime(config: WorkerConfig) -> WorkerRuntime:
    return WorkerRuntime(config, serving_client=_UnusedServingClient())


def _camera(runtime: WorkerRuntime) -> CameraRuntimeConfig:
    return runtime.config.cameras[0]


def test_fall_domain_is_ungated_24_7_when_no_window_configured() -> None:
    runtime = _runtime(_config())

    decider = runtime._build_decider("fall", _camera(runtime), _FakeFallModel())  # noqa: SLF001

    assert isinstance(decider, FallV2DomainDecider)
    assert isinstance(decider.policy, FallPolicyDeciderV2)


def test_fall_domain_is_gated_by_the_common_wrapper_once_a_window_is_configured() -> None:
    runtime = _runtime(
        _config(detection_windows={"fall": {"start": "21:00", "end": "06:00", "tz": "UTC"}})
    )

    decider = runtime._build_decider("fall", _camera(runtime), _FakeFallModel())  # noqa: SLF001

    assert isinstance(decider, worker_module._WindowGatedDecider)  # noqa: SLF001
    assert decider.window == DetectionWindow(start="21:00", end="06:00", tz="UTC")
    assert isinstance(decider.decider, FallV2DomainDecider)
    assert isinstance(decider.decider.policy, FallPolicyDeciderV2)


def test_bed_exit_is_never_wrapped_by_the_common_gate_even_with_a_window_configured() -> None:
    runtime = _runtime(
        _config(detection_windows={"bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"}})
    )

    decider = runtime._build_decider("bed_exit", _camera(runtime), _FakeFallModel())  # noqa: SLF001

    assert isinstance(decider, BedExitMonitor)
    assert not isinstance(decider, worker_module._WindowGatedDecider)  # noqa: SLF001
