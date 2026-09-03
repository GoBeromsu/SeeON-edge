"""``CompositeExtractor``'s optional watchdog guard (issue #46).

``CompositeExtractor.process`` wraps each scheduled module's forward pass
(``NamedExtractor.extract``) in ``watchdog.guard(...)`` when a watchdog is
injected at construction. This is the same fault boundary a direct CUDA
fault drives -- ``check()`` finding an overdue in-flight entry calls
``FaultHandler.handle()`` -- so a forward pass that never returns is caught
by the *same* mechanism instead of hanging the process forever.

Without an injected watchdog (``watchdog=None``, the default), ``process``
calls ``extract`` directly: existing compositions and unit tests that never
pass ``watchdog=`` keep behaving exactly as before this change.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from worker.pipeline.analytics import NamedExtractor
from worker.pipeline.analytics.composite import CompositeExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.faults import FaultHandler
from worker.runtime.watchdog import InferenceWatchdog, InFlightInference
from worker.types import FramePacket


class _RecordingFaultHandler:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def handle(self, error: object, record: object) -> None:
        self.calls.append((error, record))


def _packet(frame_index: int) -> FramePacket:
    image = np.full((24, 32, 3), frame_index, dtype=np.uint8)
    return FramePacket(
        camera_id="camera-a",
        frame=Frame(index=frame_index, time_sec=float(frame_index), image=image),
        pts=float(frame_index),
        seq=frame_index,
        width=32,
        height=24,
        decode_time_ms=0.5,
    )


def _pose_result() -> RunnerResult:
    keypoints = tuple((index, index + 1, 0.8) for index in range(17))
    flattened = tuple(value for point in keypoints for value in point)
    return pose_result((flattened,), ((1, 2, 11, 22, 0.8),))


def test_process_without_a_watchdog_calls_extract_directly() -> None:
    """The default (``watchdog=None``) composition is unchanged."""
    calls: list[int] = []

    def call(_image: Image) -> RunnerResult:
        calls.append(1)
        return _pose_result()

    extractor = NamedExtractor(module_name="pose", runner=object(), _call=call, _clock=lambda: 0.0)
    composite = CompositeExtractor(
        extractors=(extractor,),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-a"),
    )

    result = composite.process(_packet(0))

    assert calls == [1]
    assert tuple(item.module_name for item in result.module_results) == ("pose",)


def test_process_guards_the_forward_pass_and_check_drives_the_fault_handler() -> None:
    """A forward pass that overruns its deadline trips the shared watchdog.

    The clock is a controllable stub (not real time): the scripted runner
    body advances it past the deadline and calls ``watchdog.check()`` itself,
    which is exactly what the watchdog's monitor thread does on a real
    schedule -- this makes the trip deterministic without any real sleeping.
    """
    handler = _RecordingFaultHandler()
    clock_box = [0.0]

    def clock() -> float:
        return clock_box[0]

    watchdog = InferenceWatchdog(
        cast(FaultHandler, handler), profile="cpu", deadline_sec=0.01, clock=clock
    )
    observed_in_flight: list[tuple[InFlightInference, ...]] = []

    def slow_call(_image: Image) -> RunnerResult:
        # The guard must have registered this forward pass as in-flight
        # before the call runs, and it must still be in-flight while the
        # call is executing.
        observed_in_flight.append(watchdog.in_flight())
        clock_box[0] = 10.0  # jump well past the 0.01s deadline
        tripped = watchdog.check()
        assert tripped is not None
        return _pose_result()

    extractor = NamedExtractor(
        module_name="pose", runner=object(), _call=slow_call, _clock=lambda: 0.0
    )
    composite = CompositeExtractor(
        extractors=(extractor,),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-a"),
        watchdog=watchdog,
    )

    result = composite.process(_packet(0))

    assert len(observed_in_flight) == 1
    (entry,) = observed_in_flight[0]
    assert entry.camera_id == "camera-a"
    assert entry.task == "pose"
    assert entry.frame_index == 0
    assert len(handler.calls) == 1
    # The guard's ``finally`` clears the token even though the pass "overran".
    assert watchdog.in_flight() == ()
    assert tuple(item.module_name for item in result.module_results) == ("pose",)


def test_process_completes_the_guard_token_even_when_extract_raises() -> None:
    handler = _RecordingFaultHandler()
    watchdog = InferenceWatchdog(cast(FaultHandler, handler), profile="cpu", deadline_sec=30.0)

    def raising_call(_image: Image) -> RunnerResult:
        raise ValueError("boom")

    extractor = NamedExtractor(
        module_name="pose", runner=object(), _call=raising_call, _clock=lambda: 0.0
    )
    composite = CompositeExtractor(
        extractors=(extractor,),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-a"),
        watchdog=watchdog,
    )

    try:
        composite.process(_packet(0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected the fake runner's ValueError to propagate")

    assert watchdog.in_flight() == ()
