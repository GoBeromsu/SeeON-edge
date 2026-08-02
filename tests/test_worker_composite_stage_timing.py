"""``CompositeExtractor``'s optional stage-timing recorder (issue #45).

``CompositeExtractor.process`` hands each scheduled module's already-computed
``ModuleResult.elapsed_ms`` to an injected ``stage_timing_recorder`` (a
structural match for ``WorkerDiagnostics.record_stage_timing``), when one is
present. Without one (``stage_timing_recorder=None``, the default) nothing
changes: existing compositions and unit tests that never pass it keep
behaving exactly as before.
"""

from __future__ import annotations

import numpy as np

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from worker.pipeline.analytics import NamedExtractor
from worker.pipeline.analytics.composite import CompositeExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.telemetry.runtime_diagnostics import WorkerDiagnostics
from worker.types import FramePacket


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


def _pose_extractor(call: object, clock: object = lambda: 0.0) -> NamedExtractor:
    return NamedExtractor(module_name="pose", runner=object(), _call=call, _clock=clock)


def test_process_without_a_recorder_does_not_touch_diagnostics() -> None:
    """The default (``stage_timing_recorder=None``) composition is unchanged."""

    def call(_image: Image) -> RunnerResult:
        return _pose_result()

    composite = CompositeExtractor(
        extractors=(_pose_extractor(call),),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-a"),
    )

    result = composite.process(_packet(0))

    assert tuple(item.module_name for item in result.module_results) == ("pose",)


def test_process_records_each_scheduled_module_elapsed_time() -> None:
    """A real ``WorkerDiagnostics`` observes the module's elapsed time.

    The clock is a controllable stub advancing by a fixed step per call, so
    the recorded elapsed time is deterministic instead of depending on real
    wall-clock scheduling noise.
    """
    ticks = iter((0.0, 0.25))

    def clock() -> float:
        return next(ticks)

    def call(_image: Image) -> RunnerResult:
        return _pose_result()

    diagnostics = WorkerDiagnostics()
    composite = CompositeExtractor(
        extractors=(_pose_extractor(call, clock),),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-a"),
        stage_timing_recorder=diagnostics,
    )

    _ = composite.process(_packet(0))

    (camera,) = diagnostics.snapshot().cameras
    assert camera.camera_id == "camera-a"
    (stage,) = camera.stage_timings
    assert stage.stage == "pose"
    assert stage.samples == 1
    assert stage.last_sec == 0.25


def test_process_records_a_stage_timing_per_camera_scene_state() -> None:
    """Stage timings are keyed by the composite's own camera, not shared."""

    def call(_image: Image) -> RunnerResult:
        return _pose_result()

    diagnostics = WorkerDiagnostics()
    composite = CompositeExtractor(
        extractors=(_pose_extractor(call),),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-b"),
        stage_timing_recorder=diagnostics,
    )

    _ = composite.process(_packet(0))

    (camera,) = diagnostics.snapshot().cameras
    assert camera.camera_id == "camera-b"
