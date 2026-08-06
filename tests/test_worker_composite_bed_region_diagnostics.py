"""``CompositeExtractor``'s optional bed-region recorder (issue #207).

Mirrors ``test_worker_composite_stage_timing.py``'s shape: `process()` hands
`scene_state`'s just-updated bed-region freshness/counters to an injected
`bed_region_recorder` (a structural match for
`WorkerDiagnostics.record_bed_region`), when one is present. Without one
(`bed_region_recorder=None`, the default) nothing changes.
"""

from __future__ import annotations

import numpy as np

from contracts.frame import Frame
from contracts.observation import BedRegionCacheState
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


def _pose_extractor() -> NamedExtractor:
    def call(_image: Image) -> RunnerResult:
        return _pose_result()

    return NamedExtractor(module_name="pose", runner=object(), _call=call, _clock=lambda: 0.0)


def test_process_without_a_recorder_does_not_touch_diagnostics() -> None:
    """The default (``bed_region_recorder=None``) composition is unchanged."""
    composite = CompositeExtractor(
        extractors=(_pose_extractor(),),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-a"),
    )

    result = composite.process(_packet(0))

    assert tuple(item.module_name for item in result.module_results) == ("pose",)


def test_process_records_fresh_from_the_persisted_polygon_short_circuit() -> None:
    """A real ``WorkerDiagnostics`` observes the persisted-polygon short-circuit.

    A camera with a persisted bed polygon always resolves FRESH from a
    zero-cost short-circuit (`scene_state.py`'s `resolve_bed_regions`) without
    ever touching `scene_state.bed_region_counters` -- that cache-cycle
    machinery is never engaged on this path. The recorder must read this
    frame's actually-resolved state (`decision_input.bed_region.source`), not
    `scene_state.bed_region_freshness` directly, or every persisted-polygon
    camera (~10/13 of the live fleet, #207/#208) would misreport as
    permanently EMPTY.
    """
    diagnostics = WorkerDiagnostics()
    scene = SceneState("camera-a", persisted_bed_regions=((1.0, 2.0, 3.0, 4.0),))
    composite = CompositeExtractor(
        extractors=(_pose_extractor(),),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=scene,
        bed_region_recorder=diagnostics,
    )

    _ = composite.process(_packet(0))

    (camera,) = diagnostics.snapshot().cameras
    assert camera.camera_id == "camera-a"
    assert camera.bed_region is not None
    assert camera.bed_region.freshness == BedRegionCacheState.FRESH
    # Counters stay all-zero: the persisted-polygon short-circuit never
    # engages the live-detection cache-cycle machinery they track.
    assert camera.bed_region.counters["fresh"] == 0


def test_process_records_bed_region_per_camera_scene_state() -> None:
    """Bed-region diagnostics are keyed by the composite's own camera, not shared."""
    diagnostics = WorkerDiagnostics()
    composite = CompositeExtractor(
        extractors=(_pose_extractor(),),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState("camera-b", persisted_bed_regions=((1.0, 2.0, 3.0, 4.0),)),
        bed_region_recorder=diagnostics,
    )

    _ = composite.process(_packet(0))

    (camera,) = diagnostics.snapshot().cameras
    assert camera.camera_id == "camera-b"
