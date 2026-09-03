from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.observation import BoundingBox
from contracts.runner import (
    Image,
    RunnerProtocol,
    RunnerResult,
    bed_result,
    person_result,
    pose_result,
)
from worker.pipeline.analytics import (
    CompositeExtractor,
    ExtractorNameConflictError,
    ExtractorSpec,
    NamedExtractor,
    provision_extractors,
)
from worker.pipeline.analytics.merge import merge_module_results
from worker.pipeline.bus import Scheduler
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.types import FramePacket, ModuleResult

ServingOption = str | int | float | bool | None


class _StepClock:
    def __init__(self, step: float = 0.002) -> None:
        self._value: float = 0.0
        self._step: float = step

    def __call__(self) -> float:
        value = self._value
        self._value += self._step
        return value


class _Runner:
    def __init__(self, result: RunnerResult) -> None:
        self.result: RunnerResult = result
        self.calls: int = 0
        self.images: list[Image] = []

    def run(self, image: Image) -> RunnerResult:
        self.calls += 1
        self.images.append(image)
        return self.result


class _ServingClient:
    def __init__(self, runners: Mapping[str, RunnerProtocol]) -> None:
        self._runners: Mapping[str, RunnerProtocol] = runners
        self.create_calls: list[tuple[str, dict[str, ServingOption]]] = []

    def create(self, task: str, **kwargs: ServingOption) -> RunnerProtocol:
        self.create_calls.append((task, dict(kwargs)))
        return self._runners[task]


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


def _pose_output(box: tuple[int, int, int, int, float]) -> RunnerResult:
    keypoints = tuple((index, index + 1, 0.8) for index in range(17))
    flattened = tuple(value for point in keypoints for value in point)
    return pose_result((flattened,), (box,))


def _composite(
    extractors: Sequence[NamedExtractor],
    scheduler: Scheduler,
    *,
    camera_id: str = "camera-a",
) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=extractors,
        scheduler=scheduler,
        tracker=GreedyIouTracker(),
        scene_state=SceneState(camera_id),
    )


def test_composite_discards_live_bed_segmentation_and_preserves_pose_keypoints() -> None:
    pose_box = (1, 2, 11, 22, 0.55)
    person_box = (100, 110, 200, 220, 0.95)
    polygon = ((300, 310), (400, 310), (400, 410), (300, 410))
    runners = {
        "pose": _Runner(_pose_output(pose_box)),
        "person": _Runner(person_result((person_box,))),
        "bed": _Runner(bed_result(((300, 310, 400, 410, 0.9, polygon),))),
    }
    client = _ServingClient(runners)
    extractors = provision_extractors(
        client,
        (
            ExtractorSpec("pose", "pose"),
            ExtractorSpec("person", "person"),
            ExtractorSpec("bed", "bed"),
        ),
        clock=_StepClock(),
    )
    composite = _composite(extractors, Scheduler({"pose": 1, "person": 1, "bed": 30}))

    result = composite.process(_packet(0))

    assert tuple(item.module_name for item in result.module_results) == (
        "pose",
        "person",
        "bed",
    )
    assert all(abs(item.elapsed_ms - 2.0) < 1e-9 for item in result.module_results)
    assert result.observation.boxes == (BoundingBox(*person_box),)
    assert result.observation.keypoints == (tuple((index, index + 1, 0.8) for index in range(17)),)
    assert result.observation.track_ids == (0,)
    # Live segmentation is not runtime bed truth; only persisted polygons reach decisions.
    assert result.observation.bed_boxes == ()
    assert result.observation is result.decision_input.observation
    assert result.observation is composite.scene_state.latest_observation
    assert composite.scene_state.track_ids == (0,)
    assert result.decision_input.bed_region.source == "empty"


def test_box_source_person_merge_winner_is_person_not_pose() -> None:
    """Issue #108 regression (gap found reviewing PR #106 / issue #44).

    With box_source="person", ``WorkerRuntime._build_camera`` builds the
    schedule's ``intervals`` dict from the domain-required extractors first
    (pose, then bed) and appends "person" last. ``Scheduler.tasks_for_frame``
    preserves that insertion order, and ``merge_module_results`` is
    unconditional last-writer-wins on ``raw_boxes`` -- so "person always
    beats pose" is a structural consequence of construction + iteration
    order, never asserted at the merged-result level. Pin the outcome here:
    if ``Scheduler`` ever reorders tasks (e.g. sorts names -- "person" sorts
    before "pose" alphabetically), this must fail, because issue #44's fix
    would silently regress for every box_source="person" camera.
    """
    pose_box = (1, 2, 11, 22, 0.55)
    person_box = (100, 110, 200, 220, 0.95)
    runners = {
        "pose": _Runner(_pose_output(pose_box)),
        "bed": _Runner(bed_result(())),
        "person": _Runner(person_result((person_box,))),
    }
    client = _ServingClient(runners)
    extractors = provision_extractors(
        client,
        (
            ExtractorSpec("pose", "pose"),
            ExtractorSpec("bed", "bed"),
            ExtractorSpec("person", "person"),
        ),
    )
    # Mirrors WorkerRuntime._build_camera's real construction shape for
    # box_source="person": domain-required extractors built first (pose,
    # then bed), "person" inserted last -- not hand-picked for the test.
    intervals: dict[str, int] = {"pose": 1, "bed": 30}
    intervals["person"] = 1
    composite = _composite(extractors, Scheduler(intervals))

    result = composite.process(_packet(0))

    assert tuple(item.module_name for item in result.module_results) == (
        "pose",
        "bed",
        "person",
    )
    assert result.observation.boxes == (BoundingBox(*person_box),)
    assert result.observation.boxes != (BoundingBox(*pose_box),)


def test_merge_module_results_pose_only_keeps_pose_raw_boxes() -> None:
    """Issue #44 regression: when person is never scheduled (box_source is
    "pose", the default), pose's raw_boxes survive intact -- structurally,
    because there is no person merge step to run, not merely because of
    scheduling order.
    """
    pose_box = (1, 2, 11, 22, 0.55)
    result = ModuleResult(
        module_name="pose",
        result=_pose_output(pose_box),
        elapsed_ms=1.0,
    )

    merged = merge_module_results((result,))

    assert merged.raw_boxes == (pose_box,)


def test_composite_runs_only_scheduled_extractors_and_one_pose_call_per_frame() -> None:
    pose = _Runner(_pose_output((1, 2, 11, 22, 0.8)))
    person = _Runner(person_result(((1, 2, 11, 22, 0.9),)))
    bed = _Runner(bed_result(()))
    client = _ServingClient({"pose": pose, "person": person, "bed": bed})
    extractors = provision_extractors(
        client,
        tuple(ExtractorSpec(name, name) for name in ("pose", "person", "bed")),
    )
    composite = _composite(extractors, Scheduler({"pose": 1, "person": 2, "bed": 30}))

    first = composite.process(_packet(1))
    second = composite.process(_packet(2))

    assert tuple(item.module_name for item in first.module_results) == ("pose",)
    assert tuple(item.module_name for item in second.module_results) == ("pose", "person")
    assert (pose.calls, person.calls, bed.calls) == (2, 1, 0)


def test_live_bed_segmentation_is_not_cached_for_unscheduled_frames() -> None:
    bed = BoundingBox(10, 11, 20, 21, 0.9, ((10, 11), (20, 21)))
    polygon = ((10, 11), (20, 21))
    pose = _Runner(_pose_output((1, 2, 11, 22, 0.8)))
    bed_runner = _Runner(bed_result(((bed.x1, bed.y1, bed.x2, bed.y2, bed.confidence, polygon),)))
    client = _ServingClient({"pose": pose, "bed": bed_runner})
    extractors = provision_extractors(
        client,
        (ExtractorSpec("pose", "pose"), ExtractorSpec("bed", "bed")),
    )
    composite = _composite(extractors, Scheduler({"pose": 1, "bed": 30}))

    fresh = composite.process(_packet(0))
    cached = composite.process(_packet(1))

    # The deleted live-segmentation cache must not leak a bed into later frames.
    assert fresh.observation.bed_boxes == ()
    assert fresh.decision_input.bed_region.source == "empty"
    assert cached.observation.bed_boxes == ()
    assert cached.decision_input.bed_region.source == "empty"
    assert bed_runner.calls == 1


def test_fake_mediapipe_like_extractor_changes_only_module_results() -> None:
    pose = _Runner(_pose_output((1, 2, 11, 22, 0.8)))
    mediapipe = _Runner(_pose_output((200, 210, 300, 310, 0.99)))
    client = _ServingClient({"pose": pose, "mediapipe": mediapipe})
    extractors = provision_extractors(
        client,
        (
            ExtractorSpec("pose", "pose"),
            ExtractorSpec("mediapipe", "mediapipe"),
        ),
    )
    without_fake = _composite(extractors[:1], Scheduler({"pose": 1}), camera_id="base")
    with_fake = _composite(
        extractors,
        Scheduler({"pose": 1, "mediapipe": 1}),
        camera_id="extended",
    )

    base_result = without_fake.process(_packet(0))
    extended_result = with_fake.process(_packet(0))

    assert base_result.observation == extended_result.observation
    assert base_result.decision_input == extended_result.decision_input
    assert tuple(item.module_name for item in base_result.module_results) == ("pose",)
    assert tuple(item.module_name for item in extended_result.module_results) == (
        "pose",
        "mediapipe",
    )


def test_duplicate_module_names_fail_before_serving_client_creation() -> None:
    client = _ServingClient({"pose": _Runner(_pose_output((1, 2, 3, 4, 0.8)))})

    with pytest.raises(ExtractorNameConflictError, match="pose"):
        _ = provision_extractors(
            client,
            (ExtractorSpec("pose", "pose"), ExtractorSpec("pose", "pose")),
        )

    assert client.create_calls == []
