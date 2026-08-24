"""Characterization of today's pose-frame, DecisionInput, tracker, and bed path.

These pins are the DeepStream C1 baseline. They must stay byte/value equivalent
after PerceptionFrameV1 lands. There is no ``PoseFrame`` symbol today: the
current pose-frame envelope is ``PoseRunnerResult``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import get_type_hints

import numpy as np
import pytest

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.runner import (
    PoseRunnerResult,
    bed_result,
    person_result,
    pose_result,
)
from worker.adapters.model.yolo_pose import (
    COCO_KEYPOINT_COUNT,
    POSE_MODEL_CONFIDENCE,
    POSE_PREPROCESSING_IDENTITY,
    YoloPoseRunner,
)
from worker.domains.registry import DETECTION_MODULE_DEFINITIONS
from worker.pipeline.analytics.merge import authoritative_boxes, merge_module_results
from worker.pipeline.perception.features.geometry import greedy_match
from worker.pipeline.perception.scene_state import SceneState
from worker.pipeline.perception.tracker import GreedyIouTracker
from worker.types import CURRENT_TEMPORAL_PROFILE, DecisionInput, ModuleResult
from worker.types.bed_pose_features import FrameBedPoseFeatures


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def _coco17(*, origin: int = 0, confidence: float = 0.8) -> tuple[tuple[int, int, float], ...]:
    return tuple((origin + index, origin + index + 1, confidence) for index in range(17))


class _Tensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(
        self,
        xyxy: tuple[tuple[float, ...], ...],
        confidence: tuple[float, ...],
    ) -> None:
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(confidence)

    def __len__(self) -> int:
        return len(self.conf.numpy())


class _Keypoints:
    def __init__(
        self,
        xy: tuple[tuple[tuple[float, float], ...], ...],
        confidence: tuple[tuple[float, ...], ...],
    ) -> None:
        self.xy = _Tensor(xy)
        self.conf = _Tensor(confidence)


class _Result:
    def __init__(self, *, boxes: _Boxes | None, keypoints: _Keypoints | None) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


class _Model:
    def __init__(self, result: _Result) -> None:
        self.result = result
        self.calls: list[tuple[tuple[int, ...], float]] = []

    def predict(
        self,
        *,
        source: np.ndarray,
        conf: float,
        verbose: bool,
        device: str,
    ) -> tuple[_Result, ...]:
        del verbose, device
        self.calls.append((source.shape, conf))
        return (self.result,)


def test_current_pose_frame_is_pose_runner_result_not_a_poseframe_symbol() -> None:
    result = pose_result(
        poses=(_coco17(),),
        boxes=((10.9, 20.1, 110.7, 220.3, 0.83),),
    )

    assert type(result).__name__ == "PoseRunnerResult"
    assert result.kind == "pose"
    assert tuple(item.name for item in fields(PoseRunnerResult)) == ("kind", "poses", "boxes")
    assert result.poses == (_coco17(),)
    assert result.boxes == ((10.9, 20.1, 110.7, 220.3, 0.83),)
    with pytest.raises(FrozenInstanceError):
        result.kind = "detection"  # type: ignore[misc]
    assert not hasattr(result, "track_ids")
    assert not hasattr(result, "bed_boxes")
    import contracts.runner as runner_mod
    import worker.types as worker_types

    assert not hasattr(runner_mod, "PoseFrame")
    assert not hasattr(worker_types, "PoseFrame")


def test_yolo_pose_frame_truncates_source_order_and_keeps_conf_threshold() -> None:
    first_points = tuple((float(index) + 0.8, float(index) + 4.2) for index in range(17))
    second_points = tuple((float(index) + 20.6, float(index) + 30.9) for index in range(17))
    first_conf = tuple(0.11 + index / 100 for index in range(17))
    second_conf = tuple(0.51 + index / 100 for index in range(17))
    model = _Model(
        _Result(
            boxes=_Boxes(
                ((10.8, 20.2, 110.6, 220.9), (3.1, 4.9, 13.2, 24.8)),
                (0.91, 0.44),
            ),
            keypoints=_Keypoints((first_points, second_points), (first_conf, second_conf)),
        )
    )
    runner = YoloPoseRunner(confidence=POSE_MODEL_CONFIDENCE, device="cpu", model=model)

    output = runner.run(np.zeros((6, 8, 3), dtype=np.uint8))

    assert POSE_MODEL_CONFIDENCE == 0.05
    assert COCO_KEYPOINT_COUNT == 17
    assert POSE_PREPROCESSING_IDENTITY == "rgb24-to-coco17.v1"
    assert model.calls == [((6, 8, 3), 0.05)]
    assert output.kind == "pose"
    assert len(output.poses) == 2
    assert len(output.poses[0]) == 17
    assert output.poses[0][0] == (0, 4, 0.11)
    assert output.poses[1][0] == (20, 30, 0.51)
    assert output.boxes == ((10, 20, 110, 220, 0.91), (3, 4, 13, 24, 0.44))


def test_empty_pose_observation_is_an_empty_tuple_not_a_missing_channel() -> None:
    model = _Model(_Result(boxes=_Boxes((), ()), keypoints=None))
    runner = YoloPoseRunner(model=model)

    output = runner.run(np.zeros((2, 2, 3), dtype=np.uint8))

    assert output.kind == "pose"
    assert output.poses == ()
    assert output.boxes == ()


def test_decision_input_keeps_original_seven_fields_and_image_free_boundary() -> None:
    observation = FrameObservation(
        detections=((_box(1, 2, 3, 4),), ()),
        poses=(_coco17(),),
        track_ids=(7,),
    )
    decision_input = DecisionInput(
        observation=observation,
        frame_width=640,
        frame_height=360,
        live_track_ids=(7,),
        time_sec=1.25,
        frame_index=4,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.EMPTY),
    )

    names = tuple(item.name for item in fields(DecisionInput))
    assert names[:7] == (
        "observation",
        "frame_width",
        "frame_height",
        "live_track_ids",
        "time_sec",
        "frame_index",
        "bed_region",
    )
    assert names == (
        "observation",
        "frame_width",
        "frame_height",
        "live_track_ids",
        "time_sec",
        "frame_index",
        "bed_region",
        "bed_pose_features",
    )
    hints = get_type_hints(DecisionInput)
    assert hints["observation"] is FrameObservation
    assert hints["bed_region"] is BedRegionDebugSnapshot
    assert hints["bed_pose_features"] is FrameBedPoseFeatures
    assert decision_input.observation.track_ids == (7,)
    assert decision_input.live_track_ids == (7,)
    assert not hasattr(decision_input, "frame")
    assert not hasattr(decision_input, "image")
    with pytest.raises(FrozenInstanceError):
        decision_input.frame_index = 99  # type: ignore[misc]


def test_decision_input_live_ids_are_sorted_while_observation_tracks_keep_box_order() -> None:
    """Composite hands box-order ids to observation and sorted live ids to decisions."""
    tracker = GreedyIouTracker(max_misses=5)
    box_a = _box(0, 0, 40, 40)
    box_b = _box(200, 0, 240, 40)
    assert tracker.observe((box_a, box_b)) == (0, 1)
    assert tracker.observe((box_b,)) == (1,)
    assert tracker.live_ids == frozenset({0, 1})
    assert tuple(sorted(tracker.live_ids)) == (0, 1)


def test_tracker_returns_ids_in_incoming_box_order_not_match_order() -> None:
    tracker = GreedyIouTracker()
    first = _box(0, 0, 50, 50)
    second = _box(200, 200, 250, 250)
    assert tracker.observe((first, second)) == (0, 1)

    moved_second = _box(205, 205, 255, 255)
    moved_first = _box(4, 4, 54, 54)
    assert tracker.observe((moved_second, moved_first)) == (1, 0)


def test_tracker_equal_iou_tie_keeps_lower_existing_track_index() -> None:
    tracker = GreedyIouTracker()
    left = _box(0, 0, 100, 100)
    right = _box(50, 0, 150, 100)
    assert tracker.observe((left, right)) == (0, 1)

    center = _box(25, 0, 125, 100)
    assert greedy_match((left, right), (center,), min_iou=0.3) == ((0, 0),)
    assert tracker.observe((center,)) == (0,)


def test_tracker_update_empty_coasts_while_observe_empty_counts_a_miss() -> None:
    box = _box(0, 0, 20, 20)
    coasting = GreedyIouTracker(max_misses=1)
    assert coasting.update((box,)) == (0,)
    assert coasting.update(()) == ()
    assert coasting.update(()) == ()
    assert coasting.live_ids == frozenset({0})
    assert coasting.update((box,)) == (0,)

    observing = GreedyIouTracker(max_misses=1)
    assert observing.observe((box,)) == (0,)
    assert observing.observe(()) == ()
    assert observing.observe(()) == ()
    assert observing.live_ids == frozenset()
    assert observing.observe((box,)) == (1,)


def test_authoritative_boxes_are_person_or_pose_never_bed_regions() -> None:
    pose_box = (1, 2, 11, 22, 0.55)
    person_box = (100, 110, 200, 220, 0.95)
    bed_box = (300, 310, 400, 410, 0.9, ((300, 310), (400, 310), (400, 410), (300, 410)))
    merged = merge_module_results(
        (
            ModuleResult("pose", pose_result((_coco17(),), (pose_box,)), 0.0, "pose"),
            ModuleResult("person", person_result((person_box,)), 0.0, "person"),
            ModuleResult("bed", bed_result((bed_box,)), 0.0, "bed"),
        )
    )

    boxes = authoritative_boxes(merged)
    assert boxes == (BoundingBox(100, 110, 200, 220, 0.95),)
    assert merged.bed_boxes == (
        BoundingBox(300, 310, 400, 410, 0.9, ((300, 310), (400, 310), (400, 410), (300, 410))),
    )
    tracker = GreedyIouTracker()
    assert tracker.observe(boxes) == (0,)
    assert tracker.observe(authoritative_boxes(merge_module_results(()))) == ()


def test_bed_only_merge_cannot_create_or_evict_a_person_track() -> None:
    tracker = GreedyIouTracker(max_misses=1)
    person = _box(10, 10, 40, 40)
    assert tracker.observe((person,)) == (0,)
    bed_only = merge_module_results(
        (ModuleResult("bed", bed_result(((1, 1, 9, 9, 1.0),)), 0.0, "bed"),)
    )
    assert authoritative_boxes(bed_only) == ()
    assert tracker.update(authoritative_boxes(bed_only)) == ()
    assert tracker.live_ids == frozenset({0})
    assert tracker.observe((person,)) == (0,)


def test_bed_region_schedule_is_independent_of_pose_and_person_tracks() -> None:
    fall = next(
        definition
        for definition in DETECTION_MODULE_DEFINITIONS
        if definition.module_id == "fall"
    )
    bed_exit = next(
        definition
        for definition in DETECTION_MODULE_DEFINITIONS
        if definition.module_id == "bed_exit"
    )
    assert CURRENT_TEMPORAL_PROFILE.ingest_fps == 15.0
    assert CURRENT_TEMPORAL_PROFILE.task_intervals() == {"pose": 1, "bed": 90}
    assert fall.required_observation_channels == frozenset(
        {"person_boxes", "poses", "track_ids"}
    )
    assert bed_exit.required_observation_channels == frozenset(
        {"person_boxes", "poses", "track_ids", "bed_regions"}
    )
    bed_rule = next(
        rule
        for definition in DETECTION_MODULE_DEFINITIONS
        for rule in definition.schedule_rules
        if rule.component_id == "bed"
    )
    assert bed_rule.interval_source == "temporal-profile"
    assert bed_rule.skip_when_flag == "persisted-bed-region"
    assert bed_rule.resolve(1, CURRENT_TEMPORAL_PROFILE) == 90


def test_scene_state_two_scheduled_empty_cycles_expire_cache_without_touching_tracks() -> None:
    bed = _box(0, 0, 10, 10)
    person = _box(20, 20, 40, 40)
    tracker = GreedyIouTracker(max_misses=1)
    assert tracker.observe((person,)) == (0,)
    state = SceneState("cam-1")
    first = FrameObservation(regions=((bed,), ()), track_ids=(0,))
    resolved, debug = state.resolve_bed_regions(
        first, frame_index=0, bed_scheduled=True, bed_interval=90
    )
    assert resolved.bed_boxes == (bed,)
    assert debug.source == "fresh"
    # resolve_bed_regions writes SceneState via update() without track_ids, so
    # the scene cache is not the person-track owner. Tracker state is.
    assert "track_ids" not in state.resolve_bed_regions.__code__.co_varnames

    empty = FrameObservation(track_ids=(0,))
    _, first_empty = state.resolve_bed_regions(
        empty, frame_index=90, bed_scheduled=True, bed_interval=90
    )
    assert first_empty.source == "empty"
    expired, second_empty = state.resolve_bed_regions(
        empty, frame_index=180, bed_scheduled=True, bed_interval=90
    )
    assert second_empty.source == "expired"
    assert expired.bed_boxes == ()
    assert tracker.live_ids == frozenset({0})
    assert tracker.observe((person,)) == (0,)
