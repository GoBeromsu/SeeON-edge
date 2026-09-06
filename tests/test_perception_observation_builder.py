from __future__ import annotations

import pytest

from contracts.observation import BoundingBox, DetectionLabel, DetectionResult, FrameObservation
from worker.pipeline.perception import SceneState, WindowBuffer, build_frame_observation
from worker.pipeline.perception.observation_builder import observation_from_detection_result


def test_observation_builder_maps_detection_result_and_overrides_runner_outputs() -> None:
    base = DetectionResult(
        boxes=(BoundingBox(1, 2, 3, 4, 0.5),),
        labels=(DetectionLabel("NORMAL", 0.7, False),),
        keypoints=(tuple((idx, idx, 0.1) for idx in range(17)),),
        bed_boxes=(BoundingBox(10, 20, 30, 40, 0.9),),
        bed_exit_statuses=("empty",),
    )
    labels = (DetectionLabel("FALL", 0.95, True),)
    poses = (tuple((idx, idx + 2, 0.8) for idx in range(17)),)
    bed_boxes = (BoundingBox(100, 101, 200, 201, 0.87),)

    observation = build_frame_observation(
        detections=base,
        raw_boxes=((5, 6, 70, 80, 0.99),),
        labels=labels,
        poses=poses,
        bed_boxes=bed_boxes,
        bed_exit_statuses=("exit",),
    )

    assert isinstance(observation, FrameObservation)
    assert observation.detections == ((BoundingBox(5, 6, 70, 80, 0.99),), labels)
    assert observation.poses == poses
    assert observation.regions == (bed_boxes, ("exit",))
    assert observation_from_detection_result(base) == FrameObservation.from_detection_result(base)


def test_window_buffer_emits_windows_at_configured_stride() -> None:
    buffer = WindowBuffer(window=3, stride=2)

    assert buffer.append((1, 1, 1)) == ()
    assert buffer.append((2, 2, 2)) == ()
    assert buffer.append((3, 3, 3)) == (((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)),)
    assert buffer.append((4, 4, 4)) == ()
    assert buffer.append((5, 5, 5)) == (((3.0, 3.0, 3.0), (4.0, 4.0, 4.0), (5.0, 5.0, 5.0)),)


def test_window_buffer_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="window"):
        WindowBuffer(window=0, stride=1)
    with pytest.raises(ValueError, match="stride"):
        WindowBuffer(window=1, stride=0)


def test_scene_state_updates_tracks_without_mutating_persisted_bed_polygon() -> None:
    persisted_bed = BoundingBox(1, 2, 30, 40, 0.9)
    live_bed = BoundingBox(50, 60, 70, 80, 0.8)
    first = FrameObservation(regions=((live_bed,), ("occupied",)))
    second = FrameObservation()
    state = SceneState(camera_id="cam-1", persisted_bed_regions=(persisted_bed,))

    assert state.update(first, track_ids=(10, 11)) is first
    assert state.latest_observation == first
    assert state.track_ids == (10, 11)
    assert state.persisted_bed_regions == (persisted_bed,)
    assert state.bed_polygon_source == "persisted"

    state.update(second, track_ids=(12,))

    assert state.latest_observation == second
    assert state.track_ids == (12,)
    assert state.persisted_bed_regions == (persisted_bed,)
