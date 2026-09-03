from __future__ import annotations

from contracts.observation import BoundingBox, FrameObservation
from worker.pipeline.perception.scene_state import SceneState

PERSISTED = BoundingBox(1, 1, 9, 9, 1.0, polygon=((1, 1), (9, 1), (9, 9), (1, 9)))
SEGMENTED = BoundingBox(0, 0, 10, 10, 0.9)


def _observation(beds: tuple[BoundingBox, ...] = ()) -> FrameObservation:
    return FrameObservation(regions=(beds, ()))


def test_scene_state_uses_persisted_polygon_and_ignores_segmentation() -> None:
    state = SceneState("cam-1", persisted_bed_regions=(PERSISTED,))

    observation, debug = state.resolve_bed_regions(
        _observation((SEGMENTED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )

    assert observation.bed_boxes == (PERSISTED,)
    assert debug.source == "fresh"
    assert state.bed_polygon_source == "persisted"


def test_scene_state_without_persisted_polygon_never_uses_segmentation() -> None:
    state = SceneState("cam-1")

    observation, debug = state.resolve_bed_regions(
        _observation((SEGMENTED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )

    assert observation.bed_boxes == ()
    assert debug.source == "empty"
    assert state.bed_polygon_source == "none"


def test_persisted_polygon_survives_source_reset() -> None:
    state = SceneState("cam-1", persisted_bed_regions=(PERSISTED,))
    state.reset_for_new_source()

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=False, bed_interval=30
    )

    assert observation.bed_boxes == (PERSISTED,)
    assert debug.source == "fresh"
