from __future__ import annotations

from contracts.observation import BoundingBox, FrameObservation
from worker.pipeline.perception.scene_state import SceneState

BED = BoundingBox(0, 0, 10, 10, 0.9)


def _observation(beds: tuple[BoundingBox, ...] = ()) -> FrameObservation:
    return FrameObservation(regions=(beds, ()))


def test_scene_state_carries_bed_roi_until_frame_index_ttl() -> None:
    state = SceneState("cam-1")
    observation, debug = state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "fresh"

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=60, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "cached"
    assert debug.age_frames == 60

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=61, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == ()
    assert debug.source == "expired"


def test_scene_state_invalidates_after_two_scheduled_empty_bed_cycles() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    first, first_debug = state.resolve_bed_regions(
        _observation(), frame_index=30, bed_scheduled=True, bed_interval=30
    )
    assert first.bed_boxes == ()
    assert first_debug.source == "empty"
    second, second_debug = state.resolve_bed_regions(
        _observation(), frame_index=60, bed_scheduled=True, bed_interval=30
    )
    assert second.bed_boxes == ()
    assert second_debug.source == "expired"
    carried, carried_debug = state.resolve_bed_regions(
        _observation(), frame_index=61, bed_scheduled=False, bed_interval=30
    )
    assert carried.bed_boxes == ()
    assert carried_debug.source == "empty"


def test_scene_state_resets_on_frame_index_discontinuity() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    state.resolve_bed_regions(_observation(), frame_index=1, bed_scheduled=False, bed_interval=30)
    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == ()
    assert debug.reset_reason == "frame_index_discontinuity"
    assert state.bed_region_counters.reset == 1


def test_persisted_bed_regions_short_circuit_win_over_live_scheduling() -> None:
    """An operator-recognized, persisted polygon (camera-build-time, never
    mutated per-frame) is authoritative: it always reports ``fresh`` and
    never expires, regardless of what the live segmentation cycle sees on
    that frame -- including a scheduled cycle that found no beds at all.
    """
    persisted = BoundingBox(1, 1, 9, 9, 1.0, polygon=((1, 1), (9, 1), (9, 9), (1, 9)))
    state = SceneState("cam-1", persisted_bed_regions=(persisted,))

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (persisted,)
    assert debug.source == "fresh"
    assert debug.age_frames == 0
    assert debug.empty_cycles == 0
    assert debug.reset_reason is None

    # A live detection on the same frame is ignored -- the persisted polygon
    # still wins.
    observation, debug = state.resolve_bed_regions(
        _observation((BED,)), frame_index=1, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (persisted,)
    assert debug.source == "fresh"

    # Two scheduled-empty cycles would normally expire the live cache; the
    # persisted polygon still never expires.
    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=61, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (persisted,)
    assert debug.source == "fresh"

    # A frame-index discontinuity would normally reset the live cache; the
    # persisted polygon is unaffected.
    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (persisted,)
    assert debug.source == "fresh"
    assert debug.reset_reason is None


def test_cached_roi_freshness_counters_are_observable() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=1
    )
    state.resolve_bed_regions(_observation(), frame_index=1, bed_scheduled=False, bed_interval=1)
    state.resolve_bed_regions(_observation(), frame_index=3, bed_scheduled=False, bed_interval=1)
    state.resolve_bed_regions(_observation(), frame_index=0, bed_scheduled=False, bed_interval=1)
    assert state.bed_region_counters.snapshot()["fresh"] == 1
    assert state.bed_region_counters.snapshot()["cached"] == 1
    assert state.bed_region_counters.snapshot()["expired"] == 1
    assert state.bed_region_counters.snapshot()["reset"] == 1
