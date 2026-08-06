from __future__ import annotations

from contracts.observation import BoundingBox, FrameObservation
from worker.pipeline.perception.scene_state import SceneState

BED = BoundingBox(0, 0, 10, 10, 0.9)


def _observation(beds: tuple[BoundingBox, ...] = ()) -> FrameObservation:
    return FrameObservation(regions=(beds, ()))


def test_scene_state_carries_bed_roi_with_no_frame_index_or_wall_clock_ttl() -> None:
    """The cache has no age limit of its own -- see #208.

    A large, arbitrary frame-index gap between the detection and the next
    read must not expire an otherwise-good cache: only two consecutive
    scheduled-empty cycles (tested separately) may do that. This is the
    replacement for the deleted ``_cached_boxes_if_fresh`` age comparison,
    which used to expire the cache once ``frame_index - last_bed_frame_index``
    exceeded ``2 * bed_interval`` even though nothing had actually proven the
    bed was gone.
    """
    state = SceneState("cam-1")
    observation, debug = state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "fresh"

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=10_000, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "cached"

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=1, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "cached"


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


def test_scene_state_survives_a_reconnect_that_restarts_frame_index_near_zero() -> None:
    """This is the regression test for #208.

    A reconnect that restarts the decode session's frame counter must not
    wipe an otherwise-good cache. Two stacked mechanisms used to make this
    fail before #208:

    1. ``_reset_if_discontinuous`` treated any non-increasing ``frame_index``
       as a discontinuity and wiped the cache outright.
    2. Even without (1), ``_cached_boxes_if_fresh``'s age comparison computed
       ``frame_index - last_bed_frame_index``, which goes negative the moment
       the new session's frame_index is smaller than the old session's
       ``last_bed_frame_index`` -- and a negative age also failed the cache.

    Neither mechanism exists anymore. A frame-index regression is now
    indistinguishable from any other cached read: the cache is served exactly
    as if no reconnect had happened, because nothing about a reconnect alone
    is evidence the bed is gone.
    """
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=5_000, bed_scheduled=True, bed_interval=30
    )
    state.resolve_bed_regions(
        _observation(), frame_index=5_001, bed_scheduled=False, bed_interval=30
    )

    # A reconnect: the new decode session's frame_index restarts near 0,
    # far below the previous session's last_bed_frame_index (5_000).
    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=1, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (BED,)
    assert debug.source == "cached"
    assert state.bed_region_counters.snapshot()["reset"] == 0

    # The scheduled-empty mechanism is still the only way to invalidate it,
    # and it still works identically post-reconnect.
    state.resolve_bed_regions(_observation(), frame_index=31, bed_scheduled=True, bed_interval=30)
    expired, expired_debug = state.resolve_bed_regions(
        _observation(), frame_index=61, bed_scheduled=True, bed_interval=30
    )
    assert expired.bed_boxes == ()
    assert expired_debug.source == "expired"


def test_persisted_bed_regions_short_circuit_win_over_live_scheduling() -> None:
    """An operator-recognized, persisted polygon (camera-build-time, never
    mutated per-frame) is authoritative: it always reports ``fresh`` and
    never expires, regardless of what the live segmentation cycle sees on
    that frame -- including a scheduled cycle that found no beds at all, or
    a frame-index regression that would otherwise look like a reconnect.
    """
    persisted = BoundingBox(1, 1, 9, 9, 1.0, polygon=((1, 1), (9, 1), (9, 9), (1, 9)))
    state = SceneState("cam-1", persisted_bed_regions=(persisted,))

    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=True, bed_interval=30
    )
    assert observation.bed_boxes == (persisted,)
    assert debug.source == "fresh"
    assert debug.empty_cycles == 0

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

    # A frame-index regression is unaffected either way.
    observation, debug = state.resolve_bed_regions(
        _observation(), frame_index=0, bed_scheduled=False, bed_interval=30
    )
    assert observation.bed_boxes == (persisted,)
    assert debug.source == "fresh"


def test_cached_roi_freshness_counters_are_observable() -> None:
    state = SceneState("cam-1")
    state.resolve_bed_regions(
        _observation((BED,)), frame_index=0, bed_scheduled=True, bed_interval=1
    )
    state.resolve_bed_regions(_observation(), frame_index=1, bed_scheduled=False, bed_interval=1)
    state.resolve_bed_regions(_observation(), frame_index=3, bed_scheduled=False, bed_interval=1)
    # A frame-index regression no longer triggers a reset; it is still just
    # a cached read (see the reconnect regression test above).
    state.resolve_bed_regions(_observation(), frame_index=0, bed_scheduled=False, bed_interval=1)
    assert state.bed_region_counters.snapshot()["fresh"] == 1
    assert state.bed_region_counters.snapshot()["cached"] == 3
    assert state.bed_region_counters.snapshot()["expired"] == 0
    assert state.bed_region_counters.snapshot()["reset"] == 0

    # `reset` now only comes from the explicit external API (e.g. an ingest
    # lifecycle hook calling this on a real, known source restart) --
    # never implicitly from frame-index shape alone.
    state.reset_for_new_source("source_restart")
    assert state.bed_region_counters.snapshot()["reset"] == 1
    assert state.bed_regions == ()
