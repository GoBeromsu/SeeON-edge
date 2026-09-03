"""Differential parity: native strategy vs the Python tracker oracle.

`GreedyIouTracker` (`worker/pipeline/perception/tracker.py`) remains the
oracle for differential shadow comparison (Task 4 guardrail) -- it is
imported directly here, never duplicated. The candidate is driven through
its real typed seam (`PerceptionFrameIdentity` + `PersonBoxChannel` ->
`AssociationResult`), and every frame's `track_ids`, `live_ids` and
`selected_cue_indexes` are compared against the oracle via
`AssociationFrameTrace`/`compare_traces`.
"""

from __future__ import annotations

from contracts.observation import BoundingBox
from worker.native.deepstream.association import (
    AssociationFrameTrace,
    build_active_association_strategy,
    compare_traces,
)
from worker.native.deepstream.association.legacy_greedy_iou import LegacyGreedyBboxIouStrategy
from worker.pipeline.perception.tracker import GreedyIouTracker
from worker.types.perception_frame import (
    ChannelState,
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
)

_IDENTITY = PerceptionFrameIdentity(
    worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
)


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def _person_box_channel(boxes: tuple[BoundingBox, ...]) -> PersonBoxChannel:
    state = ChannelState.INFERRED if boxes else ChannelState.INFERRED_EMPTY
    cues = tuple(
        PersonBox(x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, confidence=box.confidence)
        for box in boxes
    )
    return PersonBoxChannel(state=state, boxes=cues)


def _trace_from_oracle(
    oracle: GreedyIouTracker, frames: tuple[tuple[BoundingBox, ...] | None, ...]
) -> tuple[AssociationFrameTrace, ...]:
    traces: list[AssociationFrameTrace] = []
    for boxes in frames:
        if boxes is None:
            oracle.coast()
            traces.append(AssociationFrameTrace(track_ids=(), live_ids=oracle.live_ids))
            continue
        track_ids = oracle.observe(boxes)
        traces.append(
            AssociationFrameTrace(
                track_ids=track_ids,
                live_ids=oracle.live_ids,
                selected_cue_indexes=tuple(range(len(boxes))),
                identity=_IDENTITY,
            )
        )
    return tuple(traces)


def _trace_from_native(
    frames: tuple[tuple[BoundingBox, ...] | None, ...],
) -> tuple[AssociationFrameTrace, ...]:
    strategy = build_active_association_strategy()
    traces: list[AssociationFrameTrace] = []
    for boxes in frames:
        if boxes is None:
            strategy.coast()
            traces.append(AssociationFrameTrace(track_ids=(), live_ids=strategy.live_ids))
            continue
        result = strategy.observe(_IDENTITY, _person_box_channel(boxes))
        traces.append(
            AssociationFrameTrace(
                track_ids=result.track_ids,
                live_ids=strategy.live_ids,
                selected_cue_indexes=result.selected_cue_indexes,
                identity=result.identity,
            )
        )
    return tuple(traces)


def _assert_parity(frames: tuple[tuple[BoundingBox, ...] | None, ...]) -> None:
    reference = _trace_from_oracle(GreedyIouTracker(), frames)
    candidate = _trace_from_native(frames)
    mismatches = compare_traces(reference, candidate)
    assert mismatches == (), f"parity mismatches: {mismatches}"


def test_native_matches_oracle_on_greedy_order_and_box_order_return() -> None:
    first = _box(0, 0, 50, 50)
    second = _box(200, 200, 250, 250)
    moved_second = _box(205, 205, 255, 255)
    moved_first = _box(4, 4, 54, 54)
    _assert_parity(((first, second), (moved_second, moved_first)))


def test_native_matches_oracle_on_equal_iou_tie_keeping_lower_track_index() -> None:
    left = _box(0, 0, 100, 100)
    right = _box(50, 0, 150, 100)
    center = _box(25, 0, 125, 100)
    _assert_parity(((left, right), (center,)))


def test_native_matches_oracle_on_inferred_empty_counts_a_miss() -> None:
    box = _box(0, 0, 20, 20)
    _assert_parity(((box,), (), (), (box,)))


def test_native_matches_oracle_on_skipped_frame_coasts_without_a_miss() -> None:
    box = _box(0, 0, 20, 20)
    _assert_parity(((box,), None, None, (box,)))


def test_native_matches_oracle_on_eviction_after_max_misses() -> None:
    box = _box(0, 0, 20, 20)
    oracle = GreedyIouTracker(max_misses=1)
    native = LegacyGreedyBboxIouStrategy(max_misses=1)

    assert oracle.observe((box,)) == (0,)
    assert native.observe(_IDENTITY, _person_box_channel((box,))).track_ids == (0,)
    assert oracle.observe(()) == ()
    assert native.observe(_IDENTITY, _person_box_channel(())).track_ids == ()
    assert oracle.observe(()) == ()
    assert native.observe(_IDENTITY, _person_box_channel(())).track_ids == ()
    assert oracle.live_ids == frozenset()
    assert native.live_ids == frozenset()
    assert oracle.observe((box,)) == (1,)
    assert native.observe(_IDENTITY, _person_box_channel((box,))).track_ids == (1,)


def test_native_matches_oracle_on_missing_pose_never_creates_or_evicts() -> None:
    """A frame with no pose/person cues at all (None) never mutates tracks."""
    person = _box(10, 10, 40, 40)
    _assert_parity(((person,), None, None, None, (person,)))


def test_native_matches_oracle_on_multi_person_crossing_and_occlusion() -> None:
    """Two people cross paths with real per-frame IoU overlap and briefly merge."""
    left = _box(0, 0, 40, 100)
    right = _box(80, 0, 120, 100)
    closing_left = _box(20, 0, 60, 100)
    closing_right = _box(60, 0, 100, 100)
    near_overlap_left = _box(30, 0, 70, 100)
    near_overlap_right = _box(50, 0, 90, 100)
    occluded_single = _box(35, 0, 85, 100)
    reemerging_left = _box(20, 0, 60, 100)
    reemerging_right = _box(60, 0, 100, 100)
    _assert_parity(
        (
            (left, right),
            (closing_left, closing_right),
            (near_overlap_left, near_overlap_right),
            (occluded_single,),
            None,
            (reemerging_left, reemerging_right),
        )
    )


def test_native_reset_matches_a_fresh_oracle_instance_after_reconnect() -> None:
    """Reconnect / stream-epoch rollover: reset must not resume prior ids."""
    box = _box(0, 0, 20, 20)
    native = build_active_association_strategy()
    _ = native.observe(_IDENTITY, _person_box_channel((box,)))
    _ = native.observe(_IDENTITY, _person_box_channel((box,)))
    assert native.live_ids == frozenset({0})

    native.reset()

    fresh_oracle = GreedyIouTracker()
    assert native.live_ids == frozenset()
    reset_result = native.observe(_IDENTITY, _person_box_channel((box,)))
    assert reset_result.track_ids == fresh_oracle.observe((box,))
    assert native.live_ids == fresh_oracle.live_ids == frozenset({0})


def test_boot_reset_cannot_observe_prior_tracker_state() -> None:
    """A fresh boot id must start from empty state, proven by two independent instances."""
    box = _box(0, 0, 20, 20)
    previous_boot = build_active_association_strategy()
    _ = previous_boot.observe(_IDENTITY, _person_box_channel((box,)))
    _ = previous_boot.observe(_IDENTITY, _person_box_channel((box,)))
    assert previous_boot.live_ids == frozenset({0})

    next_boot = build_active_association_strategy()
    assert next_boot.live_ids == frozenset()
    assert next_boot.observe(_IDENTITY, _person_box_channel((box,))).track_ids == (0,)


def test_bed_shaped_cue_never_resolves_to_the_prior_person_id_by_special_case() -> None:
    """The strategy treats every person-box cue uniformly; it does not know "bed"."""
    native = build_active_association_strategy()
    person = _box(10, 10, 40, 40)
    assert native.observe(_IDENTITY, _person_box_channel((person,))).track_ids == (0,)
    far_away_shape = _box(500, 500, 508, 508)
    outcome = native.observe(_IDENTITY, _person_box_channel((far_away_shape,)))
    assert outcome.track_ids != (0,), "a non-overlapping cue must mint a new id, not reuse 0"
