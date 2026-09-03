"""Characterization pins for the shipped `GreedyIouTracker` (Task 4 oracle).

These pins run against the UNCHANGED `worker.pipeline.perception.tracker`
module only -- no candidate/native code is imported here. They exist to prove
the C4 native association work has a byte/value-stable oracle to diff against
before any new production code lands: greedy descending-IoU match order,
equal-IoU tie behavior, `observe(())` vs `coast()` (inferred-empty vs
skipped), eviction after `max_misses`, reconnect/epoch-reset-shaped restart
(a fresh `GreedyIouTracker()` instance), and the downstream fall/bed-exit
event timelines the tracker's `track_ids` feed.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor, NightWindow
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallWindowClassifierV2
from worker.interfaces.fall_model import FallV2Probabilities
from worker.pipeline.perception.features.geometry import greedy_match
from worker.pipeline.perception.tracker import GreedyIouTracker
from worker.types import BusinessEvent, DecisionInput


def _box(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


# --------------------------------------------------------------------------
# Greedy order / tie / skip-empty / eviction pins (byte/value stable).
# --------------------------------------------------------------------------


def test_oracle_returns_ids_in_incoming_box_order_not_match_order() -> None:
    tracker = GreedyIouTracker()
    first = _box(0, 0, 50, 50)
    second = _box(200, 200, 250, 250)
    assert tracker.observe((first, second)) == (0, 1)

    moved_second = _box(205, 205, 255, 255)
    moved_first = _box(4, 4, 54, 54)
    assert tracker.observe((moved_second, moved_first)) == (1, 0)


def test_oracle_equal_iou_tie_keeps_lower_existing_track_index() -> None:
    tracker = GreedyIouTracker()
    left = _box(0, 0, 100, 100)
    right = _box(50, 0, 150, 100)
    assert tracker.observe((left, right)) == (0, 1)

    center = _box(25, 0, 125, 100)
    assert greedy_match((left, right), (center,), min_iou=0.3) == ((0, 0),)
    assert tracker.observe((center,)) == (0,)


def test_oracle_update_empty_coasts_while_observe_empty_counts_a_miss() -> None:
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


def test_oracle_evicts_only_after_exceeding_max_misses() -> None:
    tracker = GreedyIouTracker(max_misses=1)
    box = _box(0, 0, 20, 20)
    assert tracker.observe((box,)) == (0,)
    assert tracker.observe(()) == ()
    assert tracker.live_ids == frozenset({0})
    assert tracker.observe(()) == ()
    assert tracker.live_ids == frozenset()
    assert tracker.observe((box,)) == (1,)


def test_oracle_missing_pose_frame_never_creates_or_evicts_via_coast() -> None:
    """A frame carrying no inference result at all coasts every live track."""
    person = _box(10, 10, 40, 40)
    tracker = GreedyIouTracker(max_misses=1)
    assert tracker.observe((person,)) == (0,)
    tracker.coast()
    tracker.coast()
    assert tracker.live_ids == frozenset({0})
    assert tracker.observe((person,)) == (0,)


def test_oracle_multi_person_crossing_and_occlusion_preserves_identity() -> None:
    """Two people cross paths (each step keeps IoU > min_iou with its last
    box) and briefly merge into one fully-occluded detection.

    Greedy IoU has no motion model: a jump with zero overlap mints a new
    track. This fixture pins the real crossing shape -- small per-frame
    displacement so consecutive boxes always overlap their own predecessor.
    """
    tracker = GreedyIouTracker()
    left = _box(0, 0, 40, 100)
    right = _box(80, 0, 120, 100)
    assert tracker.observe((left, right)) == (0, 1)

    closing_left = _box(20, 0, 60, 100)
    closing_right = _box(60, 0, 100, 100)
    assert tracker.observe((closing_left, closing_right)) == (0, 1)

    near_overlap_left = _box(30, 0, 70, 100)
    near_overlap_right = _box(50, 0, 90, 100)
    assert tracker.observe((near_overlap_left, near_overlap_right)) == (0, 1)

    occluded_single = _box(35, 0, 85, 100)
    assert tracker.observe((occluded_single,)) == (0,)

    tracker.coast()

    reemerging_left = _box(20, 0, 60, 100)
    reemerging_right = _box(60, 0, 100, 100)
    assert tracker.observe((reemerging_left, reemerging_right)) == (0, 1)


def test_oracle_reconnect_shaped_restart_is_a_fresh_instance_with_empty_state() -> None:
    """Reconnect / stream-epoch rollover pins to constructing a fresh tracker.

    `GreedyIouTracker` has no `reset()` method today; the shipped restart
    shape is simply a new instance. This pins that a fresh instance starts
    from empty `live_ids` and re-mints id 0, so a native `reset()` seam has an
    exact byte/value target to reproduce.
    """
    box = _box(0, 0, 20, 20)
    previous = GreedyIouTracker()
    assert previous.observe((box,)) == (0,)
    assert previous.observe((box,)) == (0,)
    assert previous.live_ids == frozenset({0})

    fresh = GreedyIouTracker()
    assert fresh.live_ids == frozenset()
    assert fresh.observe((box,)) == (0,)


# --------------------------------------------------------------------------
# Downstream fall/bed-exit event timeline pins driven by oracle track_ids.
# --------------------------------------------------------------------------


_FRAME_SEC = 1 / 15
# V2 confirms on the third stride-5 prediction after the 30-row window fills.
_ONSET_FRAME = 40


class _FallModel:
    def predict(self, features: object) -> FallV2Probabilities:
        del features
        return FallV2Probabilities(background=0.1, fall_transition=0.9, fallen=0.0)


def _fall_decision_input(
    track_ids: tuple[int, ...], boxes: tuple[BoundingBox, ...], frame_index: int
) -> DecisionInput:
    pose = tuple((90, 50, 0.9) for _ in range(17))
    return DecisionInput(
        observation=FrameObservation(detections=(boxes, ()), poses=(pose,), track_ids=track_ids),
        frame_width=200,
        frame_height=200,
        live_track_ids=track_ids,
        time_sec=frame_index * _FRAME_SEC,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_oracle_driven_fall_timeline_emits_exactly_one_rising_edge() -> None:
    boxes = (_box(70, 20, 110, 130),)
    tracker = GreedyIouTracker()
    decider = FallV2DomainDecider(
        classifier=FallWindowClassifierV2(_FallModel()),
        policy=FallPolicyDeciderV2(
            camera_id="camera-1",
            facility_id="facility-1",
            boot_id="boot-1",
            stream_epoch="0",
            source_generation=0,
        ),
    )

    events: list[tuple[BusinessEvent, ...]] = []
    for frame_index in range(_ONSET_FRAME + 20):
        track_ids = tracker.observe(boxes)
        events.append(decider.update(_fall_decision_input(track_ids, boxes, frame_index)))

    onsets = [index for index, emitted in enumerate(events) if emitted]
    assert onsets == [_ONSET_FRAME]
    (event,) = events[_ONSET_FRAME]
    assert event.event_type == "fall"
    assert event.identity == "boot-1:0:0:0:0:1"
    assert decider.last_trace_snapshots[0].track_id == 0


def _bed_exit_monitor() -> BedExitMonitor:
    return BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-1",
            facility_id="facility-1",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
            night_window=NightWindow(start="21:00", end="05:00", tz="Asia/Seoul"),
        ),
        clock=lambda: datetime(2026, 7, 31, 22, 0, tzinfo=ZoneInfo("Asia/Seoul")),
    )


def _bed_exit_decision_input(
    track_ids: tuple[int, ...],
    boxes: tuple[BoundingBox, ...],
    bed: BoundingBox,
    frame_index: int,
) -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(
            detections=(boxes, ()),
            regions=((bed,), ()),
            track_ids=track_ids,
        ),
        frame_width=200,
        frame_height=200,
        live_track_ids=track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.FRESH),
    )


def test_oracle_driven_bed_exit_timeline_emits_exactly_one_event_on_exit() -> None:
    """Person walks out of bed in small steps so IoU tracking never breaks.

    Each consecutive step keeps IoU > 0.3 with its predecessor (the tracker
    has no motion model), while containment against ``bed`` drops from 1.0 to
    0.125, below ``min_containment=0.5`` -- a real exit, not a teleport.
    """
    bed = _box(0, 0, 80, 100)
    in_bed = _box(10, 10, 70, 90)
    step = _box(10, 30, 70, 110)
    outside = _box(10, 70, 70, 150)
    tracker = GreedyIouTracker()
    monitor = _bed_exit_monitor()

    frame_0 = tracker.observe((in_bed,))
    events_0 = monitor.update(_bed_exit_decision_input(frame_0, (in_bed,), bed, 0))
    frame_1 = tracker.observe((step,))
    events_1 = monitor.update(_bed_exit_decision_input(frame_1, (step,), bed, 1))
    frame_2 = tracker.observe((outside,))
    events_2 = monitor.update(_bed_exit_decision_input(frame_2, (outside,), bed, 2))

    assert frame_0 == frame_1 == frame_2 == (0,), "the same identity must persist across the exit"
    assert events_0 == ()
    assert events_1 == ()
    assert len(events_2) == 1
    assert events_2[0].event_type == "bed-exit"
    assert events_2[0].person_id == 0
