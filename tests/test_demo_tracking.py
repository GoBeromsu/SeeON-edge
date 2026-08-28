"""Unit tests for the greedy-IoU tracker (worker/pipeline/perception/tracker.py).

Tests are pure — no YOLO, no numpy, no heavy deps.  Covers:
- iou() math correctness (identical, non-overlapping, half-overlap, contained, touching)
- id stability across overlapping frames
- fresh id for non-overlapping box
- miss-then-recover keeps same id within max_misses
- eviction after max_misses
- evicted id absent from live_ids
- two spatially separated people keep distinct ids
- live_ids reflects missed-but-alive tracks
- empty input produces empty output
"""

from __future__ import annotations

import pytest

from contracts import BoundingBox
from worker.pipeline.perception.tracker import GreedyIouTracker, iou


def _box(x1: int, y1: int, x2: int, y2: int, conf: float = 0.9) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


# ---------------------------------------------------------------------------
# iou() — math correctness
# ---------------------------------------------------------------------------


class TestIou:
    def test_identical_boxes_return_one(self) -> None:
        b = _box(0, 0, 100, 100)
        assert iou(b, b) == pytest.approx(1.0)

    def test_non_overlapping_boxes_return_zero(self) -> None:
        a = _box(0, 0, 50, 50)
        b = _box(100, 100, 200, 200)
        assert iou(a, b) == 0.0

    def test_half_overlap_both_same_size(self) -> None:
        # a: [0,0]->[100,100]; b: [50,0]->[150,100]
        # intersection: [50,0]->[100,100] = 50*100 = 5 000
        # union: 10 000 + 10 000 - 5 000 = 15 000  →  IoU = 1/3
        a = _box(0, 0, 100, 100)
        b = _box(50, 0, 150, 100)
        assert iou(a, b) == pytest.approx(1.0 / 3.0, rel=1e-5)

    def test_contained_box_has_correct_iou(self) -> None:
        # outer [0,0]->[100,100]=10 000; inner [25,25]->[75,75]=2 500
        # intersection=2 500, union=10 000; IoU=0.25
        outer = _box(0, 0, 100, 100)
        inner = _box(25, 25, 75, 75)
        assert iou(outer, inner) == pytest.approx(0.25, rel=1e-5)

    def test_touching_edges_return_zero(self) -> None:
        # Boxes share an edge but do not overlap in area.
        a = _box(0, 0, 50, 50)
        b = _box(50, 0, 100, 50)
        assert iou(a, b) == 0.0

    def test_symmetry(self) -> None:
        a = _box(0, 0, 80, 60)
        b = _box(40, 20, 120, 100)
        assert iou(a, b) == pytest.approx(iou(b, a), rel=1e-9)


# ---------------------------------------------------------------------------
# GreedyIouTracker
# ---------------------------------------------------------------------------


class TestGreedyIouTracker:
    def test_first_box_gets_id_zero(self) -> None:
        tracker = GreedyIouTracker()
        ids = tracker.update((_box(0, 0, 100, 100),))
        assert ids == (0,)

    def test_second_fresh_box_gets_id_one(self) -> None:
        tracker = GreedyIouTracker()
        tracker.update((_box(0, 0, 100, 100),))
        ids = tracker.update((_box(500, 500, 600, 600),))
        assert ids == (1,)

    def test_stable_id_across_overlapping_frames(self) -> None:
        """Same box position in consecutive frames keeps the same track id."""
        tracker = GreedyIouTracker()
        b = _box(0, 0, 100, 100)
        (id1,) = tracker.update((b,))
        (id2,) = tracker.update((b,))
        assert id1 == id2

    def test_non_overlapping_box_gets_fresh_id(self) -> None:
        """A new box far from existing tracks gets a new id."""
        tracker = GreedyIouTracker()
        (id1,) = tracker.update((_box(0, 0, 100, 100),))
        (id2,) = tracker.update((_box(500, 500, 600, 600),))
        assert id2 != id1

    def test_miss_then_recover_keeps_same_id_within_max_misses(self) -> None:
        """A track missed for fewer than max_misses frames recovers its id."""
        tracker = GreedyIouTracker(max_misses=5)
        b = _box(0, 0, 100, 100)
        (orig_id,) = tracker.update((b,))

        # Miss 3 frames — still within max_misses=5.
        tracker.update(())
        tracker.update(())
        tracker.update(())

        (recovered_id,) = tracker.update((b,))
        assert recovered_id == orig_id

    def test_eviction_after_max_misses(self) -> None:
        """A track evicted after max_misses gets a fresh id on the next detection."""
        tracker = GreedyIouTracker(max_misses=2)
        b = _box(0, 0, 100, 100)
        (orig_id,) = tracker.update((b,))

        # Three observed-empty frames trigger eviction (misses > max_misses).
        tracker.observe(())  # misses=1
        tracker.observe(())  # misses=2
        tracker.observe(())  # misses=3 > 2 → evicted
        assert tracker.live_ids == frozenset()

        (new_id,) = tracker.update((b,))
        assert new_id != orig_id
        assert tracker.live_ids == frozenset({new_id})

    def test_evicted_id_not_in_live_ids(self) -> None:
        """live_ids does not contain evicted track ids."""
        tracker = GreedyIouTracker(max_misses=1)
        (tid,) = tracker.update((_box(0, 0, 100, 100),))
        tracker.observe(())  # misses=1
        tracker.observe(())  # misses=2 > 1 → evicted
        assert tracker.live_ids == frozenset()
        assert tid not in tracker.live_ids

    def test_two_crossing_people_keep_distinct_ids(self) -> None:
        """Two spatially separated boxes in consecutive frames keep their track ids."""
        tracker = GreedyIouTracker()
        box_a = _box(0, 0, 100, 100)
        box_b = _box(300, 300, 400, 400)

        # Frame 1: establish two tracks.
        (tid_a, tid_b) = tracker.update((box_a, box_b))
        assert tid_a != tid_b

        # Frame 2: same positions — should recover same ids.
        (id_a2, id_b2) = tracker.update((box_a, box_b))
        assert id_a2 == tid_a
        assert id_b2 == tid_b

    def test_live_ids_includes_missed_but_alive_track(self) -> None:
        """live_ids reflects missed tracks that haven't yet exceeded max_misses."""
        tracker = GreedyIouTracker(max_misses=5)
        (tid,) = tracker.update((_box(0, 0, 100, 100),))
        tracker.update(())  # miss but not evicted
        assert tid in tracker.live_ids

    def test_empty_input_produces_empty_output(self) -> None:
        tracker = GreedyIouTracker()
        assert tracker.update(()) == ()

    def test_empty_input_on_empty_tracker(self) -> None:
        tracker = GreedyIouTracker()
        assert tracker.update(()) == ()
        assert tracker.live_ids == frozenset()

    def test_multiple_frames_ids_are_stable(self) -> None:
        """Track ids remain stable over many frames with the same boxes."""
        tracker = GreedyIouTracker()
        box_a = _box(10, 10, 90, 90)
        box_b = _box(200, 200, 300, 300)
        first_ids = tracker.update((box_a, box_b))
        for _ in range(10):
            ids = tracker.update((box_a, box_b))
        assert ids == first_ids
