from __future__ import annotations

from typing import final

from contracts.observation import BoundingBox
from worker.pipeline.perception.features.geometry import greedy_match, iou


@final
class _Track:
    __slots__: tuple[str, ...] = ("last_box", "misses", "track_id")

    track_id: int
    last_box: BoundingBox
    misses: int

    def __init__(self, track_id: int, last_box: BoundingBox) -> None:
        self.track_id = track_id
        self.last_box = last_box
        self.misses = 0


@final
class GreedyIouTracker:
    """Assign persistent IDs to index-ordered boxes using greedy IoU matches."""

    def __init__(self, min_iou: float = 0.3, max_misses: int = 30) -> None:
        self._min_iou: float = min_iou
        self._max_misses: int = max_misses
        self._tracks: list[_Track] = []
        self._next_id: int = 0

    @property
    def live_ids(self) -> frozenset[int]:
        return frozenset(track.track_id for track in self._tracks)

    def update(self, boxes: tuple[BoundingBox, ...]) -> tuple[int, ...]:
        """Track detections, treating an empty tuple as no observation.

        Call :meth:`observe` when inference explicitly observed an empty scene;
        call :meth:`coast` when no inference result exists for the frame.
        ``update(())`` remains the neutral compatibility surface so callers
        cannot manufacture misses from an absent result.
        """
        if not boxes:
            self.coast()
            return ()
        return self.observe(boxes)

    def observe(self, boxes: tuple[BoundingBox, ...]) -> tuple[int, ...]:
        """Apply one actual inference observation, including an empty one."""
        existing = list(self._tracks)
        matches = greedy_match(
            tuple(track.last_box for track in existing),
            boxes,
            self._min_iou,
        )
        matched_track_indices: set[int] = set()
        matched_box_indices: set[int] = set()
        box_to_track_id: dict[int, int] = {}

        for track_index, box_index in matches:
            matched_track_indices.add(track_index)
            matched_box_indices.add(box_index)
            track = existing[track_index]
            track.last_box = boxes[box_index]
            track.misses = 0
            box_to_track_id[box_index] = track.track_id

        new_tracks: list[_Track] = []
        for box_index, box in enumerate(boxes):
            if box_index in matched_box_indices:
                continue
            track_id = self._next_id
            self._next_id += 1
            box_to_track_id[box_index] = track_id
            new_tracks.append(_Track(track_id, box))

        surviving_tracks: list[_Track] = []
        for track_index, track in enumerate(existing):
            if track_index in matched_track_indices:
                surviving_tracks.append(track)
                continue
            track.misses += 1
            if track.misses <= self._max_misses:
                surviving_tracks.append(track)

        self._tracks = surviving_tracks + new_tracks
        return tuple(box_to_track_id[box_index] for box_index in range(len(boxes)))

    def coast(self) -> None:
        """Preserve all tracks when this frame carried no inference result."""


__all__ = ["GreedyIouTracker", "iou"]
