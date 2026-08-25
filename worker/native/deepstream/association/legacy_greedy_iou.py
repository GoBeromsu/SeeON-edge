"""``legacy-greedy-bbox-iou.v1``: the native reference for the shipped tracker.

This is the executable specification the custom `GstBaseTransform` association
stage implements. It targets the exact observable behavior of
`worker/pipeline/perception/tracker.py`'s `GreedyIouTracker` -- same greedy
descending-IoU match order, same tie rule (a tie keeps the lower existing-track
index), same `inferred`/`inferred_empty` vs `skipped` split (an empty
person-box observation still counts a miss; `coast()` never does), same
eviction (`misses > max_misses` drops a track), and the same box-order-
preserving return shape -- but it is an INDEPENDENT implementation: it does
not import `worker.pipeline.perception.features.geometry.greedy_match`, the
exact matching routine the oracle's tracker calls. Sharing that routine would
let an order/tie regression move both the oracle and this candidate together
and leave the differential comparator green on a real drift.

`observe()` takes the caller's `PerceptionFrameIdentity` plus a typed
`PersonBoxChannel` and returns a real C1 `AssociationResult`
(`worker/types/perception_frame.py`) bound to that identity, with
`cue_source` fixed at `"person_box"` and `selected_cue_indexes` in
person-box input order. There is no parameter shape a `BedRegionChannel` can
satisfy here: bed regions are structurally unrepresentable at this boundary,
so they cannot create, update, or evict a person track.

Divergence from the documented behavior is a parity break, not an improvement
-- see `worker/pipeline/perception/tracker.py` module docstring and the
characterization pins in `tests/test_association_tracker_characterization.py`.
"""

from __future__ import annotations

from typing import Final, final

from contracts.observation import BoundingBox
from worker.types.perception_frame import (
    AssociationResult,
    PerceptionFrameIdentity,
    PersonBox,
    PersonBoxChannel,
)

LEGACY_GREEDY_BBOX_IOU_V1: Final = "legacy-greedy-bbox-iou.v1"
DEFAULT_MIN_IOU: Final = 0.3
DEFAULT_MAX_MISSES: Final = 30


def _intersection_over_union(a: BoundingBox, b: BoundingBox) -> float:
    """Independent IoU primitive: no import of the oracle's `iou`/`greedy_match`."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    if inter_width <= 0 or inter_height <= 0:
        return 0.0
    inter_area = inter_width * inter_height

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _own_greedy_match(
    existing_boxes: tuple[BoundingBox, ...],
    boxes: tuple[BoundingBox, ...],
    min_iou: float,
) -> tuple[tuple[int, int], ...]:
    """Own greedy descending-IoU matcher: candidate pairs sorted, taken greedily.

    Same observable contract as the oracle's `greedy_match`: pairs sort by
    descending score, and on an exact tie the pair discovered first (lower
    existing-track index, since track index is the outer loop) wins because
    Python's `sort` is stable. Implemented independently -- no shared call
    into `worker.pipeline.perception.features.geometry`.
    """
    candidates: list[tuple[float, int, int]] = []
    for track_index, existing_box in enumerate(existing_boxes):
        for box_index, box in enumerate(boxes):
            score = _intersection_over_union(existing_box, box)
            if score > 0.0:
                candidates.append((score, track_index, box_index))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    taken_tracks: set[int] = set()
    taken_boxes: set[int] = set()
    matches: list[tuple[int, int]] = []
    for score, track_index, box_index in candidates:
        if score < min_iou:
            break
        if track_index in taken_tracks or box_index in taken_boxes:
            continue
        taken_tracks.add(track_index)
        taken_boxes.add(box_index)
        matches.append((track_index, box_index))
    return tuple(matches)


def _person_box_as_bounding_box(box: PersonBox) -> BoundingBox:
    """Narrow the C1 `PersonBox` cue to the geometry the matcher needs."""
    return BoundingBox(x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, confidence=box.confidence)


@final
class _NativeTrack:
    __slots__: tuple[str, ...] = ("last_box", "misses", "track_id")

    track_id: int
    last_box: BoundingBox
    misses: int

    def __init__(self, track_id: int, last_box: BoundingBox) -> None:
        self.track_id = track_id
        self.last_box = last_box
        self.misses = 0


@final
class LegacyGreedyBboxIouStrategy:
    """Native-side, independently-implemented greedy-IoU association."""

    identity: str = LEGACY_GREEDY_BBOX_IOU_V1

    def __init__(
        self,
        min_iou: float = DEFAULT_MIN_IOU,
        max_misses: int = DEFAULT_MAX_MISSES,
    ) -> None:
        self._min_iou: float = min_iou
        self._max_misses: int = max_misses
        self._tracks: list[_NativeTrack] = []
        self._next_id: int = 0

    @property
    def live_ids(self) -> frozenset[int]:
        return frozenset(track.track_id for track in self._tracks)

    def observe(
        self,
        identity: PerceptionFrameIdentity,
        person_box: PersonBoxChannel,
    ) -> AssociationResult:
        """Associate one frame's person-box cues, returning a real `AssociationResult`.

        `selected_cue_indexes` is every cue index in input order -- this
        strategy uses (and can identify) every person-box cue it receives, so
        the selection is simply `range(len(person_box.boxes))`. `track_ids` is
        parallel to `selected_cue_indexes`, matching `AssociationResult`'s
        contract (`worker/types/perception_frame.py::association_failure`).
        """
        cues = person_box.boxes
        boxes = tuple(_person_box_as_bounding_box(cue) for cue in cues)
        existing = list(self._tracks)
        matches = _own_greedy_match(
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

        new_tracks: list[_NativeTrack] = []
        for box_index, box in enumerate(boxes):
            if box_index in matched_box_indices:
                continue
            track_id = self._next_id
            self._next_id += 1
            box_to_track_id[box_index] = track_id
            new_tracks.append(_NativeTrack(track_id, box))

        surviving_tracks: list[_NativeTrack] = []
        for track_index, track in enumerate(existing):
            if track_index in matched_track_indices:
                surviving_tracks.append(track)
                continue
            track.misses += 1
            if track.misses <= self._max_misses:
                surviving_tracks.append(track)

        self._tracks = surviving_tracks + new_tracks
        selected_cue_indexes = tuple(range(len(cues)))
        track_ids = tuple(box_to_track_id[box_index] for box_index in selected_cue_indexes)
        return AssociationResult(
            strategy=self.identity,
            track_ids=track_ids,
            selected_cue_indexes=selected_cue_indexes,
            identity=identity,
            live_track_ids=tuple(sorted(self.live_ids)),
        )

    def coast(self) -> None:
        """Preserve all tracks when this frame carried no inference result."""

    def reset(self) -> None:
        """Drop all track state and restart durable id minting from zero.

        Called on reconnect or a rolled stream epoch (Task 4 guardrail): the
        next `observe()` after `reset()` must not be able to reuse or resume
        a track id minted before this boot/epoch.
        """
        self._tracks = []
        self._next_id = 0


__all__ = [
    "DEFAULT_MAX_MISSES",
    "DEFAULT_MIN_IOU",
    "LEGACY_GREEDY_BBOX_IOU_V1",
    "LegacyGreedyBboxIouStrategy",
]
