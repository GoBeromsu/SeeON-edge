from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias, TypeVar

from contracts.observation import BoundingBox, DetectionLabel, DetectionResult, FrameObservation

RawBox: TypeAlias = tuple[int, int, int, int, float]
Pose: TypeAlias = tuple[tuple[int, int, float], ...]
BedStatus = TypeVar("BedStatus")


def observation_from_detection_result(result: DetectionResult) -> FrameObservation:
    """Convert an authoritative detection result without changing payload order."""
    return FrameObservation.from_detection_result(result)


def build_frame_observation(
    *,
    detections: DetectionResult | None = None,
    boxes: Iterable[BoundingBox] | None = None,
    labels: Iterable[DetectionLabel] | None = None,
    raw_boxes: Iterable[RawBox] | None = None,
    poses: Iterable[Sequence[tuple[int, int, float]]] | None = None,
    bed_boxes: Iterable[BoundingBox] | None = None,
    bed_exit_statuses: Iterable[BedStatus] | None = None,
    track_ids: Iterable[int | None] | None = None,
) -> FrameObservation:
    """Build one numeric observation while preserving every iterable's index order."""
    base = (
        FrameObservation.from_detection_result(detections)
        if detections is not None
        else FrameObservation()
    )
    observation_boxes = tuple(boxes) if boxes is not None else base.boxes
    if raw_boxes is not None:
        observation_boxes = tuple(
            BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
            for x1, y1, x2, y2, confidence in raw_boxes
        )
    return FrameObservation(
        detections=(
            observation_boxes,
            tuple(labels) if labels is not None else base.labels,
        ),
        poses=tuple(tuple(pose) for pose in poses) if poses is not None else base.keypoints,
        regions=(
            tuple(bed_boxes) if bed_boxes is not None else base.bed_boxes,
            (tuple(bed_exit_statuses) if bed_exit_statuses is not None else base.bed_exit_statuses),
        ),
        track_ids=tuple(track_ids) if track_ids is not None else base.track_ids,
    )


__all__ = ["build_frame_observation", "observation_from_detection_result"]
