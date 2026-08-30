from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final, TypeAlias

COCO17_KEYPOINTS: Final = 17
POSE_BBOX56_DIM: Final = 56
POSE_BBOX56_CONFIDENCE_GATE: Final = 0.5
POSE_BBOX56_PREPROCESSING_IDENTITY: Final = "coco17-xyc-plus-pose-head-xyxy-valid-f32-v1"

Keypoint: TypeAlias = tuple[float, float, float]
PoseBbox: TypeAlias = tuple[float, float, float, float]
PoseBbox56Row: TypeAlias = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PoseBbox56Track:
    """One pose-head observation, deliberately excluding pixel data."""

    track_id: str | int
    keypoints: Sequence[Sequence[float]]
    bbox: Sequence[float] | None


def pose_bbox56_row(
    keypoints: Sequence[Sequence[float]],
    bbox: Sequence[float] | None,
    frame_width: int,
    frame_height: int,
) -> PoseBbox56Row:
    """Encode the G001 COCO-17 + pose-head box feature row as IEEE float32."""
    zero = _zero_row()
    if frame_width <= 0 or frame_height <= 0 or bbox is None:
        return zero
    if len(keypoints) != COCO17_KEYPOINTS or len(bbox) != 4:
        return zero
    try:
        if any(
            len(point) != 3
            or any(not isinstance(value, Real) or isinstance(value, bool) for value in point)
            for point in keypoints
        ):
            return zero
        values = tuple(float(value) for point in keypoints for value in point)
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return zero
    if len(values) != COCO17_KEYPOINTS * 3 or not all(
        math.isfinite(value) for value in (*values, x1, y1, x2, y2)
    ):
        return zero

    max_x = float(frame_width - 1)
    max_y = float(frame_height - 1)
    x1 = min(max(x1, 0.0), max_x)
    x2 = min(max(x2, 0.0), max_x)
    y1 = min(max(y1, 0.0), max_y)
    y2 = min(max(y2, 0.0), max_y)
    if x2 <= x1 or y2 <= y1:
        return zero

    row = [0.0] * POSE_BBOX56_DIM
    for index in range(COCO17_KEYPOINTS):
        x, y, confidence = values[index * 3 : index * 3 + 3]
        if confidence >= POSE_BBOX56_CONFIDENCE_GATE:
            row[index * 3] = min(max(x, 0.0), max_x) / frame_width
            row[index * 3 + 1] = min(max(y, 0.0), max_y) / frame_height
            row[index * 3 + 2] = confidence
    row[51:55] = (
        x1 / frame_width,
        y1 / frame_height,
        x2 / frame_width,
        y2 / frame_height,
    )
    row[55] = 1.0
    try:
        return tuple(_float32(value) for value in row)
    except OverflowError:
        return zero


def pose_bbox56_tracks(
    tracks: Iterable[PoseBbox56Track], frame_width: int, frame_height: int
) -> tuple[tuple[str | int, PoseBbox56Row], ...]:
    """Produce deterministic rows sorted by the stable track identifier."""
    ordered = sorted(tracks, key=lambda track: track.track_id)
    return tuple(
        (track.track_id, pose_bbox56_row(track.keypoints, track.bbox, frame_width, frame_height))
        for track in ordered
    )


def native_pose_bbox56_row(
    keypoints: Sequence[Sequence[float]],
    bbox: Sequence[float] | None,
    frame_width: int,
    frame_height: int,
    *,
    box_source: str,
) -> PoseBbox56Row:
    """Native boundary equivalent; only a pose-head box is authoritative."""
    if box_source != "pose":
        return _zero_row()
    return pose_bbox56_row(keypoints, bbox, frame_width, frame_height)


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _zero_row() -> PoseBbox56Row:
    return tuple(_float32(0.0) for _ in range(POSE_BBOX56_DIM))


__all__ = [
    "COCO17_KEYPOINTS",
    "POSE_BBOX56_CONFIDENCE_GATE",
    "POSE_BBOX56_DIM",
    "POSE_BBOX56_PREPROCESSING_IDENTITY",
    "PoseBbox56Row",
    "PoseBbox56Track",
    "native_pose_bbox56_row",
    "pose_bbox56_row",
    "pose_bbox56_tracks",
]
