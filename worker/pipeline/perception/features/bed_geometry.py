"""Per-frame bed-relative pose scalars, computed in perception.

Numpy is allowed here. The result is a ``BedPoseFeatures`` of plain Python
scalars so ``worker.domains`` never has to import this module or numpy.

Coordinate frames: a persisted ``bed_zone_polygon`` lives in
``bed_zone_image_width`` x ``bed_zone_image_height`` space, while keypoints
arrive in ``frame_width`` x ``frame_height``. Those sizes are not guaranteed
to match. This producer always scales the polygon into frame space before
any inside/distance measurement. Live-segmentation polygons (no source size)
are already in frame space and are left unscaled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from contracts.model import DEFAULT_FALL_CONFIDENCE_THRESHOLD
from contracts.observation import BoundingBox, FrameObservation
from worker.types.bed_pose_features import (
    EMPTY_FRAME_BED_POSE_FEATURES,
    BedPoseFeatures,
    FrameBedPoseFeatures,
)

_KEYPOINT_COUNT: Final = 17
_LEFT_SHOULDER: Final = 5
_RIGHT_SHOULDER: Final = 6
_LEFT_HIP: Final = 11
_RIGHT_HIP: Final = 12
_LEFT_KNEE: Final = 13
_RIGHT_KNEE: Final = 14
_LEFT_ANKLE: Final = 15
_RIGHT_ANKLE: Final = 16
_TORSO_INDICES: Final = (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
_LOWER_INDICES: Final = (_LEFT_KNEE, _RIGHT_KNEE, _LEFT_ANKLE, _RIGHT_ANKLE)
_CONF_GATE: Final = DEFAULT_FALL_CONFIDENCE_THRESHOLD
_ZERO_AREA: Final = 1e-9

Keypoint = tuple[int, int, float]
Pose = Sequence[Keypoint]


def compute_frame_bed_pose_features(
    observation: FrameObservation,
    *,
    frame_width: int,
    frame_height: int,
    polygon_image_width: int | None = None,
    polygon_image_height: int | None = None,
) -> FrameBedPoseFeatures:
    """Compute :class:`BedPoseFeatures` for every tracked pose on this frame."""
    poses = observation.poses
    if not poses:
        return EMPTY_FRAME_BED_POSE_FEATURES
    track_ids = observation.track_ids
    items: list[BedPoseFeatures] = []
    for index, pose in enumerate(poses):
        track_id = track_ids[index] if index < len(track_ids) else index
        if track_id is None:
            continue
        items.append(
            compute_bed_pose_features(
                track_id=int(track_id),
                pose=pose,
                bed_boxes=observation.bed_boxes,
                frame_width=frame_width,
                frame_height=frame_height,
                polygon_image_width=polygon_image_width,
                polygon_image_height=polygon_image_height,
            )
        )
    return FrameBedPoseFeatures(items=tuple(items)) if items else EMPTY_FRAME_BED_POSE_FEATURES


def compute_bed_pose_features(
    *,
    track_id: int,
    pose: Pose,
    bed_boxes: Sequence[BoundingBox],
    frame_width: int,
    frame_height: int,
    polygon_image_width: int | None = None,
    polygon_image_height: int | None = None,
) -> BedPoseFeatures:
    """Compute one track's bed-relation scalars from COCO-17 keypoints."""
    valid, points = _valid_keypoints(pose)
    observability = float(int(valid.sum())) / float(_KEYPOINT_COUNT)
    torso_angle = _torso_angle(valid, points)

    candidates: list[BedPoseFeatures] = []
    for bed_id, box in enumerate(bed_boxes):
        polygon = _scaled_usable_polygon(
            box,
            frame_width=frame_width,
            frame_height=frame_height,
            polygon_image_width=polygon_image_width,
            polygon_image_height=polygon_image_height,
        )
        if polygon is None:
            continue
        candidates.append(
            _against_polygon(
                track_id=track_id,
                bed_id=bed_id,
                valid=valid,
                points=points,
                polygon=polygon,
                observability=observability,
                torso_angle=torso_angle,
            )
        )
    if not candidates:
        return _missing_bed(track_id, observability, torso_angle)
    return max(
        candidates,
        key=lambda item: (
            item.torso_in_frac,
            item.keypoint_in_frac,
            item.hip_depth,
            -int(item.bed_id or 0),
        ),
    )


def _valid_keypoints(pose: Pose) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    points = np.full((_KEYPOINT_COUNT, 2), np.nan, dtype=np.float64)
    valid = np.zeros(_KEYPOINT_COUNT, dtype=bool)
    for index, keypoint in enumerate(pose):
        if index >= _KEYPOINT_COUNT:
            break
        x_coordinate, y_coordinate, confidence = keypoint
        if float(confidence) < _CONF_GATE:
            continue
        points[index, 0] = float(x_coordinate)
        points[index, 1] = float(y_coordinate)
        valid[index] = True
    return valid, points


def _torso_angle(valid: NDArray[np.bool_], points: NDArray[np.float64]) -> float:
    """Angle of hip→shoulder from image-horizontal, radians in ``[0, π/2]``.

    Lying along the image x-axis is ~0; sitting/standing (torso along y) is
    ~π/2. Matches the todo-7 bands (in-bed ≤ 0.61, sitting-up > 0.61,
    edge-sitting > 0.87). Missing shoulders or hips yield 0.0.
    """
    shoulder = _midpoint(valid, points, (_LEFT_SHOULDER, _RIGHT_SHOULDER))
    hip = _midpoint(valid, points, (_LEFT_HIP, _RIGHT_HIP))
    if shoulder is None or hip is None:
        return 0.0
    delta_x = abs(shoulder[0] - hip[0])
    delta_y = abs(shoulder[1] - hip[1])
    if delta_x == 0.0 and delta_y == 0.0:
        return 0.0
    return float(math.atan2(delta_y, delta_x))


def _midpoint(
    valid: NDArray[np.bool_],
    points: NDArray[np.float64],
    indices: tuple[int, ...],
) -> tuple[float, float] | None:
    selected = [index for index in indices if valid[index]]
    if not selected:
        return None
    chosen = points[selected]
    return float(chosen[:, 0].mean()), float(chosen[:, 1].mean())


def _scaled_usable_polygon(
    box: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
    polygon_image_width: int | None,
    polygon_image_height: int | None,
) -> NDArray[np.float64] | None:
    source = _polygon_vertices(box)
    if source is None:
        return None
    scaled = _scale_into_frame(
        source,
        frame_width=frame_width,
        frame_height=frame_height,
        polygon_image_width=polygon_image_width,
        polygon_image_height=polygon_image_height,
    )
    if scaled is None or not _polygon_usable(scaled):
        return None
    return scaled


def _polygon_vertices(box: object) -> NDArray[np.float64] | None:
    if not isinstance(box, BoundingBox):
        return None
    if box.polygon is not None:
        if len(box.polygon) < 3:
            return None
        return np.asarray(box.polygon, dtype=np.float64)
    if box.x2 > box.x1 and box.y2 > box.y1:
        return np.asarray(
            (
                (box.x1, box.y1),
                (box.x2, box.y1),
                (box.x2, box.y2),
                (box.x1, box.y2),
            ),
            dtype=np.float64,
        )
    return None


def _scale_into_frame(
    polygon: NDArray[np.float64],
    *,
    frame_width: int,
    frame_height: int,
    polygon_image_width: int | None,
    polygon_image_height: int | None,
) -> NDArray[np.float64] | None:
    if polygon_image_width is None or polygon_image_height is None:
        return polygon
    if polygon_image_width <= 0 or polygon_image_height <= 0:
        return None
    if frame_width <= 0 or frame_height <= 0:
        return None
    scale = np.asarray(
        (
            float(frame_width) / float(polygon_image_width),
            float(frame_height) / float(polygon_image_height),
        ),
        dtype=np.float64,
    )
    return polygon * scale


def _polygon_usable(polygon: NDArray[np.float64]) -> bool:
    if len(polygon) < 3:
        return False
    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    span = max_xy - min_xy
    if float(span[0]) <= 0.0 or float(span[1]) <= 0.0:
        return False
    if _all_collinear(polygon):
        return False
    return abs(_shoelace_area(polygon)) > _ZERO_AREA


def _all_collinear(polygon: NDArray[np.float64]) -> bool:
    origin = polygon[0]
    direction = None
    for point in polygon[1:]:
        candidate = point - origin
        if float(np.dot(candidate, candidate)) > 0.0:
            direction = candidate
            break
    if direction is None:
        return True
    crosses = (polygon[:, 0] - origin[0]) * direction[1] - (polygon[:, 1] - origin[1]) * direction[
        0
    ]
    return bool(np.all(np.abs(crosses) <= _ZERO_AREA))


def _shoelace_area(polygon: NDArray[np.float64]) -> float:
    rolled = np.roll(polygon, -1, axis=0)
    return 0.5 * float(np.dot(polygon[:, 0], rolled[:, 1]) - np.dot(polygon[:, 1], rolled[:, 0]))


def _against_polygon(
    *,
    track_id: int,
    bed_id: int,
    valid: NDArray[np.bool_],
    points: NDArray[np.float64],
    polygon: NDArray[np.float64],
    observability: float,
    torso_angle: float,
) -> BedPoseFeatures:
    inside = np.zeros(_KEYPOINT_COUNT, dtype=bool)
    if valid.any():
        inside[valid] = _points_inside(points[valid], polygon)
    torso_in_frac = _group_fraction(valid, inside, _TORSO_INDICES)
    lower_in_frac = _group_fraction(valid, inside, _LOWER_INDICES)
    keypoint_in_frac = _group_fraction(valid, inside, tuple(range(_KEYPOINT_COUNT)))

    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    span = max_xy - min_xy
    diagonal = float(np.linalg.norm(span))
    hip = _midpoint(valid, points, (_LEFT_HIP, _RIGHT_HIP))
    if hip is None or diagonal <= 0.0:
        hip_depth = 0.0
        hip_x_rel = 0.0
        hip_y_rel = 0.0
    else:
        distance = _min_edge_distance(hip[0], hip[1], polygon)
        sign = 1.0 if _points_inside(np.asarray([hip], dtype=np.float64), polygon)[0] else -1.0
        hip_depth = sign * distance / diagonal
        hip_x_rel = (hip[0] - float(min_xy[0])) / float(span[0]) if float(span[0]) > 0.0 else 0.0
        hip_y_rel = (hip[1] - float(min_xy[1])) / float(span[1]) if float(span[1]) > 0.0 else 0.0

    if (not valid.any()) or diagonal <= 0.0:
        centroid_displacement = 0.0
    else:
        centroid = points[valid].mean(axis=0)
        bed_center = (min_xy + max_xy) / 2.0
        centroid_displacement = float(np.linalg.norm(centroid - bed_center)) / diagonal

    return BedPoseFeatures(
        track_id=track_id,
        bed_id=bed_id,
        torso_in_frac=float(torso_in_frac),
        lower_in_frac=float(lower_in_frac),
        keypoint_in_frac=float(keypoint_in_frac),
        hip_depth=float(hip_depth),
        torso_angle=float(torso_angle),
        centroid_displacement=float(centroid_displacement),
        hip_x_rel=float(hip_x_rel),
        hip_y_rel=float(hip_y_rel),
        observability=float(observability),
        bed_polygon_valid=True,
    )


def _group_fraction(
    valid: NDArray[np.bool_],
    inside: NDArray[np.bool_],
    indices: tuple[int, ...],
) -> float:
    usable = [index for index in indices if valid[index]]
    if not usable:
        return 0.0
    return float(sum(1 for index in usable if inside[index])) / float(len(usable))


def _points_inside(points: NDArray[np.float64], polygon: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Even-odd inclusion for ``points`` shape ``(N, 2)``."""
    x_coordinates = points[:, 0]
    y_coordinates = points[:, 1]
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(polygon[:, 0], -1)
    y1 = np.roll(polygon[:, 1], -1)
    crosses_vertical = (y0 > y_coordinates[:, None]) != (y1 > y_coordinates[:, None])
    denom = y1 - y0
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = (x1 - x0) * (y_coordinates[:, None] - y0) / denom + x0
    crosses = crosses_vertical & (x_coordinates[:, None] < x_intersect)
    return np.asarray(crosses.sum(axis=1) % 2 == 1)


def _min_edge_distance(
    x_coordinate: float,
    y_coordinate: float,
    polygon: NDArray[np.float64],
) -> float:
    point = np.asarray((x_coordinate, y_coordinate), dtype=np.float64)
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segments = ends - starts
    lengths_sq = np.sum(segments * segments, axis=1)
    offsets = point - starts
    with np.errstate(divide="ignore", invalid="ignore"):
        t_param = np.where(
            lengths_sq > 0.0,
            np.clip(np.sum(offsets * segments, axis=1) / lengths_sq, 0.0, 1.0),
            0.0,
        )
    closest = starts + t_param[:, None] * segments
    return float(np.linalg.norm(closest - point, axis=1).min())


def _missing_bed(track_id: int, observability: float, torso_angle: float) -> BedPoseFeatures:
    return BedPoseFeatures(
        track_id=track_id,
        bed_id=None,
        torso_in_frac=0.0,
        lower_in_frac=0.0,
        keypoint_in_frac=0.0,
        hip_depth=0.0,
        torso_angle=float(torso_angle),
        centroid_displacement=0.0,
        hip_x_rel=0.0,
        hip_y_rel=0.0,
        observability=float(observability),
        bed_polygon_valid=False,
    )


__all__ = ["compute_bed_pose_features", "compute_frame_bed_pose_features"]
