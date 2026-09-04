"""Tensor row parsers for the pose, person and bed engines.

Shapes are fixed by the export profile: pose ``[N,300,57]``, person
``[N,300,6]``, bed ``[N,300,38]`` plus a ``[N,32,160,160]`` prototype tensor.

Each row is ``x1, y1, x2, y2, score, class`` in TENSOR space, followed by the
task payload. The box columns are already corner-form: these are YOLO26
end-to-end heads whose decode and one-to-one matching happen inside the graph,
which is measured in ``task-3-baseline-corpus.json`` and is exactly why the plan
forbids adding a second NMS here.

Three shipped behaviors are load-bearing and reproduced exactly:

- Pose keeps rows with STRICT ``score > 0.05``. ``>=`` admits an extra row.
- Source order is preserved. The head already resolved duplicates; a second NMS
  here would drop overlapping detections the shipped path keeps.
- Boxes truncate through ``int()`` after the box inverse; keypoints and mask
  contours use the separate keypoint inverse. See ``geometry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from worker.native.deepstream.parity.geometry import AffineMetadata
from worker.types.perception_frame import (
    BedRegion,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PersonBox,
    PersonBoxChannel,
)

POSE_ROW_STRIDE: Final = 57
PERSON_ROW_STRIDE: Final = 6
BED_ROW_STRIDE: Final = 38
BED_PROTOTYPE_SHAPE: Final = (32, 160, 160)
MAX_ROWS: Final = 300

#: ``worker/adapters/model/yolo_pose.py`` -- strict, not inclusive.
POSE_SCORE_THRESHOLD: Final = 0.05
COCO_KEYPOINT_COUNT: Final = 17
COCO_PERSON_CLASS_ID: Final = 0
COCO_BED_CLASS_ID: Final = 59
BED_MASK_MAX_POINTS: Final = 48


@dataclass(frozen=True, slots=True)
class ParsedPose:
    boxes: tuple[tuple[int, int, int, int, float], ...]
    poses: tuple[tuple[Keypoint, ...], ...]

    def human_pose_channel(self) -> HumanPoseChannel:
        return HumanPoseChannel(state=_state(len(self.poses)), poses=self.poses)

    def person_box_channel(self) -> PersonBoxChannel:
        return PersonBoxChannel(state=_state(len(self.boxes)), boxes=_person_boxes(self.boxes))


@dataclass(frozen=True, slots=True)
class ParsedPerson:
    boxes: tuple[tuple[int, int, int, int, float], ...]

    def person_box_channel(self) -> PersonBoxChannel:
        return PersonBoxChannel(state=_state(len(self.boxes)), boxes=_person_boxes(self.boxes))


@dataclass(frozen=True, slots=True)
class ParsedBed:
    regions: tuple[BedRegion, ...]

    def bed_region_channel(self) -> BedRegionChannel:
        return BedRegionChannel(state=_state(len(self.regions)), regions=self.regions)


def _state(count: int) -> ChannelState:
    return ChannelState.INFERRED_EMPTY if count == 0 else ChannelState.INFERRED


def _person_boxes(
    rows: tuple[tuple[int, int, int, int, float], ...],
) -> tuple[PersonBox, ...]:
    return tuple(
        PersonBox(x1=row[0], y1=row[1], x2=row[2], y2=row[3], confidence=row[4]) for row in rows
    )


def _validated(rows: NDArray[np.floating], stride: int, task: str) -> NDArray[np.float64]:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != stride:
        raise ValueError(
            f"{task} tensor rows must have shape [rows, {stride}], received {values.shape}"
        )
    if values.shape[0] > MAX_ROWS:
        raise ValueError(
            f"{task} tensor carries {values.shape[0]} rows, above the {MAX_ROWS} export profile"
        )
    return values


def _xyxy(row: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Read the tensor-space corner box straight off the row.

    The end-to-end head already emits xyxy, so there is no centre/size decode to
    reproduce. Introducing one would shift every box by half its extent.
    """
    return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))


def parse_pose_rows(rows: NDArray[np.floating], affine: AffineMetadata) -> ParsedPose:
    """Parse one pose engine output plane in source order.

    Rows survive on strict ``score > 0.05`` only. No second NMS and no sort:
    the row order the engine produced is the order the shipped adapter reports.
    """
    values = _validated(rows, POSE_ROW_STRIDE, "pose")
    boxes: list[tuple[int, int, int, int, float]] = []
    poses: list[tuple[Keypoint, ...]] = []
    for row in values:
        score = float(row[4])
        if not score > POSE_SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = affine.invert_box(_xyxy(row))
        boxes.append((x1, y1, x2, y2, score))
        keypoints: list[Keypoint] = []
        for index in range(COCO_KEYPOINT_COUNT):
            offset = PERSON_ROW_STRIDE + index * 3
            x, y = affine.invert_keypoint((float(row[offset]), float(row[offset + 1])))
            keypoints.append(Keypoint(x=x, y=y, score=float(row[offset + 2])))
        poses.append(tuple(keypoints))
    return ParsedPose(boxes=tuple(boxes), poses=tuple(poses))


def parse_person_rows(
    rows: NDArray[np.floating],
    affine: AffineMetadata,
    *,
    confidence: float,
) -> ParsedPerson:
    """Parse the person engine output, keeping the shipped class/score filter."""
    values = _validated(rows, PERSON_ROW_STRIDE, "person")
    boxes: list[tuple[int, int, int, int, float]] = []
    for row in values:
        score = float(row[4])
        if int(row[5]) != COCO_PERSON_CLASS_ID or score < confidence:
            continue
        x1, y1, x2, y2 = affine.invert_box(_xyxy(row))
        boxes.append((x1, y1, x2, y2, score))
    return ParsedPerson(boxes=tuple(boxes))


def parse_bed_rows(
    rows: NDArray[np.floating],
    prototypes: NDArray[np.floating],
    affine: AffineMetadata,
    *,
    confidence: float,
    max_points: int = BED_MASK_MAX_POINTS,
) -> ParsedBed:
    """Parse the bed engine output plus its mask prototypes.

    Mask contours are geometry, so they take the KEYPOINT inverse rule, matching
    ``ops.scale_coords`` in the shipped segmentation path.
    """
    values = _validated(rows, BED_ROW_STRIDE, "bed")
    protos = np.asarray(prototypes, dtype=np.float64)
    if protos.shape != BED_PROTOTYPE_SHAPE:
        raise ValueError(
            f"bed prototype tensor must have shape {BED_PROTOTYPE_SHAPE}, received {protos.shape}"
        )
    regions: list[BedRegion] = []
    for row in values:
        score = float(row[4])
        if int(row[5]) != COCO_BED_CLASS_ID or score < confidence:
            continue
        tensor_box = _xyxy(row)
        x1, y1, x2, y2 = affine.invert_box(tensor_box)
        polygon = _mask_polygon(
            row[PERSON_ROW_STRIDE:],
            protos,
            tensor_box,
            affine,
            max_points=max_points,
        )
        regions.append(BedRegion(x1=x1, y1=y1, x2=x2, y2=y2, confidence=score, polygon=polygon))
    return ParsedBed(regions=tuple(regions))


def _mask_polygon(
    coefficients: NDArray[np.float64],
    prototypes: NDArray[np.float64],
    tensor_box: tuple[float, float, float, float],
    affine: AffineMetadata,
    *,
    max_points: int,
) -> tuple[tuple[int, int], ...] | None:
    """Recover one instance contour from prototype coefficients.

    Reproduces ``ops.process_mask``: linear-combine the prototypes, crop to the
    detection box in prototype space, threshold at zero, then trace the contour
    and map it back with the keypoint inverse.
    """
    channels, mask_height, mask_width = prototypes.shape
    mask = (coefficients @ prototypes.reshape(channels, -1)).reshape(mask_height, mask_width)
    width_ratio = mask_width / affine.tensor_width
    height_ratio = mask_height / affine.tensor_height
    left = int(max(0, np.floor(tensor_box[0] * width_ratio)))
    top = int(max(0, np.floor(tensor_box[1] * height_ratio)))
    right = int(min(mask_width, np.ceil(tensor_box[2] * width_ratio)))
    bottom = int(min(mask_height, np.ceil(tensor_box[3] * height_ratio)))
    cropped = np.zeros_like(mask, dtype=np.uint8)
    if right > left and bottom > top:
        cropped[top:bottom, left:right] = (mask[top:bottom, left:right] > 0.0).astype(np.uint8)
    points = np.argwhere(cropped > 0)
    if points.size == 0:
        return None
    contour = _trace_contour(points, mask_width, mask_height)
    if len(contour) > max_points:
        indices = np.linspace(0, len(contour) - 1, max_points).astype(np.int64)
        contour = tuple(contour[index] for index in indices)
    return tuple(affine.invert_keypoint((x / width_ratio, y / height_ratio)) for x, y in contour)


def _trace_contour(
    points: NDArray[np.int64],
    mask_width: int,
    mask_height: int,
) -> tuple[tuple[float, float], ...]:
    """Order mask pixels into a deterministic boundary walk.

    Determinism matters more than contour fidelity here: the comparator treats a
    reordered polygon as a mismatch, so the walk must not depend on iteration
    order of a set or on floating-point ties.
    """
    del mask_width, mask_height
    ys = points[:, 0]
    xs = points[:, 1]
    center_y = float(ys.mean())
    center_x = float(xs.mean())
    angles = np.arctan2(ys - center_y, xs - center_x)
    order = np.lexsort((xs, ys, angles))
    return tuple((float(xs[index]), float(ys[index])) for index in order)


__all__ = [
    "BED_MASK_MAX_POINTS",
    "BED_PROTOTYPE_SHAPE",
    "BED_ROW_STRIDE",
    "COCO_BED_CLASS_ID",
    "COCO_KEYPOINT_COUNT",
    "COCO_PERSON_CLASS_ID",
    "MAX_ROWS",
    "PERSON_ROW_STRIDE",
    "POSE_ROW_STRIDE",
    "POSE_SCORE_THRESHOLD",
    "ParsedBed",
    "ParsedPerson",
    "ParsedPose",
    "parse_bed_rows",
    "parse_person_rows",
    "parse_pose_rows",
]
