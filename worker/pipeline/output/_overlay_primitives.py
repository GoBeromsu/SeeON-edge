from __future__ import annotations

import math
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.observation import BoundingBox

Color = tuple[int, int, int]

PERSON_COLOR: Final[Color] = (0, 255, 0)
BED_COLOR: Final[Color] = (255, 0, 0)
BED_ROI_TEXT_COLOR: Final[Color] = (255, 255, 0)
BED_EXIT_STATUS_COLOR: Final[Color] = (0, 0, 255)
BED_PRESENT_STATUS_COLOR: Final[Color] = (0, 255, 255)
# bedexit-mode bed outline (teal, BGR) -- distinct from the legacy filled
# BED_COLOR box so the dashed polygon reads as a configured zone, not a
# live detection.
BED_DASHED_COLOR: Final[Color] = (128, 128, 0)
# fall-mode per-track label colors: danger red for FALL, neutral gray for NORMAL.
FALL_LABEL_COLOR: Final[Color] = (0, 0, 255)
NORMAL_LABEL_COLOR: Final[Color] = (180, 180, 180)
POSE_COLOR: Final[Color] = (80, 160, 255)
POSE_DOT_COLOR: Final[Color] = (255, 255, 255)
CAPTION_TEXT_COLOR: Final[Color] = (16, 16, 16)
CAPTION_FONT_SCALE: Final = 0.5
CAPTION_THICKNESS: Final = 1
LABEL_FONT_SCALE: Final = 0.45
MASK_FILL_ALPHA: Final = 0.3
MIN_KEYPOINT_CONFIDENCE: Final = 0.2
DASH_LENGTH: Final = 10
DASH_GAP_LENGTH: Final = 6

POSE_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def draw_box(
    image: NDArray[np.uint8],
    box: BoundingBox,
    color: Color,
    *,
    thickness: int = 2,
) -> None:
    """Draw an axis-aligned detection outline."""
    cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, thickness)


def draw_region(
    image: NDArray[np.uint8],
    box: BoundingBox,
    color: Color,
    *,
    fill: bool = False,
    thickness: int = 2,
) -> None:
    """Draw a segmentation polygon, or a box when no polygon is present."""
    if box.polygon:
        points = np.array(box.polygon, dtype=np.int32).reshape((-1, 1, 2))
        if fill:
            mask = image.copy()
            cv2.fillPoly(mask, [points], color)
            cv2.addWeighted(
                mask,
                MASK_FILL_ALPHA,
                image,
                1 - MASK_FILL_ALPHA,
                0,
                image,
            )
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=color,
            thickness=thickness,
        )
        return
    draw_box(image, box, color, thickness=thickness)


def draw_dashed_region(
    image: NDArray[np.uint8],
    box: BoundingBox,
    color: Color,
    *,
    thickness: int = 2,
    dash_length: int = DASH_LENGTH,
    gap_length: int = DASH_GAP_LENGTH,
) -> None:
    """Draw a dashed polygon outline, or a dashed box when no polygon is present."""
    points = (
        list(box.polygon)
        if box.polygon
        else [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]
    )
    _draw_dashed_polygon(
        image,
        points,
        color,
        thickness=thickness,
        dash_length=dash_length,
        gap_length=gap_length,
    )


def _draw_dashed_polygon(
    image: NDArray[np.uint8],
    points: list[tuple[int, int]],
    color: Color,
    *,
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    if len(points) < 2:
        return
    edges = list(zip(points, points[1:] + points[:1], strict=True))
    for start, end in edges:
        _draw_dashed_line(
            image,
            start,
            end,
            color,
            thickness=thickness,
            dash_length=dash_length,
            gap_length=gap_length,
        )


def _draw_dashed_line(
    image: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    color: Color,
    *,
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return
    stride = dash_length + gap_length
    offset = 0.0
    while offset < length:
        dash_end = min(offset + dash_length, length)
        start_point = (
            int(x1 + (x2 - x1) * (offset / length)),
            int(y1 + (y2 - y1) * (offset / length)),
        )
        end_point = (
            int(x1 + (x2 - x1) * (dash_end / length)),
            int(y1 + (y2 - y1) * (dash_end / length)),
        )
        cv2.line(image, start_point, end_point, color, thickness)
        offset += stride


def draw_caption(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: Color,
) -> None:
    """Draw a filled caption chip with contrasting text."""
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        CAPTION_FONT_SCALE,
        CAPTION_THICKNESS,
    )
    top = max(y - text_height - baseline - 2, 0)
    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 2, top + text_height + baseline + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 1, top + text_height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        CAPTION_FONT_SCALE,
        CAPTION_TEXT_COLOR,
        CAPTION_THICKNESS,
        cv2.LINE_AA,
    )


def draw_label(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: Color,
    *,
    scale: float = LABEL_FONT_SCALE,
) -> None:
    """Draw a plain text label."""
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def draw_pose(
    image: NDArray[np.uint8],
    keypoints: tuple[tuple[int, int, float], ...],
    *,
    color: Color = POSE_COLOR,
    dot_radius: int = 3,
    skeleton: bool = True,
    min_confidence: float = MIN_KEYPOINT_CONFIDENCE,
) -> None:
    """Draw confident COCO-17 keypoints and an optional skeleton."""
    if skeleton:
        for start, end in POSE_EDGES:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            start_point = keypoints[start]
            end_point = keypoints[end]
            if start_point[2] < min_confidence or end_point[2] < min_confidence:
                continue
            cv2.line(
                image,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                color,
                2,
            )
    for x, y, confidence in keypoints:
        if confidence >= min_confidence:
            cv2.circle(image, (int(x), int(y)), dot_radius, color, -1)


__all__ = [
    "BED_COLOR",
    "BED_DASHED_COLOR",
    "BED_EXIT_STATUS_COLOR",
    "BED_PRESENT_STATUS_COLOR",
    "BED_ROI_TEXT_COLOR",
    "FALL_LABEL_COLOR",
    "NORMAL_LABEL_COLOR",
    "PERSON_COLOR",
    "POSE_DOT_COLOR",
    "draw_box",
    "draw_caption",
    "draw_dashed_region",
    "draw_label",
    "draw_pose",
    "draw_region",
]
