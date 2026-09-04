"""Numpy decoding for the fixed-shape YOLO26 end-to-end segmentation head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

COCO_BED_CLASS_ID: Final = 59
MODEL_SIZE: Final = 640
PROTOTYPE_CHANNELS: Final = 32
MASK_THRESHOLD: Final = 0.5
BedPolygon: TypeAlias = tuple[tuple[int, int], ...]
BedInstance: TypeAlias = tuple[int, int, int, int, float, BedPolygon]


@dataclass(frozen=True, slots=True)
class Letterbox:
    source_height: int
    source_width: int
    scale: float
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int


def letterbox_rgb(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], Letterbox]:
    """Return YOLO's square RGB tensor and its reversible letterbox metadata."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("bed image must be an HxWx3 uint8 RGB array")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("bed image geometry must be positive")
    scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    total_pad_x = MODEL_SIZE - resized_width
    total_pad_y = MODEL_SIZE - resized_height
    pad_left = round(total_pad_x / 2 - 0.1)
    pad_top = round(total_pad_y / 2 - 0.1)
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = _resize_bilinear(image, resized_height, resized_width).astype(np.uint8)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
    tensor /= 255.0
    return tensor, Letterbox(
        source_height=height,
        source_width=width,
        scale=scale,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_left=pad_left,
    )


def decode_end_to_end_segmentation(
    detections: object,
    prototypes: object,
    letterbox: Letterbox,
    *,
    confidence: float,
    max_points: int,
    bed_class_id: int = COCO_BED_CLASS_ID,
) -> tuple[BedInstance, ...]:
    """Decode end-to-end (already NMS-resolved) YOLO26 segmentation outputs."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    rows = np.asarray(detections, dtype=np.float32)
    protos = np.asarray(prototypes, dtype=np.float32)
    if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[2] != 6 + PROTOTYPE_CHANNELS:
        raise ValueError("YOLO26 segmentation detections must have shape (1, N, 38)")
    if protos.shape != (1, PROTOTYPE_CHANNELS, 160, 160):
        raise ValueError("YOLO26 segmentation prototypes must have shape (1, 32, 160, 160)")
    if not np.isfinite(rows).all() or not np.isfinite(protos).all():
        raise ValueError("YOLO26 segmentation outputs must be finite")

    flattened_protos = protos[0].reshape(PROTOTYPE_CHANNELS, -1)
    instances: list[BedInstance] = []
    for row in rows[0]:
        x1, y1, x2, y2, score, class_id = (float(value) for value in row[:6])
        if int(class_id) != bed_class_id or score < confidence or x2 <= x1 or y2 <= y1:
            continue
        mask = _sigmoid((row[6:] @ flattened_protos).reshape(160, 160))
        mask = _crop_mask(mask, (x1, y1, x2, y2))
        mask = _unletterbox_mask(_resize_bilinear(mask, MODEL_SIZE, MODEL_SIZE), letterbox)
        polygon = simplify_polygon(largest_external_contour(mask > MASK_THRESHOLD), max_points)
        instances.append(
            (
                _inverse_x(x1, letterbox),
                _inverse_y(y1, letterbox),
                _inverse_x(x2, letterbox),
                _inverse_y(y2, letterbox),
                score,
                polygon,
            )
        )
    return tuple(instances)


def largest_external_contour(mask: NDArray[np.bool_]) -> BedPolygon:
    """Trace the largest clockwise boundary loop from a binary pixel mask."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not values.any():
        return ()
    height, width = values.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(start: tuple[int, int], end: tuple[int, int]) -> None:
        edges.setdefault(start, []).append(end)

    for y, x in np.argwhere(values):
        if y == 0 or not values[y - 1, x]:
            add((int(x), int(y)), (int(x + 1), int(y)))
        if x == width - 1 or not values[y, x + 1]:
            add((int(x + 1), int(y)), (int(x + 1), int(y + 1)))
        if y == height - 1 or not values[y + 1, x]:
            add((int(x + 1), int(y + 1)), (int(x), int(y + 1)))
        if x == 0 or not values[y, x - 1]:
            add((int(x), int(y + 1)), (int(x), int(y)))

    loops: list[BedPolygon] = []
    while edges:
        start = next(iter(edges))
        current = start
        loop: list[tuple[int, int]] = []
        while True:
            loop.append(current)
            ends = edges.get(current)
            if not ends:
                break
            next_point = ends.pop()
            if not ends:
                del edges[current]
            current = next_point
            if current == start:
                loops.append(tuple(loop))
                break
    return max(loops, key=_area, default=())


def simplify_polygon(points: BedPolygon, max_points: int) -> BedPolygon:
    """Simplify a closed contour with Douglas-Peucker, respecting point capacity."""
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return points
    values = np.asarray(points, dtype=np.float64)
    first, second = _furthest_pair(values)
    chain_one = _ring_chain(values, first, second)
    chain_two = _ring_chain(values, second, first)
    epsilon_low, epsilon_high = 0.0, float(max(np.ptp(values, axis=0)))
    best = values
    for _ in range(32):
        candidate = np.vstack(
            (
                _douglas_peucker(chain_one, (epsilon_low + epsilon_high) / 2)[:-1],
                _douglas_peucker(chain_two, (epsilon_low + epsilon_high) / 2)[:-1],
            )
        )
        if len(candidate) > max_points:
            epsilon_low = (epsilon_low + epsilon_high) / 2
        else:
            best = candidate
            epsilon_high = (epsilon_low + epsilon_high) / 2
    if len(best) > max_points:
        indices = np.linspace(0, len(best) - 1, max_points).round().astype(np.intp)
        best = best[indices]
    return tuple((round(float(x)), round(float(y))) for x, y in best)


def _crop_mask(
    mask: NDArray[np.float32], box: tuple[float, float, float, float]
) -> NDArray[np.float32]:
    height, width = mask.shape
    x1, y1, x2, y2 = (value / MODEL_SIZE for value in box)
    x = np.arange(width, dtype=np.float32)[np.newaxis, :]
    y = np.arange(height, dtype=np.float32)[:, np.newaxis]
    return np.where(
        (x >= x1 * width) & (x < x2 * width) & (y >= y1 * height) & (y < y2 * height),
        mask,
        0.0,
    )


def _sigmoid(values: NDArray[np.float32]) -> NDArray[np.float32]:
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _unletterbox_mask(
    mask_logits: NDArray[np.float32], letterbox: Letterbox
) -> NDArray[np.float32]:
    content = mask_logits[
        letterbox.pad_top : letterbox.pad_top + letterbox.resized_height,
        letterbox.pad_left : letterbox.pad_left + letterbox.resized_width,
    ]
    return _resize_bilinear(content, letterbox.source_height, letterbox.source_width)


def _inverse_x(value: float, letterbox: Letterbox) -> int:
    return int(np.clip((value - letterbox.pad_left) / letterbox.scale, 0, letterbox.source_width))


def _inverse_y(value: float, letterbox: Letterbox) -> int:
    return int(np.clip((value - letterbox.pad_top) / letterbox.scale, 0, letterbox.source_height))


def _resize_bilinear(values: NDArray[np.generic], height: int, width: int) -> NDArray[np.float32]:
    if values.shape[:2] == (height, width):
        return np.asarray(values, dtype=np.float32)
    source_height, source_width = values.shape[:2]
    y = np.clip(
        (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5,
        0,
        source_height - 1,
    )
    x = np.clip(
        (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5, 0, source_width - 1
    )
    y0, x0 = np.floor(y).astype(np.intp), np.floor(x).astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, source_height - 1), np.minimum(x0 + 1, source_width - 1)
    wy, wx = y - y0, x - x0
    trailing = (1,) * (values.ndim - 2)
    x_weights = wx.reshape((1, width, *trailing))
    y_weights = wy.reshape((height, 1, *trailing))
    top = (
        values[y0[:, np.newaxis], x0[np.newaxis, :]] * (1 - x_weights)
        + values[y0[:, np.newaxis], x1[np.newaxis, :]] * x_weights
    )
    bottom = (
        values[y1[:, np.newaxis], x0[np.newaxis, :]] * (1 - x_weights)
        + values[y1[:, np.newaxis], x1[np.newaxis, :]] * x_weights
    )
    return np.asarray(top * (1 - y_weights) + bottom * y_weights, dtype=np.float32)


def _area(points: BedPolygon) -> float:
    if len(points) < 3:
        return 0.0
    values = np.asarray(points, dtype=np.float64)
    return abs(
        float(
            np.dot(values[:, 0], np.roll(values[:, 1], -1))
            - np.dot(values[:, 1], np.roll(values[:, 0], -1))
        )
    )


def _furthest_pair(points: NDArray[np.float64]) -> tuple[int, int]:
    distances = ((points[:, np.newaxis] - points[np.newaxis, :]) ** 2).sum(axis=2)
    return tuple(int(value) for value in np.unravel_index(np.argmax(distances), distances.shape))


def _ring_chain(points: NDArray[np.float64], start: int, end: int) -> NDArray[np.float64]:
    if start <= end:
        return points[start : end + 1]
    return np.vstack((points[start:], points[: end + 1]))


def _douglas_peucker(points: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    vector = end - start
    length = float(np.hypot(*vector))
    distances = (
        np.abs(vector[0] * (start[1] - points[1:-1, 1]) - (start[0] - points[1:-1, 0]) * vector[1])
        if length
        else np.hypot(*(points[1:-1] - start).T)
    )
    index = int(np.argmax(distances)) + 1
    if distances[index - 1] <= epsilon * length if length else distances[index - 1] <= epsilon:
        return np.vstack((start, end))
    return np.vstack(
        (
            _douglas_peucker(points[: index + 1], epsilon)[:-1],
            _douglas_peucker(points[index:], epsilon),
        )
    )


__all__ = [
    "COCO_BED_CLASS_ID",
    "BedInstance",
    "BedPolygon",
    "Letterbox",
    "decode_end_to_end_segmentation",
    "largest_external_contour",
    "letterbox_rgb",
    "simplify_polygon",
]
