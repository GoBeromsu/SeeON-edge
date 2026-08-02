from __future__ import annotations

import numpy as np

from contracts.observation import BoundingBox
from worker.pipeline.output.overlay import (
    draw_box,
    draw_caption,
    draw_dashed_region,
    draw_label,
    draw_pose,
    draw_region,
)


def _blank(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_box_paints_outline_not_interior() -> None:
    image = _blank()
    draw_box(image, BoundingBox(4, 4, 28, 28, 0.9), (0, 255, 0))
    assert int(image[4, 4:28].sum()) > 0  # top edge painted
    assert int(image[16, 16].sum()) == 0  # interior untouched (outline only)


def test_draw_region_fills_polygon_interior_only() -> None:
    polygon = ((4, 4), (28, 4), (28, 28), (4, 28))
    image = _blank()
    draw_region(image, BoundingBox(4, 4, 28, 28, 0.9, polygon), (255, 0, 0), fill=True)
    assert int(image[16, 16][0]) > 0  # interior tinted by translucent mask (blue channel)
    assert int(image[1, 1].sum()) == 0  # outside polygon stays untouched


def test_draw_region_without_polygon_falls_back_to_box() -> None:
    image = _blank()
    draw_region(image, BoundingBox(4, 4, 28, 28, 0.9), (255, 0, 0))
    assert int(image[4, 4:28].sum()) > 0  # rectangle outline painted
    assert int(image[16, 16].sum()) == 0  # no fill for a plain box


def test_draw_dashed_region_outlines_polygon_without_filling_interior() -> None:
    polygon = ((4, 4), (28, 4), (28, 28), (4, 28))
    image = _blank()
    draw_dashed_region(image, BoundingBox(4, 4, 28, 28, 0.9, polygon), (128, 128, 0))
    assert int(image[4, 6].sum()) > 0  # dash drawn at the start of the top edge
    assert int(image[16, 16].sum()) == 0  # interior never filled, unlike draw_region
    assert int(image[1, 1].sum()) == 0  # outside the polygon stays untouched


def test_draw_dashed_region_has_gaps_along_a_straight_edge() -> None:
    image = _blank(32, 64)
    box = BoundingBox(4, 10, 60, 20, 0.9)
    draw_dashed_region(image, box, (128, 128, 0))
    top_row = image[10, 4:60]
    painted = np.any(top_row.reshape(-1, 3) != 0, axis=-1)
    assert bool(painted.any())  # at least one dash segment painted
    assert bool((~painted).any())  # at least one gap left unpainted


def test_draw_dashed_region_without_polygon_falls_back_to_dashed_box() -> None:
    image = _blank()
    draw_dashed_region(image, BoundingBox(4, 4, 28, 28, 0.9), (128, 128, 0))
    assert int(image[4, 6].sum()) > 0
    assert int(image[16, 16].sum()) == 0


def test_draw_caption_and_label_paint_pixels() -> None:
    chip = _blank(48, 96)
    draw_caption(chip, "BED 1 exit", 4, 30, (255, 180, 32))
    assert int(chip.sum()) > 0
    plain = _blank(32, 96)
    draw_label(plain, "person", 2, 20, (0, 255, 0))
    assert int(plain.sum()) > 0


def test_draw_pose_skeleton_adds_pixels_over_dots_only() -> None:
    keypoints = ((10, 10, 0.9), (20, 10, 0.9), *(((0, 0, 0.0),) * 15))
    with_skeleton = _blank()
    draw_pose(with_skeleton, keypoints)
    dots_only = _blank()
    draw_pose(dots_only, keypoints, skeleton=False)
    assert int(with_skeleton[10, 10:21].sum()) > 0  # skeleton line along the edge row
    assert int(with_skeleton.sum()) > int(dots_only.sum())  # skeleton adds pixels


def test_draw_pose_skips_low_confidence_keypoints() -> None:
    low = ((10, 10, 0.1), *(((0, 0, 0.0),) * 16))
    image = _blank()
    draw_pose(image, low)
    assert int(image.sum()) == 0  # below MIN_KEYPOINT_CONFIDENCE → nothing drawn
