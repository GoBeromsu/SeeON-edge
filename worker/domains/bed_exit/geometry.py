from __future__ import annotations

import logging
from functools import lru_cache
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.observation import BoundingBox

LOGGER: Final = logging.getLogger(__name__)

# Rasterized-mask cache for bed polygons (see `_bed_polygon_mask`). Bounded
# so a long-running worker with many cameras/polygon updates cannot grow
# this unboundedly; entries are cheap (one small uint8 array each) so a
# generous size costs little.
_MASK_CACHE_SIZE: Final = 64


def best_bed_id(containments: tuple[float, ...], min_containment: float) -> int | None:
    candidates = (
        (ratio, bed_id)
        for bed_id, ratio in enumerate(containments)
        if ratio >= min_containment
    )
    best = max(candidates, key=lambda item: (item[0], -item[1]), default=None)
    return None if best is None else best[1]


def containment_ratio(person: BoundingBox, bed: BoundingBox) -> float:
    """Fraction of `person`'s box area that lies inside `bed`.

    Uses `bed.polygon` (the operator-traced bed outline) when present: the
    bed's `x1..y2` alone is an axis-aligned bounding box, which measured
    1.57x-2.07x (mean 1.94x) the true polygon area across this site's 10
    persisted bed zones (#219), inflating containment for anyone standing
    just outside the real bed but inside its AABB. Falls back to the exact
    axis-aligned-rectangle intersection -- bit-for-bit the prior formula --
    when no polygon is recorded (or the recorded polygon turns out to be
    unusable, see `_bed_polygon_mask`), so cameras with no drawn bed zone
    keep working exactly as before.
    """
    person_area = max(0, person.x2 - person.x1) * max(0, person.y2 - person.y1)
    if person_area <= 0:
        return 0.0
    if bed.polygon:
        mask_info = _bed_polygon_mask(bed.polygon)
        if mask_info is not None:
            return _mask_containment_ratio(person, mask_info, person_area)
    return _aabb_containment_ratio(person, bed, person_area)


def _aabb_containment_ratio(person: BoundingBox, bed: BoundingBox, person_area: int) -> float:
    left = max(person.x1, bed.x1)
    top = max(person.y1, bed.y1)
    right = min(person.x2, bed.x2)
    bottom = min(person.y2, bed.y2)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    return intersection / person_area


# (mask, origin_x, origin_y): `mask[y - origin_y, x - origin_x]` is 1 iff
# pixel (x, y) in the bed's own coordinate space is inside its polygon.
_BedMask = tuple[NDArray[np.uint8], int, int]


@lru_cache(maxsize=_MASK_CACHE_SIZE)
def _bed_polygon_mask(polygon: tuple[tuple[int, int], ...]) -> _BedMask | None:
    """Rasterize a bed polygon into a binary mask, local to its own AABB.

    Why rasterization and not exact analytic clipping: production polygons
    on this site are confirmed non-convex -- each one's convex-hull area
    exceeds its own polygon area by 11-31% (mean ~20%), which a rotated
    rectangle (hull == polygon) could not produce -- and one persisted
    camera's trace self-intersects at its closing seam (a ~2.2px
    trace-closure artifact, not a real self-crossing shape). A convex-only
    clip (e.g. Sutherland-Hodgman) is silently wrong for the non-convex
    majority; a general polygon-clipping algorithm correct for both
    non-convexity and self-intersection
    (Weiler-Atherton and friends) is a subtle-bug factory nobody here could
    review with confidence. `cv2.fillPoly`'s scanline rasterizer handles
    both cases correctly with no special-casing, using a dependency this
    project already has (`opencv-python-headless`) -- adding an exact
    polygon-intersection library (e.g. shapely) was judged not worth a new
    dependency for this alone.

    Deliberately permissive: a polygon that self-intersects still gets
    rasterized (a sane filled region, just not what shoelace-style analytic
    area would call "valid") -- it is NOT rejected into the AABB fallback.
    Silently reverting an already-broken camera back to the AABB path while
    everything else (PR, tests, dashboard) reports the bug fixed is a worse
    failure than an ambiguous few pixels at a seam. Only genuinely unusable
    input falls back here: fewer than 3 points, all points collinear, or an
    empty mask after fill -- each logged once (this function is cached, so
    the warning does not repeat every frame) so a fallback is never silent.

    Cached (`lru_cache`) because polygons are static per camera between
    bed-zone re-recognitions; this runs per person per bed per frame at
    5fps, and re-rasterizing on every call would be wasted, repeated work.
    """
    if len(polygon) < 3 or _all_collinear(polygon):
        LOGGER.warning(
            "bed polygon has fewer than 3 points or is degenerate (all "
            "points collinear); falling back to AABB containment for this "
            "bed region: %r",
            polygon,
        )
        return None

    xs = tuple(point[0] for point in polygon)
    ys = tuple(point[1] for point in polygon)
    origin_x, origin_y = min(xs), min(ys)
    width = max(xs) - origin_x
    height = max(ys) - origin_y
    if width <= 0 or height <= 0:
        LOGGER.warning(
            "bed polygon has zero-area bounding box; falling back to AABB "
            "containment for this bed region: %r",
            polygon,
        )
        return None

    mask: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
    shifted = np.array(
        [[point[0] - origin_x, point[1] - origin_y] for point in polygon],
        dtype=np.int32,
    ).reshape((1, -1, 2))
    cv2.fillPoly(mask, shifted, 1)
    if not mask.any():
        LOGGER.warning(
            "bed polygon rasterized to an empty mask; falling back to AABB "
            "containment for this bed region: %r",
            polygon,
        )
        return None
    return mask, origin_x, origin_y


def _all_collinear(points: tuple[tuple[int, int], ...]) -> bool:
    origin_x, origin_y = points[0]
    direction_x, direction_y = 0, 0
    for point_x, point_y in points[1:]:
        direction_x, direction_y = point_x - origin_x, point_y - origin_y
        if direction_x != 0 or direction_y != 0:
            break
    else:
        return True  # every point coincides with the first
    return all(
        (point_x - origin_x) * direction_y - (point_y - origin_y) * direction_x == 0
        for point_x, point_y in points
    )


def _mask_containment_ratio(
    person: BoundingBox, mask_info: _BedMask, person_area: int
) -> float:
    mask, origin_x, origin_y = mask_info
    height, width = mask.shape
    left = max(person.x1 - origin_x, 0)
    top = max(person.y1 - origin_y, 0)
    right = min(person.x2 - origin_x, width)
    bottom = min(person.y2 - origin_y, height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = int(mask[top:bottom, left:right].sum())
    if intersection == 0:
        return 0.0
    return intersection / person_area


__all__ = ["best_bed_id", "containment_ratio"]
