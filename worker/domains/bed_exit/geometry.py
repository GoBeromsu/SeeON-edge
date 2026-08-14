from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from contracts.observation import BoundingBox

LOGGER: Final = logging.getLogger(__name__)

# Rasterized-mask cache for bed polygons (see `_bed_polygon_mask`). Bounded
# so a long-running worker with many cameras/polygon updates cannot grow
# this unboundedly; entries are cheap (one small per-row prefix-sum tuple
# each) so a generous size costs little.
_MASK_CACHE_SIZE: Final = 64


def best_bed_id(containments: tuple[float, ...], min_containment: float) -> int | None:
    candidates = (
        (ratio, bed_id) for bed_id, ratio in enumerate(containments) if ratio >= min_containment
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


@dataclass(frozen=True, slots=True)
class _BedMask:
    """A rasterized bed polygon, local to its own AABB.

    Pure-Python (no ndarray): `row_prefix_sums[y]` is a length-`width + 1`
    tuple of cumulative filled-pixel counts for rasterized row `y`, so the
    number of filled pixels in columns `[left, right)` of row `y` is
    `row_prefix_sums[y][right] - row_prefix_sums[y][left]` -- an O(height)
    replacement for what used to be a single vectorized ndarray-slice sum.
    """

    origin_x: int
    origin_y: int
    width: int
    height: int
    row_prefix_sums: tuple[tuple[int, ...], ...]


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
    non-convexity and self-intersection (Weiler-Atherton and friends) is a
    subtle-bug factory nobody here could review with confidence. A plain
    scanline fill (edge-intersection per row, even-odd rule, pixel-center
    sampling) handles both cases correctly with no special-casing and no
    ndarray/OpenCV dependency, which `worker/domains` may not import
    (architecture-audit H3): the domain layer stays numeric/hardware-
    agnostic, independent of which inference/vision library an
    infrastructure profile happens to ship.

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

    row_prefix_sums = _rasterize_rows(polygon, origin_x, origin_y, width, height)
    if not any(row[-1] > 0 for row in row_prefix_sums):
        LOGGER.warning(
            "bed polygon rasterized to an empty mask; falling back to AABB "
            "containment for this bed region: %r",
            polygon,
        )
        return None
    return _BedMask(origin_x, origin_y, width, height, row_prefix_sums)


def _rasterize_rows(
    polygon: tuple[tuple[int, int], ...],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> tuple[tuple[int, ...], ...]:
    """Scanline-fill `polygon` (local to its AABB) into per-row prefix sums.

    Standard even-odd scanline polygon fill: for each integer row `y`,
    intersect every non-horizontal edge against the pixel-center scanline
    `y + 0.5`, sort the intersection x-coordinates, and fill columns between
    each consecutive pair. Edge y-intervals are treated half-open
    (`[low, high)`) so a shared vertex between two edges on the same
    scanline is counted exactly once, matching the usual scan-conversion
    convention.
    """
    shifted = tuple((point[0] - origin_x, point[1] - origin_y) for point in polygon)
    edge_count = len(shifted)
    edges = tuple(
        (shifted[index], shifted[(index + 1) % edge_count]) for index in range(edge_count)
    )
    rows: list[tuple[int, ...]] = []
    for y in range(height):
        intersections: list[float] = []
        for (x0, y0), (x1, y1) in edges:
            if y0 == y1:
                continue
            low_y, high_y = (y0, y1) if y0 < y1 else (y1, y0)
            if not (low_y <= y < high_y):
                continue
            t = (y - y0) / (y1 - y0)
            intersections.append(x0 + t * (x1 - x0))
        intersections.sort()
        # `delta` is a difference array over filled columns: +1 at each
        # fill-run's start, -1 at its end, so a running sum reconstructs
        # which columns are filled without ever materializing a row buffer.
        delta = [0] * (width + 1)
        for pair_index in range(0, len(intersections) - 1, 2):
            start_column = max(math.ceil(intersections[pair_index] - 0.5), 0)
            end_column = min(math.ceil(intersections[pair_index + 1] - 0.5), width)
            if end_column > start_column:
                delta[start_column] += 1
                delta[end_column] -= 1
        cumulative = [0] * (width + 1)
        running = 0
        for column in range(width):
            running += delta[column]
            cumulative[column + 1] = cumulative[column] + (1 if running > 0 else 0)
        rows.append(tuple(cumulative))
    return tuple(rows)


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


def _mask_containment_ratio(person: BoundingBox, mask_info: _BedMask, person_area: int) -> float:
    left = max(person.x1 - mask_info.origin_x, 0)
    top = max(person.y1 - mask_info.origin_y, 0)
    right = min(person.x2 - mask_info.origin_x, mask_info.width)
    bottom = min(person.y2 - mask_info.origin_y, mask_info.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = sum(
        mask_info.row_prefix_sums[row][right] - mask_info.row_prefix_sums[row][left]
        for row in range(top, bottom)
    )
    if intersection == 0:
        return 0.0
    return intersection / person_area


__all__ = ["best_bed_id", "containment_ratio"]
