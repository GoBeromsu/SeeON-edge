"""Unit tests for `worker.domains.bed_exit.geometry.containment_ratio`.

Covers the polygon-containment fix for #219: a bed's `x1..y2` alone is an
axis-aligned bounding box (AABB) that can be up to ~2x the true polygon area
for a rotated or irregular bed outline. `containment_ratio` now measures
against `bed.polygon` (rasterized with `cv2.fillPoly` into a cached mask)
when present, falling back to the exact AABB formula (bit-for-bit unchanged)
when it is absent or the polygon is unusable.
"""

from __future__ import annotations

import logging
from typing import Final

import pytest

from contracts.observation import BoundingBox
from worker.domains.bed_exit.geometry import (
    _bed_polygon_mask,
    containment_ratio,
)

# A square rotated 45 degrees ("diamond"), centered at (50, 50), inscribed in
# the AABB [0, 100] x [0, 100]. Diamond area = 100*100/2 = 5,000; AABB area =
# 10,000 -- a 2.0x inflation, matching the worst end of the #219 measured
# range (1.57x-2.07x, mean 1.94x) for a simple convex rotated rectangle.
DIAMOND: Final = ((50, 0), (100, 50), (50, 100), (0, 50))

# An "L" shape: a 100x100 square with its top-right 40x40 corner notched out
# (e.g. tracing around a nightstand at the head of the bed). This is
# non-convex -- the notch is inside the AABB but outside the polygon.
NOTCHED_L: Final = ((0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100))


def _bed(polygon: tuple[tuple[int, int], ...] | None) -> BoundingBox:
    return BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=polygon)


def test_convex_polygon_person_fully_inside_is_fully_contained() -> None:
    # Given: a person box entirely within the diamond (all 4 corners satisfy
    # |x-50| + |y-50| <= 50).
    bed = _bed(DIAMOND)
    person = BoundingBox(40, 40, 60, 60, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 1.0


def test_convex_polygon_person_outside_polygon_but_inside_aabb() -> None:
    # Given: this is the #219 regression case -- a person box that is fully
    # outside the diamond (every point has x+y >= 160, past the diamond's
    # x+y<=150 edge) yet fully inside the bed's AABB [0,100]x[0,100]. The
    # pre-fix AABB-only formula measures this as full containment
    # (intersection == person box == 225px^2, ratio == 1.0); a person
    # standing here should NOT read as "in bed".
    bed = _bed(DIAMOND)
    person = BoundingBox(80, 80, 95, 95, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 0.0


def test_non_convex_polygon_person_inside_main_body() -> None:
    # Given: a person box well within the L-shape's main body, away from the
    # notch.
    bed = _bed(NOTCHED_L)
    person = BoundingBox(10, 10, 50, 50, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 1.0


def test_non_convex_polygon_person_in_the_notch_is_not_contained() -> None:
    # Given: a person box sitting entirely in the notched-out corner --
    # inside the L-shape's AABB, but outside the actual (non-convex)
    # polygon. A convex-only algorithm (e.g. Sutherland-Hodgman clipping)
    # would get this wrong; ray-casting point sampling does not.
    bed = _bed(NOTCHED_L)
    person = BoundingBox(70, 70, 90, 90, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 0.0


def test_no_polygon_falls_back_to_exact_aabb_formula() -> None:
    # Given: a bed with no persisted polygon (the pre-#219 shape, and still
    # the shape for any camera with no drawn bed zone) and a person box that
    # partially overlaps it.
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=None)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then: intersection is [50,100]x[50,100] = 2,500; person_area is
    # 100*100 = 10,000 -- the exact pre-fix formula, unchanged.
    assert ratio == 2_500 / 10_000


def test_no_polygon_person_fully_outside_is_zero() -> None:
    # Given
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=None)
    person = BoundingBox(200, 200, 250, 250, 0.9)

    # When / Then
    assert containment_ratio(person, bed) == 0.0


def test_degenerate_person_box_is_zero_regardless_of_polygon() -> None:
    # Given: a zero-area person box (guards the person_area <= 0 short
    # circuit still applies before any polygon rasterization is attempted).
    bed = _bed(DIAMOND)
    person = BoundingBox(50, 50, 50, 80, 0.9)

    # When / Then
    assert containment_ratio(person, bed) == 0.0


def test_person_box_fully_outside_bed_aabb_is_zero() -> None:
    # Given: a person box with a generous margin beyond the bed's own AABB,
    # diagonally down-right. This is the case `_mask_containment_ratio`'s
    # four-sided clamp (`max(person.x1 - origin_x, 0)`, `min(person.x2 -
    # origin_x, width)`, mirrored on y) plus its `right <= left or bottom <=
    # top` short-circuit exist to protect: numpy silently wraps negative
    # slice indices instead of raising, so a person nowhere near the bed
    # could read a plausible-looking nonzero containment off the far side of
    # the mask if either the clamps or the guard were ever removed. Distinct
    # from `test_convex_polygon_person_outside_polygon_but_inside_aabb`
    # above, which is outside the *polygon* but still inside the AABB -- this
    # one is outside the AABB entirely.
    bed = _bed(DIAMOND)
    person = BoundingBox(150, 150, 180, 180, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 0.0


def test_person_box_one_pixel_outside_bed_aabb_is_zero() -> None:
    # Given: a person box immediately adjacent to (not overlapping) the bed's
    # own AABB, up-left of the origin by exactly one pixel on both axes --
    # the tightest case for the clamp, and the one most likely to slip
    # through a future off-by-one if someone "simplifies" the `right <= left`
    # guard next to the `max`/`min` clamps.
    bed = _bed(DIAMOND)
    person = BoundingBox(-20, -20, 0, 0, 0.9)

    ratio = containment_ratio(person, bed)

    assert ratio == 0.0


def test_self_intersecting_polygon_is_rasterized_not_rejected() -> None:
    # Given: a polygon whose closing edge crosses its first edge near the
    # origin -- a synthetic stand-in for the real "08 205호" production
    # camera, whose 48-point trace self-intersects by ~2.2px at its closing
    # seam (point 47 wrapping back to point 0). That is a trace-closure
    # artifact, not a real bowtie fold, and must NOT be rejected into the
    # AABB fallback: silently reverting an already-broken camera to the AABB
    # path while the PR/tests/dashboard all report the bug fixed is a worse
    # failure than an ambiguous few pixels at a seam.
    self_intersecting: Final = ((0, 0), (100, 0), (100, 100), (0, 100), (5, -5))

    # When
    mask_info = _bed_polygon_mask(self_intersecting)

    # Then: rasterized, not rejected.
    assert mask_info is not None
    # And: containment still works end to end through the public API for a
    # person clearly within the polygon's main body.
    bed = _bed(self_intersecting)
    person = BoundingBox(20, 20, 80, 80, 0.9)
    assert containment_ratio(person, bed) == 1.0


def test_polygon_with_fewer_than_three_points_falls_back_to_aabb() -> None:
    # Given: an unusable "polygon" (a single edge, not a shape).
    degenerate: Final = ((10, 10), (90, 90))
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=degenerate)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then: falls back to the exact AABB formula (same as the no-polygon
    # case): intersection [50,100]x[50,100] = 2,500 / person_area 10,000.
    assert ratio == 2_500 / 10_000
    assert _bed_polygon_mask(degenerate) is None


def test_collinear_polygon_falls_back_to_aabb() -> None:
    # Given: points that trace a straight line, not a shape.
    collinear: Final = ((0, 0), (25, 25), (50, 50), (75, 75), (100, 100))
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=1.0, polygon=collinear)
    person = BoundingBox(50, 50, 150, 150, 0.9)

    # When
    ratio = containment_ratio(person, bed)

    # Then
    assert ratio == 2_500 / 10_000
    assert _bed_polygon_mask(collinear) is None


def test_unusable_polygon_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    # Given: a fresh degenerate polygon (not reused by another test, since
    # `_bed_polygon_mask` is cached and would otherwise short-circuit before
    # logging again).
    degenerate: Final = ((11, 11), (91, 91))

    # When
    with caplog.at_level(logging.WARNING, logger="worker.domains.bed_exit.geometry"):
        result = _bed_polygon_mask(degenerate)

    # Then
    assert result is None
    assert "falling back to AABB" in caplog.text
