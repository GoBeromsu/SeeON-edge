"""Pixel-level contracts for the CPU live-preview renderer."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from worker.pipeline.output.preview_renderer import (
    BedZoneGeometry,
    PreviewRenderer,
    PreviewTrack,
)
from worker.types.perception_frame import PersonBox
from worker.types.preview import FallPreviewState, OverlaySelection

FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


@pytest.fixture
def jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (320, 180), "black").save(output, format="JPEG", quality=95)
    return output.getvalue()


@pytest.fixture
def renderer() -> PreviewRenderer:
    return PreviewRenderer(FONT)


def _pixels(jpeg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))


def _track(track_id: int | None = 7) -> PreviewTrack:
    return PreviewTrack(PersonBox(25, 25, 75, 75, 0.9), 100, 100, track_id)


def test_person_box_scales_from_source_pixels_and_cjk_label_has_ink(
    renderer: PreviewRenderer, jpeg: bytes
) -> None:
    rendered = _pixels(
        renderer.render(
            jpeg,
            OverlaySelection(person=True, bed=False),
            (_track(),),
            None,
            {7: FallPreviewState(7, "normal")},
        )
    )

    # Source x=25 and y=25 scale independently into a 320x180 JPEG.
    assert rendered[45, 80, 1] > 100
    # The label background sits above the box; bright text pixels prove the
    # Korean glyphs were actually rasterized rather than omitted.
    label = rendered[20:45, 80:220]
    assert int(np.count_nonzero(np.all(label > 210, axis=2))) > 8


def test_independent_toggles_draw_only_the_selected_geometry(
    renderer: PreviewRenderer, jpeg: bytes
) -> None:
    bed = BedZoneGeometry((((5, 80), (45, 80), (45, 95), (5, 95)),), 100, 100)
    state = {7: FallPreviewState(7, "normal")}

    assert renderer.render(jpeg, OverlaySelection(False, False), (_track(),), bed, state) == jpeg
    person_only = _pixels(
        renderer.render(jpeg, OverlaySelection(True, False), (_track(),), bed, state)
    )
    bed_only = _pixels(
        renderer.render(jpeg, OverlaySelection(False, True), (_track(),), bed, state)
    )
    assert person_only[45, 80, 1] > 100
    assert int(person_only[144, 16].max()) < 30
    assert bed_only[144, 16, 0] > 80
    assert int(bed_only[45, 80].max()) < 30


def test_suspected_uses_a_different_colour_and_renders_probability(
    renderer: PreviewRenderer, jpeg: bytes
) -> None:
    normal = _pixels(
        renderer.render(
            jpeg,
            OverlaySelection(True, False),
            (_track(),),
            None,
            {7: FallPreviewState(7, "normal", 0.12)},
        )
    )
    suspected = _pixels(
        renderer.render(
            jpeg,
            OverlaySelection(True, False),
            (_track(),),
            None,
            {7: FallPreviewState(7, "suspected", 0.91)},
        )
    )
    assert not np.array_equal(normal[45, 80], suspected[45, 80])
    assert normal[45, 80, 1] > normal[45, 80, 0]
    assert suspected[45, 80, 0] > suspected[45, 80, 1]


def test_missing_geometry_never_invents_a_bed(renderer: PreviewRenderer, jpeg: bytes) -> None:
    assert renderer.render(jpeg, OverlaySelection(False, True), (), None, {}) == jpeg


def test_four_persisted_beds_are_all_drawn(renderer: PreviewRenderer, jpeg: bytes) -> None:
    beds = BedZoneGeometry(
        (
            ((5, 10), (20, 10), (20, 25), (5, 25)),
            ((30, 10), (45, 10), (45, 25), (30, 25)),
            ((55, 10), (70, 10), (70, 25), (55, 25)),
            ((80, 10), (95, 10), (95, 25), (80, 25)),
        ),
        100,
        100,
    )

    rendered = _pixels(renderer.render(jpeg, OverlaySelection(False, True), (), beds, {}))

    # Each source polygon's top-left outline is independently scaled and drawn.
    for x in (16, 96, 176, 256):
        pixel = rendered[18, x]
        assert pixel[0] > 80
        assert pixel[0] > pixel[2]


def test_missing_font_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        PreviewRenderer(tmp_path / "missing-font.ttc")


def test_person_label_reads_confidence_bed_and_state_without_tracker_counter() -> None:
    label = PreviewRenderer._person_label  # noqa: SLF001 - label text is the contract under test
    assert label(0.84, None, None) == "사람 84%"
    assert label(0.28, FallPreviewState(7, "normal", 0.08), None) == "사람 28% · 정상"
    assert (
        label(0.9, FallPreviewState(7, "suspected", 0.83), 2) == "사람 90% · 침대2 · 낙상 의심 0.83"
    )
    assert label(0.9, FallPreviewState(7, "suspected", None), None) == "사람 90% · 낙상 의심"
    assert "#" not in label(0.5, FallPreviewState(7, "normal", None), 1)


def test_bed_number_uses_the_box_foot_point_inside_a_saved_polygon() -> None:
    beds = PreviewRenderer._bed_number_at  # noqa: SLF001
    square = (((10, 10), (50, 10), (50, 50), (10, 50)),)
    two = (((10, 10), (50, 10), (50, 50), (10, 50)), ((60, 10), (90, 10), (90, 50), (60, 50)))
    assert beds((30, 30), square) == 1
    assert beds((55, 30), square) is None
    assert beds((70, 40), two) == 2
    assert beds((30, 5), two) is None
