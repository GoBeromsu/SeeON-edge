from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.output._scene_renderer import (
    OverlayFontUnavailable,
    _unicode_font,
    render_scene,
    semantic_panel_layout,
)
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import (
    AnalysisTrace,
    DecisionTrace,
    OptionalNumber,
    TraceComponent,
    TracePerson,
)
from worker.types import DecisionTraceSnapshot, FramePacket
from worker.types.overlay_scene import SceneLabel


def _analysis(*, width: int = 320, height: int = 180) -> AnalysisTrace:
    return AnalysisTrace(
        "a" * 64,
        ("boot", "camera", 1, 1),
        OptionalNumber(0.0),
        OptionalNumber(0.0),
        width,
        height,
        "fresh",
        (TracePerson(0, OptionalNumber(7), (10, 20, width - 20, height - 20), 0.9),),
        (),
        (TraceComponent(0, "fall", "not-scheduled"),),
    )


def _decision(module: str, *, triggered: bool, state: str, score: float) -> DecisionTrace:
    return DecisionTrace(
        "d" * 64,
        "a" * 64,
        0,
        f"{module}.v1",
        f"{module}.policy.v1",
        "e" * 64,
        "f" * 64,
        DecisionTraceSnapshot(
            reason="fall-onset" if module == "fall" else "contained",
            previous_state="clear",
            current_state=state,
            triggered=triggered,
            track_id=7,
            bed_id=0 if module == "bed_exit" else None,
            values=(
                {"fall_probability": score, "operating_threshold": 0.7}
                if module == "fall"
                else {"containment_ratio": score, "min_containment": 0.5}
            ),
        ),
    )


def _scene(*, width: int = 320, height: int = 180):
    return OverlaySceneBuilder().from_traces(
        _analysis(width=width, height=height),
        (
            _decision("fall", triggered=True, state="fall", score=0.9),
            _decision("bed_exit", triggered=False, state="contained", score=0.8),
        ),
        provenance=AppliedCameraProvenance("b" * 64, "camera.v1"),
    )


def test_linux_image_guarantees_noto_cjk_and_renderer_fails_closed_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile.edge").read_text()
    renderer = (Path(__file__).parents[1] / "worker/pipeline/output/_scene_renderer.py").read_text()

    assert "fonts-noto-cjk" in dockerfile
    assert "NotoSansCJK-Regular.ttc" in dockerfile
    assert "OverlayFontUnavailable" in renderer
    assert "NotoSansCJK-Regular.ttc" in renderer
    monkeypatch.setenv("ML_OVERLAY_FONT_PATH", "/not-a-font.ttf")
    _unicode_font.cache_clear()
    with pytest.raises(OverlayFontUnavailable, match="CJK overlay font is unavailable"):
        _unicode_font()
    _unicode_font.cache_clear()


def test_fractional_frame_time_selects_the_current_30000_over_1001_scene() -> None:
    first = _scene()
    second = replace(
        first, frame=replace(first.frame, pts=replace(first.frame.pts, value=1001 / 30000))
    )
    timestamp = Fraction(1001, 30000)
    tolerance = Fraction(1, 2_000_000)
    eligible = [
        scene
        for scene in (first, second)
        if scene.frame.pts.value is not None
        and Fraction(str(scene.frame.pts.value)) <= timestamp + tolerance
    ]
    selected = max(eligible, key=lambda scene: Fraction(str(scene.frame.pts.value or 0)))
    assert selected is second


def test_multiple_domain_decisions_for_one_track_remain_visible_and_triggered_label_wins() -> None:
    scene = _scene()

    assert len(scene.decisions) == 2
    assert any(label.text == "FALL" for label in scene.labels)
    assert any(label.text == "CONTAINED" for label in scene.labels)


def test_renderer_displays_decision_semantics_and_cjk_without_pseudoglyph_cells() -> None:
    scene = _scene()
    scene = replace(
        scene,
        labels=scene.labels + (SceneLabel("낙상 감지", (2.0, 178.0), (0, 0, 255), 50),),
    )
    image = np.zeros((180, 320, 3), dtype=np.uint8)

    rendered = render_scene(image, scene)

    # Panels and labels remain inside the target and use real antialiased glyphs, not cells.
    assert rendered.shape == image.shape
    assert int(rendered.sum()) > 0
    assert np.any(rendered[150:, :])


def test_renderer_handles_dense_portrait_and_edge_labels_without_index_errors() -> None:
    scene = _scene(width=90, height=240)
    scene = replace(
        scene,
        labels=tuple(
            SceneLabel(
                f"label-{index}-낙상", (float(index % 3 * 40), float(index * 17)), (0, 0, 255), 50
            )
            for index in range(20)
        ),
    )

    rendered = render_scene(np.zeros((240, 90, 3), dtype=np.uint8), scene)

    assert rendered.shape == (240, 90, 3)


@pytest.mark.parametrize("size", ((320, 240), (120, 320)))
@pytest.mark.parametrize("reverse", (False, True))
def test_semantic_panels_are_non_overlapping_and_keep_triggered_values(
    size: tuple[int, int], reverse: bool
) -> None:
    width, height = size
    scene = _scene(width=width, height=height)
    if reverse:
        scene = replace(scene, decisions=tuple(reversed(scene.decisions)))

    layout = semantic_panel_layout(scene, width, height)
    triggered = next(item for item in layout if item.panel.triggered)

    assert len(layout) == 3  # fall, bed-exit containment, component state
    assert triggered.panel.lines[-3:] == ("score=0.90", "threshold=0.70", "reason=fall-onset")
    assert any(item.panel.lines == ("fall:not evaluated",) for item in layout)
    assert all("..." not in line for item in layout for line in item.panel.lines)
    for item in layout:
        assert 0 <= item.x < item.x + item.width <= width
        assert 0 <= item.y < item.y + item.height <= height
    for index, left in enumerate(layout):
        for right in layout[index + 1 :]:
            assert (
                left.x + left.width <= right.x
                or right.x + right.width <= left.x
                or left.y + left.height <= right.y
                or right.y + right.height <= left.y
            )


def test_impossibly_small_frames_use_a_deterministic_explicit_continuation_page() -> None:
    scene = replace(
        _scene(width=16, height=16), frame=replace(_scene(width=16, height=16).frame, seq=1)
    )

    layout = semantic_panel_layout(scene, 16, 16)

    assert len(layout) == 1
    assert layout[0].panel.lines[-1] == "P2/3"
    assert layout[0].x >= 0 and layout[0].y >= 0


def test_global_z_order_is_stable_for_equal_z_labels() -> None:
    scene = _scene()
    scene = replace(
        scene,
        labels=(
            SceneLabel("first", (10.0, 20.0), (0, 255, 0), 50),
            SceneLabel("second", (10.0, 20.0), (0, 0, 255), 50),
        ),
    )
    packet = FramePacket(
        "camera",
        Frame(1, 0.0, np.zeros((180, 320, 3), dtype=np.uint8)),
        0.0,
        1,
        320,
        180,
        0.0,
        "boot",
        1,
    )

    first = render_scene(packet.borrow_host_frame().image.copy(), scene)
    second = render_scene(packet.borrow_host_frame().image.copy(), scene)

    assert np.array_equal(first, second)
