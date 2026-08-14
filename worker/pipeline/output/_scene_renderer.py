from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from contracts.observation import BoundingBox
from worker.pipeline.output._overlay_primitives import (
    POSE_EDGES,
    draw_box,
    draw_dashed_region,
    draw_label,
)
from worker.types.overlay_scene import (
    ObservationSemantics,
    OverlayScene,
    SceneBed,
    SceneDecision,
    SceneKeypoint,
    SceneLabel,
    SceneValue,
)

# Overlay visual tokens. All dimensions derive from this compact 4px scale.
_SPACE: Final = 4
_PANEL_MARGIN: Final = _SPACE * 2
_PANEL_GAP: Final = _SPACE
_PANEL_WIDTH: Final = _SPACE * 66
_PANEL_LINE: Final = _SPACE * 4
_PANEL_BACKGROUND: Final = (20, 20, 20)
_PANEL_BORDER: Final = (235, 235, 235)
_TEXT: Final = (255, 255, 255)
_OUTLINE: Final = (0, 0, 0)
_FONT_CANDIDATES: Final = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


class OverlayFontUnavailable(RuntimeError):
    """Raised when a requested Unicode label cannot be rendered as real text."""


@dataclass(frozen=True, slots=True)
class _DrawCommand:
    z_order: int
    kind: int
    ordinal: int
    draw: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _SemanticPanel:
    identity: str
    lines: tuple[str, ...]
    color: tuple[int, int, int]
    triggered: bool


@dataclass(frozen=True, slots=True)
class _PanelPlacement:
    panel: _SemanticPanel
    x: int
    y: int
    width: int
    height: int
    font_scale: float


def render_scene(image: NDArray[np.uint8], scene: OverlayScene) -> NDArray[np.uint8]:
    """Render the canonical scene with stable global ordering and no policy lookup."""
    if image.shape[:2] != (scene.transform.target_height, scene.transform.target_width):
        raise ValueError("overlay target dimensions do not match the rendered frame")
    commands: list[_DrawCommand] = []
    for bed in scene.beds:
        commands.append(
            _DrawCommand(bed.z_order, 0, bed.ordinal, lambda bed=bed: _draw_bed(image, scene, bed))
        )
    for person in scene.persons:
        commands.append(
            _DrawCommand(
                person.z_order,
                1,
                person.ordinal,
                lambda person=person: _draw_person(
                    image, scene, person.keypoints, person.box, person.confidence, person.color
                ),
            )
        )
    panel_layout = semantic_panel_layout(scene, image.shape[1], image.shape[0])
    occupied_labels = [_rectangle(item.x, item.y, item.width, item.height) for item in panel_layout]
    for ordinal, label in enumerate(scene.labels):
        commands.append(
            _DrawCommand(
                label.z_order,
                2,
                ordinal,
                lambda label=label: _draw_label(image, scene, label, occupied_labels),
            )
        )
    for ordinal, placement in enumerate(panel_layout):
        commands.append(
            _DrawCommand(
                40,
                3,
                ordinal,
                lambda placement=placement: _draw_semantic_panel(image, placement),
            )
        )
    for command in sorted(commands, key=lambda item: (item.z_order, item.kind, item.ordinal)):
        command.draw()
    return image


def _draw_bed(image: NDArray[np.uint8], scene: OverlayScene, bed: SceneBed) -> None:
    points = bed.polygon or _box_points(bed.box)
    mapped = tuple(scene.transform.point(point) for point in points)
    draw_dashed_region(
        image, BoundingBox(*_bounds(mapped, bed.confidence), polygon=mapped), bed.color
    )


def _draw_person(
    image: NDArray[np.uint8],
    scene: OverlayScene,
    points: tuple[SceneKeypoint, ...],
    box: tuple[float, float, float, float],
    confidence: float,
    color: tuple[int, int, int],
) -> None:
    mapped = tuple(scene.transform.point(point) for point in _box_points(box))
    draw_box(image, BoundingBox(*_bounds(mapped, confidence)), color)
    _draw_keypoints(image, scene, points)


def _draw_keypoints(
    image: NDArray[np.uint8], scene: OverlayScene, points: tuple[SceneKeypoint, ...]
) -> None:
    known = {
        point.index: scene.transform.point(point.point)
        for point in points
        if point.semantics is ObservationSemantics.PRESENT and point.point is not None
    }
    for start, end in POSE_EDGES:
        if start in known and end in known:
            _ = cv2.line(image, known[start], known[end], _TEXT, 2, cv2.LINE_AA)
    for index in sorted(known):
        _ = cv2.circle(image, known[index], 2, _TEXT, -1, cv2.LINE_AA)


def _draw_label(
    image: NDArray[np.uint8],
    scene: OverlayScene,
    label: SceneLabel,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    x, y = scene.transform.point(label.anchor)
    if scene.frame.camera_configuration_id == "not-recorded":
        draw_label(image, label.text, x, max(12, y - _SPACE), label.color)
        return
    if bounds := _draw_text_plate(image, label.text, x, y - _SPACE, label.color, tuple(occupied)):
        occupied.append(bounds)


def semantic_panel_layout(
    scene: OverlayScene, width: int, height: int
) -> tuple[_PanelPlacement, ...]:
    """Lay out every semantic card once, reserving its exact geometry for all layers.

    Triggered decisions sort first independent of source-domain order. Wide frames use
    two columns; narrow frames stack compact cards. The required score, threshold and
    reason are compact tokens rather than truncated prose, so 120px-wide frames retain
    every semantic value without a continuation page.
    """
    panels = sorted(_semantic_panels(scene), key=lambda item: (not item.triggered, item.identity))
    if not panels:
        return ()
    compact_page = min(width, height) < _SPACE * 20
    narrow = width < _SPACE * 48
    columns = 1 if narrow else 2
    margin = 1 if compact_page else _PANEL_MARGIN
    gap = 1 if compact_page else _PANEL_GAP
    font_scale = 0.12 if compact_page else (0.27 if narrow else 0.34)
    line_height = 2 if compact_page else max(_SPACE * 2, round(_SPACE * 3 * font_scale / 0.34))
    if compact_page:
        page = scene.frame.seq % len(panels)
        selected = panels[page]
        panels = (
            _SemanticPanel(
                selected.identity,
                selected.lines + (f"P{page + 1}/{len(_semantic_panels(scene))}",),
                selected.color,
                selected.triggered,
            ),
        )
    card_width = (width - margin * 2 - gap * (columns - 1)) // columns
    placements: list[_PanelPlacement] = []
    column_heights = [margin] * columns
    for ordinal, panel in enumerate(panels):
        column = ordinal % columns
        card_height = margin + line_height * len(panel.lines) + margin
        x = margin + column * (card_width + gap)
        y = column_heights[column]
        if y + card_height > height - margin:
            raise ValueError("overlay semantic panels exceed the target frame")
        placements.append(_PanelPlacement(panel, x, y, card_width, card_height, font_scale))
        column_heights[column] = y + card_height + gap
    return tuple(placements)


def _semantic_panels(scene: OverlayScene) -> tuple[_SemanticPanel, ...]:
    decisions = tuple(
        _SemanticPanel(
            decision.module_qualified_id,
            _decision_lines(decision),
            decision.color,
            decision.triggered,
        )
        for decision in scene.decisions
    )
    components = tuple(
        f"{_public_component(component.qualified_id)}:{component.semantics.value.replace('-', ' ')}"
        for component in sorted(scene.components, key=lambda item: item.qualified_id)
    )
    component_panel = (
        _SemanticPanel("components", components, (180, 180, 180), False) if components else None
    )
    return decisions + ((component_panel,) if component_panel else ())


def _draw_semantic_panel(image: NDArray[np.uint8], placement: _PanelPlacement) -> None:
    _panel(
        image, placement.x, placement.y, placement.width, placement.height, placement.panel.color
    )
    line_height = max(_SPACE * 2, round(_SPACE * 3 * placement.font_scale / 0.34))
    for index, line in enumerate(placement.panel.lines):
        _draw_ascii(
            image,
            line,
            placement.x + _SPACE,
            placement.y + _PANEL_MARGIN + line_height * (index + 1) - _SPACE,
            _TEXT,
            placement.font_scale,
        )


def _rectangle(x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
    return x, y, x + width, y + height


def _decision_lines(decision: SceneDecision) -> tuple[str, ...]:
    domain = _public_component(decision.module_qualified_id).upper()
    metric = "contain" if decision.module_qualified_id.startswith("bed_exit.") else "score"
    identity = (
        f"track={decision.track_id.value}" if decision.track_id.value is not None else "scene"
    )
    if decision.bed_id.value is not None:
        identity += f" bed={decision.bed_id.value}"
    return (
        f"{domain} {'ALERT' if decision.triggered else decision.current_state.upper()}",
        identity,
        f"{metric}={_value_number(decision.score)}",
        f"threshold={_value_number(decision.threshold)}",
        f"reason={decision.reason}",
    )


def _value_number(value: SceneValue) -> str:
    if value.value is None:
        return f"--:{_semantic(value.semantics, value.reason)}"
    return f"{float(value.value):.2f}"


def _semantic(semantics: ObservationSemantics, reason: str | None) -> str:
    result = semantics.value
    return f"{result}: {reason}" if reason else result


def _public_component(value: str) -> str:
    """Show the domain/component name, never a policy hash or provenance identifier."""
    return value.split(".", 1)[0].replace("_", " ")


def _panel(
    image: NDArray[np.uint8], x: int, y: int, width: int, height: int, color: tuple[int, int, int]
) -> None:
    x2, y2 = min(image.shape[1] - 1, x + width), min(image.shape[0] - 1, y + height)
    overlay = image.copy()
    _ = cv2.rectangle(overlay, (x, y), (x2, y2), _PANEL_BACKGROUND, -1, cv2.LINE_AA)
    _ = cv2.addWeighted(overlay, 0.86, image, 0.14, 0, dst=image)
    _ = cv2.rectangle(image, (x, y), (x2, y2), color, 1, cv2.LINE_AA)


def _draw_text_plate(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    occupied: tuple[tuple[int, int, int, int], ...],
) -> tuple[int, int, int, int] | None:
    text = text[:96]
    if text.isascii():
        size = _ascii_size(text)
    else:
        size = _unicode_size(text)
    position = _label_position(image.shape[1], image.shape[0], x, y, size, occupied)
    if position is None:
        return None
    x, y = position
    _panel(
        image,
        x - _SPACE,
        y - size[1] - _SPACE,
        size[0] + _PANEL_MARGIN,
        size[1] + _PANEL_MARGIN,
        _OUTLINE,
    )
    if text.isascii():
        _draw_ascii(image, text, x, y, color)
    else:
        _draw_unicode(image, text, x, y - size[1], color)
    return (x - _SPACE, y - size[1] - _SPACE, x + size[0], y + _SPACE)


def _label_position(
    width: int,
    height: int,
    x: int,
    y: int,
    size: tuple[int, int],
    occupied: tuple[tuple[int, int, int, int], ...],
) -> tuple[int, int] | None:
    text_width, text_height = size
    max_x = max(_SPACE, width - text_width - _SPACE)
    max_y = max(text_height + _SPACE, height - _SPACE)
    candidates = [(x, y)]
    candidates.extend(
        (candidate_x, candidate_y)
        for candidate_y in range(text_height + _SPACE, max_y + 1, text_height + _SPACE)
        for candidate_x in range(_SPACE, max_x + 1, text_width + _SPACE)
    )
    for candidate_x, candidate_y in candidates:
        candidate_x = min(max(_SPACE, candidate_x), max_x)
        candidate_y = min(max(text_height + _SPACE, candidate_y), max_y)
        if not any(
            candidate_x < right
            and candidate_x + text_width > left
            and candidate_y - text_height < bottom
            and candidate_y > top
            for left, top, right, bottom in occupied
        ):
            return candidate_x, candidate_y
    return None


def _draw_ascii(
    image: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.42,
) -> None:
    _ = cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, _OUTLINE, 3, cv2.LINE_8)
    _ = cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_8)


def _ascii_size(text: str) -> tuple[int, int]:
    width, height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    return int(width), int(height) + _SPACE


def _unicode_size(text: str) -> tuple[int, int]:
    font = _unicode_font()
    left, top, right, bottom = font.getbbox(text, stroke_width=1)
    return int(right - left), int(bottom - top)


def _draw_unicode(
    image: NDArray[np.uint8], text: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (x, y),
        text,
        font=_unicode_font(),
        fill=color[::-1],
        stroke_width=1,
        stroke_fill=_OUTLINE[::-1],
    )
    image[...] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=1)
def _unicode_font() -> ImageFont.FreeTypeFont:
    configured = os.environ.get("ML_OVERLAY_FONT_PATH")
    candidates = (configured,) if configured else _FONT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, _SPACE * 4)
            except OSError:
                continue
    raise OverlayFontUnavailable("a real CJK overlay font is unavailable")


def _ellipsize(text: str, available: int) -> str:
    result = text
    while result and _ascii_size(result)[0] > available:
        result = result[:-4] + "..." if len(result) > 3 else ""
    return result


def _box_points(box: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = box
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _bounds(
    points: tuple[tuple[int, int], ...], confidence: float
) -> tuple[int, int, int, int, float]:
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        confidence,
    )


__all__ = ["OverlayFontUnavailable", "render_scene"]
