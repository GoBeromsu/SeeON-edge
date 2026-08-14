from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from contracts.frame import Frame
from worker.pipeline.output.overlay import OverlayRenderer
from worker.pipeline.output.overlay_scene import AppliedCameraProvenance, OverlaySceneBuilder
from worker.pipeline.trace.models import AnalysisTrace, OptionalNumber, TracePerson
from worker.types import FramePacket
from worker.types.overlay_scene import SceneLabel, fit_scene_transform


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    (
        ((1920, 1080), (1280, 720), (2 / 3, 2 / 3, 0.0, 0.0)),
        ((640, 480), (1280, 720), (1.5, 1.5, 160.0, 0.0)),
        ((1080, 1920), (720, 1280), (2 / 3, 2 / 3, 0.0, 0.0)),
    ),
)
def test_contain_transform_is_explicit_across_aspect_ratios(
    source: tuple[int, int],
    target: tuple[int, int],
    expected: tuple[float, float, float, float],
) -> None:
    transform = fit_scene_transform(*source, *target, mode="contain")

    assert (
        transform.scale_x,
        transform.scale_y,
        transform.offset_x,
        transform.offset_y,
    ) == pytest.approx(expected)


def test_cpu_still_render_is_byte_deterministic_with_cjk_label() -> None:
    analysis = AnalysisTrace(
        "a" * 64,
        ("boot-a", "camera-a", 1, 1),
        OptionalNumber(0.0),
        OptionalNumber(0.0),
        320,
        180,
        "fresh",
        (TracePerson(0, OptionalNumber(7), (20, 20, 120, 160), 0.9),),
        (),
        (),
    )
    scene = OverlaySceneBuilder().from_traces(
        analysis,
        (),
        provenance=AppliedCameraProvenance("b" * 64, "camera.v1"),
    )
    scene = replace(
        scene,
        labels=scene.labels + (SceneLabel("낙상 감지", (20.0, 170.0), (0, 0, 255), 50),),
    )
    packet = FramePacket(
        "camera-a",
        Frame(1, 0.0, np.zeros((180, 320, 3), dtype=np.uint8)),
        0.0,
        1,
        320,
        180,
        0.0,
        "boot-a",
        1,
    )
    renderer = OverlayRenderer()

    first = renderer.render_scene(packet, scene)
    second = renderer.render_scene(packet, scene)

    assert np.array_equal(first, second)
    assert int(first[160:180, 20:100].sum()) > 0
