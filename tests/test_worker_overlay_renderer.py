from __future__ import annotations

import numpy as np

from contracts.frame import Frame
from contracts.observation import (
    BedRegionDebugSnapshot,
    BoundingBox,
    DetectionLabel,
    FrameObservation,
)
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.schema import BedExitDebugSnapshot, BedStatus
from worker.pipeline.output.overlay import (
    FALL_LABEL_COLOR,
    NORMAL_LABEL_COLOR,
    OverlayRenderer,
)
from worker.types import FramePacket


def _packet(image: np.ndarray) -> FramePacket:
    return FramePacket(
        camera_id="cam-1",
        frame=Frame(index=1, time_sec=1.0, image=image),
        pts=1.0,
        seq=1,
        width=image.shape[1],
        height=image.shape[0],
        decode_time_ms=0.0,
    )


def test_overlay_renderer_none_mode_draws_nothing() -> None:
    bed = BoundingBox(1, 1, 20, 20, 0.9)
    person = BoundingBox(2, 2, 10, 10, 0.9)
    observation = FrameObservation(detections=((person,), ()), regions=((bed,), ()))

    rendered = OverlayRenderer(mode="none").render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (),
    )

    assert rendered.shape == (32, 32, 3)
    assert int(rendered.sum()) == 0


def test_overlay_renderer_bedexit_does_not_call_bed_exit_update(monkeypatch) -> None:
    # worker's BedExitMonitor collapsed edge's separate update_boxes() into
    # update() (worker/domains/bed_exit/detector.py:52-92 has no update_boxes
    # method at all), so only update() can be guarded here.
    def fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("renderer must not mutate bed-exit detector")

    monkeypatch.setattr(BedExitMonitor, "update", fail)
    bed = BoundingBox(1, 1, 20, 20, 0.9)
    person = BoundingBox(2, 2, 10, 10, 0.9)
    observation = FrameObservation(detections=((person,), ()), regions=((bed,), ()))
    snapshot = BedExitDebugSnapshot(
        frame_index=1,
        person_boxes=(person,),
        bed_boxes=(bed,),
        statuses=(BedStatus(0, bed, "occupied", person_id=1),),
        bed_region=BedRegionDebugSnapshot(source="fresh"),
    )

    rendered = OverlayRenderer(mode="bedexit").render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (snapshot,),
    )
    assert rendered.shape == (32, 32, 3)
    assert int(rendered.sum()) > 0


def test_overlay_renderer_encodes_jpeg() -> None:
    observation = FrameObservation()
    jpeg = OverlayRenderer(mode="bedexit").encode_jpeg(
        _packet(np.zeros((16, 16, 3), dtype=np.uint8)),
        observation,
        (),
    )
    assert jpeg.startswith(b"\xff\xd8")


def test_overlay_renderer_bedexit_draws_dashed_bed_outline_not_filled() -> None:
    # bedexit mode draws a dashed outline (worker/pipeline/output/
    # _overlay_primitives.py draw_dashed_region), not the legacy translucent
    # segmentation fill -- the interior of the polygon must stay untouched.
    polygon = ((4, 4), (28, 4), (28, 28), (4, 28))
    bed = BoundingBox(4, 4, 28, 28, 0.9, polygon)
    observation = FrameObservation(detections=((), ()), regions=((bed,), ()))

    rendered = OverlayRenderer(mode="bedexit").render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (),
    )

    # A dash starts at the very first edge point (4, 4) -> nearby pixels along
    # the top edge are drawn.
    assert int(rendered[4, 6].sum()) > 0
    # The polygon interior is never filled in dashed/outline mode.
    assert int(rendered[16, 16].sum()) == 0
    # A pixel outside the polygon entirely stays black.
    assert int(rendered[1, 1].sum()) == 0


def test_overlay_renderer_fall_mode_labels_are_not_positionally_naive() -> None:
    # FallWindowClassifier.classify() filters out None track ids when building
    # `labels`, so `labels[k]` is not necessarily `boxes[k]`'s label. Use two
    # live tracks so the renderer must replay the track-id walk rather than
    # zip(boxes, labels) directly.
    fall_box = BoundingBox(2, 2, 10, 10, 0.9)
    normal_box = BoundingBox(15, 2, 25, 10, 0.9)
    observation = FrameObservation(
        detections=(
            (fall_box, normal_box),
            (
                DetectionLabel(text="FALL", confidence=0.9, is_fall=True),
                DetectionLabel(text="NORMAL", confidence=0.9, is_fall=False),
            ),
        ),
        track_ids=(1, 2),
    )

    rendered = OverlayRenderer(mode="fall").render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (),
    )

    fall_pixels = np.all(rendered == np.array(FALL_LABEL_COLOR), axis=-1)
    normal_pixels = np.all(rendered == np.array(NORMAL_LABEL_COLOR), axis=-1)
    assert bool(np.any(fall_pixels))
    assert bool(np.any(normal_pixels))


def test_overlay_renderer_fall_mode_draws_no_bed() -> None:
    bed = BoundingBox(1, 1, 20, 20, 0.9)
    observation = FrameObservation(detections=((), ()), regions=((bed,), ()))

    rendered = OverlayRenderer(mode="fall").render(
        _packet(np.zeros((32, 32, 3), dtype=np.uint8)),
        observation,
        (),
    )

    assert int(rendered.sum()) == 0
