"""Best-effort CPU overlays consuming the canonical hardware-neutral scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output._overlay_primitives import (
    FALL_LABEL_COLOR,
    NORMAL_LABEL_COLOR,
    draw_box,
    draw_caption,
    draw_dashed_region,
    draw_label,
    draw_pose,
    draw_region,
)
from worker.pipeline.output._scene_renderer import render_scene
from worker.pipeline.output.overlay_scene import OverlaySceneBuilder
from worker.types import FramePacket
from worker.types.overlay_scene import OverlayScene

MAX_SNAPSHOT_BYTES: Final = 200 * 1024
OverlayMode = Literal["none", "bedexit", "fall"]


class OverlayEncodingError(RuntimeError):
    """Raised when OpenCV cannot encode a rendered overlay."""


@dataclass(frozen=True, slots=True)
class OverlayRenderer:
    """CPU reference renderer; both live stills and video frames use one scene."""

    mode: OverlayMode = "none"
    backend_id: str = "opencv-cpu"
    render_version: str = "overlay-cpu.v1"
    input_memory_kind: str = "host"

    def render(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> NDArray[np.uint8]:
        image = packet.borrow_host_frame().image.copy()
        if self.mode == "none":
            return image
        scene = OverlaySceneBuilder().from_live(
            packet,
            observation,
            debug_snapshots,
            mode=self.mode,
        )
        return render_scene(image, scene)

    def render_scene(
        self,
        packet: FramePacket,
        scene: OverlayScene,
    ) -> NDArray[np.uint8]:
        image = packet.borrow_host_frame().image.copy()
        return render_scene(image, scene)

    def encode_scene_jpeg(self, packet: FramePacket, scene: OverlayScene) -> bytes:
        return _encode_jpeg(self.render_scene(packet, scene))

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bytes:
        return _encode_jpeg(self.render(packet, observation, debug_snapshots))

    def encode_jpeg_bounded(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
        *,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
    ) -> bytes | None:
        try:
            image = self.render(packet, observation, debug_snapshots)
            encoded = _encode_bounded_image(image, max_bytes=max_bytes)
            if encoded is not None:
                return encoded
            downscaled = np.asarray(
                cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                dtype=np.uint8,
            )
            return _encode_bounded_image(downscaled, max_bytes=max_bytes)
        except (cv2.error, OverlayEncodingError, ValueError):
            return None


def _encode_jpeg(image: NDArray[np.uint8]) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise OverlayEncodingError("failed to encode overlay JPEG")
    return bytes(encoded)


def _encode_bounded_image(image: NDArray[np.uint8], *, max_bytes: int) -> bytes | None:
    for quality in (85, 70, 55, 40, 25):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            jpeg = bytes(encoded)
            if len(jpeg) <= max_bytes:
                return jpeg
    return None


__all__ = [
    "FALL_LABEL_COLOR",
    "MAX_SNAPSHOT_BYTES",
    "NORMAL_LABEL_COLOR",
    "OverlayEncodingError",
    "OverlayMode",
    "OverlayRenderer",
    "draw_box",
    "draw_caption",
    "draw_dashed_region",
    "draw_label",
    "draw_pose",
    "draw_region",
]
