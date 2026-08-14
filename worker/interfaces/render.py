from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import FramePacket
from worker.types.overlay_scene import OverlayScene


@runtime_checkable
class OverlaySceneRenderer(Protocol):
    """Hardware seam: render one canonical scene without domain decisions."""

    backend_id: str
    render_version: str
    input_memory_kind: str

    def render_scene(self, packet: FramePacket, scene: OverlayScene) -> object: ...


__all__ = ["OverlaySceneRenderer"]
