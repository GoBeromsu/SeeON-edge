"""Profile-scoped ownership of the authoritative NVIDIA native child."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

from worker.pipeline.output.evidence.packet_repository import PacketRingRepository
from worker.pipeline.output.evidence.scene_repository import SceneRingRepository
from worker.pipeline.output.live_view import LatestFrameStore
from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.errors import ChildFatalError
from worker.runtime.deepstream.supervisor import (
    DeepStreamChildSupervisor,
    SharedSupervisorResources,
)


@dataclass(frozen=True, slots=True)
class NvidiaMediaResources:
    packet_repository: PacketRingRepository
    scene_repository: SceneRingRepository
    preview_frames: LatestFrameStore
    fatal_exit: Callable[[int], None]


@final
class NvidiaMediaPlane:
    """Own exactly one child and project its stores into existing surfaces."""

    def __init__(self, config: ChildConfig, resources: NvidiaMediaResources) -> None:
        self._child = DeepStreamChildSupervisor(
            config,
            SharedSupervisorResources(
                resources.packet_repository,
                resources.scene_repository,
                resources.preview_frames,
            ),
        )
        self._fatal_exit = resources.fatal_exit
        self._monitor: threading.Thread | None = None

    @property
    def child(self) -> DeepStreamChildSupervisor:
        return self._child

    def start(self) -> None:
        self._child.start()
        monitor = threading.Thread(
            target=self._wait_child,
            name="nvidia-media-plane-fatal",
            daemon=True,
        )
        monitor.start()
        self._monitor = monitor

    def stop(self) -> None:
        self._child.stop()
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=5.0)
        self._monitor = None

    def _wait_child(self) -> None:
        try:
            _ = self._child.wait()
        except ChildFatalError:
            self._fatal_exit(4)


__all__ = ["NvidiaMediaPlane", "NvidiaMediaResources"]
