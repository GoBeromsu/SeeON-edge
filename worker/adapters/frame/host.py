from __future__ import annotations

from contracts.frame import Frame
from worker.types.capabilities import HOST_RGB, ConverterCapabilities
from worker.types.copy_metrics import CopyMetrics
from worker.types.frame_memory import FrameLease, MemoryKind


class HostFrameMaterializer:
    """Named host boundary; ``view`` is zero-copy and ``materialize`` is counted."""

    __slots__ = ("capabilities", "metrics", "name")

    def __init__(self, *, name: str, metrics: CopyMetrics) -> None:
        if not name:
            raise ValueError("host materializer name must be non-empty")
        self.name = name
        self.metrics = metrics
        self.capabilities = ConverterCapabilities(name, HOST_RGB, HOST_RGB, True)

    def view(self, lease: FrameLease) -> Frame:
        if lease.descriptor.memory_kind is not MemoryKind.HOST:
            raise RuntimeError(
                f"materializer {self.name!r} cannot view non-host frame "
                f"{lease.descriptor.memory_kind.value!r} without a copy implementation"
            )
        return lease.host_frame

    def materialize(self, lease: FrameLease) -> FrameLease:
        source = self.view(lease)
        image = source.image.copy()
        self.metrics.record_materialization(adapter=self.name, size_bytes=int(image.nbytes))
        return FrameLease.from_host(
            Frame(index=source.index, time_sec=source.time_sec, image=image),
            pixel_format=lease.descriptor.pixel_format,
        )


__all__ = ["HostFrameMaterializer"]
