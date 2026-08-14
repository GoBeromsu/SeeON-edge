from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types import FramePacket


@runtime_checkable
class FrameSubscription(Protocol):
    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None: ...

    def close(self) -> None: ...


@runtime_checkable
class FrameBus(Protocol):
    """Fan out packets; ``publish`` consumes the caller's lease handle."""

    def subscribe(
        self,
        name: str,
        *,
        capacity: int,
        latest_only: bool = False,
    ) -> FrameSubscription: ...

    def publish(self, packet: FramePacket) -> None: ...


__all__ = ["FrameBus", "FrameSubscription"]
