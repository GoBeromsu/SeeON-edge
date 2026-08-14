from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types.source_packet import SourcePacket


@runtime_checkable
class SourcePacketSink(Protocol):
    def append(self, packet: SourcePacket) -> bool: ...


__all__ = ["SourcePacketSink"]
