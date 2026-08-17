from __future__ import annotations

from typing import Protocol, runtime_checkable

from worker.types.source_packet import SourcePacket, StreamEpoch


@runtime_checkable
class SourcePacketSink(Protocol):
    def append(self, packet: SourcePacket) -> bool: ...


@runtime_checkable
class EpochRollingSourcePacketSink(SourcePacketSink, Protocol):
    def roll_epoch(self, epoch: StreamEpoch) -> None: ...


__all__ = ["EpochRollingSourcePacketSink", "SourcePacketSink"]
