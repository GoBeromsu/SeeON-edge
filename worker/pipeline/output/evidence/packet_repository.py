from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

from worker.pipeline.output.evidence.packet_ring import PacketRingLimits, SourcePacketRing
from worker.types.source_packet import SourcePacket, StreamEpoch


@dataclass(slots=True)
class PacketRepositoryMetrics:
    global_limit_drops: int = 0
    global_limit_drop_bytes: int = 0
    global_evicted_packets: int = 0
    global_evicted_bytes: int = 0
    unknown_camera_drops: int = 0


@final
class PacketRingRepository:
    """One bounded packet ring per camera under a process-wide byte ceiling."""

    def __init__(
        self,
        camera_ids: tuple[str, ...],
        *,
        per_camera_limits: PacketRingLimits,
        global_max_bytes: int,
    ) -> None:
        if global_max_bytes <= 0:
            raise ValueError("global packet byte limit must be positive")
        self._rings = {
            camera_id: SourcePacketRing(camera_id, per_camera_limits) for camera_id in camera_ids
        }
        self._per_camera_limits = per_camera_limits
        self._global_max_bytes = global_max_bytes
        self._lock = threading.RLock()
        self._closed = False
        self._epoch_listeners: list[Callable[[StreamEpoch, StreamEpoch], None]] = []
        self.metrics = PacketRepositoryMetrics()

    def register_camera(self, camera_id: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot register a closed packet repository")
            if camera_id not in self._rings:
                self._rings[camera_id] = SourcePacketRing(camera_id, self._per_camera_limits)

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            ring = self._rings.pop(camera_id, None)
            if ring is not None:
                ring.close()

    def subscribe_epoch_roll(
        self, listener: Callable[[StreamEpoch, StreamEpoch], None]
    ) -> None:
        with self._lock:
            self._epoch_listeners.append(listener)

    def append(self, packet: SourcePacket) -> bool:
        with self._lock:
            if self._closed:
                return False
            ring = self._rings.get(packet.epoch.camera_id)
            if ring is None:
                self.metrics.unknown_camera_drops += 1
                return False
            if ring.active_epoch is not None and packet.epoch != ring.active_epoch:
                return ring.append(packet)
            if packet.size_bytes > self._global_max_bytes:
                self._record_global_drop(packet)
                return False
            while self._total_bytes() + packet.size_bytes > self._global_max_bytes:
                removed = self._evict_one()
                if removed is None:
                    self._record_global_drop(packet)
                    return False
                self.metrics.global_evicted_packets += 1
                self.metrics.global_evicted_bytes += removed.size_bytes
            return ring.append(packet)

    def roll_epoch(self, epoch: StreamEpoch) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot roll a closed packet repository")
            ring = self.ring(epoch.camera_id)
            previous = ring.active_epoch
            listeners = tuple(self._epoch_listeners)
        if previous is not None and previous != epoch:
            for listener in listeners:
                listener(previous, epoch)
        with self._lock:
            ring.roll_epoch(epoch)

    def ring(self, camera_id: str) -> SourcePacketRing:
        try:
            return self._rings[camera_id]
        except KeyError as exc:
            raise ValueError(f"camera {camera_id!r} has no packet ring") from exc

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes()

    def _total_bytes(self) -> int:
        return sum(ring.total_bytes for ring in self._rings.values())

    def _evict_one(self) -> SourcePacket | None:
        for ring in sorted(
            self._rings.values(),
            key=lambda candidate: candidate.total_bytes,
            reverse=True,
        ):
            removed = ring.evict_oldest_unleased()
            if removed is not None:
                return removed
        return None

    def _record_global_drop(self, packet: SourcePacket) -> None:
        self.metrics.global_limit_drops += 1
        self.metrics.global_limit_drop_bytes += packet.size_bytes

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for ring in self._rings.values():
                ring.close()


__all__ = ["PacketRepositoryMetrics", "PacketRingRepository"]
