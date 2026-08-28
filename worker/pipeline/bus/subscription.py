"""Internal lease-owning bounded subscription backing ``BoundedFrameBus``."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from worker.pipeline.bus.metrics import BusMetricsSnapshot
from worker.types import FramePacket


class BoundedSubscription:
    """A queue owns every accepted packet until take, eviction, or close."""

    def __init__(
        self,
        *,
        capacity: int,
        latest_only: bool,
        clock: Callable[[], float],
    ) -> None:
        if capacity <= 0:
            raise ValueError("subscription capacity must be positive")
        self.capacity = capacity
        self.latest_only = latest_only
        self._clock = clock
        self._condition = threading.Condition()
        self._queue: deque[tuple[FramePacket, float]] = deque()
        self._published = 0
        self._taken = 0
        self._dropped = 0
        self._closed = False

    def publish(self, packet: FramePacket) -> None:
        release_after: FramePacket | None = None
        queued = False
        try:
            with self._condition:
                self._published += 1
                if self._closed or (len(self._queue) >= self.capacity and not self.latest_only):
                    self._dropped += 1
                    release_after = packet
                else:
                    published_at = self._clock()
                    if len(self._queue) >= self.capacity:
                        self._dropped += 1
                        evicted, _evicted_at = self._queue.popleft()
                        release_after = evicted
                    self._queue.append((packet, published_at))
                    queued = True
                    self._condition.notify_all()
        except Exception:
            if not queued and not packet.released:
                packet.release()
            raise
        if release_after is not None:
            release_after.release()

    def take(self, *, timeout_sec: float | None = None) -> FramePacket | None:
        with self._condition:
            if not self._queue and not self._closed and timeout_sec != 0:
                self._condition.wait_for(
                    lambda: bool(self._queue) or self._closed,
                    timeout=timeout_sec,
                )
            if not self._queue:
                return None
            packet, _published_at = self._queue.popleft()
            self._taken += 1
            return packet

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            queued = tuple(packet for packet, _published_at in self._queue)
            self._queue.clear()
            self._condition.notify_all()
        for packet in queued:
            packet.release()

    def metrics(self) -> BusMetricsSnapshot:
        with self._condition:
            queue_age_sec = 0.0
            if self._queue:
                _oldest_packet, oldest_published_at = self._queue[0]
                queue_age_sec = self._clock() - oldest_published_at
            return BusMetricsSnapshot(
                published=self._published,
                taken=self._taken,
                dropped=self._dropped,
                queue_age_sec=queue_age_sec,
            )


__all__ = ["BoundedSubscription"]
