"""Concrete per-camera bounded frame bus with balanced lease ownership."""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import monotonic

from worker.pipeline.bus.metrics import BusMetricsSnapshot
from worker.pipeline.bus.subscription import BoundedSubscription
from worker.types import FramePacket

_DEFAULT_EVIDENCE_CAPACITY = 128


class BoundedFrameBus:
    """Consume one publisher handle and fan out precharged consumer handles."""

    __slots__ = (
        "_clock",
        "_closed",
        "_lock",
        "_subscriptions",
        "evidence",
        "inference",
        "live",
    )

    def __init__(
        self,
        *,
        evidence_capacity: int = _DEFAULT_EVIDENCE_CAPACITY,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False
        self._subscriptions: dict[str, BoundedSubscription] = {}
        self.inference = self.subscribe("inference", capacity=1, latest_only=True)
        self.live = self.subscribe("live", capacity=1, latest_only=True)
        self.evidence = self.subscribe(
            "evidence", capacity=evidence_capacity, latest_only=False
        )

    def subscribe(
        self,
        name: str,
        *,
        capacity: int,
        latest_only: bool = False,
    ) -> BoundedSubscription:
        if not name:
            raise ValueError("subscription name must be non-empty")
        subscription = BoundedSubscription(
            capacity=capacity, latest_only=latest_only, clock=self._clock
        )
        with self._lock:
            previous = self._subscriptions.get(name)
            self._subscriptions[name] = subscription
            closed = self._closed
        if previous is not None:
            previous.close()
        if closed:
            subscription.close()
        return subscription

    def publish(self, packet: FramePacket) -> None:
        if packet.released:
            raise RuntimeError("cannot publish a released frame packet")
        with self._lock:
            subscriptions = () if self._closed else tuple(self._subscriptions.values())
        lease = packet.lease
        if lease is None:  # pragma: no cover - FramePacket post-init invariant
            raise RuntimeError("frame packet has no lease")
        fanout = lease.precharge(len(subscriptions))
        failure: Exception | None = None
        try:
            for subscription in subscriptions:
                child = packet.with_lease(fanout.take())
                try:
                    subscription.publish(child)
                except Exception as error:  # noqa: BLE001 - finish balanced fanout
                    if not child.released:
                        child.release()
                    if failure is None:
                        failure = error
        finally:
            fanout.seal()
            packet.release()
        if failure is not None:
            raise failure

    def metrics(self, name: str) -> BusMetricsSnapshot:
        with self._lock:
            subscription = self._subscriptions[name]
        return subscription.metrics()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscriptions = tuple(self._subscriptions.values())
        for subscription in subscriptions:
            subscription.close()


__all__ = ["BoundedFrameBus"]
