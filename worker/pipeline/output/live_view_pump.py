"""Drain one camera's ``bus.live`` tap into the operator live view.

``BoundedFrameBus.live`` (``worker/pipeline/bus/frame_bus.py``) is a
latest-only capacity-1 subscription that ingest has always published to and
nothing ever consumed (``bus.metrics("live").taken == 0``): the preview was
published from inside ``CameraPipelinePump`` *after* ``analytics.process``,
so every preview frame queued behind a pose forward. This module is the real
consumer -- one pump per camera, on its own thread, drawing the LATEST cached
observation rather than waiting for this frame's inference.

Cached, not re-inferred: the per-camera pump records its observation into
:class:`LatestObservationStore` after analytics (a dict write, no model call),
and this pump reads it back tagged with its age. Past ``stale_after_sec`` the
skeleton is dropped instead of being drawn as if it were current -- an old
pose over a new frame is a lie about where the resident is standing.

Viewer gating (#48) stays where it already lives: ``LiveViewSubscriber.publish``
returns before touching the renderer when no viewer is attached, so with zero
viewers this loop costs one dict lookup per frame and zero JPEG encodes.

Mirrors ``ClipFrameFeeder``'s shape (bounded poll loop, ``stop_event``,
per-packet failure isolation) so it satisfies the same ``camera_id``/``run``/
``stop`` surface the composition root's ``IngestSupervisor`` already manages.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Final, Protocol, final

from contracts.observation import FrameObservation
from worker.interfaces.bus import FrameSubscription
from worker.types import FramePacket

LOGGER: Final = logging.getLogger(__name__)
DEFAULT_POLL_TIMEOUT_SEC: Final = 0.2
# One pose observation is worth ~200ms at the 5fps admission cadence; beyond
# half a second the skeleton no longer describes the frame it would be drawn
# on, so it is dropped rather than presented as current.
DEFAULT_STALE_AFTER_SEC: Final = 0.5
_EMPTY_OBSERVATION: Final = FrameObservation()


class LiveViewPublisher(Protocol):
    """Structural seam matched by ``LiveViewSubscriber`` (live_view.py)."""

    def publish(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[Any, ...] = (),
        *,
        observation_age_sec: float | None = None,
        overlay_stale: bool = False,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class CachedObservation:
    observation: FrameObservation
    debug_snapshots: tuple[Any, ...]
    frame_index: int
    recorded_at: float


@final
class LatestObservationStore:
    """Camera-keyed latest observation, written by each camera's own pump."""

    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: dict[str, CachedObservation] = {}

    def record(
        self,
        camera_id: str,
        observation: FrameObservation,
        debug_snapshots: tuple[Any, ...] = (),
        *,
        frame_index: int,
    ) -> None:
        cached = CachedObservation(
            observation=observation,
            debug_snapshots=debug_snapshots,
            frame_index=frame_index,
            recorded_at=self._clock(),
        )
        with self._lock:
            self._latest[camera_id] = cached

    def latest(self, camera_id: str) -> CachedObservation | None:
        with self._lock:
            return self._latest.get(camera_id)

    def now(self) -> float:
        return self._clock()


@final
class LiveViewPump:
    """Publish one camera's newest decoded frame with its newest observation."""

    def __init__(
        self,
        camera_id: str,
        subscription: FrameSubscription,
        live_view: LiveViewPublisher,
        observations: LatestObservationStore,
        *,
        poll_timeout_sec: float = DEFAULT_POLL_TIMEOUT_SEC,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if stale_after_sec <= 0:
            raise ValueError("live overlay staleness threshold must be positive")
        self._camera_id = camera_id
        self._subscription = subscription
        self._live_view = live_view
        self._observations = observations
        self._poll_timeout_sec = poll_timeout_sec
        self._stale_after_sec = stale_after_sec
        self._clock = clock
        self._stop_event = threading.Event()
        self.failure_count = 0
        self.published_count = 0
        self.stale_overlay_count = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()

    def run_once(self) -> bool:
        """Take at most one frame and publish it; ``False`` when none was ready."""
        packet = self._subscription.take(timeout_sec=self._poll_timeout_sec)
        if packet is None:
            return False
        try:
            self._publish_one(packet)
        finally:
            packet.release()
        return True

    def stop(self) -> None:
        self._stop_event.set()

    def _publish_one(self, packet: FramePacket) -> None:
        observation, snapshots, age, stale = self._overlay_for(packet.camera_id)
        try:
            _ = self._live_view.publish(
                packet,
                observation,
                snapshots,
                observation_age_sec=age,
                overlay_stale=stale,
            )
        except Exception as error:  # noqa: BLE001 - a cosmetic view is a tap
            self.failure_count += 1
            LOGGER.warning(
                "live view pump failed to publish a frame: camera_id=%s error=%s",
                self._camera_id,
                type(error).__name__,
                extra={"camera_id": self._camera_id, "error": type(error).__name__},
            )
            return
        self.published_count += 1

    def _overlay_for(
        self, camera_id: str
    ) -> tuple[FrameObservation, tuple[Any, ...], float | None, bool]:
        cached = self._observations.latest(camera_id)
        if cached is None:
            return _EMPTY_OBSERVATION, (), None, False
        age = max(0.0, self._clock() - cached.recorded_at)
        if age > self._stale_after_sec:
            # Stale: publish the fresh frame with no skeleton at all, and let
            # the age travel with it so the viewer can say "overlay is old"
            # instead of showing a pose from a different moment as current.
            self.stale_overlay_count += 1
            return _EMPTY_OBSERVATION, (), age, True
        return cached.observation, cached.debug_snapshots, age, False


__all__ = [
    "DEFAULT_POLL_TIMEOUT_SEC",
    "DEFAULT_STALE_AFTER_SEC",
    "CachedObservation",
    "LatestObservationStore",
    "LiveViewPublisher",
    "LiveViewPump",
]
