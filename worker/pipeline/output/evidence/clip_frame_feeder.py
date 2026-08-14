"""Drain one camera's ``bus.evidence`` tap into the shared ``ClipRecorder``.

``BoundedFrameBus.evidence`` (``worker/pipeline/bus/frame_bus.py``) is a
bounded FIFO subscription that exists for exactly one purpose: feeding
decoded frames to clip recording. Nothing reads it on its own --
``ClipRecorder`` only ever receives packets a caller pushes through
``on_frame``. This module is that caller: one feeder per camera drains its
own ``evidence`` subscription and forwards each complete packet unchanged.

Mirrors ``worker.pipeline.camera_pipeline.CameraPipelinePump``'s shape (bounded
poll loop, ``stop_event``, per-packet failure isolation) so it satisfies the
same ``camera_id``/``run``/``stop`` surface the composition root's
``IngestSupervisor`` already manages -- start/stop/join for this feeder ride
the same thread lifecycle as the ingest loop and pipeline pump, rather than
inventing a second one.
"""

from __future__ import annotations

import logging
import threading
from typing import Final, final

from worker.interfaces.bus import FrameSubscription
from worker.pipeline.output.evidence.clip_recorder import ClipRecorder
from worker.types import FramePacket

LOGGER: Final = logging.getLogger(__name__)
DEFAULT_POLL_TIMEOUT_SEC: Final = 0.2


@final
class ClipFrameFeeder:
    """Feed one camera's decoded frames into the shared ``ClipRecorder``."""

    def __init__(
        self,
        camera_id: str,
        subscription: FrameSubscription,
        recorder: ClipRecorder,
        *,
        poll_timeout_sec: float = DEFAULT_POLL_TIMEOUT_SEC,
    ) -> None:
        self._camera_id = camera_id
        self._subscription = subscription
        self._recorder = recorder
        self._poll_timeout_sec = poll_timeout_sec
        self._stop_event = threading.Event()
        self.failure_count = 0
        self.fed_count = 0

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def run(self) -> None:
        while not self._stop_event.is_set():
            packet = self._subscription.take(timeout_sec=self._poll_timeout_sec)
            if packet is None:
                continue
            try:
                self._feed_one(packet)
            finally:
                packet.release()

    def stop(self) -> None:
        self._stop_event.set()

    def _feed_one(self, packet: FramePacket) -> None:
        try:
            _ = self._recorder.on_frame(packet)
        except Exception as error:  # noqa: BLE001 - per-camera boundary, mirrors CameraPipelinePump
            self.failure_count += 1
            LOGGER.warning(
                "clip frame feeder failed to admit a frame",
                extra={"camera_id": self._camera_id, "error": type(error).__name__},
            )
            return
        self.fed_count += 1


__all__ = ["ClipFrameFeeder", "DEFAULT_POLL_TIMEOUT_SEC"]
