"""Camera-keyed latest JPEG storage and live-view publication."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.types import FramePacket

OverlayMode = Literal["none", "bedexit", "fall"]


@dataclass(frozen=True, slots=True)
class LatestFrame:
    jpeg: bytes
    seq: int
    frame_index: int
    content_type: str = "image/jpeg"
    # Age of the cached observation drawn on this frame (`LiveViewPump`).
    # `None` means "no observation has ever been cached for this camera";
    # `overlay_stale` marks a frame published deliberately WITHOUT a skeleton
    # because the newest observation was older than the pump's threshold.
    observation_age_sec: float | None = None
    overlay_stale: bool = False


class LatestFrameStore:
    """Retain one non-consuming latest JPEG value for every known camera."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frames: dict[str, LatestFrame] = {}
        self._known_camera_ids: set[str] = set()
        self._viewer_counts: dict[str, int] = {}
        self._snapshot_demand: set[str] = set()
        self._mode: dict[str, OverlayMode] = {}
        self._mode_generation: dict[str, int] = {}
        self._demand_listener: Callable[[str, int, OverlayMode, bool], None] | None = None

    def set_demand_listener(
        self,
        listener: Callable[[str, int, OverlayMode, bool], None] | None,
    ) -> None:
        with self._condition:
            self._demand_listener = listener

    def register_camera(self, camera_id: str) -> None:
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._condition.notify_all()

    def mark_viewer_connected(self, camera_id: str) -> None:
        """Count one open ``/stream/{camera_id}`` HTTP connection."""
        with self._condition:
            self._viewer_counts[camera_id] = self._viewer_counts.get(camera_id, 0) + 1
            viewers = self._viewer_counts[camera_id]
            mode = self._mode.get(camera_id, "none")
            listener = self._demand_listener
            if listener is not None:
                _ = self._frames.pop(camera_id, None)
            self._condition.notify_all()
        if listener is not None:
            listener(camera_id, viewers, mode, False)

    def mark_viewer_disconnected(self, camera_id: str) -> None:
        """Undo one ``mark_viewer_connected`` (every stream return path calls this)."""
        with self._condition:
            count = self._viewer_counts.get(camera_id, 0) - 1
            viewers = max(count, 0)
            self._viewer_counts[camera_id] = viewers
            mode = self._mode.get(camera_id, "none")
            listener = self._demand_listener
            if viewers > 0 and listener is not None:
                _ = self._frames.pop(camera_id, None)
            self._condition.notify_all()
        if listener is not None:
            listener(camera_id, viewers, mode, False)

    def has_viewers(self, camera_id: str) -> bool:
        with self._condition:
            return self._viewer_counts.get(camera_id, 0) > 0

    def request_snapshot_refresh(self, camera_id: str) -> None:
        """Treat one ``/snapshot/{camera_id}`` request as a momentary viewer.

        Lets ``LiveViewSubscriber.publish`` encode exactly one fresh frame on
        its next call even when no stream viewer is connected, so the
        dashboard's periodic snapshot polling still gets a live frame under
        viewer gating instead of an ever-staler cached one.
        """
        with self._condition:
            self._snapshot_demand.add(camera_id)
            viewers = self._viewer_counts.get(camera_id, 0)
            mode = self._mode.get(camera_id, "none")
            listener = self._demand_listener
            if listener is not None:
                _ = self._frames.pop(camera_id, None)
            self._condition.notify_all()
        if listener is not None:
            listener(camera_id, viewers, mode, True)

    def consume_snapshot_demand(self, camera_id: str) -> bool:
        """Atomically check and clear the one-frame snapshot demand flag."""
        with self._condition:
            if camera_id in self._snapshot_demand:
                self._snapshot_demand.discard(camera_id)
                return True
            return False

    def set_mode(self, camera_id: str, mode: OverlayMode) -> None:
        with self._condition:
            if self._mode.get(camera_id, "none") == mode:
                return
            self._mode[camera_id] = mode
            generation = self._mode_generation.get(camera_id, 0) + 1
            self._mode_generation[camera_id] = generation
            _ = self._frames.pop(camera_id, None)
            viewers = self._viewer_counts.get(camera_id, 0)
            listener = self._demand_listener
            self._condition.notify_all()
        if listener is not None:
            listener(camera_id, viewers, mode, True)

    def get_mode(self, camera_id: str) -> OverlayMode:
        with self._condition:
            return self._mode.get(camera_id, "none")

    def mode_generation(self, camera_id: str) -> int:
        with self._condition:
            return self._mode_generation.get(camera_id, 0)

    def publish_jpeg(
        self,
        camera_id: str,
        jpeg: bytes,
        *,
        frame_index: int,
        seq: int | None = None,
        observation_age_sec: float | None = None,
        overlay_stale: bool = False,
        expected_mode: OverlayMode | None = None,
        expected_generation: int | None = None,
    ) -> bool:
        latest = LatestFrame(
            jpeg=bytes(jpeg),
            seq=frame_index if seq is None else seq,
            frame_index=frame_index,
            observation_age_sec=observation_age_sec,
            overlay_stale=overlay_stale,
        )
        with self._condition:
            if expected_mode is not None and self._mode.get(camera_id, "none") != expected_mode:
                return False
            if (
                expected_generation is not None
                and self._mode_generation.get(camera_id, 0) != expected_generation
            ):
                return False
            self._known_camera_ids.add(camera_id)
            self._frames[camera_id] = latest
            self._condition.notify_all()
            return True

    def get_latest(self, camera_id: str) -> LatestFrame | None:
        with self._condition:
            return self._frames.get(camera_id)

    def clear_camera(self, camera_id: str) -> None:
        """Discard a preview that belongs to a retired stream identity."""
        with self._condition:
            _ = self._frames.pop(camera_id, None)
            self._snapshot_demand.discard(camera_id)
            self._condition.notify_all()

    def is_known(self, camera_id: str) -> bool:
        with self._condition:
            return camera_id in self._known_camera_ids

    def wait_for_latest(
        self,
        camera_id: str,
        *,
        previous: LatestFrame | None,
        timeout: float,
    ) -> LatestFrame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frames.get(camera_id) is not previous,
                timeout=timeout,
            )
            return self._frames.get(camera_id)


class LiveViewRenderer(Protocol):
    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...],
    ) -> bytes: ...


class LiveViewSubscriber:
    """Publish injected JPEG output without changing inference state or flow."""

    def __init__(
        self,
        store: LatestFrameStore,
        renderer: LiveViewRenderer,
    ) -> None:
        self._store = store
        self._renderer = renderer

    def publish(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
        *,
        observation_age_sec: float | None = None,
        overlay_stale: bool = False,
    ) -> bool:
        """Encode one frame for viewers, tagged with its overlay's age.

        ``observation_age_sec``/``overlay_stale`` come from ``LiveViewPump``:
        the frame is current, the overlay on it may not be. A frame whose
        newest cached observation was too old to draw arrives here already
        stripped of its skeleton and flagged, so the store never lets a viewer
        read an old pose as a present one. An empty-but-fresh observation
        (nobody in the room) is NOT stale and is not flagged.
        """
        camera_id = packet.camera_id
        # Confirmed product decision (#48): no viewers means no encoding at
        # all, not merely no transmission. A pending snapshot demand (see
        # `LatestFrameStore.request_snapshot_refresh`) counts as one momentary
        # viewer so `/snapshot/{id}` polling still gets a fresh frame.
        snapshot_requested = self._store.consume_snapshot_demand(camera_id)
        if not self._store.has_viewers(camera_id) and not snapshot_requested:
            return False
        try:
            jpeg = self._renderer.encode_jpeg(packet, observation, debug_snapshots)
        except Exception:  # noqa: BLE001 - preview output must not stop the live lane
            return False
        self._store.publish_jpeg(
            camera_id,
            jpeg,
            seq=packet.seq,
            frame_index=packet.frame.index,
            observation_age_sec=observation_age_sec,
            overlay_stale=overlay_stale,
        )
        return True


__all__ = [
    "LatestFrame",
    "LatestFrameStore",
    "LiveViewSubscriber",
]
