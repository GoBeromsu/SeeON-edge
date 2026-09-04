"""Camera-keyed latest JPEG storage and live-view publication."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import cv2

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output.overlay import (
    OverlayEncodingError,
    OverlayMode,
    OverlayRenderer,
)
from worker.types import FramePacket


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
            listener(camera_id, viewers, mode, False)

    def mark_viewer_disconnected(self, camera_id: str) -> None:
        """Undo one ``mark_viewer_connected`` (every stream return path calls this)."""
        with self._condition:
            count = self._viewer_counts.get(camera_id, 0) - 1
            viewers = max(count, 0)
            self._viewer_counts[camera_id] = viewers
            mode = self._mode.get(camera_id, "none")
            listener = self._demand_listener
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
            self._mode[camera_id] = mode
            viewers = self._viewer_counts.get(camera_id, 0)
            listener = self._demand_listener
        if listener is not None:
            listener(camera_id, viewers, mode, False)

    def get_mode(self, camera_id: str) -> OverlayMode:
        with self._condition:
            return self._mode.get(camera_id, "none")

    def publish_jpeg(
        self,
        camera_id: str,
        jpeg: bytes,
        *,
        frame_index: int,
        seq: int | None = None,
        observation_age_sec: float | None = None,
        overlay_stale: bool = False,
    ) -> None:
        latest = LatestFrame(
            jpeg=bytes(jpeg),
            seq=frame_index if seq is None else seq,
            frame_index=frame_index,
            observation_age_sec=observation_age_sec,
            overlay_stale=overlay_stale,
        )
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._frames[camera_id] = latest
            self._condition.notify_all()

    def get_latest(self, camera_id: str) -> LatestFrame | None:
        with self._condition:
            return self._frames.get(camera_id)

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


class _PerCameraOverlayRenderer:
    """Default ``LiveViewSubscriber`` renderer: per-camera runtime overlay mode.

    ``OverlayRenderer.mode`` (overlay.py) is a frozen, per-instance
    construction switch. Sharing one ``OverlayRenderer`` across every camera
    (the previous ``worker.py`` wiring) collapses the switch into a single
    process-global value. This reads each camera's mode from ``store`` -- the
    same camera-keyed collaborator already threaded through the worker -- and
    constructs a fresh, stateless ``OverlayRenderer`` per call, so switching
    modes for one camera never affects another, and ``"none"`` still skips
    every drawing loop entirely (overlay.py's early return), not just the
    final composite.
    """

    def __init__(self, store: LatestFrameStore) -> None:
        self._store = store

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bytes:
        mode = self._store.get_mode(packet.camera_id)
        return OverlayRenderer(mode=mode).encode_jpeg(packet, observation, debug_snapshots)


class LiveViewSubscriber:
    """Render optional live output without changing inference state or flow."""

    def __init__(
        self,
        store: LatestFrameStore,
        renderer: LiveViewRenderer | None = None,
    ) -> None:
        self._store = store
        self._renderer = renderer if renderer is not None else _PerCameraOverlayRenderer(store)

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
        except (cv2.error, OverlayEncodingError):
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
