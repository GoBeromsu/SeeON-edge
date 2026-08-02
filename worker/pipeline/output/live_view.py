"""Camera-keyed latest JPEG storage and live-view publication."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import cv2

from contracts.observation import FrameObservation
from worker.domains.bed_exit import BedExitDebugSnapshot
from worker.pipeline.output.overlay import OverlayEncodingError, OverlayRenderer
from worker.types import FramePacket


@dataclass(frozen=True, slots=True)
class LatestFrame:
    jpeg: bytes
    seq: int
    frame_index: int
    content_type: str = "image/jpeg"


class LatestFrameStore:
    """Retain one non-consuming latest JPEG value for every known camera."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frames: dict[str, LatestFrame] = {}
        self._known_camera_ids: set[str] = set()
        self._viewer_counts: dict[str, int] = {}
        self._snapshot_demand: set[str] = set()
        self._show_pose: dict[str, bool] = {}

    def register_camera(self, camera_id: str) -> None:
        with self._condition:
            self._known_camera_ids.add(camera_id)
            self._condition.notify_all()

    def mark_viewer_connected(self, camera_id: str) -> None:
        """Count one open ``/stream/{camera_id}`` HTTP connection."""
        with self._condition:
            self._viewer_counts[camera_id] = self._viewer_counts.get(camera_id, 0) + 1

    def mark_viewer_disconnected(self, camera_id: str) -> None:
        """Undo one ``mark_viewer_connected`` (every stream return path calls this)."""
        with self._condition:
            count = self._viewer_counts.get(camera_id, 0) - 1
            self._viewer_counts[camera_id] = max(count, 0)

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

    def consume_snapshot_demand(self, camera_id: str) -> bool:
        """Atomically check and clear the one-frame snapshot demand flag."""
        with self._condition:
            if camera_id in self._snapshot_demand:
                self._snapshot_demand.discard(camera_id)
                return True
            return False

    def set_show_pose(self, camera_id: str, show_pose: bool) -> None:
        with self._condition:
            self._show_pose[camera_id] = show_pose

    def get_show_pose(self, camera_id: str) -> bool:
        with self._condition:
            return self._show_pose.get(camera_id, False)

    def publish_jpeg(
        self,
        camera_id: str,
        jpeg: bytes,
        *,
        frame_index: int,
        seq: int | None = None,
    ) -> None:
        latest = LatestFrame(
            jpeg=bytes(jpeg),
            seq=frame_index if seq is None else seq,
            frame_index=frame_index,
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


class _PerCameraPoseRenderer:
    """Default ``LiveViewSubscriber`` renderer: per-camera runtime pose toggle.

    ``OverlayRenderer.show_pose`` (overlay.py:40) is a frozen, per-instance
    construction switch. Sharing one ``OverlayRenderer`` across every camera
    (the previous ``worker.py`` wiring) collapses the toggle into a single
    process-global switch. This reads each camera's toggle from ``store`` --
    the same camera-keyed collaborator already threaded through the worker --
    and constructs a fresh, stateless ``OverlayRenderer`` per call, so turning
    pose off for one camera never affects another, and "off" still skips the
    keypoint loop entirely (overlay.py:57-58's existing ``if self.show_pose``
    gate), not just the drawing.
    """

    def __init__(self, store: LatestFrameStore) -> None:
        self._store = store

    def encode_jpeg(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bytes:
        show_pose = self._store.get_show_pose(packet.camera_id)
        return OverlayRenderer(show_pose=show_pose).encode_jpeg(
            packet, observation, debug_snapshots
        )


class LiveViewSubscriber:
    """Render optional live output without changing inference state or flow."""

    def __init__(
        self,
        store: LatestFrameStore,
        renderer: LiveViewRenderer | None = None,
    ) -> None:
        self._store = store
        self._renderer = (
            renderer if renderer is not None else _PerCameraPoseRenderer(store)
        )

    def publish(
        self,
        packet: FramePacket,
        observation: FrameObservation,
        debug_snapshots: tuple[BedExitDebugSnapshot, ...] = (),
    ) -> bool:
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
        )
        return True


__all__ = [
    "LatestFrame",
    "LatestFrameStore",
    "LiveViewSubscriber",
]
