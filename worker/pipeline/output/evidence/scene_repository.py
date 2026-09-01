from __future__ import annotations

import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import final

from worker.pipeline.output.evidence.scene_ring import (
    DEFAULT_SCENE_RING_LIMITS,
    CameraSceneRing,
    SceneRingLimits,
)
from worker.types import SceneRecord

SCENE_RING_GLOBAL_MAX_BYTES = 48 * 1024 * 1024


@dataclass(slots=True)
class SceneRepositoryMetrics:
    global_limit_drops: int = 0
    global_limit_drop_bytes: int = 0
    global_evicted_frames: int = 0
    global_evicted_bytes: int = 0
    unknown_camera_drops: int = 0
    active_ring_evictions: int = 0


@final
class SceneRingRepository:
    """Camera scene rings bounded by a process-wide, active-aware byte ceiling."""

    def __init__(
        self,
        camera_ids: tuple[str, ...] = (),
        *,
        per_camera_limits: SceneRingLimits = DEFAULT_SCENE_RING_LIMITS,
        global_max_bytes: int = SCENE_RING_GLOBAL_MAX_BYTES,
    ) -> None:
        if global_max_bytes <= 0:
            raise ValueError("global scene byte limit must be positive")
        self._rings = {
            camera_id: CameraSceneRing(camera_id, per_camera_limits) for camera_id in camera_ids
        }
        self._per_camera_limits = per_camera_limits
        self._global_max_bytes = global_max_bytes
        self._active_cameras: set[str] = set()
        self.metrics = SceneRepositoryMetrics()
        self._closed = False
        self._lock = threading.RLock()

    def register_camera(self, camera_id: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot register a closed scene repository")
            self._rings.setdefault(camera_id, CameraSceneRing(camera_id, self._per_camera_limits))

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            self._active_cameras.discard(camera_id)
            ring = self._rings.pop(camera_id, None)
            if ring is not None:
                ring.close()

    def append(self, record: SceneRecord) -> bool:
        with self._lock:
            if self._closed:
                return False
            ring = self._rings.get(record.camera_id)
            if ring is None:
                self.metrics.unknown_camera_drops += 1
                return False
            if record.size_bytes > self._global_max_bytes:
                self._record_global_drop(record)
                return False
            while self._total_bytes() + record.size_bytes > self._global_max_bytes:
                removed = self._evict_one()
                if removed is None:
                    self._record_global_drop(record)
                    return False
            return ring.append(record)

    def roll_epoch(self, camera_id: str, epoch: int) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot roll a closed scene repository")
            self.ring(camera_id).roll_epoch(epoch)

    def select(
        self, camera_id: str, trigger_epoch: int, start_pts_sec: Fraction, end_pts_sec: Fraction
    ) -> tuple[SceneRecord, ...]:
        return self.ring(camera_id).select(trigger_epoch, start_pts_sec, end_pts_sec)

    def mark_active(self, camera_id: str) -> None:
        with self._lock:
            self.ring(camera_id)
            self._active_cameras.add(camera_id)

    def clear_active(self, camera_id: str) -> None:
        with self._lock:
            self._active_cameras.discard(camera_id)

    def ring(self, camera_id: str) -> CameraSceneRing:
        try:
            return self._rings[camera_id]
        except KeyError as exc:
            raise ValueError(f"camera {camera_id!r} has no scene ring") from exc

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._active_cameras.clear()
            for ring in self._rings.values():
                ring.close()

    def _total_bytes(self) -> int:
        return sum(ring.total_bytes for ring in self._rings.values())

    def _evict_one(self) -> SceneRecord | None:
        rings = tuple(self._rings.values())
        non_active = tuple(ring for ring in rings if ring.camera_id not in self._active_cameras)
        for ring in sorted(non_active, key=lambda candidate: candidate.total_bytes, reverse=True):
            removed = ring.evict_oldest()
            if removed is not None:
                self.metrics.global_evicted_frames += 1
                self.metrics.global_evicted_bytes += removed.size_bytes
                return removed
        for ring in sorted(rings, key=lambda candidate: candidate.total_bytes, reverse=True):
            if ring.camera_id not in self._active_cameras:
                continue
            removed = ring.evict_oldest()
            if removed is not None:
                self.metrics.global_evicted_frames += 1
                self.metrics.global_evicted_bytes += removed.size_bytes
                self.metrics.active_ring_evictions += 1
                return removed
        return None

    def _record_global_drop(self, record: SceneRecord) -> None:
        self.metrics.global_limit_drops += 1
        self.metrics.global_limit_drop_bytes += record.size_bytes


__all__ = [
    "SCENE_RING_GLOBAL_MAX_BYTES",
    "SceneRepositoryMetrics",
    "SceneRingRepository",
]
