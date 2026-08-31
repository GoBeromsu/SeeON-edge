from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import final

from worker.types import SceneRecord


@dataclass(frozen=True, slots=True)
class SceneRingLimits:
    max_frames: int = 1200
    max_bytes: int = 4 * 1024 * 1024
    max_duration_seconds: float = 70.0

    def __post_init__(self) -> None:
        if self.max_frames <= 0 or self.max_bytes <= 0:
            raise ValueError("scene ring frame and byte limits must be positive")
        if not math.isfinite(self.max_duration_seconds) or self.max_duration_seconds <= 0:
            raise ValueError("scene ring duration limit must be finite and positive")


DEFAULT_SCENE_RING_LIMITS = SceneRingLimits()


@dataclass(slots=True)
class SceneRingMetrics:
    accepted_frames: int = 0
    accepted_bytes: int = 0
    dropped_frames: int = 0
    dropped_bytes: int = 0
    evicted_frames: int = 0
    evicted_bytes: int = 0
    epoch_rolls: int = 0


@final
class CameraSceneRing:
    """Per-camera, lease-free history of immutable pre-serialized scene records."""

    def __init__(
        self, camera_id: str, limits: SceneRingLimits = DEFAULT_SCENE_RING_LIMITS
    ) -> None:
        if not camera_id:
            raise ValueError("scene ring camera id must not be blank")
        self.camera_id = camera_id
        self.limits = limits
        self.metrics = SceneRingMetrics()
        self._entries: deque[SceneRecord] = deque()
        self._active_epoch: int | None = None
        self._active_generation: int | None = None
        self._newest_pts: Fraction | None = None
        self._total_bytes = 0
        self._max_duration = Fraction(str(limits.max_duration_seconds))
        self._closed = False
        self._lock = threading.RLock()

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def append(self, record: SceneRecord) -> bool:
        if record.camera_id != self.camera_id:
            raise ValueError("scene record camera does not match its ring")
        with self._lock:
            if self._closed or record.size_bytes > self.limits.max_bytes:
                self._drop(record)
                return False
            if self._active_epoch is not None and (
                record.stream_epoch != self._active_epoch
                or (
                    self._active_generation is not None
                    and record.generation != self._active_generation
                )
            ):
                self._drop(record)
                return False
            self._entries.append(record)
            self._total_bytes += record.size_bytes
            if self._newest_pts is None or record.source_pts_sec > self._newest_pts:
                self._newest_pts = record.source_pts_sec
            self._trim_to_limits()
            self.metrics.accepted_frames += 1
            self.metrics.accepted_bytes += record.size_bytes
            return True

    def roll_epoch(self, epoch: int, generation: int | None = None) -> None:
        if epoch < 0 or (generation is not None and generation < 0):
            raise ValueError("scene epoch and generation must be non-negative")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot roll a closed scene ring")
            if self._active_epoch is not None and epoch < self._active_epoch:
                raise ValueError("scene ring epoch cannot move backwards")
            self._evict_all()
            self._active_epoch = epoch
            self._active_generation = generation
            self._newest_pts = None
            self.metrics.epoch_rolls += 1

    def select(
        self,
        trigger_epoch: int,
        start_pts_sec: Fraction,
        end_pts_sec: Fraction,
    ) -> tuple[SceneRecord, ...]:
        if end_pts_sec < start_pts_sec:
            raise ValueError("scene selection end must not precede start")
        with self._lock:
            if self._closed:
                return ()
            if self._active_epoch is not None and trigger_epoch != self._active_epoch:
                return ()
            return tuple(
                entry
                for entry in self._entries
                if entry.stream_epoch == trigger_epoch
                and start_pts_sec <= entry.source_pts_sec <= end_pts_sec
            )

    def evict_oldest(self) -> SceneRecord | None:
        with self._lock:
            if self._closed or not self._entries:
                return None
            return self._evict_one()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._evict_all()

    def _trim_to_limits(self) -> None:
        while self._entries and (
            len(self._entries) > self.limits.max_frames
            or self._total_bytes > self.limits.max_bytes
            or (
                self._newest_pts is not None
                and self._newest_pts - self._entries[0].source_pts_sec > self._max_duration
            )
        ):
            self._evict_one()

    def _evict_all(self) -> None:
        while self._entries:
            self._evict_one()

    def _evict_one(self) -> SceneRecord:
        removed = self._entries.popleft()
        self._total_bytes -= removed.size_bytes
        self.metrics.evicted_frames += 1
        self.metrics.evicted_bytes += removed.size_bytes
        return removed

    def _drop(self, record: SceneRecord) -> None:
        self.metrics.dropped_frames += 1
        self.metrics.dropped_bytes += record.size_bytes


__all__ = ["CameraSceneRing", "SceneRingLimits", "SceneRingMetrics"]
