from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from worker.pipeline.trace.models import (
    DetailUnavailableReason,
    RecoveredCameraTrace,
    TraceFrame,
    TraceTruncation,
    trace_frame_row_count,
    trace_frame_size_bytes,
)


@dataclass(slots=True)
class _CameraCache:
    frames: dict[str, TraceFrame] = field(default_factory=dict)
    handoff_dropped: int = 0
    persistence_failed: int = 0
    pruned: int = 0
    detail_unavailable_reason: DetailUnavailableReason | None = None


@dataclass(slots=True)
class _Cache:
    cameras: dict[str, _CameraCache] = field(default_factory=dict)


_CACHES: dict[str, _Cache] = {}
_CACHES_LOCK = threading.Lock()


class TraceStore:
    """Process-local bounded cache for droppable per-frame analysis detail.

    Decision basis is deliberately not durable here: admitted events carry it in
    their delivery-queue EVENT envelope. This cache exists only to support
    best-effort annotated derivatives while the worker process is alive.
    """

    def __init__(self, cache_key: Path) -> None:
        self._cache_key = str(cache_key)

    def persist_batch(
        self,
        frames: Sequence[TraceFrame],
        *,
        max_frames_per_camera: int,
        max_age_seconds: float,
        max_cameras: int,
        max_total_frames: int,
        max_total_rows: int,
        max_total_bytes: int,
        dropped_by_camera: Mapping[str, int],
        failed_by_camera: Mapping[str, int] | None = None,
    ) -> int:
        del max_cameras  # A cache is bounded by its retained frame budget.
        failed = failed_by_camera or {}
        with _CACHES_LOCK:
            cache = _CACHES.setdefault(self._cache_key, _Cache())
            for camera_id, count in dropped_by_camera.items():
                camera = cache.cameras.setdefault(camera_id, _CameraCache())
                camera.handoff_dropped += count
                camera.detail_unavailable_reason = DetailUnavailableReason.BOUNDED_HANDOFF_CAPACITY
            for camera_id, count in failed.items():
                camera = cache.cameras.setdefault(camera_id, _CameraCache())
                camera.persistence_failed += count
                camera.detail_unavailable_reason = DetailUnavailableReason.DETAIL_CACHE_WRITE_FAILED
            inserted = 0
            for frame in frames:
                camera = cache.cameras.setdefault(frame.analysis.frame_key[1], _CameraCache())
                if frame.analysis.trace_id not in camera.frames:
                    camera.frames[frame.analysis.trace_id] = frame
                    inserted += 1
            self._prune(
                cache,
                max_frames_per_camera=max_frames_per_camera,
                max_age_seconds=max_age_seconds,
                max_total_frames=max_total_frames,
                max_total_rows=max_total_rows,
                max_total_bytes=max_total_bytes,
            )
            return inserted

    def recover_camera(self, camera_id: str) -> RecoveredCameraTrace:
        with _CACHES_LOCK:
            camera = _CACHES.get(self._cache_key, _Cache()).cameras.get(camera_id)
            if camera is None:
                return RecoveredCameraTrace((), (), TraceTruncation(0, 0, None, None))
            frames = tuple(sorted(camera.frames.values(), key=_frame_order))
            analyses = tuple(frame.analysis for frame in frames)
            decisions = tuple(
                decision for frame in frames for decision in frame.decisions
            )
            return RecoveredCameraTrace(
                analyses,
                decisions,
                TraceTruncation(
                    camera.handoff_dropped,
                    camera.pruned,
                    None if not analyses else analyses[0].frame_key[3],
                    None if not analyses else analyses[-1].frame_key[3],
                    camera.persistence_failed,
                    0,
                    None if not analyses else analyses[0].frame_key,
                    None if not analyses else analyses[-1].frame_key,
                    camera.detail_unavailable_reason,
                ),
            )

    @staticmethod
    def _prune(
        cache: _Cache,
        *,
        max_frames_per_camera: int,
        max_age_seconds: float,
        max_total_frames: int,
        max_total_rows: int,
        max_total_bytes: int,
    ) -> None:
        for camera in cache.cameras.values():
            ordered = sorted(camera.frames.values(), key=_frame_order, reverse=True)
            newest_time = next(
                (frame.analysis.source_time.value
                 for frame in ordered if frame.analysis.source_time.value is not None),
                None,
            )
            keep = ordered[:max_frames_per_camera]
            if newest_time is not None:
                keep = [
                    frame for frame in keep
                    if frame.analysis.source_time.value is None
                    or frame.analysis.source_time.value >= newest_time - max_age_seconds
                ]
            removed = set(camera.frames).difference(frame.analysis.trace_id for frame in keep)
            camera.pruned += len(removed)
            if removed:
                camera.detail_unavailable_reason = DetailUnavailableReason.RETENTION_BOUND
            for trace_id in removed:
                del camera.frames[trace_id]

        retained = sorted(
            (frame for camera in cache.cameras.values() for frame in camera.frames.values()),
            key=_frame_order,
            reverse=True,
        )
        used_rows = used_bytes = 0
        keep_ids: set[str] = set()
        for frame in retained:
            rows = trace_frame_row_count(frame)
            size = trace_frame_size_bytes(frame)
            if (
                len(keep_ids) >= max_total_frames
                or used_rows + rows > max_total_rows
                or used_bytes + size > max_total_bytes
            ):
                continue
            keep_ids.add(frame.analysis.trace_id)
            used_rows += rows
            used_bytes += size
        for camera in cache.cameras.values():
            removed = set(camera.frames).difference(keep_ids)
            camera.pruned += len(removed)
            if removed:
                camera.detail_unavailable_reason = DetailUnavailableReason.RETENTION_BOUND
            for trace_id in removed:
                del camera.frames[trace_id]


def _frame_order(frame: TraceFrame) -> tuple[str, int, int, str]:
    boot_id, _camera_id, epoch, seq = frame.analysis.frame_key
    return boot_id, epoch, seq, frame.analysis.trace_id


__all__ = ["TraceStore"]
