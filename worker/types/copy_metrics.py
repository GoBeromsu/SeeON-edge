from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CopyMetricsSnapshot:
    materializations: int
    copied_frames: int
    copied_bytes: int
    by_adapter: tuple[tuple[str, int], ...]


class CopyMetrics:
    __slots__ = ("_by_adapter", "_copied_bytes", "_copied_frames", "_lock", "_materializations")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._materializations = 0
        self._copied_frames = 0
        self._copied_bytes = 0
        self._by_adapter: dict[str, int] = {}

    def record_materialization(self, *, adapter: str, size_bytes: int) -> None:
        if not adapter:
            raise ValueError("copy adapter name must be non-empty")
        if size_bytes < 0:
            raise ValueError("copied byte count must not be negative")
        with self._lock:
            self._materializations += 1
            self._copied_frames += 1
            self._copied_bytes += size_bytes
            self._by_adapter[adapter] = self._by_adapter.get(adapter, 0) + 1

    def snapshot(self) -> CopyMetricsSnapshot:
        with self._lock:
            return CopyMetricsSnapshot(
                materializations=self._materializations,
                copied_frames=self._copied_frames,
                copied_bytes=self._copied_bytes,
                by_adapter=tuple(sorted(self._by_adapter.items())),
            )


__all__ = ["CopyMetrics", "CopyMetricsSnapshot"]
