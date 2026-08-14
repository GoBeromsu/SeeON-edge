"""Transfer/pressure accounting for the experimental device-resident pool.

Local-only counters, same convention as ``worker.types.copy_metrics.CopyMetrics``
(the production host-copy counter): thread-safe, snapshot-based, never raises
past its own boundary. Kept separate from ``CopyMetrics`` because this
prototype counts H2D/D2H byte transfers and pool pressure -- a materially
different shape than "one host materialization by adapter name" -- and
``worker/types`` cannot import this package's adapter-level concerns anyway
(import-linter: adapters never import back into types beyond what
``worker.types`` already exports).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceResidencyTelemetrySnapshot:
    h2d_transfers: int
    h2d_bytes: int
    d2h_transfers: int
    d2h_bytes: int
    pool_capacity: int
    pool_outstanding: int
    pool_high_watermark: int
    pool_exhaustion_events: int
    decode_time_ms_total: float
    decode_samples: int
    inference_time_ms_total: float
    inference_samples: int


class DeviceResidencyTelemetry:
    """Thread-safe counters for one camera's device-resident pipeline."""

    __slots__ = (
        "_d2h_bytes",
        "_d2h_transfers",
        "_decode_samples",
        "_decode_time_ms_total",
        "_h2d_bytes",
        "_h2d_transfers",
        "_inference_samples",
        "_inference_time_ms_total",
        "_lock",
        "_pool_capacity",
        "_pool_exhaustion_events",
        "_pool_high_watermark",
        "_pool_outstanding",
    )

    def __init__(self, *, pool_capacity: int) -> None:
        if pool_capacity <= 0:
            raise ValueError("pool capacity must be positive")
        self._lock = threading.Lock()
        self._h2d_transfers = 0
        self._h2d_bytes = 0
        self._d2h_transfers = 0
        self._d2h_bytes = 0
        self._pool_capacity = pool_capacity
        self._pool_outstanding = 0
        self._pool_high_watermark = 0
        self._pool_exhaustion_events = 0
        self._decode_time_ms_total = 0.0
        self._decode_samples = 0
        self._inference_time_ms_total = 0.0
        self._inference_samples = 0

    def record_h2d(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("transfer byte count must not be negative")
        with self._lock:
            self._h2d_transfers += 1
            self._h2d_bytes += size_bytes

    def record_d2h(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("transfer byte count must not be negative")
        with self._lock:
            self._d2h_transfers += 1
            self._d2h_bytes += size_bytes

    def record_acquire(self, outstanding: int) -> None:
        with self._lock:
            self._pool_outstanding = outstanding
            self._pool_high_watermark = max(self._pool_high_watermark, outstanding)

    def record_release(self, outstanding: int) -> None:
        with self._lock:
            self._pool_outstanding = outstanding

    def record_pool_exhausted(self) -> None:
        with self._lock:
            self._pool_exhaustion_events += 1

    def record_decode_time(self, elapsed_ms: float) -> None:
        if elapsed_ms < 0:
            raise ValueError("decode time must not be negative")
        with self._lock:
            self._decode_time_ms_total += elapsed_ms
            self._decode_samples += 1

    def record_inference_time(self, elapsed_ms: float) -> None:
        if elapsed_ms < 0:
            raise ValueError("inference time must not be negative")
        with self._lock:
            self._inference_time_ms_total += elapsed_ms
            self._inference_samples += 1

    def snapshot(self) -> DeviceResidencyTelemetrySnapshot:
        with self._lock:
            return DeviceResidencyTelemetrySnapshot(
                h2d_transfers=self._h2d_transfers,
                h2d_bytes=self._h2d_bytes,
                d2h_transfers=self._d2h_transfers,
                d2h_bytes=self._d2h_bytes,
                pool_capacity=self._pool_capacity,
                pool_outstanding=self._pool_outstanding,
                pool_high_watermark=self._pool_high_watermark,
                pool_exhaustion_events=self._pool_exhaustion_events,
                decode_time_ms_total=self._decode_time_ms_total,
                decode_samples=self._decode_samples,
                inference_time_ms_total=self._inference_time_ms_total,
                inference_samples=self._inference_samples,
            )


__all__ = ["DeviceResidencyTelemetry", "DeviceResidencyTelemetrySnapshot"]
