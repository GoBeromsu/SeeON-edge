"""Transfer/pressure/outcome accounting for the device-input NVENC prototype.

Same convention as ``worker.adapters.decode.nvdec_device.telemetry``: local,
thread-safe, snapshot-based, never raises past its own boundary. Kept as a
distinct type (not a reuse of ``DeviceResidencyTelemetry``) because this
encoder side additionally tracks submission outcomes, encoder-choice
provenance, and artifact bytes -- a materially different shape than a decode
pool's H2D/D2H + acquire/release counters.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceEncoderTelemetrySnapshot:
    submissions_accepted: int
    submissions_rejected_host_input: int
    d2h_transfers: int
    d2h_bytes: int
    frames_encoded: int
    encode_time_ms_total: int | float
    sessions_opened: int
    sessions_failed: int
    pool_capacity: int
    pool_outstanding: int
    pool_high_watermark: int
    pool_exhaustion_events: int
    artifacts_finalized: int
    artifact_bytes_total: int
    last_unavailable_reason: str | None


class DeviceEncoderTelemetry:
    """Thread-safe counters for one camera's device-input NVENC pipeline."""

    __slots__ = (
        "_artifact_bytes_total",
        "_artifacts_finalized",
        "_d2h_bytes",
        "_d2h_transfers",
        "_encode_time_ms_total",
        "_frames_encoded",
        "_last_unavailable_reason",
        "_lock",
        "_pool_capacity",
        "_pool_exhaustion_events",
        "_pool_high_watermark",
        "_pool_outstanding",
        "_sessions_failed",
        "_sessions_opened",
        "_submissions_accepted",
        "_submissions_rejected_host_input",
    )

    def __init__(self, *, pool_capacity: int) -> None:
        if pool_capacity <= 0:
            raise ValueError("pool capacity must be positive")
        self._lock = threading.Lock()
        self._submissions_accepted = 0
        self._submissions_rejected_host_input = 0
        self._d2h_transfers = 0
        self._d2h_bytes = 0
        self._frames_encoded = 0
        self._encode_time_ms_total = 0.0
        self._sessions_opened = 0
        self._sessions_failed = 0
        self._pool_capacity = pool_capacity
        self._pool_outstanding = 0
        self._pool_high_watermark = 0
        self._pool_exhaustion_events = 0
        self._artifacts_finalized = 0
        self._artifact_bytes_total = 0
        self._last_unavailable_reason: str | None = None

    def record_submission_accepted(self) -> None:
        with self._lock:
            self._submissions_accepted += 1

    def record_submission_rejected_host_input(self, reason: str) -> None:
        if not reason:
            raise ValueError("rejection reason must be non-empty")
        with self._lock:
            self._submissions_rejected_host_input += 1
            self._last_unavailable_reason = reason

    def record_d2h(self, size_bytes: int) -> None:
        """Record a host readback that happens only at the declared output seam.

        This prototype's zero-host-transfer invariant is about the
        decode/overlay/encode-input path, not the final encoded byte stream:
        NVENC's own compressed bitstream output must still cross to host
        memory once, to be written to the evidence store. That single,
        declared, per-artifact readback is what this counter tracks --
        distinct from a forbidden full-frame surface readback, which never
        happens in this path.
        """
        if size_bytes < 0:
            raise ValueError("transfer byte count must not be negative")
        with self._lock:
            self._d2h_transfers += 1
            self._d2h_bytes += size_bytes

    def record_frame_encoded(self, elapsed_ms: float) -> None:
        if elapsed_ms < 0:
            raise ValueError("encode time must not be negative")
        with self._lock:
            self._frames_encoded += 1
            self._encode_time_ms_total += elapsed_ms

    def record_session_opened(self) -> None:
        with self._lock:
            self._sessions_opened += 1

    def record_session_failed(self, reason: str) -> None:
        if not reason:
            raise ValueError("failure reason must be non-empty")
        with self._lock:
            self._sessions_failed += 1
            self._last_unavailable_reason = reason

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

    def record_artifact_finalized(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("artifact byte count must not be negative")
        with self._lock:
            self._artifacts_finalized += 1
            self._artifact_bytes_total += size_bytes

    def snapshot(self) -> DeviceEncoderTelemetrySnapshot:
        with self._lock:
            return DeviceEncoderTelemetrySnapshot(
                submissions_accepted=self._submissions_accepted,
                submissions_rejected_host_input=self._submissions_rejected_host_input,
                d2h_transfers=self._d2h_transfers,
                d2h_bytes=self._d2h_bytes,
                frames_encoded=self._frames_encoded,
                encode_time_ms_total=self._encode_time_ms_total,
                sessions_opened=self._sessions_opened,
                sessions_failed=self._sessions_failed,
                pool_capacity=self._pool_capacity,
                pool_outstanding=self._pool_outstanding,
                pool_high_watermark=self._pool_high_watermark,
                pool_exhaustion_events=self._pool_exhaustion_events,
                artifacts_finalized=self._artifacts_finalized,
                artifact_bytes_total=self._artifact_bytes_total,
                last_unavailable_reason=self._last_unavailable_reason,
            )


__all__ = ["DeviceEncoderTelemetry", "DeviceEncoderTelemetrySnapshot"]
