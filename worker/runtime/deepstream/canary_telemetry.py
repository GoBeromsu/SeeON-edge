"""Opt-in raw native timing telemetry for the host-side canary harness."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict, final

TELEMETRY_ENV: Final = "SEEON_CANARY_TELEMETRY_PATH"
WINDOW_SECONDS: Final = 10.0


class CanaryWindowRecord(TypedDict):
    schema_version: int
    camera_id: str
    window_started_ns: int
    window_ended_ns: int
    decision_count: int
    latency_samples_ms: list[float]
    metadata_published: int
    metadata_overwritten: int
    timestamp_discontinuities: int


@final
class NativeCanaryTelemetry:
    """Mutable per-camera accumulator that seals non-overlapping decision windows."""

    def __init__(self, camera_id: str, path: Path) -> None:
        self._camera_id = camera_id
        self._path = path
        self._window_started_monotonic = time.monotonic()
        self._window_started_ns = time.time_ns()
        self._base_pts: int | None = None
        self._base_source_time_ns: int | None = None
        self._last_pts: int | None = None
        self._last_publish_sequence: int | None = None
        self._latencies_ms: list[float] = []
        self._published = 0
        self._overwritten = 0
        self._timestamp_discontinuities = 0

    @classmethod
    def from_environment(
        cls,
        camera_id: str,
        environ: Mapping[str, str] = os.environ,
    ) -> NativeCanaryTelemetry | None:
        raw = environ.get(TELEMETRY_ENV, "").strip()
        return None if not raw else cls(camera_id, Path(raw))

    def record(
        self,
        source_pts: int,
        source_time_ns: int,
        publish_sequence: int,
    ) -> None:
        now_ns = time.time_ns()
        if self._base_pts is None or self._base_source_time_ns is None:
            self._base_pts = source_pts
            self._base_source_time_ns = source_time_ns
        last_pts = self._last_pts
        if last_pts is not None and source_pts <= last_pts:
            self._timestamp_discontinuities += 1
            self._base_pts = source_pts
            self._base_source_time_ns = source_time_ns
        expected_source_ns = self._base_source_time_ns + source_pts - self._base_pts
        self._latencies_ms.append(max(0.0, (now_ns - expected_source_ns) / 1_000_000))
        previous_sequence = self._last_publish_sequence
        sequence_delta = 1 if previous_sequence is None else publish_sequence - previous_sequence
        self._published += max(1, sequence_delta)
        self._overwritten += max(0, sequence_delta - 1)
        self._last_pts = source_pts
        self._last_publish_sequence = publish_sequence
        if time.monotonic() - self._window_started_monotonic >= WINDOW_SECONDS:
            self._flush(now_ns)

    def _flush(self, ended_ns: int) -> None:
        record = CanaryWindowRecord(
            schema_version=1,
            camera_id=self._camera_id,
            window_started_ns=self._window_started_ns,
            window_ended_ns=ended_ns,
            decision_count=len(self._latencies_ms),
            latency_samples_ms=self._latencies_ms,
            metadata_published=self._published,
            metadata_overwritten=self._overwritten,
            timestamp_discontinuities=self._timestamp_discontinuities,
        )
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "ab") as target:
            _ = target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        self._window_started_monotonic = time.monotonic()
        self._window_started_ns = ended_ns
        self._latencies_ms = []
        self._published = 0
        self._overwritten = 0
        self._timestamp_discontinuities = 0


__all__ = ["TELEMETRY_ENV", "WINDOW_SECONDS", "NativeCanaryTelemetry"]
