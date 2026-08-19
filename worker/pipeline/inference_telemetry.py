"""Geometry observation and physical-batch histograms for pose inference."""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from math import ceil
from typing import Final, TypeAlias, final

LOGGER: Final = logging.getLogger(__name__)
FrameGeometry: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class CameraInferenceTelemetry:
    admitted: int
    overwritten: int
    inferred: int
    queue_age_sec: float
    observed_geometry: FrameGeometry | None = None


@dataclass(frozen=True, slots=True)
class InferenceTelemetrySnapshot:
    cameras: dict[str, CameraInferenceTelemetry]
    batch_sizes: dict[int, int]
    forward_p50_sec: float
    forward_p95_sec: float
    geometry_batch_sizes: dict[FrameGeometry, dict[int, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InferenceTelemetryCounters:
    observed_geometries: dict[str, FrameGeometry]
    batch_sizes: dict[int, int]
    geometry_batch_sizes: dict[FrameGeometry, dict[int, int]]
    forward_p50_sec: float
    forward_p95_sec: float


@final
class InferenceGeometryTelemetry:
    """Accumulate last-seen geometries and physical model-call histograms.

    Mutation is the documented purpose: one ledger per coordinator lifetime.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observed: dict[str, FrameGeometry] = {}
        self._batch_sizes: Counter[int] = Counter()
        self._geometry_batch_sizes: dict[FrameGeometry, Counter[int]] = {}
        self._forward_times: deque[float] = deque(maxlen=1024)

    def observe_geometry(self, camera_id: str, geometry: FrameGeometry) -> None:
        """Record one camera's latest frame geometry and warn on a change."""
        with self._lock:
            previous = self._observed.get(camera_id)
            self._observed[camera_id] = geometry
        if previous is None or previous == geometry:
            return
        LOGGER.warning(
            "camera geometry changed: camera_id=%s previous=%sx%s current=%sx%s",
            camera_id,
            previous[0],
            previous[1],
            geometry[0],
            geometry[1],
            extra={
                "camera_id": camera_id,
                "previous_geometry": f"{previous[0]}x{previous[1]}",
                "current_geometry": f"{geometry[0]}x{geometry[1]}",
            },
        )

    def record_physical_batch(
        self, geometry: FrameGeometry, batch_size: int, elapsed_sec: float
    ) -> None:
        """Count one successful homogeneous model call."""
        with self._lock:
            self._batch_sizes[batch_size] += 1
            self._geometry_batch_sizes.setdefault(geometry, Counter())[batch_size] += 1
            self._forward_times.append(elapsed_sec)

    def counters(self) -> InferenceTelemetryCounters:
        """Return a frozen copy of accumulated inference telemetry."""
        with self._lock:
            times = tuple(self._forward_times)
            observed = dict(self._observed)
            sizes = dict(sorted(self._batch_sizes.items()))
            geometry_sizes = {
                geometry: dict(sorted(counts.items()))
                for geometry, counts in sorted(self._geometry_batch_sizes.items())
            }
        return InferenceTelemetryCounters(
            observed_geometries=observed,
            batch_sizes=sizes,
            geometry_batch_sizes=geometry_sizes,
            forward_p50_sec=_percentile(times, 0.50),
            forward_p95_sec=_percentile(times, 0.95),
        )


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[max(0, ceil(len(values) * quantile) - 1)]


__all__ = [
    "CameraInferenceTelemetry",
    "FrameGeometry",
    "InferenceGeometryTelemetry",
    "InferenceTelemetryCounters",
    "InferenceTelemetrySnapshot",
]
