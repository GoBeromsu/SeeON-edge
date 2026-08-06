"""Immutable worker-local telemetry models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

from contracts.encode_diagnostics import EncodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.perception.scene_state import BedRegionCacheCounterSnapshot


@dataclass(frozen=True, slots=True)
class BedRegionDiagnostics:
    """One camera's latest bed-region cache state and cumulative counters.

    Local-only, same as ``encode`` below: never crosses the relay boundary
    (worker/runtime/telemetry/wire.py's ``RelayCameraPayload`` has no field
    for it), so nothing here needs to satisfy that strict wire contract.
    Deliberately holds only ``BedRegionCacheState``'s four named states and
    plain counts (issue #207) -- never the polygon coordinates, an RTSP URL,
    or anything else that would turn a pasted-into-an-issue log line into a
    camera-location or credential leak.
    """

    freshness: BedRegionCacheState
    # NOTE: ``counters.expired`` counts scheduled-empty cycles observed while
    # already EXPIRED, not distinct expiry transitions -- one continuous
    # occlusion re-increments it every cycle, so `expired: 47` can mean one
    # 47-cycle occlusion rather than 47 separate flaps. See issue #226.
    counters: BedRegionCacheCounterSnapshot
    updated_at_sec: float


@dataclass(frozen=True, slots=True)
class StageTimingSnapshot:
    """Aggregated elapsed time for one camera pipeline stage."""

    stage: str
    samples: int
    total_sec: float
    last_sec: float
    max_sec: float


@dataclass(frozen=True, slots=True)
class BusSubscriptionSnapshot:
    """Local queue counters for one named bus subscription."""

    name: str
    published: int
    taken: int
    dropped: int
    queue_age_sec: float


@dataclass(frozen=True, slots=True)
class EncoderLifecycleSnapshot:
    """Worker-local encoder process and segment lifecycle counters."""

    process_starts: int = 0
    recreates: int = 0
    failures: int = 0
    active_sessions: int = 0
    finalized_segments: int = 0
    unavailable_cameras: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CameraDiagnosticsSnapshot:
    """Rich local diagnostics for one camera."""

    camera_id: str
    failure_category: str | None
    stage_timings: tuple[StageTimingSnapshot, ...]
    bus: tuple[BusSubscriptionSnapshot, ...]
    # Local-only (#53): unlike decode's selection, which is also projected
    # onto the strict backend relay payload (RelayDecodePayload in
    # worker/runtime/telemetry/wire.py), this never crosses the relay
    # boundary -- see worker/runtime/AGENTS.md's byte-for-byte compatibility
    # rule for RelayRuntimeStatusRequest.
    encode: EncodeSelection | None = None
    # Local-only, same reasoning as `encode` above (#207).
    bed_region: BedRegionDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosticsSnapshot:
    """Immutable worker-local metrics that never cross the relay boundary."""

    cameras: tuple[CameraDiagnosticsSnapshot, ...]
    encoder: EncoderLifecycleSnapshot


class SubscriptionMetrics(Protocol):
    """Structural view of bounded bus counters."""

    @property
    def published(self) -> int: ...

    @property
    def taken(self) -> int: ...

    @property
    def dropped(self) -> int: ...

    @property
    def queue_age_sec(self) -> float: ...


class BusMetricsSource(Protocol):
    """Source of consistent metrics for a named bus subscription."""

    def metrics(self, name: str) -> SubscriptionMetrics: ...


@final
class InvalidStageTimingError(ValueError):
    """Raised when a stage reports an impossible negative elapsed time."""

    __slots__ = ("elapsed_sec",)

    def __init__(self, elapsed_sec: float) -> None:
        self.elapsed_sec = elapsed_sec
        super().__init__(f"stage timing must be non-negative: {elapsed_sec}")


__all__ = [
    "BedRegionDiagnostics",
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "EncoderLifecycleSnapshot",
    "InvalidStageTimingError",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
]
