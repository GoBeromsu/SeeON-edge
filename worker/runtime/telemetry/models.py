"""Immutable worker-local telemetry models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

from contracts.encode_diagnostics import EncodeSelection


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
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "EncoderLifecycleSnapshot",
    "InvalidStageTimingError",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
]
