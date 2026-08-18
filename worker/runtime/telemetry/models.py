"""Immutable worker-local telemetry models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

from contracts.encode_diagnostics import EncodeSelection
from contracts.observation import BedRegionCacheState
from worker.pipeline.inference_coordinator import (
    CameraInferenceTelemetry,
    InferenceTelemetrySnapshot,
)
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
class BedExitScoringDiagnostics:
    """One camera's cumulative bed_exit scoring signal (issue #238).

    Distinguishes, from logs alone, (b) "person never scored inside the bed
    polygon" from (c) "scored inside, but the exit counter never crossed the
    grace threshold" when bed_exit fires zero events overnight --
    ``BedRegionDiagnostics`` above only covers whether the region itself was
    usable, not what ``BedExitMonitor`` did with it once it was. All three
    numeric fields accumulate since worker boot (same convention as
    ``BedRegionCacheCounterSnapshot``), never reset per `RuntimeStatusSender`
    tick. Deliberately holds only scores/counts and the camera_id -- never a
    bounding box, polygon coordinate, or track identity.
    """

    max_containment_observed: float
    grace_positive_transitions: int
    assignments_made: int
    updated_at_sec: float


@dataclass(frozen=True, slots=True)
class DecodeBackendObservability:
    """One camera's boot-time requested-versus-actual decode selection.

    This stays in the worker-local snapshot: the relay's ``decode`` payload
    is a strict legacy wire contract and must not gain the concrete class.
    """

    requested_profile_decode: str
    resolved_backend: str
    actual_adapter_class: str


@dataclass(frozen=True, slots=True)
class StageTimingSnapshot:
    """Aggregated elapsed time for one camera pipeline stage."""

    stage: str
    samples: int
    total_sec: float
    last_sec: float
    max_sec: float


@dataclass(frozen=True, slots=True)
class DeviceResidencyDiagnostics:
    """One camera's experimental NVIDIA device-resident pipeline counters (Todo 17).

    Local-only, same convention as ``bed_region``/``bed_exit_scoring`` below:
    only ever populated for a camera running the opt-in
    ``nvidia-device-experimental`` profile, never crosses the relay boundary,
    and holds only bounded counters/timings -- never a device pointer, a
    tensor, or any RTSP/path value.
    """

    residency_path: str
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
    unavailable_reason: str | None
    updated_at_sec: float


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
    # Local-only: the strict relay decode payload must not expose an
    # implementation class or profile-resolution detail.
    decode_backend: DecodeBackendObservability | None = None
    # Local-only (#53): unlike decode's selection, which is also projected
    # onto the strict backend relay payload (RelayDecodePayload in
    # worker/runtime/telemetry/wire.py), this never crosses the relay
    # boundary -- see worker/runtime/AGENTS.md's byte-for-byte compatibility
    # rule for RelayRuntimeStatusRequest.
    encode: EncodeSelection | None = None
    # Local-only, same reasoning as `encode` above (#207).
    bed_region: BedRegionDiagnostics | None = None
    # Local-only, same reasoning as `bed_region` above (#238).
    bed_exit_scoring: BedExitScoringDiagnostics | None = None
    # Local-only (Todo 17): populated only for a camera running the opt-in
    # nvidia-device-experimental profile.
    device_residency: DeviceResidencyDiagnostics | None = None
    # Local-only (issue #330): monotonic per-camera count of successful
    # decision.update() returns. Zero-event completions increment; later
    # evidence/sink failures do not retract the increment.
    decision_completed: int = 0
    inference: CameraInferenceTelemetry | None = None
    batch_sizes: tuple[tuple[int, int], ...] = ()
    forward_p50_sec: float = 0.0
    forward_p95_sec: float = 0.0


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


class InferenceMetricsSource(Protocol):
    """Structural view of the pipeline coordinator's local snapshot."""

    def snapshot(self) -> InferenceTelemetrySnapshot: ...


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
    "BedExitScoringDiagnostics",
    "BedRegionDiagnostics",
    "BusMetricsSource",
    "BusSubscriptionSnapshot",
    "CameraDiagnosticsSnapshot",
    "DecodeBackendObservability",
    "DeviceResidencyDiagnostics",
    "EncoderLifecycleSnapshot",
    "InferenceMetricsSource",
    "InvalidStageTimingError",
    "RuntimeDiagnosticsSnapshot",
    "StageTimingSnapshot",
    "SubscriptionMetrics",
]
