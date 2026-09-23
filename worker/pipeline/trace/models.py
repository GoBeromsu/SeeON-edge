from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from worker.types import DecisionTraceSnapshot

_SHA256: Final = re.compile(r"[0-9a-f]{64}")


class TraceContractError(ValueError):
    pass


class DetailUnavailableReason(StrEnum):
    BOUNDED_HANDOFF_CAPACITY = "bounded-handoff-capacity"
    DETAIL_CACHE_WRITE_FAILED = "detail-cache-write-failed"
    RETENTION_BOUND = "retention-bound"


@dataclass(frozen=True, slots=True)
class OptionalNumber:
    value: int | float | None
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.missing_reason is None):
            raise TraceContractError(
                "optional numeric trace fields require exactly one value or missing reason"
            )
        if self.value is not None:
            if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
                raise TraceContractError("numeric trace values must be finite")
        elif not self.missing_reason:
            raise TraceContractError("missing numeric trace fields require a reason")


@dataclass(frozen=True, slots=True)
class TraceKeypoint:
    index: int
    x: int
    y: int
    confidence: float


@dataclass(frozen=True, slots=True)
class TracePerson:
    ordinal: int
    track_id: OptionalNumber
    box: tuple[int, int, int, int]
    confidence: float
    keypoints: tuple[TraceKeypoint, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceBed:
    ordinal: int
    box: tuple[int, int, int, int]
    confidence: float
    provenance: str
    polygon: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class TraceComponent:
    ordinal: int
    qualified_id: str
    observation_state: str


@dataclass(frozen=True, slots=True)
class AnalysisTrace:
    trace_id: str
    frame_key: tuple[str, str, int, int]
    pts: OptionalNumber
    source_time: OptionalNumber
    frame_width: int
    frame_height: int
    bed_region_provenance: str
    persons: tuple[TracePerson, ...]
    beds: tuple[TraceBed, ...]
    components: tuple[TraceComponent, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    trace_id: str
    analysis_trace_id: str
    identity_index: int
    module_qualified_id: str
    policy_qualified_id: str
    effective_policy_id: str
    runtime_manifest_sha256: str
    snapshot: DecisionTraceSnapshot
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class TraceTruncation:
    handoff_dropped_frames: int
    pruned_frames: int
    oldest_retained_seq: int | None
    newest_retained_seq: int | None
    persistence_failed_frames: int = 0
    retention_blocked_frames: int = 0
    oldest_retained_key: tuple[str, str, int, int] | None = None
    newest_retained_key: tuple[str, str, int, int] | None = None
    detail_unavailable_reason: DetailUnavailableReason | None = None


@dataclass(frozen=True, slots=True)
class RecoveredCameraTrace:
    frames: tuple[AnalysisTrace, ...]
    decisions: tuple[DecisionTrace, ...]
    truncation: TraceTruncation


def require_sha256(value: str, field: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise TraceContractError(f"{field} must be a lowercase SHA-256")
    return value


__all__ = [
    "AnalysisTrace",
    "DecisionTrace",
    "DetailUnavailableReason",
    "OptionalNumber",
    "RecoveredCameraTrace",
    "TraceBed",
    "TraceComponent",
    "TraceContractError",
    "TraceKeypoint",
    "TracePerson",
    "TraceTruncation",
    "require_sha256",
]
