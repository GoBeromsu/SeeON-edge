from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias

from worker.types import DecisionTraceSnapshot

_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_QUALIFIED: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_COMPONENT_QUALIFIED: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\.sha256\.[0-9a-f]{64}")


class TraceContractError(ValueError):
    pass


class TracePersistenceError(RuntimeError):
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
class TraceFrame:
    analysis: AnalysisTrace
    decisions: tuple[DecisionTrace, ...]


TraceFrameKey: TypeAlias = tuple[str, str, int, int]


@dataclass(frozen=True, slots=True)
class TraceTruncation:
    handoff_dropped_frames: int
    pruned_frames: int
    oldest_retained_seq: int | None
    newest_retained_seq: int | None
    persistence_failed_frames: int = 0
    retention_blocked_frames: int = 0
    oldest_retained_key: TraceFrameKey | None = None
    newest_retained_key: TraceFrameKey | None = None
    detail_unavailable_reason: DetailUnavailableReason | None = None


@dataclass(frozen=True, slots=True)
class RecoveredCameraTrace:
    frames: tuple[AnalysisTrace, ...]
    decisions: tuple[DecisionTrace, ...]
    truncation: TraceTruncation


@dataclass(frozen=True, slots=True)
class TraceWriterStats:
    handoff_dropped_frames: int
    persisted_frames: int
    failed_batches: int
    persistence_failed_frames: int = 0
    retry_attempts: int = 0
    rejected_frames: int = 0
    duplicate_frames: int = 0
    #: Publication attempts that failed. Traces stay local; QA replay
    #: simply has less to work with, and detection is unaffected.
    publication_failures: int = 0
    #: Frames dropped from the bounded republication buffer while the
    #: relay was unreachable. Without this the loss is invisible: the
    #: backend's copy is quietly less complete than the worker's.
    unpublished_shed_frames: int = 0


def trace_frame_row_count(frame: TraceFrame) -> int:
    return (
        1
        + len(frame.analysis.components)
        + len(frame.analysis.persons)
        + len(frame.analysis.beds)
        + sum(len(person.keypoints) for person in frame.analysis.persons)
        + sum(len(bed.polygon) for bed in frame.analysis.beds)
        + len(frame.decisions)
        + sum(
            len(decision.snapshot.values) + len(decision.snapshot.missing_values)
            for decision in frame.decisions
        )
    )


def trace_frame_size_bytes(frame: TraceFrame) -> int:
    analysis = frame.analysis
    payload = {
        "analysis": [
            analysis.trace_id,
            analysis.schema_version,
            *analysis.frame_key,
            analysis.pts.value,
            analysis.pts.missing_reason,
            analysis.source_time.value,
            analysis.source_time.missing_reason,
            analysis.frame_width,
            analysis.frame_height,
            analysis.bed_region_provenance,
            [
                [
                    person.ordinal,
                    person.track_id.value,
                    person.track_id.missing_reason,
                    *person.box,
                    person.confidence,
                    [
                        [point.index, point.x, point.y, point.confidence]
                        for point in person.keypoints
                    ],
                ]
                for person in analysis.persons
            ],
            [
                [bed.ordinal, *bed.box, bed.confidence, bed.provenance, list(bed.polygon)]
                for bed in analysis.beds
            ],
            [
                [component.ordinal, component.qualified_id, component.observation_state]
                for component in analysis.components
            ],
        ],
        "decisions": [
            [
                decision.trace_id,
                decision.schema_version,
                decision.analysis_trace_id,
                decision.identity_index,
                decision.module_qualified_id,
                decision.policy_qualified_id,
                decision.effective_policy_id,
                decision.runtime_manifest_sha256,
                decision.snapshot.reason,
                decision.snapshot.previous_state,
                decision.snapshot.current_state,
                decision.snapshot.triggered,
                decision.snapshot.track_id,
                decision.snapshot.bed_id,
                dict(decision.snapshot.values),
                dict(decision.snapshot.missing_values),
            ]
            for decision in frame.decisions
        ],
    }
    return len(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    )


def require_sha256(value: str, field: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise TraceContractError(f"{field} must be a lowercase SHA-256")
    return value


def require_qualified(value: str, field: str) -> str:
    if (
        _QUALIFIED.fullmatch(value) is None
        or "://" in value
        or value.startswith(("/", "\\"))
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise TraceContractError(f"{field} contains an unsafe or unqualified identity")
    return value


def require_component_qualified(value: str) -> str:
    if _COMPONENT_QUALIFIED.fullmatch(value) is None:
        raise TraceContractError(
            "component_qualified_ids contains an unsafe or unqualified identity"
        )
    return value


def content_id(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AnalysisTrace",
    "DecisionTrace",
    "DetailUnavailableReason",
    "OptionalNumber",
    "RecoveredCameraTrace",
    "TraceBed",
    "TraceComponent",
    "TraceContractError",
    "TraceFrame",
    "TraceFrameKey",
    "TraceKeypoint",
    "TracePersistenceError",
    "TracePerson",
    "TraceTruncation",
    "TraceWriterStats",
    "content_id",
    "require_component_qualified",
    "require_qualified",
    "require_sha256",
    "trace_frame_row_count",
    "trace_frame_size_bytes",
]
