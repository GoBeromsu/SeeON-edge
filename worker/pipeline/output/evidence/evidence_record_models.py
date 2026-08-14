from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from shared.edge_db.reviews import EvidenceReview


class EvidenceLifecycle(StrEnum):
    STAGING = "STAGING"
    MEDIA_READY = "MEDIA_READY"
    PUBLISHED = "PUBLISHED"
    DERIVATIVE_PENDING = "DERIVATIVE_PENDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ArtifactState(StrEnum):
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    CORRUPT = "CORRUPT"


@dataclass(frozen=True, slots=True)
class PrimaryEvidence:
    clip_id: str
    media_sha256: str | None
    media_size_bytes: int | None
    media_relpath: str | None
    manifest_sha256: str | None
    manifest_size_bytes: int | None
    manifest_relpath: str | None
    codec: str | None
    audio_codec: str | None
    duration_ms: int | None
    source_packet_preserved: bool
    source_missing_reason: str | None
    source_media: dict[str, Any] | None
    time_origin: dict[str, Any] | None
    truncation_reasons: tuple[str, ...]
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    incident_id: str
    schema_version: int
    edge_event_id: str
    camera_id: str
    event_type: str
    detected_at: str
    runtime_manifest_sha256: str | None
    decision_trace_id: str | None
    module_qualified_id: str | None
    policy_qualified_id: str | None
    effective_policy_id: str | None
    provenance_state: str
    provenance_missing_reason: str | None
    lifecycle: EvidenceLifecycle
    revision: int
    failure_reason: str | None
    primary_state: ArtifactState
    snapshot_state: ArtifactState
    derivative_state: ArtifactState | None
    event_delivery_state: str
    event_attempt_count: int
    clip_publish_state: str | None
    clip_publish_attempt_count: int | None
    retention_state: str | None
    primary: PrimaryEvidence | None
    review: EvidenceReview | None


@dataclass(slots=True)
class EvidenceRecordConflictError(RuntimeError):
    incident_id: str
    detail: str

    def __str__(self) -> str:
        return f"central evidence {self.incident_id}: {self.detail}"


__all__ = [
    "ArtifactState",
    "EvidenceLifecycle",
    "EvidenceRecord",
    "EvidenceRecordConflictError",
    "PrimaryEvidence",
]
