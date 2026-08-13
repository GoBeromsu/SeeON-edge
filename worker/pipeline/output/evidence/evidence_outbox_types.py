"""Typed values and failures for durable evidence delivery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NewType

EdgeEventId = NewType("EdgeEventId", str)
ClipId = NewType("ClipId", str)


class ClipLocalState(StrEnum):
    AWAITING_FINALIZE = "AWAITING_FINALIZE"
    VERIFIED = "VERIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    CORRUPT = "CORRUPT"


class EventDeliveryState(StrEnum):
    PENDING = "PENDING"
    ACKED = "ACKED"
    PERMANENT = "PERMANENT"
    COMPATIBILITY = "COMPATIBILITY"


class ClipPublishState(StrEnum):
    WAITING = "WAITING"
    IN_FLIGHT = "IN_FLIGHT"
    PUBLISHED = "PUBLISHED"
    PERMANENT = "PERMANENT"
    COMPATIBILITY = "COMPATIBILITY"


class EvidenceReasonCode(StrEnum):
    ENCODER_FAILED = "ENCODER_FAILED"
    NO_FRAMES = "NO_FRAMES"
    FINALIZE_FAILED = "FINALIZE_FAILED"
    INTERRUPTED_FINALIZE = "INTERRUPTED_FINALIZE"
    MISSING = "MISSING"
    CORRUPT = "CORRUPT"


@dataclass(frozen=True, slots=True)
class StagedEvent:
    edge_event_id: EdgeEventId
    detected_at: str
    payload_json: str
    queued_at: float


@dataclass(frozen=True, slots=True)
class ClaimLease:
    owner: str
    now: float
    duration: float


@dataclass(frozen=True, slots=True)
class ClaimedEvent:
    edge_event_id: EdgeEventId
    detected_at: str
    payload_json: str
    clip_id: ClipId | None
    lease_owner: str
    lease_expires_at: float
    attempt_count: int


@dataclass(frozen=True, slots=True)
class ClaimedClip:
    clip_id: ClipId
    local_state: ClipLocalState
    state_version: int
    media_relpath: str | None
    sha256: str | None
    size_bytes: int | None
    mime_type: str | None
    codec: str | None
    duration_ms: int | None
    clip_start_at: str | None
    clip_end_at: str | None
    finalized_at: str | None
    unavailable_reason: EvidenceReasonCode | None
    event_ids: tuple[EdgeEventId, ...]
    event_payload_json: str
    lease_owner: str
    lease_expires_at: float
    attempt_count: int


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    journal_mode: str
    synchronous: int
    foreign_keys: int


@dataclass(frozen=True, slots=True)
class ClipOutcome:
    clip_id: ClipId
    local_state: ClipLocalState
    manifest_path: str | None
    state_version: int
    media_relpath: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    codec: str | None = None
    duration_ms: int | None = None
    clip_start_at: str | None = None
    clip_end_at: str | None = None
    finalized_at: str | None = None
    unavailable_reason: EvidenceReasonCode | None = None


@dataclass(slots=True)
class NewerSchemaVersionError(Exception):
    found: int
    supported: int

    def __str__(self) -> str:
        return f"evidence outbox schema {self.found} is newer than supported {self.supported}"


@dataclass(slots=True)
class EventClipConflictError(Exception):
    edge_event_id: EdgeEventId
    existing_clip_id: ClipId
    requested_clip_id: ClipId

    def __str__(self) -> str:
        return (
            f"event {self.edge_event_id} is bound to {self.existing_clip_id}, "
            f"not {self.requested_clip_id}"
        )


@dataclass(slots=True)
class StagedEventConflictError(Exception):
    edge_event_id: EdgeEventId
    mismatched_fields: tuple[Literal["detected_at", "payload_json"], ...]

    def __str__(self) -> str:
        fields = ", ".join(self.mismatched_fields)
        return f"event {self.edge_event_id} replay conflicts in fields: {fields}"


@dataclass(slots=True)
class MissingStagedEventError(Exception):
    edge_event_id: EdgeEventId

    def __str__(self) -> str:
        return f"event {self.edge_event_id} must be staged before clip binding"


@dataclass(slots=True)
class ClipOutcomeConflictError(Exception):
    clip_id: ClipId
    existing_state: ClipLocalState
    requested_state: ClipLocalState

    def __str__(self) -> str:
        return (
            f"clip {self.clip_id} outcome is {self.existing_state}, "
            f"not {self.requested_state}"
        )


__all__ = [
    "ClaimLease",
    "ClaimedClip",
    "ClaimedEvent",
    "ClipId",
    "ClipLocalState",
    "ClipOutcome",
    "ClipOutcomeConflictError",
    "ClipPublishState",
    "DatabaseSettings",
    "EdgeEventId",
    "EventDeliveryState",
    "EventClipConflictError",
    "EvidenceReasonCode",
    "MissingStagedEventError",
    "NewerSchemaVersionError",
    "StagedEvent",
    "StagedEventConflictError",
]
