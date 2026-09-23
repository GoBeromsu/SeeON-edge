"""Frozen diagnostics vocabulary and batch value types.

No Pydantic. Enums mirror ``execution_records_ddl``. Late-ACK unit ids
are ``sha256(original_unit_id + ":late:" + batch_id)`` hex because the
DDL identity check is 1..128 chars.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from shared.events.execution_records import canonical_json

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_FIELDS: Final = (
    "backend_build_revision",
    "calibration_digest",
    "config_digest",
    "model_digest",
    "policy_identity",
    "preprocessing_identity",
    "worker_build_revision",
    "worker_image_digest",
)


class RecordKind(StrEnum):
    SDK_FRAME = "sdk.frame"
    POLICY_CONSUME = "policy.consume"
    MODEL_SCORE = "model.score"
    POLICY_DECISION = "policy.decision"
    EVENT_DELIVERY = "event.delivery"
    BACKEND_ACCEPTANCE = "backend.acceptance"


class CoverageKind(StrEnum):
    MISSING_NOT_RECORDED = "MISSING_NOT_RECORDED"
    DELETED_BY_CAPACITY = "DELETED_BY_CAPACITY"
    UNKNOWN_COARSENED = "UNKNOWN_COARSENED"
    UNKNOWN = "UNKNOWN"
    ACK_OBSERVED_PARENT_DELETED = "ACK_OBSERVED_PARENT_DELETED"
    ACK_OBSERVED_PARENT_UNKNOWN_COARSENED = "ACK_OBSERVED_PARENT_UNKNOWN_COARSENED"
    REJECTED_OVERSIZE = "REJECTED_OVERSIZE"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"


class SegmentStorageState(StrEnum):
    OPEN = "OPEN"
    SEALED_PENDING = "SEALED_PENDING"
    SEALED_FINAL = "SEALED_FINAL"
    PRUNED_SUMMARY = "PRUNED_SUMMARY"


class UnitCausalState(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE_KNOWN = "INCOMPLETE_KNOWN"
    INCOMPLETE_UNKNOWN = "INCOMPLETE_UNKNOWN"


class StorageState(StrEnum):
    COMMITTED = "committed"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"


class AvailabilityKind(StrEnum):
    AVAILABLE = "AVAILABLE"
    MISSING_NOT_RECORDED = "MISSING_NOT_RECORDED"
    DELETED_BY_CAPACITY = "DELETED_BY_CAPACITY"
    UNKNOWN_COARSENED = "UNKNOWN_COARSENED"
    UNKNOWN = "UNKNOWN"


def require_sha256(value: str, *, what: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid {what}")
    return value


def require_identity(value: str, *, what: str) -> str:
    if not value or len(value) > 128 or "\x00" in value:
        raise ValueError(f"invalid {what}")
    return value


def late_ack_unit_id(original_unit_id: str, batch_id: str) -> str:
    return hashlib.sha256(f"{original_unit_id}:late:{batch_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Provenance:
    worker_build_revision: str
    worker_image_digest: str
    model_digest: str
    calibration_digest: str
    preprocessing_identity: str
    config_digest: str
    policy_identity: str
    backend_build_revision: str

    def __post_init__(self) -> None:
        for name in _PROVENANCE_FIELDS:
            require_identity(getattr(self, name), what=name)

    @property
    def provenance_id(self) -> str:
        payload = {name: getattr(self, name) for name in _PROVENANCE_FIELDS}
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionRecordInput:
    record_id: str
    record_kind: RecordKind
    camera_id: str
    worker_boot_id: str
    source_generation: int
    stream_epoch: int
    producer: str
    producer_sequence: int
    observed_at_ns: int
    time_quality: str
    causal_unit_id: str
    outcome: str
    payload: Mapping[str, object]
    frame_seq: int | None = None
    source_pts_ns: int | None = None
    parent_record_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.record_id, what="record_id")
        if not isinstance(self.record_kind, RecordKind):
            raise TypeError("invalid record_kind")
        for name in (
            "camera_id",
            "worker_boot_id",
            "producer",
            "time_quality",
            "causal_unit_id",
            "outcome",
        ):
            require_identity(getattr(self, name), what=name)
        if self.parent_record_id is not None:
            require_sha256(self.parent_record_id, what="parent_record_id")
        if self.reason is not None:
            require_identity(self.reason, what="reason")
        for name in (
            "source_generation",
            "stream_epoch",
            "producer_sequence",
            "observed_at_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {name}")
        if self.frame_seq is not None and (type(self.frame_seq) is not int or self.frame_seq < 0):
            raise ValueError("invalid frame_seq")
        if self.source_pts_ns is not None and type(self.source_pts_ns) is not int:
            raise ValueError("invalid source_pts_ns")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        canonical_json(dict(self.payload))


@dataclass(frozen=True, slots=True)
class GapReport:
    producer: str
    from_sequence: int
    to_sequence: int
    from_ns: int
    to_ns: int
    record_count: int
    cause: str

    def __post_init__(self) -> None:
        require_identity(self.producer, what="producer")
        require_identity(self.cause, what="cause")
        for name in ("from_sequence", "to_sequence", "from_ns", "to_ns", "record_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {name}")
        if self.to_sequence < self.from_sequence or self.to_ns < self.from_ns:
            raise ValueError("invalid gap range")


@dataclass(frozen=True, slots=True)
class IngestBatch:
    batch_id: str
    camera_id: str
    worker_boot_id: str
    provenance: Provenance
    records: tuple[ExecutionRecordInput, ...]
    gaps: tuple[GapReport, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(self.batch_id, what="batch_id")
        require_identity(self.camera_id, what="camera_id")
        require_identity(self.worker_boot_id, what="worker_boot_id")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "gaps", tuple(self.gaps))
        for record in self.records:
            if record.camera_id != self.camera_id or record.worker_boot_id != self.worker_boot_id:
                raise ValueError("record camera/boot does not match batch")


@dataclass(frozen=True, slots=True)
class BatchReceipt:
    batch_id: str
    accepted: int
    duplicates: int
    rejected: tuple[tuple[str, str], ...]
    storage_state: StorageState
    committed_at_ns: int


__all__ = [
    "AvailabilityKind",
    "BatchReceipt",
    "CoverageKind",
    "ExecutionRecordInput",
    "GapReport",
    "IngestBatch",
    "Provenance",
    "RecordKind",
    "SegmentStorageState",
    "StorageState",
    "UnitCausalState",
    "canonical_json",
    "late_ack_unit_id",
    "require_identity",
    "require_sha256",
]
