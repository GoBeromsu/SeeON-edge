from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypeAlias

from worker.pipeline.output.evidence.evidence_manifest import ClipManifest
from worker.pipeline.output.evidence.evidence_metadata import (
    validate_runtime_manifest_sha256,
)
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId, EdgeEventId

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class PublicationStage(StrEnum):
    MEDIA_FSYNCED = "MEDIA_FSYNCED"
    MEDIA_RENAMED = "MEDIA_RENAMED"
    THUMBNAIL_RENAMED = "THUMBNAIL_RENAMED"
    MANIFEST_RENAMED = "MANIFEST_RENAMED"


class PublicationBarrier(Protocol):
    def __call__(self, stage: PublicationStage, path: Path, /) -> None: ...


@dataclass(frozen=True, slots=True)
class ClipTimeOrigin:
    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    generation: int
    media_origin_pts_sec: float
    event_pts_sec: float
    requested_start_pts_sec: float
    requested_end_pts_sec: float

    @property
    def event_media_time_ms(self) -> float:
        return (self.event_pts_sec - self.media_origin_pts_sec) * 1000.0


@dataclass(frozen=True, slots=True)
class ClipPublicationMetadata:
    camera_id: str
    event_refs: tuple[EdgeEventId, ...]
    event_type: str | None
    clip_start_at: datetime
    clip_end_at: datetime
    finalized_at: datetime
    started_at: datetime
    duration_s: float
    encoder: str
    runtime_manifest_sha256: str | None = None
    decision_trace_id: str | None = None
    time_origin: ClipTimeOrigin | None = None
    source_media: dict[str, JsonValue] | None = None
    source_error_reason: str | None = None
    truncation_reasons: tuple[str, ...] = ()
    domain: str | None = None

    def __post_init__(self) -> None:
        validate_runtime_manifest_sha256(self.runtime_manifest_sha256)


@dataclass(frozen=True, slots=True)
class PublishedClip:
    clip_id: ClipId
    manifest: ClipManifest
    manifest_path: Path
    video_path: Path | None


@dataclass(slots=True)
class ClipPublicationConflictError(Exception):
    clip_id: ClipId
    detail: str

    def __str__(self) -> str:
        return f"clip {self.clip_id} publication conflict: {self.detail}"


__all__ = [
    "JsonValue",
    "ClipPublicationConflictError",
    "ClipPublicationMetadata",
    "ClipTimeOrigin",
    "PublicationBarrier",
    "PublicationStage",
    "PublishedClip",
]
