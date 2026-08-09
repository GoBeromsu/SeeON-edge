from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.evidence_manifest import ClipManifest
from worker.pipeline.output.evidence.evidence_outbox_types import ClipId, EdgeEventId


class PublicationStage(StrEnum):
    MEDIA_FSYNCED = "MEDIA_FSYNCED"
    MEDIA_RENAMED = "MEDIA_RENAMED"
    THUMBNAIL_RENAMED = "THUMBNAIL_RENAMED"
    MANIFEST_RENAMED = "MANIFEST_RENAMED"


class PublicationBarrier(Protocol):
    def __call__(self, stage: PublicationStage, path: Path, /) -> None: ...


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
    "ClipPublicationConflictError",
    "ClipPublicationMetadata",
    "PublicationBarrier",
    "PublicationStage",
    "PublishedClip",
]
