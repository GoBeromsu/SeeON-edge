"""Schema-v2 manifests for re-encoded derivative evidence clips."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from worker.pipeline.output.evidence.clip_consistency_io import read_strict_json
from worker.pipeline.output.evidence.clip_consistency_types import ClipConsistencyError
from worker.pipeline.output.evidence.evidence_media import (
    ClipEvidenceError,
    inspect_finalized_media,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    EdgeEventId,
    EvidenceReasonCode,
)
from worker.pipeline.output.evidence.manifest_models import (
    ClipManifest,
    ReadyClipManifest,
    UnavailableClipManifest,
    coalesce_event_refs,
    rfc3339_milliseconds,
)

MAX_MANIFEST_BYTES: Final = 64 * 1024


def finalize_ready_manifest(
    *,
    video_path: Path,
    clip_id: ClipId,
    camera_id: str,
    event_refs: tuple[EdgeEventId, ...],
    clip_start_at: datetime,
    clip_end_at: datetime,
    finalized_at: datetime,
    ffprobe_bin: str = "ffprobe",
) -> ReadyClipManifest:
    """Describe verified derivative bytes without implying source preservation."""
    facts = inspect_finalized_media(video_path, ffprobe_bin=ffprobe_bin)
    return ReadyClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_refs=coalesce_event_refs(tuple(str(value) for value in event_refs)),
        clip_start_at=rfc3339_milliseconds(clip_start_at),
        clip_end_at=rfc3339_milliseconds(clip_end_at),
        finalized_at=rfc3339_milliseconds(finalized_at),
        sha256=facts.sha256,
        size_bytes=facts.size_bytes,
        duration_ms=facts.duration_ms,
    )


def unavailable_manifest(
    *,
    clip_id: ClipId,
    camera_id: str,
    event_refs: tuple[EdgeEventId, ...],
    clip_start_at: datetime,
    clip_end_at: datetime,
    finalized_at: datetime,
    reason_code: EvidenceReasonCode,
) -> UnavailableClipManifest:
    return UnavailableClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_refs=coalesce_event_refs(tuple(str(value) for value in event_refs)),
        clip_start_at=rfc3339_milliseconds(clip_start_at),
        clip_end_at=rfc3339_milliseconds(clip_end_at),
        finalized_at=rfc3339_milliseconds(finalized_at),
        reason_code=reason_code,
    )


def verify_ready_manifest(
    manifest: ReadyClipManifest,
    video_path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
) -> None:
    facts = inspect_finalized_media(video_path, ffprobe_bin=ffprobe_bin)
    if facts.sha256 != manifest.sha256 or facts.size_bytes != manifest.size_bytes:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "immutable bytes mismatch")
    if facts.duration_ms != manifest.duration_ms:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "duration mismatch")


def parse_manifest(path: Path) -> ClipManifest:
    try:
        payload = read_strict_json(
            path,
            max_bytes=MAX_MANIFEST_BYTES,
            error_code="manifest_invalid",
        )
        state = payload.get("state")
        match state:
            case "READY":
                return ReadyClipManifest.model_validate(payload)
            case "UNAVAILABLE":
                return UnavailableClipManifest.model_validate(payload)
            case _:
                raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest state invalid")
    except (OSError, UnicodeDecodeError, ValidationError, ClipConsistencyError) as exc:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest invalid") from exc


__all__ = [
    "MAX_MANIFEST_BYTES",
    "ClipEvidenceError",
    "ClipManifest",
    "EvidenceReasonCode",
    "ReadyClipManifest",
    "UnavailableClipManifest",
    "finalize_ready_manifest",
    "parse_manifest",
    "unavailable_manifest",
    "verify_ready_manifest",
]
