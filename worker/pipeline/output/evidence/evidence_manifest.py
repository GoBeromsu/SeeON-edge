"""Schema-v2 manifests for re-encoded derivative evidence clips."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Final

from pydantic import ValidationError

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
    runtime_manifest_sha256: str | None = None,
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
        codec=facts.video_codec,
        audio_codec=facts.audio_codec,
        duration_ms=facts.duration_ms,
        runtime_manifest_sha256=runtime_manifest_sha256,
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
    runtime_manifest_sha256: str | None = None,
) -> UnavailableClipManifest:
    return UnavailableClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_refs=coalesce_event_refs(tuple(str(value) for value in event_refs)),
        clip_start_at=rfc3339_milliseconds(clip_start_at),
        clip_end_at=rfc3339_milliseconds(clip_end_at),
        finalized_at=rfc3339_milliseconds(finalized_at),
        reason_code=reason_code,
        runtime_manifest_sha256=runtime_manifest_sha256,
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
    manifest, _, _ = parse_manifest_content(path)
    return manifest


def parse_manifest_content(path: Path) -> tuple[ClipManifest, bytes, dict[str, object]]:
    """Parse and retain the exact bounded manifest inode bytes used for validation."""
    try:
        content = read_manifest_bytes(path)
        payload = json.loads(content, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest state invalid")
        state = payload.get("state")
        match state:
            case "READY":
                manifest: ClipManifest = ReadyClipManifest.model_validate(payload)
            case "UNAVAILABLE":
                manifest = UnavailableClipManifest.model_validate(payload)
            case _:
                raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest state invalid")
        return manifest, content, payload  # noqa: TRY300 - one guarded parse boundary
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest invalid") from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_manifest_bytes(path: Path) -> bytes:
    """Read one bounded immutable manifest inode without following links."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_MANIFEST_BYTES:
            raise OSError("manifest file shape invalid")
        chunks: list[bytes] = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_MANIFEST_BYTES:
            raise OSError("manifest exceeds size limit")
        return payload
    finally:
        os.close(descriptor)


__all__ = [
    "MAX_MANIFEST_BYTES",
    "ClipEvidenceError",
    "ClipManifest",
    "EvidenceReasonCode",
    "ReadyClipManifest",
    "UnavailableClipManifest",
    "finalize_ready_manifest",
    "parse_manifest",
    "parse_manifest_content",
    "read_manifest_bytes",
    "unavailable_manifest",
    "verify_ready_manifest",
]
