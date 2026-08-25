"""Validation and canonicalization helpers for evidence restart reconciliation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.evidence_manifest import parse_manifest
from worker.pipeline.output.evidence.evidence_media import ClipEvidenceError
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    ClipLocalState,
    ClipOutcome,
    ClipOutcomeConflictError,
    EdgeEventId,
    EventClipConflictError,
    EvidenceReasonCode,
    MissingStagedEventError,
)
from worker.pipeline.output.evidence.evidence_reconciliation_port import ReconciliationOutbox


def record_corrupt(
    outbox: ReconciliationOutbox,
    clip_id: ClipId,
    manifest_path: str | None,
    event_ids: tuple[EdgeEventId, ...],
) -> None:
    outcome = ClipOutcome(
        clip_id=clip_id,
        local_state=ClipLocalState.CORRUPT,
        manifest_path=manifest_path,
        state_version=2,
        unavailable_reason=EvidenceReasonCode.CORRUPT,
    )
    if event_ids:
        try:
            outbox.reconcile_clip(event_ids, outcome)
        except (
            ClipOutcomeConflictError, EventClipConflictError,
            MissingStagedEventError, ValueError,
        ):
            pass
        else:
            return
    outbox.record_clip_outcome(outcome)


def validate_clip_directory(clip_dir: Path) -> None:
    if clip_dir.is_symlink() or not clip_dir.is_dir():
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "clip directory invalid")


def validate_clip_identity(manifest_clip_id: str, expected_clip_id: ClipId) -> None:
    if manifest_clip_id != expected_clip_id:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "clip identity mismatch")


def validated_video_path(clip_dir: Path) -> Path:
    video_path = clip_dir / "clip.mp4"
    if video_path.parent.resolve() != clip_dir.resolve():
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media path escaped clip")
    return video_path


def unavailable_outcome(
    clip_id: ClipId, reason_code: EvidenceReasonCode, manifest_path: str | None
) -> ClipOutcome:
    return ClipOutcome(
        clip_id=clip_id,
        local_state=ClipLocalState.UNAVAILABLE,
        manifest_path=manifest_path,
        state_version=2,
        unavailable_reason=reason_code,
    )


def relative_or_none(store_dir: Path, path: Path) -> str | None:
    try:
        return str(path.relative_to(store_dir))
    except ValueError:
        return None


def belongs_to_staged_event(clip_dir: Path, outbox: ReconciliationOutbox) -> bool:
    try:
        manifest = parse_manifest(clip_dir / "manifest.json")
    except ClipEvidenceError:
        return False
    return any(outbox.has_event(EdgeEventId(event_ref)) for event_ref in manifest.event_refs)


def quarantine(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ClipEvidenceError(
            EvidenceReasonCode.CORRUPT,
            f"quarantine destination already exists: {destination.name}",
        )
    os.replace(source, destination)
    fsync_directory(destination.parent)
    fsync_directory(source.parent)


def canonical_object(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("central evidence manifest object facts are invalid")
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_truncations(value: object) -> str:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list) and all(isinstance(reason, str) and reason for reason in value):
        values = value
    else:
        raise ValueError("central evidence truncation facts are invalid")
    if len(values) != len(set(values)):
        raise ValueError("central evidence truncation facts are duplicated")
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("central evidence missing reason is invalid")
    return value


__all__ = [
    "belongs_to_staged_event", "canonical_object", "canonical_truncations", "optional_text",
    "quarantine", "record_corrupt", "relative_or_none", "unavailable_outcome",
    "validate_clip_directory", "validate_clip_identity", "validated_video_path",
]
