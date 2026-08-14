"""Restart reconciliation between atomic clip manifests and the WAL outbox."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from worker.pipeline.output.evidence.durability import fsync_directory
from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    ReadyClipManifest,
    UnavailableClipManifest,
    parse_manifest,
    parse_manifest_content,
    verify_ready_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox import EvidenceOutbox
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
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    verified: int = 0
    unavailable: int = 0
    corrupt: int = 0
    quarantined: int = 0
    snapshot_corrupt: int = 0
    snapshot_attached: int = 0
    snapshot_discarded: int = 0
    snapshot_purged: int = 0
    completed: int = 0


def reconcile_finalized_clip(
    store_dir: Path,
    clip_id: ClipId,
    outbox: EvidenceOutbox,
) -> ClipLocalState:
    clip_dir = store_dir / "clips" / clip_id
    return _reconcile_final_dir(store_dir, clip_dir, clip_id, outbox)


def reconcile_event_evidence(
    store_dir: Path,
    outbox: EvidenceOutbox,
) -> ReconciliationReport:
    clips_root = store_dir / "clips"
    staging_root = clips_root / ".staging"
    clips_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    quarantine_root = clips_root / ".quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    verified = 0
    unavailable = 0
    corrupt = 0
    quarantined = 0
    seen: set[ClipId] = set()
    retained = set(outbox.retained_clip_ids())
    for clip_dir in sorted(clips_root.iterdir(), key=lambda path: path.name):
        if clip_dir.name in {".staging", ".quarantine"}:
            continue
        clip_id = ClipId(clip_dir.name)
        if outbox.clip_retention_state(clip_id) == "PURGED":
            _quarantine(clip_dir, quarantine_root / f"retained-{clip_dir.name}")
            quarantined += 1
            continue
        if outbox.clip_outcome(clip_id) is None and not _belongs_to_staged_event(clip_dir, outbox):
            _quarantine(clip_dir, quarantine_root / f"final-{clip_dir.name}")
            quarantined += 1
            continue
        seen.add(clip_id)
        outcome = _reconcile_final_dir(store_dir, clip_dir, clip_id, outbox)
        match outcome:
            case ClipLocalState.VERIFIED:
                verified += 1
            case ClipLocalState.UNAVAILABLE:
                unavailable += 1
            case ClipLocalState.CORRUPT:
                corrupt += 1
            case ClipLocalState.AWAITING_FINALIZE:
                raise AssertionError("reconciliation cannot retain awaiting finalization")
    for staging_dir in sorted(staging_root.iterdir(), key=lambda path: path.name):
        clip_id = ClipId(staging_dir.name)
        outcome = outbox.clip_outcome(clip_id)
        if clip_id not in seen and outcome is not None:
            outbox.record_clip_outcome(
                _unavailable_outcome(
                    clip_id,
                    EvidenceReasonCode.INTERRUPTED_FINALIZE,
                    _relative_or_none(store_dir, staging_dir / "manifest.json"),
                )
            )
            seen.add(clip_id)
            unavailable += 1
        _quarantine(staging_dir, quarantine_root / f"staging-{staging_dir.name}")
        quarantined += 1
    for clip_id in outbox.awaiting_clip_ids():
        if clip_id in seen:
            continue
        outbox.record_clip_outcome(
            _unavailable_outcome(clip_id, EvidenceReasonCode.MISSING, None)
        )
        unavailable += 1
    reconciled_at = _utc_now()
    for clip_id in retained:
        if (
            outbox.clip_retention_state(clip_id) == "PENDING"
            and not (clips_root / clip_id).exists()
        ):
            outbox.complete_clip_retention(clip_id, updated_at=reconciled_at)
    for clip_id in outbox.finalized_clip_ids():
        if clip_id in seen or clip_id in retained:
            continue
        current = outbox.clip_outcome(clip_id)
        if current is None or current.local_state is not ClipLocalState.VERIFIED:
            continue
        outbox.record_clip_outcome(
            replace(
                current,
                local_state=ClipLocalState.CORRUPT,
                unavailable_reason=EvidenceReasonCode.MISSING,
            )
        )
        corrupt += 1
    snapshot_report = outbox.reconcile_snapshots(
        SnapshotStore(store_dir),
        now=datetime.now(UTC),
    )
    completed = outbox.complete_published_records(updated_at=reconciled_at)
    return ReconciliationReport(
        verified=verified,
        unavailable=unavailable,
        corrupt=corrupt,
        quarantined=quarantined,
        snapshot_corrupt=snapshot_report.corrupt,
        snapshot_attached=snapshot_report.attached,
        snapshot_discarded=snapshot_report.discarded,
        snapshot_purged=snapshot_report.purged,
        completed=completed,
    )


def _reconcile_final_dir(
    store_dir: Path,
    clip_dir: Path,
    clip_id: ClipId,
    outbox: EvidenceOutbox,
) -> ClipLocalState:
    manifest_path = clip_dir / "manifest.json"
    relative_manifest = _relative_or_none(store_dir, manifest_path)
    event_ids = outbox.ordered_event_ids(clip_id)
    try:
        _validate_clip_directory(clip_dir)
        manifest, manifest_content, manifest_payload = parse_manifest_content(manifest_path)
        _validate_clip_identity(manifest.clip_id, clip_id)
        event_ids = tuple(EdgeEventId(event_ref) for event_ref in manifest.event_refs)
        outbox.validate_recovery_manifest(manifest)
        match manifest:
            case ReadyClipManifest():
                video_path = _validated_video_path(clip_dir)
                verify_ready_manifest(manifest, video_path)
                outbox.reconcile_clip(
                    event_ids,
                    ClipOutcome(
                        clip_id=clip_id,
                        local_state=ClipLocalState.VERIFIED,
                        manifest_path=relative_manifest,
                        state_version=manifest.state_version,
                        media_relpath=_relative_or_none(store_dir, video_path),
                        sha256=manifest.sha256,
                        size_bytes=manifest.size_bytes,
                        mime_type=manifest.mime_type,
                        codec=manifest.codec,
                        duration_ms=manifest.duration_ms,
                        clip_start_at=manifest.clip_start_at,
                        clip_end_at=manifest.clip_end_at,
                        finalized_at=manifest.finalized_at,
                        manifest_sha256=hashlib.sha256(manifest_content).hexdigest(),
                        manifest_size_bytes=len(manifest_content),
                        audio_codec=manifest.audio_codec,
                        runtime_manifest_sha256=manifest.runtime_manifest_sha256,
                        source_media_json=_canonical_object(manifest_payload.get("source_media")),
                        time_origin_json=_canonical_object(manifest_payload.get("time_origin")),
                        truncation_json=_canonical_truncations(
                            manifest_payload.get("truncation_reasons")
                        ),
                        source_missing_reason=_optional_text(
                            manifest_payload.get("source_error_reason")
                        ),
                    ),
                )
                return ClipLocalState.VERIFIED
            case UnavailableClipManifest():
                outbox.reconcile_clip(
                    event_ids,
                    ClipOutcome(
                        clip_id=clip_id,
                        local_state=ClipLocalState.UNAVAILABLE,
                        manifest_path=relative_manifest,
                        state_version=manifest.state_version,
                        clip_start_at=manifest.clip_start_at,
                        clip_end_at=manifest.clip_end_at,
                        finalized_at=manifest.finalized_at,
                        unavailable_reason=manifest.reason_code,
                        manifest_sha256=hashlib.sha256(manifest_content).hexdigest(),
                        manifest_size_bytes=len(manifest_content),
                        runtime_manifest_sha256=manifest.runtime_manifest_sha256,
                        source_media_json=_canonical_object(manifest_payload.get("source_media")),
                        time_origin_json=_canonical_object(manifest_payload.get("time_origin")),
                        truncation_json=_canonical_truncations(
                            manifest_payload.get("truncation_reasons")
                        ),
                        source_missing_reason=(
                            _optional_text(manifest_payload.get("source_error_reason"))
                            or manifest.reason_code.value
                        ),
                    ),
                )
                return ClipLocalState.UNAVAILABLE
    except (
        ClipEvidenceError,
        ClipOutcomeConflictError,
        EventClipConflictError,
        MissingStagedEventError,
        ValueError,
    ):
        _record_corrupt(outbox, clip_id, relative_manifest, event_ids)
        return ClipLocalState.CORRUPT


def _record_corrupt(
    outbox: EvidenceOutbox,
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
            ClipOutcomeConflictError,
            EventClipConflictError,
            MissingStagedEventError,
            ValueError,
        ):
            pass
        else:
            return
    outbox.record_clip_outcome(outcome)


def _validate_clip_directory(clip_dir: Path) -> None:
    if clip_dir.is_symlink() or not clip_dir.is_dir():
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "clip directory invalid")


def _validate_clip_identity(manifest_clip_id: str, expected_clip_id: ClipId) -> None:
    if manifest_clip_id != expected_clip_id:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "clip identity mismatch")


def _validated_video_path(clip_dir: Path) -> Path:
    video_path = clip_dir / "clip.mp4"
    if video_path.parent.resolve() != clip_dir.resolve():
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "media path escaped clip")
    return video_path


def _unavailable_outcome(
    clip_id: ClipId,
    reason_code: EvidenceReasonCode,
    manifest_path: str | None,
) -> ClipOutcome:
    return ClipOutcome(
        clip_id=clip_id,
        local_state=ClipLocalState.UNAVAILABLE,
        manifest_path=manifest_path,
        state_version=2,
        unavailable_reason=reason_code,
    )


def _relative_or_none(store_dir: Path, path: Path) -> str | None:
    try:
        return str(path.relative_to(store_dir))
    except ValueError:
        return None


def _belongs_to_staged_event(clip_dir: Path, outbox: EvidenceOutbox) -> bool:
    try:
        manifest = parse_manifest(clip_dir / "manifest.json")
    except ClipEvidenceError:
        return False
    return any(outbox.has_event(EdgeEventId(event_ref)) for event_ref in manifest.event_refs)


def _quarantine(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ClipEvidenceError(
            EvidenceReasonCode.CORRUPT,
            f"quarantine destination already exists: {destination.name}",
        )
    os.replace(source, destination)
    fsync_directory(destination.parent)
    fsync_directory(source.parent)


def _canonical_object(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("central evidence manifest object facts are invalid")
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _canonical_truncations(value: object) -> str:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list) and all(isinstance(reason, str) and reason for reason in value):
        values = value
    else:
        raise ValueError("central evidence truncation facts are invalid")
    if len(values) != len(set(values)):
        raise ValueError("central evidence truncation facts are duplicated")
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("central evidence missing reason is invalid")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ReconciliationReport",
    "reconcile_event_evidence",
    "reconcile_finalized_clip",
]
