"""Restart reconciliation between atomic clip manifests and the WAL outbox."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from worker.pipeline.output.evidence.evidence_manifest import (
    ClipEvidenceError,
    ReadyClipManifest,
    UnavailableClipManifest,
    parse_manifest_content,
    verify_ready_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    ClipLocalState,
    ClipOutcome,
    ClipOutcomeConflictError,
    EdgeEventId,
    EventClipConflictError,
    MissingStagedEventError,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    canonical_object as _canonical_object,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    canonical_truncations as _canonical_truncations,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    optional_text as _optional_text,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    record_corrupt as _record_corrupt,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    relative_or_none as _relative_or_none,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    validate_clip_directory as _validate_clip_directory,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    validate_clip_identity as _validate_clip_identity,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    validated_video_path as _validated_video_path,
)
from worker.pipeline.output.evidence.evidence_reconciliation_port import ReconciliationOutbox
from worker.pipeline.output.evidence.evidence_reconciliation_scan import scan_evidence
from worker.pipeline.output.evidence.terminal_outcome import (
    TerminalClipOutcome,
    TerminalClipState,
    TerminalOutcomeConflictError,
    commit_corrupt_terminal_outcome,
    commit_terminal_outcome,
)


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
    outbox: ReconciliationOutbox,
) -> ClipLocalState:
    clip_dir = store_dir / "clips" / clip_id
    return _reconcile_final_dir(store_dir, clip_dir, clip_id, outbox)


def reconcile_event_evidence(
    store_dir: Path,
    outbox: ReconciliationOutbox,
) -> ReconciliationReport:
    values = scan_evidence(store_dir, outbox, _reconcile_final_dir, _utc_now())
    return ReconciliationReport(*values)


def _reconcile_final_dir(
    store_dir: Path,
    clip_dir: Path,
    clip_id: ClipId,
    outbox: ReconciliationOutbox,
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
        manifest_digest = hashlib.sha256(manifest_content).hexdigest()
        match manifest:
            case ReadyClipManifest():
                video_path = _validated_video_path(clip_dir)
                verify_ready_manifest(manifest, video_path)
                _ = commit_terminal_outcome(
                    clip_dir,
                    TerminalClipOutcome(
                        str(clip_id), tuple(str(value) for value in event_ids),
                        TerminalClipState.READY, manifest_digest,
                    ),
                )
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
                _ = commit_terminal_outcome(
                    clip_dir,
                    TerminalClipOutcome(
                        str(clip_id), tuple(str(value) for value in event_ids),
                        TerminalClipState.UNAVAILABLE, manifest_digest,
                    ),
                )
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
        TerminalOutcomeConflictError,
    ):
        if event_ids:
            try:
                content = manifest_path.read_bytes()
                _ = commit_corrupt_terminal_outcome(
                    clip_dir,
                    TerminalClipOutcome(
                        str(clip_id), tuple(str(value) for value in event_ids),
                        TerminalClipState.CORRUPT, hashlib.sha256(content).hexdigest(),
                    ),
                )
            except (OSError, TerminalOutcomeConflictError):
                pass
        _record_corrupt(outbox, clip_id, relative_manifest, event_ids)
        return ClipLocalState.CORRUPT


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ReconciliationReport",
    "reconcile_event_evidence",
    "reconcile_finalized_clip",
]
