"""Directory scan phase of restart evidence reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    ClipLocalState,
    EvidenceReasonCode,
)
from worker.pipeline.output.evidence.evidence_reconciliation_helpers import (
    belongs_to_staged_event,
    quarantine,
    relative_or_none,
    unavailable_outcome,
)
from worker.pipeline.output.evidence.evidence_reconciliation_port import ReconciliationOutbox
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore


def scan_evidence(
    store_dir: Path,
    outbox: ReconciliationOutbox,
    reconcile_one: Callable[[Path, Path, ClipId, ReconciliationOutbox], ClipLocalState],
    reconciled_at: str,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    clips_root, staging_root = store_dir / "clips", store_dir / "clips" / ".staging"
    clips_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    quarantine_root = clips_root / ".quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    verified = unavailable = corrupt = quarantined = 0
    seen: set[ClipId] = set()
    retained = set(outbox.retained_clip_ids())
    for clip_dir in sorted(clips_root.iterdir(), key=lambda path: path.name):
        if clip_dir.name in {".staging", ".quarantine"}:
            continue
        clip_id = ClipId(clip_dir.name)
        if outbox.clip_retention_state(clip_id) == "PURGED":
            quarantine(clip_dir, quarantine_root / f"retained-{clip_dir.name}")
            quarantined += 1
            continue
        if outbox.clip_outcome(clip_id) is None and not belongs_to_staged_event(clip_dir, outbox):
            quarantine(clip_dir, quarantine_root / f"final-{clip_dir.name}")
            quarantined += 1
            continue
        seen.add(clip_id)
        state = reconcile_one(store_dir, clip_dir, clip_id, outbox)
        verified += state is ClipLocalState.VERIFIED
        unavailable += state is ClipLocalState.UNAVAILABLE
        corrupt += state is ClipLocalState.CORRUPT
    for staging_dir in sorted(staging_root.iterdir(), key=lambda path: path.name):
        clip_id = ClipId(staging_dir.name)
        if clip_id not in seen and outbox.clip_outcome(clip_id) is not None:
            outbox.record_clip_outcome(
                unavailable_outcome(
                    clip_id, EvidenceReasonCode.INTERRUPTED_FINALIZE,
                    relative_or_none(store_dir, staging_dir / "manifest.json"),
                )
            )
            seen.add(clip_id)
            unavailable += 1
        quarantine(staging_dir, quarantine_root / f"staging-{staging_dir.name}")
        quarantined += 1
    for clip_id in outbox.awaiting_clip_ids():
        if clip_id not in seen:
            outbox.record_clip_outcome(
                unavailable_outcome(clip_id, EvidenceReasonCode.MISSING, None)
            )
            unavailable += 1
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
        if current is not None and current.local_state is ClipLocalState.VERIFIED:
            outbox.record_clip_outcome(
                replace(current, local_state=ClipLocalState.CORRUPT,
                        unavailable_reason=EvidenceReasonCode.MISSING)
            )
            corrupt += 1
    snapshots = outbox.reconcile_snapshots(SnapshotStore(store_dir), now=datetime.now(UTC))
    completed = outbox.complete_published_records(updated_at=reconciled_at)
    return (
        verified, unavailable, corrupt, quarantined, snapshots.corrupt, snapshots.attached,
        snapshots.discarded, snapshots.purged, completed,
    )


__all__ = ["scan_evidence"]
