"""Narrow durable outbox port consumed by restart reconciliation."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from worker.pipeline.output.evidence.evidence_manifest import ClipManifest
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipId,
    ClipOutcome,
    EdgeEventId,
)
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore


class SnapshotReconciliation(Protocol):
    corrupt: int
    attached: int
    discarded: int
    purged: int


class ReconciliationOutbox(Protocol):
    def ordered_event_ids(self, clip_id: ClipId) -> tuple[EdgeEventId, ...]: ...
    def validate_recovery_manifest(self, manifest: ClipManifest) -> None: ...
    def reconcile_clip(
        self, event_ids: tuple[EdgeEventId, ...], outcome: ClipOutcome
    ) -> None: ...
    def record_clip_outcome(self, outcome: ClipOutcome) -> None: ...
    def has_event(self, event_id: EdgeEventId) -> bool: ...
    def retained_clip_ids(self) -> tuple[ClipId, ...]: ...
    def clip_retention_state(self, clip_id: ClipId) -> str | None: ...
    def clip_outcome(self, clip_id: ClipId) -> ClipOutcome | None: ...
    def awaiting_clip_ids(self) -> tuple[ClipId, ...]: ...
    def finalized_clip_ids(self) -> tuple[ClipId, ...]: ...
    def complete_clip_retention(self, clip_id: ClipId, *, updated_at: str) -> None: ...
    def reconcile_snapshots(
        self, store: SnapshotStore, *, now: datetime
    ) -> SnapshotReconciliation: ...
    def complete_published_records(self, *, updated_at: str) -> int: ...


__all__ = ["ReconciliationOutbox"]
