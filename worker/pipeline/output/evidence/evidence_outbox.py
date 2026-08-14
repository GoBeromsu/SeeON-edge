"""Durable worker-owned evidence outbox."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from worker.pipeline.output.evidence import evidence_outbox_clips as clip_store
from worker.pipeline.output.evidence import evidence_outbox_delivery as delivery_store
from worker.pipeline.output.evidence import evidence_outbox_events as event_store
from worker.pipeline.output.evidence.evidence_manifest_validation import (
    validate_recovery_manifest,
)
from worker.pipeline.output.evidence.evidence_outbox_database import (
    database_settings,
    open_connection,
)
from worker.pipeline.output.evidence.evidence_outbox_stage import stage_event
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClaimedClip,
    ClaimedEvent,
    ClaimLease,
    ClipId,
    ClipLocalState,
    ClipOutcome,
    ClipOutcomeConflictError,
    DatabaseSettings,
    EdgeEventId,
    EventClipConflictError,
    EvidenceReasonCode,
    MissingStagedEventError,
    NewerSchemaVersionError,
    StagedEvent,
    StagedEventConflictError,
)
from worker.pipeline.output.evidence.evidence_record_publish import (
    complete_published_records,
)
from worker.pipeline.output.evidence.evidence_record_recovery import verify_snapshot_records
from worker.pipeline.output.evidence.evidence_record_retention import (
    begin_clip_retention,
    clip_retention_state,
    complete_clip_retention,
    fail_clip_retention,
    retained_clip_ids,
)
from worker.pipeline.output.evidence.evidence_record_stage import attach_snapshot_record
from worker.pipeline.output.evidence.manifest_models import ClipManifest
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction
from worker.pipeline.output.evidence.snapshot_reconciliation import (
    SnapshotReconciliationReport,
    reconcile_snapshots,
)
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore


class EvidenceOutbox:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path) -> Self:
        return cls(open_connection(path))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        self._connection.close()

    def stage(
        self,
        event: StagedEvent,
        *,
        required_runtime_manifest_sha256: str | None = None,
        required_decision_trace_id: str | None = None,
    ) -> None:
        stage_event(
            self._connection,
            event,
            required_runtime_manifest_sha256=required_runtime_manifest_sha256,
            required_decision_trace_id=required_decision_trace_id,
        )

    def bind_clip(self, edge_event_id: EdgeEventId, clip_id: ClipId) -> int:
        return clip_store.bind_clip(self._connection, edge_event_id, clip_id)

    def mark_ready(self, edge_event_id: EdgeEventId) -> bool:
        return event_store.mark_ready(self._connection, edge_event_id)

    def claim(self, lease: ClaimLease) -> ClaimedEvent | None:
        return event_store.claim(self._connection, lease)

    def schedule_retry(self, claim: ClaimedEvent, *, next_attempt_at: float) -> bool:
        return event_store.schedule_retry(
            self._connection,
            claim,
            next_attempt_at=next_attempt_at,
        )

    def acknowledge(
        self,
        claim: ClaimedEvent,
        *,
        backend_event_id: str | None = None,
    ) -> bool:
        return event_store.acknowledge(
            self._connection,
            claim,
            backend_event_id=backend_event_id,
        )

    def mark_event_failure(
        self,
        claim: ClaimedEvent,
        *,
        state: str,
        error_code: str,
    ) -> bool:
        return event_store.mark_failure(
            self._connection,
            claim,
            state=state,
            error_code=error_code,
        )

    def event_delivery_state(self, edge_event_id: EdgeEventId) -> str | None:
        return event_store.delivery_state(self._connection, edge_event_id)

    def event_attempt_count(self, edge_event_id: EdgeEventId) -> int | None:
        return event_store.attempt_count(self._connection, edge_event_id)

    def claim_clip(self, lease: ClaimLease) -> ClaimedClip | None:
        return delivery_store.claim_clip(self._connection, lease)

    def release_clip_claim(self, claim: ClaimedClip) -> bool:
        return delivery_store.release_clip_claim(self._connection, claim)

    def schedule_clip_retry(
        self,
        claim: ClaimedClip,
        *,
        next_attempt_at: float,
        error_code: str,
    ) -> bool:
        return delivery_store.schedule_clip_retry(
            self._connection,
            claim,
            next_attempt_at=next_attempt_at,
            error_code=error_code,
        )

    def acknowledge_clip(
        self,
        claim: ClaimedClip,
        *,
        acknowledged_at: float,
        remote_state: str,
    ) -> bool:
        return delivery_store.acknowledge_clip(
            self._connection,
            claim,
            acknowledged_at=acknowledged_at,
            remote_state=remote_state,
        )

    def mark_clip_failure(
        self,
        claim: ClaimedClip,
        *,
        state: str,
        error_code: str,
    ) -> bool:
        return delivery_store.mark_clip_failure(
            self._connection,
            claim,
            state=state,
            error_code=error_code,
        )

    def clip_publish_state(self, clip_id: ClipId) -> str | None:
        return delivery_store.clip_publish_state(self._connection, clip_id)

    def is_clip_held(self, clip_id: ClipId) -> bool:
        return delivery_store.is_clip_held(self._connection, clip_id)

    def begin_clip_retention(self, clip_id: ClipId, *, updated_at: str) -> bool:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            begun = begin_clip_retention(
                self._connection,
                clip_id,
                updated_at=updated_at,
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return begun

    def complete_clip_retention(self, clip_id: ClipId, *, updated_at: str) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            complete_clip_retention(
                self._connection,
                clip_id,
                updated_at=updated_at,
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def fail_clip_retention(
        self,
        clip_id: ClipId,
        *,
        reason: str,
        updated_at: str,
    ) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            fail_clip_retention(
                self._connection,
                clip_id,
                reason=reason,
                updated_at=updated_at,
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def clip_retention_state(self, clip_id: ClipId) -> str | None:
        return clip_retention_state(self._connection, clip_id)

    def retained_clip_ids(self) -> tuple[ClipId, ...]:
        return tuple(ClipId(clip_id) for clip_id in retained_clip_ids(self._connection))

    def clip_remote_state(self, clip_id: ClipId) -> str | None:
        row = self._connection.execute(
            "SELECT remote_state FROM evidence_clips WHERE clip_id = ?",
            (clip_id,),
        ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def held_clip_ids(self) -> tuple[ClipId, ...]:
        rows = self._connection.execute(
            """
            SELECT clip_id FROM evidence_clips
            WHERE publish_state != 'PUBLISHED' ORDER BY clip_id
            """
        ).fetchall()
        return tuple(ClipId(str(row[0])) for row in rows)

    def release_compatibility(self) -> None:
        delivery_store.release_compatibility(self._connection)

    def pending_count(self) -> int:
        return event_store.pending_count(self._connection)

    def has_event(self, edge_event_id: EdgeEventId) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM evidence_events WHERE edge_event_id = ?",
                (edge_event_id,),
            ).fetchone()
            is not None
        )

    def attach_snapshot(
        self,
        edge_event_id: EdgeEventId,
        snapshot: dict[str, object],
    ) -> None:
        with ImmediateTransaction(self._connection):
            attach_snapshot_record(self._connection, str(edge_event_id), snapshot)

    def reconcile_snapshots(
        self,
        store: SnapshotStore,
        *,
        now: datetime,
    ) -> SnapshotReconciliationReport:
        return reconcile_snapshots(self._connection, store, now=now)

    def verify_snapshot_records(self, store_dir: Path, *, updated_at: str) -> int:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            corrupt = verify_snapshot_records(
                self._connection,
                store_dir,
                updated_at=updated_at,
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        else:
            return corrupt

    def complete_published_records(self, *, updated_at: str) -> int:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            completed = complete_published_records(
                self._connection,
                updated_at=updated_at,
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        else:
            return completed

    def validate_recovery_manifest(self, manifest: ClipManifest) -> None:
        validate_recovery_manifest(self._connection, manifest)

    def ordered_event_ids(self, clip_id: ClipId) -> tuple[EdgeEventId, ...]:
        return clip_store.ordered_event_ids(self._connection, clip_id)

    def awaiting_clip_ids(self) -> tuple[ClipId, ...]:
        return clip_store.awaiting_clip_ids(self._connection)

    def finalized_clip_ids(self) -> tuple[ClipId, ...]:
        return clip_store.finalized_clip_ids(self._connection)

    def record_clip_outcome(self, outcome: ClipOutcome) -> None:
        clip_store.record_clip_outcome(self._connection, outcome)

    def reconcile_clip(
        self,
        event_ids: tuple[EdgeEventId, ...],
        outcome: ClipOutcome,
    ) -> None:
        clip_store.reconcile_clip(self._connection, event_ids, outcome)

    def clip_outcome(self, clip_id: ClipId) -> ClipOutcome | None:
        return clip_store.clip_outcome(self._connection, clip_id)

    def database_settings(self) -> DatabaseSettings:
        return database_settings(self._connection)


__all__ = [
    "ClaimLease",
    "ClaimedClip",
    "ClaimedEvent",
    "ClipId",
    "ClipLocalState",
    "ClipOutcome",
    "ClipOutcomeConflictError",
    "DatabaseSettings",
    "EdgeEventId",
    "EventClipConflictError",
    "EvidenceOutbox",
    "EvidenceReasonCode",
    "MissingStagedEventError",
    "NewerSchemaVersionError",
    "StagedEvent",
    "StagedEventConflictError",
]
