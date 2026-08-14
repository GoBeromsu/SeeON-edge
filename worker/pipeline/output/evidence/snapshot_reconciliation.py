"""Crash/restart reconciliation for staged snapshot publication."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from worker.pipeline.output.evidence.evidence_record_recovery import (
    available_snapshot_records,
    mark_pending_snapshot_unavailable,
    pending_snapshot_records,
    verify_snapshot_records,
)
from worker.pipeline.output.evidence.evidence_record_stage import attach_snapshot_record
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction
from worker.pipeline.output.evidence.snapshot_retention import maintain_snapshot_retention
from worker.pipeline.output.evidence.snapshot_store import (
    SnapshotConflictError,
    SnapshotStore,
    StoredSnapshot,
)


@dataclass(frozen=True, slots=True)
class SnapshotReconciliationReport:
    attached: int = 0
    discarded: int = 0
    corrupt: int = 0
    purged: int = 0
    held: int = 0
    retention_failures: int = 0


def reconcile_snapshots(
    connection: sqlite3.Connection,
    store: SnapshotStore,
    *,
    now: datetime,
) -> SnapshotReconciliationReport:
    """Resume every durable transition and classify every unreferenced stage."""
    timestamp = now.isoformat().replace("+00:00", "Z")
    attached = 0
    transition_corrupt = 0
    with ImmediateTransaction(connection):
        pending = pending_snapshot_records(connection)
        available = available_snapshot_records(connection)
        referenced = {record.snapshot_id for record in (*pending, *available)}
        discarded = store.discard_unreferenced_staging(referenced, now=now)
        for record in pending:
            try:
                store.publish(record)
                attach_snapshot_record(
                    connection,
                    str(record.edge_event_id),
                    _snapshot_payload(record),
                )
                store.commit(record)
                attached += 1
            except (OSError, SnapshotConflictError, ValueError):
                if mark_pending_snapshot_unavailable(
                    connection,
                    record.snapshot_id,
                    reason="MISSING_OR_MUTATED_STAGING",
                    updated_at=timestamp,
                ):
                    transition_corrupt += 1
        for record in available:
            try:
                store.publish(record)
                store.commit(record)
            except (OSError, SnapshotConflictError, ValueError):
                # The hash verifier below records the explicit mutable state.
                pass
    # Retention markers precede deletion, so resume them before classifying a
    # post-delete/pre-commit crash as externally missing evidence.
    retention = maintain_snapshot_retention(connection, store, now=now)
    with ImmediateTransaction(connection):
        verified_corrupt = verify_snapshot_records(
            connection,
            store.store_dir,
            updated_at=timestamp,
        )
    return SnapshotReconciliationReport(
        attached=attached,
        discarded=discarded.discarded,
        corrupt=transition_corrupt + discarded.corrupt + verified_corrupt,
        purged=retention.purged,
        held=retention.held,
        retention_failures=retention.failures,
    )


def _snapshot_payload(record: StoredSnapshot) -> dict[str, object]:
    return {
        "snapshot_id": record.snapshot_id,
        "path": record.path,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "mime_type": record.mime_type,
        "captured_at": record.captured_at,
        "camera_id": record.camera_id,
        "edge_event_id": record.edge_event_id,
    }


__all__ = ["SnapshotReconciliationReport", "reconcile_snapshots"]
