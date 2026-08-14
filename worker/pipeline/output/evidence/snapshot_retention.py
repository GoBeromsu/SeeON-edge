"""Policy-bounded snapshot retention with central tombstones."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from worker.pipeline.output.evidence.evidence_record_stage import central_records_available
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore, StoredSnapshot


@dataclass(frozen=True, slots=True)
class SnapshotRetentionReport:
    purged: int = 0
    held: int = 0
    failures: int = 0


def maintain_snapshot_retention(
    connection: sqlite3.Connection,
    store: SnapshotStore,
    *,
    now: datetime,
) -> SnapshotRetentionReport:
    """Purge only released evidence until every configured bound is met."""
    identities = {record.snapshot_id: record for record in store.identity_records()}
    markers = {record.snapshot_id: record for record in store.retention_records()}
    records = sorted(
        {**identities, **markers}.values(),
        key=lambda record: (_captured(record), record.snapshot_id),
    )
    total_files = len(records)
    total_bytes = sum(record.size_bytes for record in records)
    camera_files: dict[str, int] = {}
    camera_bytes: dict[str, int] = {}
    for record in records:
        camera_files[record.camera_id] = camera_files.get(record.camera_id, 0) + 1
        camera_bytes[record.camera_id] = camera_bytes.get(record.camera_id, 0) + record.size_bytes

    purged = 0
    held = 0
    failures = 0
    for record in records:
        old = now - _captured(record) >= store.limits.max_age
        exceeds = (
            total_files > store.limits.max_files_global
            or total_bytes > store.limits.max_bytes_global
            or camera_files[record.camera_id] > store.limits.max_files_per_camera
            or camera_bytes[record.camera_id] > store.limits.max_bytes_per_camera
        )
        if not old and not exceeds and record.snapshot_id not in markers:
            continue
        if record.snapshot_id in identities and not store.matches_committed(record):
            if record.snapshot_id in markers:
                store.cancel_retention(record)
            failures += 1
            continue
        store.stage_retention(record)
        try:
            with ImmediateTransaction(connection):
                if not _begin_retention(connection, record, updated_at=_timestamp(now)):
                    store.cancel_retention(record)
                    held += 1
                    continue
                store.remove_committed(record)
                _complete_retention(connection, record, updated_at=_timestamp(now))
        except (OSError, sqlite3.Error, ValueError):
            failures += 1
            continue
        store.commit_retention(record)
        purged += 1
        total_files -= 1
        total_bytes -= record.size_bytes
        camera_files[record.camera_id] -= 1
        camera_bytes[record.camera_id] -= record.size_bytes
    return SnapshotRetentionReport(purged=purged, held=held, failures=failures)


def _begin_retention(
    connection: sqlite3.Connection,
    snapshot: StoredSnapshot,
    *,
    updated_at: str,
) -> bool:
    if not central_records_available(connection):
        return False
    review_clause = ""
    if _table_exists(connection, "control_evidence_review_state"):
        review_clause = (
            "AND NOT EXISTS (SELECT 1 FROM control_evidence_review_state AS review "
            "WHERE review.incident_id = incident.incident_id)"
        )
    row = connection.execute(
        f"""
        SELECT incident.incident_id, slot.state, slot.reason, slot.revision
        FROM evidence_incident_snapshots AS relation
        JOIN evidence_incidents AS incident USING (incident_id)
        JOIN evidence_events AS event USING (edge_event_id)
        JOIN evidence_clips AS clip ON clip.clip_id = incident.primary_clip_id
        JOIN evidence_artifact_slots AS slot
          ON slot.incident_id = incident.incident_id AND slot.slot_name = 'SNAPSHOT'
        WHERE relation.snapshot_id = ?
          AND relation.camera_id = ?
          AND incident.edge_event_id = ?
          AND incident.lifecycle_state = 'COMPLETE'
          AND event.delivery_state = 'ACKED'
          AND clip.publish_state = 'PUBLISHED'
          {review_clause}
        """,
        (snapshot.snapshot_id, snapshot.camera_id, snapshot.edge_event_id),
    ).fetchone()
    if row is None:
        return False
    state = str(row[1])
    reason = None if row[2] is None else str(row[2])
    if state == "UNAVAILABLE" and reason == "RETENTION_PENDING":
        return True
    if state == "UNAVAILABLE" and reason == "RETENTION_PURGED":
        return True
    if state != "AVAILABLE":
        return False
    changed = connection.execute(
        """
        UPDATE evidence_artifact_slots
        SET state = 'UNAVAILABLE', reason = 'RETENTION_PENDING',
            revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND slot_name = 'SNAPSHOT'
          AND state = 'AVAILABLE' AND revision = ?
        """,
        (updated_at, str(row[0]), int(row[3])),
    ).rowcount
    return changed == 1


def _complete_retention(
    connection: sqlite3.Connection,
    snapshot: StoredSnapshot,
    *,
    updated_at: str,
) -> None:
    row = connection.execute(
        """
        SELECT slot.incident_id, slot.state, slot.reason, slot.revision
        FROM evidence_incident_snapshots AS relation
        JOIN evidence_artifact_slots AS slot
          ON slot.incident_id = relation.incident_id AND slot.slot_name = 'SNAPSHOT'
        WHERE relation.snapshot_id = ?
        """,
        (snapshot.snapshot_id,),
    ).fetchone()
    if row is None:
        raise ValueError("central snapshot retention relation is missing")
    if str(row[1]) == "UNAVAILABLE" and str(row[2]) == "RETENTION_PURGED":
        return
    if str(row[1]) != "UNAVAILABLE" or str(row[2]) != "RETENTION_PENDING":
        raise ValueError("central snapshot retention was not staged")
    changed = connection.execute(
        """
        UPDATE evidence_artifact_slots
        SET reason = 'RETENTION_PURGED', revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND slot_name = 'SNAPSHOT'
          AND state = 'UNAVAILABLE' AND reason = 'RETENTION_PENDING' AND revision = ?
        """,
        (updated_at, str(row[0]), int(row[3])),
    ).rowcount
    if changed != 1:
        raise sqlite3.IntegrityError("central snapshot retention revision changed")


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _captured(record: StoredSnapshot) -> datetime:
    parsed = datetime.fromisoformat(record.captured_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot captured_at is not timezone-aware")
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["SnapshotRetentionReport", "maintain_snapshot_retention"]
