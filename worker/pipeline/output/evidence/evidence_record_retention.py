"""Crash-recoverable retention state for authoritative central evidence."""

from __future__ import annotations

import sqlite3

from worker.pipeline.output.evidence.evidence_record_stage import central_records_available


def begin_clip_retention(
    connection: sqlite3.Connection,
    clip_id: str,
    *,
    updated_at: str,
) -> bool:
    """Persist intent before deleting bytes; return false while an incident is held."""
    if not central_records_available(connection):
        return True
    clip = connection.execute(
        "SELECT publish_state FROM evidence_clips WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    if clip is None or str(clip[0]) != "PUBLISHED":
        return False
    incomplete = connection.execute(
        "SELECT 1 FROM evidence_incidents "
        "WHERE primary_clip_id = ? AND lifecycle_state != 'COMPLETE' LIMIT 1",
        (clip_id,),
    ).fetchone()
    if incomplete is not None:
        return False
    derivative_pending = connection.execute(
        "SELECT 1 FROM derivative_jobs AS job "
        "JOIN evidence_incidents AS incident USING(incident_id) "
        "WHERE incident.primary_clip_id=? "
        "AND job.state IN ('PENDING','RUNNING') LIMIT 1",
        (clip_id,),
    ).fetchone()
    if derivative_pending is not None:
        return False
    row = connection.execute(
        "SELECT state, revision FROM evidence_retention_states WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO evidence_retention_states "
            "(clip_id, state, requested_at, updated_at) VALUES (?, 'PENDING', ?, ?)",
            (clip_id, updated_at, updated_at),
        )
        return True
    state = str(row[0])
    if state in {"PENDING", "PURGED"}:
        return True
    changed = connection.execute(
        "UPDATE evidence_retention_states "
        "SET state = 'PENDING', reason = NULL, revision = revision + 1, updated_at = ? "
        "WHERE clip_id = ? AND state = 'FAILED' AND revision = ?",
        (updated_at, clip_id, int(row[1])),
    ).rowcount
    return changed == 1


def complete_clip_retention(
    connection: sqlite3.Connection,
    clip_id: str,
    *,
    updated_at: str,
) -> None:
    if not central_records_available(connection):
        return
    row = connection.execute(
        "SELECT state, revision FROM evidence_retention_states WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    if row is None:
        raise ValueError("central evidence retention was not staged")
    if str(row[0]) == "PURGED":
        return
    if str(row[0]) != "PENDING":
        raise ValueError("central evidence retention is not pending")
    changed = connection.execute(
        "UPDATE evidence_retention_states "
        "SET state = 'PURGED', revision = revision + 1, updated_at = ? "
        "WHERE clip_id = ? AND state = 'PENDING' AND revision = ?",
        (updated_at, clip_id, int(row[1])),
    ).rowcount
    if changed != 1:
        raise sqlite3.IntegrityError("central evidence retention revision changed")
    connection.execute(
        """
        UPDATE evidence_artifact_slots
        SET state = 'UNAVAILABLE', reason = 'RETENTION_PURGED',
            revision = revision + 1, updated_at = ?
        WHERE slot_name = 'PRIMARY_CLIP' AND state = 'AVAILABLE'
          AND incident_id IN (
              SELECT incident_id FROM evidence_incidents WHERE primary_clip_id = ?
          )
        """,
        (updated_at, clip_id),
    )


def fail_clip_retention(
    connection: sqlite3.Connection,
    clip_id: str,
    *,
    reason: str,
    updated_at: str,
) -> None:
    if not central_records_available(connection):
        return
    if not reason:
        raise ValueError("central evidence retention failure requires a reason")
    row = connection.execute(
        "SELECT state, revision FROM evidence_retention_states WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    if row is None or str(row[0]) != "PENDING":
        return
    connection.execute(
        "UPDATE evidence_retention_states "
        "SET state = 'FAILED', reason = ?, revision = revision + 1, updated_at = ? "
        "WHERE clip_id = ? AND state = 'PENDING' AND revision = ?",
        (reason, updated_at, clip_id, int(row[1])),
    )


def clip_retention_state(connection: sqlite3.Connection, clip_id: str) -> str | None:
    if not central_records_available(connection):
        return None
    row = connection.execute(
        "SELECT state FROM evidence_retention_states WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    return None if row is None else str(row[0])


def retained_clip_ids(connection: sqlite3.Connection) -> tuple[str, ...]:
    if not central_records_available(connection):
        return ()
    rows = connection.execute(
        "SELECT clip_id FROM evidence_retention_states "
        "WHERE state IN ('PENDING','PURGED') ORDER BY clip_id"
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


__all__ = [
    "begin_clip_retention",
    "clip_retention_state",
    "complete_clip_retention",
    "fail_clip_retention",
    "retained_clip_ids",
]
