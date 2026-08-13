"""Lease-guarded event delivery state transitions."""

from __future__ import annotations

import sqlite3

from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClaimedEvent,
    ClaimLease,
    ClipId,
    EdgeEventId,
    MissingStagedEventError,
)
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction


def mark_ready(connection: sqlite3.Connection, edge_event_id: EdgeEventId) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_events
        SET state = 'READY'
        WHERE edge_event_id = ? AND state = 'STAGED'
        """,
        (edge_event_id,),
    )
    return result.rowcount == 1


def claim(connection: sqlite3.Connection, lease: ClaimLease) -> ClaimedEvent | None:
    with ImmediateTransaction(connection):
        candidate = connection.execute(
            """
            SELECT edge_event_id
            FROM evidence_events
            WHERE next_attempt_at <= ?
              AND delivery_state = 'PENDING'
              AND (
                state = 'READY'
                OR (state = 'IN_FLIGHT' AND lease_expires_at <= ?)
              )
            ORDER BY queued_at, edge_event_id
            LIMIT 1
            """,
            (lease.now, lease.now),
        ).fetchone()
        if candidate is None:
            return None
        edge_event_id = EdgeEventId(str(candidate[0]))
        lease_expires_at = lease.now + lease.duration
        connection.execute(
            """
            UPDATE evidence_events
            SET state = 'IN_FLIGHT', lease_owner = ?, lease_expires_at = ?,
                attempt_count = attempt_count + 1
            WHERE edge_event_id = ?
            """,
            (lease.owner, lease_expires_at, edge_event_id),
        )
        claimed = connection.execute(
            """
            SELECT event.edge_event_id, event.detected_at, event.payload_json,
                   relation.clip_id, event.attempt_count
            FROM evidence_events AS event
            LEFT JOIN clip_events AS relation USING (edge_event_id)
            WHERE event.edge_event_id = ?
            """,
            (edge_event_id,),
        ).fetchone()
        if claimed is None:
            raise MissingStagedEventError(edge_event_id)
    clip_id = None if claimed[3] is None else ClipId(str(claimed[3]))
    return ClaimedEvent(
        edge_event_id=EdgeEventId(str(claimed[0])),
        detected_at=str(claimed[1]),
        payload_json=str(claimed[2]),
        clip_id=clip_id,
        lease_owner=lease.owner,
        lease_expires_at=lease_expires_at,
        attempt_count=int(claimed[4]),
    )


def schedule_retry(
    connection: sqlite3.Connection,
    claim: ClaimedEvent,
    *,
    next_attempt_at: float,
) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_events
        SET state = 'READY', next_attempt_at = ?,
            lease_owner = NULL, lease_expires_at = NULL
        WHERE edge_event_id = ? AND state = 'IN_FLIGHT'
          AND lease_owner = ? AND lease_expires_at = ?
        """,
        (
            next_attempt_at,
            claim.edge_event_id,
            claim.lease_owner,
            claim.lease_expires_at,
        ),
    )
    return result.rowcount == 1


def acknowledge(
    connection: sqlite3.Connection,
    claim: ClaimedEvent,
    *,
    backend_event_id: str | None = None,
) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_events
        SET state = 'ACKED', delivery_state = 'ACKED', backend_event_id = ?,
            lease_owner = NULL, lease_expires_at = NULL, last_error_code = NULL
        WHERE edge_event_id = ? AND state = 'IN_FLIGHT'
          AND lease_owner = ? AND lease_expires_at = ?
        """,
        (
            backend_event_id,
            claim.edge_event_id,
            claim.lease_owner,
            claim.lease_expires_at,
        ),
    )
    return result.rowcount == 1


def mark_failure(
    connection: sqlite3.Connection,
    claim: ClaimedEvent,
    *,
    state: str,
    error_code: str,
) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_events
        SET state = 'READY', delivery_state = ?, last_error_code = ?,
            lease_owner = NULL, lease_expires_at = NULL
        WHERE edge_event_id = ? AND state = 'IN_FLIGHT'
          AND lease_owner = ? AND lease_expires_at = ?
        """,
        (state, error_code, claim.edge_event_id, claim.lease_owner, claim.lease_expires_at),
    )
    return result.rowcount == 1


def delivery_state(connection: sqlite3.Connection, edge_event_id: EdgeEventId) -> str | None:
    row = connection.execute(
        "SELECT delivery_state FROM evidence_events WHERE edge_event_id = ?",
        (edge_event_id,),
    ).fetchone()
    return None if row is None else str(row[0])


def attempt_count(connection: sqlite3.Connection, edge_event_id: EdgeEventId) -> int | None:
    row = connection.execute(
        "SELECT attempt_count FROM evidence_events WHERE edge_event_id = ?",
        (edge_event_id,),
    ).fetchone()
    return None if row is None else int(row[0])


def pending_count(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT COUNT(*) FROM evidence_events WHERE state != 'ACKED'"
    ).fetchone()
    return 0 if row is None else int(row[0])


__all__ = [
    "acknowledge",
    "attempt_count",
    "claim",
    "delivery_state",
    "mark_failure",
    "mark_ready",
    "pending_count",
    "schedule_retry",
]
