"""Lease-guarded clip publication state transitions."""

from __future__ import annotations

import sqlite3

from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClaimedClip,
    ClaimLease,
    ClipId,
    ClipLocalState,
    EdgeEventId,
    EvidenceReasonCode,
)


def claim_clip(connection: sqlite3.Connection, lease: ClaimLease) -> ClaimedClip | None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            """
            SELECT clip.clip_id
            FROM evidence_clips AS clip
            WHERE clip.local_state IN ('VERIFIED', 'UNAVAILABLE', 'CORRUPT')
              AND clip.publish_next_attempt_at <= ?
              AND (
                clip.publish_state = 'WAITING'
                OR (
                  clip.publish_state = 'IN_FLIGHT'
                  AND clip.publish_lease_expires_at <= ?
                )
              )
              AND EXISTS (
                SELECT 1 FROM clip_events WHERE clip_id = clip.clip_id
              )
              AND NOT EXISTS (
                SELECT 1
                FROM clip_events AS relation
                JOIN evidence_events AS event USING (edge_event_id)
                WHERE relation.clip_id = clip.clip_id
                  AND event.delivery_state != 'ACKED'
              )
            ORDER BY clip.clip_id
            LIMIT 1
            """,
            (lease.now, lease.now),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        clip_id = ClipId(str(row[0]))
        expires_at = lease.now + lease.duration
        connection.execute(
            """
            UPDATE evidence_clips
            SET publish_state = 'IN_FLIGHT', publish_lease_owner = ?,
                publish_lease_expires_at = ?,
                publish_attempt_count = publish_attempt_count + 1
            WHERE clip_id = ?
            """,
            (lease.owner, expires_at, clip_id),
        )
        data = connection.execute(
            """
            SELECT local_state, state_version, media_relpath, sha256, size_bytes,
                   mime_type, codec, duration_ms, clip_start_at, clip_end_at,
                   finalized_at, unavailable_reason, publish_attempt_count
            FROM evidence_clips WHERE clip_id = ?
            """,
            (clip_id,),
        ).fetchone()
        events = connection.execute(
            """
            SELECT event.edge_event_id, event.payload_json
            FROM clip_events AS relation
            JOIN evidence_events AS event USING (edge_event_id)
            WHERE relation.clip_id = ?
            ORDER BY relation.ordinal
            """,
            (clip_id,),
        ).fetchall()
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    if data is None or not events:
        return None
    reason = None if data[11] is None else EvidenceReasonCode(str(data[11]))
    return ClaimedClip(
        clip_id=clip_id,
        local_state=ClipLocalState(str(data[0])),
        state_version=int(data[1]),
        media_relpath=None if data[2] is None else str(data[2]),
        sha256=None if data[3] is None else str(data[3]),
        size_bytes=None if data[4] is None else int(data[4]),
        mime_type=None if data[5] is None else str(data[5]),
        codec=None if data[6] is None else str(data[6]),
        duration_ms=None if data[7] is None else int(data[7]),
        clip_start_at=None if data[8] is None else str(data[8]),
        clip_end_at=None if data[9] is None else str(data[9]),
        finalized_at=None if data[10] is None else str(data[10]),
        unavailable_reason=reason,
        event_ids=tuple(EdgeEventId(str(event[0])) for event in events),
        event_payload_json=str(events[0][1]),
        lease_owner=lease.owner,
        lease_expires_at=expires_at,
        attempt_count=int(data[12]),
    )


def release_clip_claim(connection: sqlite3.Connection, claim: ClaimedClip) -> bool:
    """Undo an undispatched claim when live export turns off.

    The lease CAS prevents releasing a claim that another sender has already
    recovered. Since no dispatch occurred, restore the attempt counter too.
    """
    result = connection.execute(
        """
        UPDATE evidence_clips
        SET publish_state = 'WAITING', publish_lease_owner = NULL,
            publish_lease_expires_at = NULL,
            publish_attempt_count = publish_attempt_count - 1
        WHERE clip_id = ? AND publish_state = 'IN_FLIGHT'
          AND publish_lease_owner = ? AND publish_lease_expires_at = ?
          AND publish_attempt_count > 0
        """,
        (claim.clip_id, claim.lease_owner, claim.lease_expires_at),
    )
    return result.rowcount == 1


def schedule_clip_retry(
    connection: sqlite3.Connection,
    claim: ClaimedClip,
    *,
    next_attempt_at: float,
    error_code: str,
) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_clips
        SET publish_state = 'WAITING', publish_next_attempt_at = ?,
            publish_lease_owner = NULL, publish_lease_expires_at = NULL,
            last_error_code = ?
        WHERE clip_id = ? AND publish_state = 'IN_FLIGHT'
          AND publish_lease_owner = ? AND publish_lease_expires_at = ?
        """,
        (next_attempt_at, error_code, claim.clip_id, claim.lease_owner, claim.lease_expires_at),
    )
    return result.rowcount == 1


def acknowledge_clip(
    connection: sqlite3.Connection,
    claim: ClaimedClip,
    *,
    acknowledged_at: float,
    remote_state: str,
) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_clips
        SET publish_state = 'PUBLISHED', remote_state = ?, backend_ack_at = ?,
            publish_lease_owner = NULL, publish_lease_expires_at = NULL,
            last_error_code = NULL
        WHERE clip_id = ? AND publish_state = 'IN_FLIGHT'
          AND publish_lease_owner = ? AND publish_lease_expires_at = ?
        """,
        (
            remote_state,
            acknowledged_at,
            claim.clip_id,
            claim.lease_owner,
            claim.lease_expires_at,
        ),
    )
    return result.rowcount == 1


def mark_clip_failure(
    connection: sqlite3.Connection,
    claim: ClaimedClip,
    *,
    state: str,
    error_code: str,
) -> bool:
    result = connection.execute(
        """
        UPDATE evidence_clips
        SET publish_state = ?, publish_lease_owner = NULL,
            publish_lease_expires_at = NULL, last_error_code = ?
        WHERE clip_id = ? AND publish_state = 'IN_FLIGHT'
          AND publish_lease_owner = ? AND publish_lease_expires_at = ?
        """,
        (state, error_code, claim.clip_id, claim.lease_owner, claim.lease_expires_at),
    )
    return result.rowcount == 1


def clip_publish_state(connection: sqlite3.Connection, clip_id: ClipId) -> str | None:
    row = connection.execute(
        "SELECT publish_state FROM evidence_clips WHERE clip_id = ?",
        (clip_id,),
    ).fetchone()
    return None if row is None else str(row[0])


def is_clip_held(connection: sqlite3.Connection, clip_id: ClipId) -> bool:
    return clip_publish_state(connection, clip_id) != "PUBLISHED"


def release_compatibility(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        UPDATE evidence_events
        SET state = 'READY', delivery_state = 'PENDING', last_error_code = NULL
        WHERE delivery_state = 'COMPATIBILITY'
        """
    )
    connection.execute(
        """
        UPDATE evidence_clips
        SET publish_state = 'WAITING', last_error_code = NULL
        WHERE publish_state = 'COMPATIBILITY'
        """
    )


__all__ = [
    "acknowledge_clip",
    "claim_clip",
    "clip_publish_state",
    "is_clip_held",
    "mark_clip_failure",
    "release_compatibility",
    "schedule_clip_retry",
]
