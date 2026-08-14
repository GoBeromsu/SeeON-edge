"""Clip-specific durable evidence outbox operations."""

from __future__ import annotations

import sqlite3

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
from worker.pipeline.output.evidence.evidence_record_publish import (
    reconcile_primary_records,
)
from worker.pipeline.output.evidence.outbox_transaction import (
    ImmediateTransaction,
    write_transaction,
)


def bind_clip(
    connection: sqlite3.Connection,
    edge_event_id: EdgeEventId,
    clip_id: ClipId,
) -> int:
    with write_transaction(connection):
        has_direct_trace_refs = (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' "
                "AND name = 'evidence_clip_trace_refs'"
            ).fetchone()
            is not None
        )
        decision_trace_id = None
        if has_direct_trace_refs:
            trace_row = connection.execute(
                "SELECT decision_trace_id FROM evidence_event_trace_refs WHERE edge_event_id = ?",
                (edge_event_id,),
            ).fetchone()
            if trace_row is None:
                raise ValueError("central evidence clip requires a decision trace reference")
            decision_trace_id = str(trace_row[0])
        existing = connection.execute(
            "SELECT clip_id, ordinal FROM clip_events WHERE edge_event_id = ?",
            (edge_event_id,),
        ).fetchone()
        if existing is not None:
            existing_clip_id = ClipId(str(existing[0]))
            if existing_clip_id != clip_id:
                raise EventClipConflictError(edge_event_id, existing_clip_id, clip_id)
            if decision_trace_id is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO evidence_clip_trace_refs "
                    "(clip_id, edge_event_id, decision_trace_id) VALUES (?, ?, ?)",
                    (clip_id, edge_event_id, decision_trace_id),
                )
            return int(existing[1])
        staged = connection.execute(
            "SELECT 1 FROM evidence_events WHERE edge_event_id = ?",
            (edge_event_id,),
        ).fetchone()
        if staged is None:
            raise MissingStagedEventError(edge_event_id)
        connection.execute(
            """
            INSERT INTO evidence_clips (clip_id, local_state, state_version)
            VALUES (?, 'AWAITING_FINALIZE', 1) ON CONFLICT DO NOTHING
            """,
            (clip_id,),
        )
        ordinal_row = connection.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM clip_events WHERE clip_id = ?",
            (clip_id,),
        ).fetchone()
        ordinal = 0 if ordinal_row is None else int(ordinal_row[0])
        connection.execute(
            "INSERT INTO clip_events (clip_id, edge_event_id, ordinal) VALUES (?, ?, ?)",
            (clip_id, edge_event_id, ordinal),
        )
        if decision_trace_id is not None:
            connection.execute(
                "INSERT INTO evidence_clip_trace_refs "
                "(clip_id, edge_event_id, decision_trace_id) VALUES (?, ?, ?)",
                (clip_id, edge_event_id, decision_trace_id),
            )
        connection.execute(
            "UPDATE evidence_events SET state = 'READY' WHERE edge_event_id = ?",
            (edge_event_id,),
        )
        return ordinal


def ordered_event_ids(
    connection: sqlite3.Connection,
    clip_id: ClipId,
) -> tuple[EdgeEventId, ...]:
    rows = connection.execute(
        "SELECT edge_event_id FROM clip_events WHERE clip_id = ? ORDER BY ordinal",
        (clip_id,),
    ).fetchall()
    return tuple(EdgeEventId(str(row[0])) for row in rows)


def finalized_clip_ids(connection: sqlite3.Connection) -> tuple[ClipId, ...]:
    rows = connection.execute(
        """
        SELECT clip_id FROM evidence_clips
        WHERE local_state != 'AWAITING_FINALIZE'
        ORDER BY clip_id
        """
    ).fetchall()
    return tuple(ClipId(str(row[0])) for row in rows)


def awaiting_clip_ids(connection: sqlite3.Connection) -> tuple[ClipId, ...]:
    rows = connection.execute(
        """
        SELECT clip_id FROM evidence_clips
        WHERE local_state = 'AWAITING_FINALIZE'
        ORDER BY clip_id
        """
    ).fetchall()
    return tuple(ClipId(str(row[0])) for row in rows)


def record_clip_outcome(connection: sqlite3.Connection, outcome: ClipOutcome) -> None:
    with write_transaction(connection):
        existing = connection.execute(
            """
            SELECT local_state, manifest_path, state_version, media_relpath,
                   sha256, size_bytes, mime_type, codec, duration_ms,
                   clip_start_at, clip_end_at, finalized_at,
                   COALESCE(unavailable_reason_code, unavailable_reason)
            FROM evidence_clips WHERE clip_id = ?
            """,
            (outcome.clip_id,),
        ).fetchone()
        if existing is None:
            _insert_clip_outcome(connection, outcome)
        else:
            current = _clip_outcome_from_row(outcome.clip_id, existing)
            if not _same_local_outcome(current, outcome):
                if (
                    current.local_state is not ClipLocalState.AWAITING_FINALIZE
                    and outcome.local_state is not ClipLocalState.CORRUPT
                ):
                    raise ClipOutcomeConflictError(
                        outcome.clip_id,
                        current.local_state,
                        outcome.local_state,
                    )
                connection.execute(
                    """
                    UPDATE evidence_clips
                    SET local_state = ?, manifest_path = COALESCE(?, manifest_path),
                        state_version = ?, media_relpath = COALESCE(?, media_relpath),
                        sha256 = COALESCE(?, sha256), size_bytes = COALESCE(?, size_bytes),
                        mime_type = COALESCE(?, mime_type), codec = COALESCE(?, codec),
                        duration_ms = COALESCE(?, duration_ms),
                        clip_start_at = COALESCE(?, clip_start_at),
                        clip_end_at = COALESCE(?, clip_end_at),
                        finalized_at = COALESCE(?, finalized_at), unavailable_reason = ?,
                        unavailable_reason_code = ?
                    WHERE clip_id = ?
                    """,
                    _clip_outcome_values(outcome),
                )
        event_ids = ordered_event_ids(connection, outcome.clip_id)
        reconcile_primary_records(connection, event_ids, outcome)


def reconcile_clip(
    connection: sqlite3.Connection,
    event_ids: tuple[EdgeEventId, ...],
    outcome: ClipOutcome,
) -> None:
    with ImmediateTransaction(connection):
        for event_id in event_ids:
            bind_clip(connection, event_id, outcome.clip_id)
        record_clip_outcome(connection, outcome)


def clip_outcome(
    connection: sqlite3.Connection,
    clip_id: ClipId,
) -> ClipOutcome | None:
    row = connection.execute(
        """
        SELECT local_state, manifest_path, state_version, media_relpath,
               sha256, size_bytes, mime_type, codec, duration_ms,
               clip_start_at, clip_end_at, finalized_at,
               COALESCE(unavailable_reason_code, unavailable_reason)
        FROM evidence_clips WHERE clip_id = ?
        """,
        (clip_id,),
    ).fetchone()
    return None if row is None else _clip_outcome_from_row(clip_id, row)


def _insert_clip_outcome(
    connection: sqlite3.Connection,
    outcome: ClipOutcome,
) -> None:
    connection.execute(
        """
        INSERT INTO evidence_clips (
            clip_id, local_state, manifest_path, state_version, media_relpath,
            sha256, size_bytes, mime_type, codec, duration_ms, clip_start_at,
            clip_end_at, finalized_at, unavailable_reason, unavailable_reason_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (outcome.clip_id, *_clip_outcome_values(outcome)[:-1]),
    )


def _clip_outcome_values(outcome: ClipOutcome) -> tuple[str | int | None, ...]:
    return (
        outcome.local_state.value,
        outcome.manifest_path,
        outcome.state_version,
        outcome.media_relpath,
        outcome.sha256,
        outcome.size_bytes,
        outcome.mime_type,
        outcome.codec,
        outcome.duration_ms,
        outcome.clip_start_at,
        outcome.clip_end_at,
        outcome.finalized_at,
        _legacy_unavailable_reason(outcome.unavailable_reason),
        None if outcome.unavailable_reason is None else outcome.unavailable_reason.value,
        outcome.clip_id,
    )


def _legacy_unavailable_reason(reason: EvidenceReasonCode | None) -> str | None:
    if reason is None or reason is EvidenceReasonCode.STREAM_EPOCH_MISMATCH:
        return None
    return reason.value


def _same_local_outcome(current: ClipOutcome, requested: ClipOutcome) -> bool:
    return (
        current.clip_id,
        current.local_state,
        current.manifest_path,
        current.state_version,
        current.media_relpath,
        current.sha256,
        current.size_bytes,
        current.mime_type,
        current.codec,
        current.duration_ms,
        current.clip_start_at,
        current.clip_end_at,
        current.finalized_at,
        current.unavailable_reason,
    ) == (
        requested.clip_id,
        requested.local_state,
        requested.manifest_path,
        requested.state_version,
        requested.media_relpath,
        requested.sha256,
        requested.size_bytes,
        requested.mime_type,
        requested.codec,
        requested.duration_ms,
        requested.clip_start_at,
        requested.clip_end_at,
        requested.finalized_at,
        requested.unavailable_reason,
    )


def _clip_outcome_from_row(clip_id: ClipId, row: sqlite3.Row) -> ClipOutcome:
    reason = None if row[12] is None else EvidenceReasonCode(str(row[12]))
    return ClipOutcome(
        clip_id=clip_id,
        local_state=ClipLocalState(str(row[0])),
        manifest_path=None if row[1] is None else str(row[1]),
        state_version=int(row[2]),
        media_relpath=None if row[3] is None else str(row[3]),
        sha256=None if row[4] is None else str(row[4]),
        size_bytes=None if row[5] is None else int(row[5]),
        mime_type=None if row[6] is None else str(row[6]),
        codec=None if row[7] is None else str(row[7]),
        duration_ms=None if row[8] is None else int(row[8]),
        clip_start_at=None if row[9] is None else str(row[9]),
        clip_end_at=None if row[10] is None else str(row[10]),
        finalized_at=None if row[11] is None else str(row[11]),
        unavailable_reason=reason,
    )


__all__ = [
    "awaiting_clip_ids",
    "bind_clip",
    "clip_outcome",
    "finalized_clip_ids",
    "ordered_event_ids",
    "reconcile_clip",
    "record_clip_outcome",
]
