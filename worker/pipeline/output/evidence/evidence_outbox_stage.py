"""Atomic immutable staging for evidence outbox events."""

from __future__ import annotations

import sqlite3
from typing import Literal

from worker.pipeline.output.evidence.evidence_outbox_types import (
    StagedEvent,
    StagedEventConflictError,
)
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction


def stage_event(connection: sqlite3.Connection, event: StagedEvent) -> None:
    with ImmediateTransaction(connection):
        existing = connection.execute(
            """
            SELECT detected_at, payload_json
            FROM evidence_events
            WHERE edge_event_id = ?
            """,
            (event.edge_event_id,),
        ).fetchone()
        if existing is not None:
            _refuse_mismatched_replay(event, existing)
            return
        connection.execute(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state,
                queued_at, next_attempt_at
            ) VALUES (?, ?, ?, 'STAGED', ?, ?)
            """,
            (
                event.edge_event_id,
                event.detected_at,
                event.payload_json,
                event.queued_at,
                event.queued_at,
            ),
        )


def _refuse_mismatched_replay(event: StagedEvent, existing: sqlite3.Row) -> None:
    mismatches: list[Literal["detected_at", "payload_json"]] = []
    if str(existing[0]) != event.detected_at:
        mismatches.append("detected_at")
    if str(existing[1]) != event.payload_json:
        mismatches.append("payload_json")
    if mismatches:
        raise StagedEventConflictError(event.edge_event_id, tuple(mismatches))


__all__ = ["stage_event"]
