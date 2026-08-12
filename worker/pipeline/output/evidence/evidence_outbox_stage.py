"""Atomic immutable staging for evidence outbox events."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Literal

from worker.pipeline.output.evidence.evidence_outbox_types import (
    EdgeEventId,
    EventDeliveryState,
    OperatorEventRegistration,
    StagedEvent,
    StagedEventConflictError,
)
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction


def stage_event(connection: sqlite3.Connection, event: StagedEvent) -> None:
    with ImmediateTransaction(connection):
        existing = connection.execute(
            """
            SELECT detected_at, payload_json, operator_only
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
                queued_at, next_attempt_at, operator_only
            ) VALUES (?, ?, ?, 'STAGED', ?, ?, ?)
            """,
            (
                event.edge_event_id,
                event.detected_at,
                event.payload_json,
                event.queued_at,
                event.queued_at,
                int(event.operator_only),
            ),
        )


def create_or_load_operator_event(
    connection: sqlite3.Connection,
    validation_run_id: str,
    event_factory: Callable[[], StagedEvent],
) -> OperatorEventRegistration:
    """Atomically bind one immutable operator event to a validation run."""
    with ImmediateTransaction(connection):
        existing = _operator_event_for_validation_run(connection, validation_run_id)
        if existing is not None:
            return existing
        event = event_factory()
        if not event.operator_only:
            raise ValueError("operator event factory must create an operator-only event")
        connection.execute(
            """
            INSERT INTO evidence_events (
                edge_event_id, detected_at, payload_json, state,
                queued_at, next_attempt_at, operator_only
            ) VALUES (?, ?, ?, 'READY', ?, ?, 1)
            """,
            (
                event.edge_event_id,
                event.detected_at,
                event.payload_json,
                event.queued_at,
                event.queued_at,
            ),
        )
        connection.execute(
            """INSERT INTO system_test_runs (validation_run_id, edge_event_id)
               VALUES (?, ?)""",
            (validation_run_id, event.edge_event_id),
        )
        return OperatorEventRegistration(
            validation_run_id=validation_run_id,
            edge_event_id=event.edge_event_id,
            payload_json=event.payload_json,
            delivery_state=EventDeliveryState.PENDING,
            backend_event_id=None,
            created=True,
        )


def _operator_event_for_validation_run(
    connection: sqlite3.Connection,
    validation_run_id: str,
) -> OperatorEventRegistration | None:
    row = connection.execute(
        """
        SELECT event.edge_event_id, event.payload_json, event.delivery_state,
               event.backend_event_id
        FROM system_test_runs AS run
        JOIN evidence_events AS event USING (edge_event_id)
        WHERE run.validation_run_id = ? AND event.operator_only = 1
        """,
        (validation_run_id,),
    ).fetchone()
    if row is None:
        return None
    return OperatorEventRegistration(
        validation_run_id=validation_run_id,
        edge_event_id=EdgeEventId(str(row[0])),
        payload_json=str(row[1]),
        delivery_state=EventDeliveryState(str(row[2])),
        backend_event_id=None if row[3] is None else str(row[3]),
        created=False,
    )


def _refuse_mismatched_replay(event: StagedEvent, existing: sqlite3.Row) -> None:
    mismatches: list[Literal["detected_at", "payload_json", "operator_only"]] = []
    if str(existing[0]) != event.detected_at:
        mismatches.append("detected_at")
    if str(existing[1]) != event.payload_json:
        mismatches.append("payload_json")
    if bool(existing[2]) is not event.operator_only:
        mismatches.append("operator_only")
    if mismatches:
        raise StagedEventConflictError(event.edge_event_id, tuple(mismatches))


__all__ = ["create_or_load_operator_event", "stage_event"]
