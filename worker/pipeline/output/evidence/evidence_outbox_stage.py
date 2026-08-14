"""Atomic immutable staging for evidence outbox events."""

from __future__ import annotations

import sqlite3
from typing import Literal

from worker.pipeline.output.evidence.decision_trace_reference import (
    require_decision_trace,
)
from worker.pipeline.output.evidence.evidence_outbox_types import (
    StagedEvent,
    StagedEventConflictError,
)
from worker.pipeline.output.evidence.evidence_record_stage import stage_central_incident
from worker.pipeline.output.evidence.outbox_transaction import ImmediateTransaction
from worker.pipeline.output.evidence.runtime_manifest_reference import (
    require_runtime_manifest_contents,
)


def stage_event(
    connection: sqlite3.Connection,
    event: StagedEvent,
    *,
    required_runtime_manifest_sha256: str | None = None,
    required_decision_trace_id: str | None = None,
) -> None:
    with ImmediateTransaction(connection):
        has_trace_references = (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' "
                "AND name = 'evidence_event_trace_refs'"
            ).fetchone()
            is not None
        )
        if required_decision_trace_id is not None and not has_trace_references:
            raise ValueError("decision trace reference requires the central edge database")
        if required_runtime_manifest_sha256 is not None:
            require_runtime_manifest_contents(
                connection,
                required_runtime_manifest_sha256,
            )
        if required_decision_trace_id is not None:
            require_decision_trace(
                connection,
                required_decision_trace_id,
                runtime_manifest_sha256=required_runtime_manifest_sha256,
            )
        trace_column = (
            "(SELECT decision_trace_id FROM evidence_event_trace_refs "
            "WHERE edge_event_id = evidence_events.edge_event_id)"
            if has_trace_references
            else "NULL"
        )
        existing = connection.execute(
            f"""
            SELECT detected_at, payload_json, {trace_column}
            FROM evidence_events
            WHERE edge_event_id = ?
            """,
            (event.edge_event_id,),
        ).fetchone()
        if existing is not None:
            _refuse_mismatched_replay(event, existing, required_decision_trace_id)
            stage_central_incident(
                connection,
                event,
                runtime_manifest_sha256=required_runtime_manifest_sha256,
                decision_trace_id=required_decision_trace_id,
            )
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
        if required_decision_trace_id is not None:
            connection.execute(
                "INSERT INTO evidence_event_trace_refs VALUES (?, ?)",
                (event.edge_event_id, required_decision_trace_id),
            )
        stage_central_incident(
            connection,
            event,
            runtime_manifest_sha256=required_runtime_manifest_sha256,
            decision_trace_id=required_decision_trace_id,
        )


def _refuse_mismatched_replay(
    event: StagedEvent,
    existing: sqlite3.Row,
    required_decision_trace_id: str | None,
) -> None:
    mismatches: list[Literal["detected_at", "payload_json", "decision_trace_id"]] = []
    if str(existing[0]) != event.detected_at:
        mismatches.append("detected_at")
    if str(existing[1]) != event.payload_json:
        mismatches.append("payload_json")
    if existing[2] != required_decision_trace_id:
        mismatches.append("decision_trace_id")
    if mismatches:
        raise StagedEventConflictError(event.edge_event_id, tuple(mismatches))


__all__ = ["stage_event"]
