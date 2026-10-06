"""Keyset reads and row decoding for immutable audit history."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.audit.catalog import AuditAction

SqlValue = str | int | float | bytes | None


@dataclass(frozen=True, slots=True)
class AuditHistoryEvent:
    audit_id: int
    occurred_at: str
    recorded_at: str
    actor_type: str
    actor_id: str
    action: AuditAction
    target_type: str
    target_id: str
    outcome: str
    detail_json: str | None
    previous_hash: str
    record_hash: str


class AuditStoredIdentityError(ValueError):
    """A verified audit row contains a non-integer identity."""


_SELECT = (
    "SELECT audit_id,occurred_at,recorded_at,actor_type,actor_id,action,target_type,"
    "target_id,outcome,detail_json,previous_hash,record_hash FROM audit_events"
)


def _audit_id(value: SqlValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AuditStoredIdentityError
    return value


def _event(row: tuple[SqlValue, ...]) -> AuditHistoryEvent:
    return AuditHistoryEvent(
        audit_id=_audit_id(row[0]),
        occurred_at=str(row[1]),
        recorded_at=str(row[2]),
        actor_type=str(row[3]),
        actor_id=str(row[4]),
        action=AuditAction(str(row[5])),
        target_type=str(row[6]),
        target_id=str(row[7]),
        outcome=str(row[8]),
        detail_json=None if row[9] is None else str(row[9]),
        previous_hash=str(row[10]),
        record_hash=str(row[11]),
    )


def list_history(
    database: PostgresDatabase, limit: int, before_id: int | None
) -> tuple[AuditHistoryEvent, ...]:
    if before_id is None:
        query = _SELECT + " ORDER BY audit_id DESC LIMIT %s"
        parameters = (limit + 1,)
    else:
        query = _SELECT + " WHERE audit_id < %s ORDER BY audit_id DESC LIMIT %s"
        parameters = (before_id, limit + 1)
    return database.read(
        lambda connection: tuple(_event(row) for row in connection.execute(query, parameters))
    )


def get_history(database: PostgresDatabase, audit_id: int) -> AuditHistoryEvent | None:
    def read_event(connection: psycopg.Connection) -> AuditHistoryEvent | None:
        row = connection.execute(_SELECT + " WHERE audit_id = %s", (audit_id,)).fetchone()
        return None if row is None else _event(row)

    return database.read(read_event)


__all__ = ["AuditHistoryEvent", "get_history", "list_history"]
