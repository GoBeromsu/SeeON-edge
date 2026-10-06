"""Native PostgreSQL audit append and snapshot verification through the database owner."""

from __future__ import annotations

import json
from collections.abc import Sequence

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.functions import audit_record_hash
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.audit.postgres_verification import (
    PostgresAuditCheckpoint,
    _verify_snapshot,
)
from backend.app.features.audit.store import AuditEvent, AuditRecord, _payload, utc_now
from backend.app.features.audit.verification import GENESIS_HASH, AuditVerificationError


def _lock_audit_chain(connection: psycopg.Connection) -> None:
    """Serialize chain reads within an already active transaction."""
    # Same transaction lock as the SQL insert trigger.
    connection.execute("SELECT pg_advisory_xact_lock('audit_events'::regclass::oid::bigint)")


def append_postgres_audit(connection: psycopg.Connection, event: AuditEvent) -> AuditRecord:
    if connection.info.transaction_status is not TransactionStatus.INTRANS:
        raise AuditVerificationError("audit append requires an owned transaction")
    if event.detail.action is not event.action:
        raise AuditVerificationError("audit action/detail variants do not match")
    _lock_audit_chain(connection)
    row = connection.execute(
        "SELECT record_hash FROM audit_events ORDER BY audit_id DESC LIMIT 1"
    ).fetchone()
    previous = GENESIS_HASH if row is None else str(row[0])
    recorded = utc_now()
    values = _payload(event, recorded, previous)
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record_hash = audit_record_hash(previous, payload)
    values["record_hash"] = record_hash
    columns = tuple(values)
    inserted = connection.execute(
        sql.SQL("INSERT INTO audit_events ({}) VALUES ({}) RETURNING audit_id").format(
            sql.SQL(",").join(map(sql.Identifier, columns)),
            sql.SQL(",").join(sql.Placeholder() for _ in columns),
        ),
        tuple(values[name] for name in columns),
    ).fetchone()
    if inserted is None:
        raise AuditVerificationError("audit insert did not return an identity")
    return AuditRecord(
        audit_id=inserted[0],
        occurred_at=event.occurred_at,
        recorded_at=recorded,
        actor_id=event.actor_id,
        action=event.action,
        target_type=values["target_type"],
        target_id=event.target_id,
        detail=event.detail,
        previous_hash=previous,
        record_hash=record_hash,
    )


class PostgresAuditStore:
    """Audit component borrowing the lifespan database and writer authority.

    Verification is stateless; readiness and checkpoint publication belong to
    the caller, not this component.
    """

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority

    def append(
        self, event: AuditEvent, *, connection: psycopg.Connection | None = None
    ) -> AuditRecord:
        """Return after owned COMMIT, or a tentative record on the caller's transaction.

        A caller-owned result is not durable until that caller commits. No
        transaction is committed, retried, or nested on the caller-owned path.
        """

        def append(owned: psycopg.Connection) -> AuditRecord:
            require_authority(owned, self.authority)
            return append_postgres_audit(owned, event)

        if connection is not None:
            if connection.info.transaction_status is not TransactionStatus.INTRANS:
                raise AuditVerificationError(
                    "caller-owned audit append requires an active transaction"
                )
            return append(connection)
        return self.database.transact(append)

    def append_batch(self, events: Sequence[AuditEvent]) -> tuple[AuditRecord, ...]:
        """Commit the complete group atomically; release records only after COMMIT."""

        def append(connection: psycopg.Connection) -> tuple[AuditRecord, ...]:
            require_authority(connection, self.authority)
            return tuple(append_postgres_audit(connection, event) for event in events)

        return self.database.transact(append)

    def verify(self, checkpoint: PostgresAuditCheckpoint | None = None) -> PostgresAuditCheckpoint:
        """Release a full-snapshot continuity anchor only after COMMIT/pool exit."""
        try:
            return self.database.read_snapshot(
                lambda connection: _verify_snapshot(connection, self.database.schema, checkpoint)
            )
        except (AuditVerificationError, psycopg.Error, OSError):
            # Include known COMMIT rejection; already-safe typed owner failures,
            # especially CommitOutcomeUnknown, propagate without replay.
            raise AuditVerificationError("audit verification failed") from None
