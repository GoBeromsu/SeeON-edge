"""Native audit sessions and recovery fences within authority-owned transactions."""

from __future__ import annotations

from uuid import uuid4

import psycopg
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import require_authority
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    recovery_detail,
    session_detail,
)
from backend.app.features.audit.postgres_store import (
    PostgresAuditStore,
    _lock_audit_chain,
    append_postgres_audit,
)
from backend.app.features.audit.sessions import AuditSession
from backend.app.features.audit.store import AuditEvent, AuditRecord, utc_now
from backend.app.features.audit.verification import AuditVerificationError


def start_session(
    store: PostgresAuditStore, connection: psycopg.Connection | None = None
) -> AuditSession:
    """Fence the latest unclosed start, then open a random opaque process session.

    With a caller-owned transaction, the returned session is tentative until
    the caller commits; otherwise it is released only after a known COMMIT.
    """
    session = AuditSession(uuid4().hex)

    def start(owned: psycopg.Connection) -> AuditSession:
        require_authority(owned, store.authority)
        _lock_audit_chain(owned)
        previous = owned.execute(
            "SELECT target_id FROM audit_events WHERE action=%s ORDER BY audit_id DESC LIMIT 1",
            (AuditAction.AUDIT_SESSION_START.value,),
        ).fetchone()
        if previous is not None:
            previous_id = str(previous[0])
            closed = owned.execute(
                "SELECT 1 FROM audit_events WHERE action=%s AND target_id=%s LIMIT 1",
                (AuditAction.AUDIT_SESSION_CLOSE.value, previous_id),
            ).fetchone()
            if closed is None:
                _fence_if_needed(owned, previous_id, "unclean_restart")
        append_postgres_audit(owned, _session_event(AuditAction.AUDIT_SESSION_START, session))
        return session

    if connection is not None:
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise AuditVerificationError("session start requires an active transaction")
        return start(connection)
    return store.database.transact(start)


def close_session(store: PostgresAuditStore, session: AuditSession) -> None:
    """Durably close this session once, serialized before the existence check."""

    def close(connection: psycopg.Connection) -> None:
        require_authority(connection, store.authority)
        _lock_audit_chain(connection)
        exists = connection.execute(
            "SELECT 1 FROM audit_events WHERE action=%s AND target_id=%s LIMIT 1",
            (AuditAction.AUDIT_SESSION_CLOSE.value, session.session_id),
        ).fetchone()
        if exists is None:
            append_postgres_audit(
                connection, _session_event(AuditAction.AUDIT_SESSION_CLOSE, session)
            )

    store.database.transact(close)


def append_with_recovery(
    store: PostgresAuditStore,
    event: AuditEvent,
    session: AuditSession,
    failure_code: str,
    connection: psycopg.Connection | None = None,
) -> AuditRecord:
    """Append one session fence and the protected event atomically, in that order.

    The trusted caller supplies a stable, non-sensitive failure code. A record
    returned on a caller-owned transaction is tentative until the caller commits.
    """

    def append(owned: psycopg.Connection) -> AuditRecord:
        require_authority(owned, store.authority)
        _lock_audit_chain(owned)
        _fence_if_needed(owned, session.session_id, failure_code)
        return append_postgres_audit(owned, event)

    if connection is not None:
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise AuditVerificationError("recovery append requires an active transaction")
        return append(connection)
    return store.database.transact(append)


def _fence_if_needed(connection: psycopg.Connection, session_id: str, failure_code: str) -> None:
    exists = connection.execute(
        "SELECT 1 FROM audit_events WHERE action=%s AND target_id=%s LIMIT 1",
        (AuditAction.RECOVERY_FENCE.value, session_id),
    ).fetchone()
    if exists is not None:
        return
    append_postgres_audit(
        connection,
        AuditEvent(
            occurred_at=utc_now(),
            actor_id="audit-readiness",
            action=AuditAction.RECOVERY_FENCE,
            target_id=session_id,
            detail=recovery_detail(failure_code, utc_now()),
            actor_type=AuditActorType.SYSTEM,
            auth_mechanism=AuditAuthMechanism.INTERNAL,
        ),
    )


def _session_event(action: AuditAction, session: AuditSession) -> AuditEvent:
    return AuditEvent(
        occurred_at=utc_now(),
        actor_id="audit-readiness",
        action=action,
        target_id=session.session_id,
        detail=session_detail(action),
        actor_type=AuditActorType.SYSTEM,
        auth_mechanism=AuditAuthMechanism.INTERNAL,
    )


__all__ = ["append_with_recovery", "close_session", "start_session"]
