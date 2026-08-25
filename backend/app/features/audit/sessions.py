"""Durable audit-only process sessions and deduplicated recovery fences."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    recovery_detail,
    session_detail,
)
from backend.app.features.audit.verification import AuditVerificationError

if TYPE_CHECKING:
    from backend.app.features.audit.store import AuditEvent, AuditRecord, AuditStore


@dataclass(frozen=True, slots=True)
class AuditSession:
    session_id: str


def start_session(
    store: AuditStore, connection: sqlite3.Connection | None = None
) -> AuditSession:
    """Fence one prior unclosed session, then durably open this process session."""
    session = AuditSession(uuid4().hex)
    if connection is not None:
        if not connection.in_transaction:
            raise AuditVerificationError("session start requires an active transaction")
        _start_session(store, connection, session)
        return session
    with closing(open_runtime_database(store.path, actor=RuntimeActor.API)) as owned:
        with write_transaction(owned):
            _start_session(store, owned, session)
    return session


def _start_session(
    store: AuditStore, connection: sqlite3.Connection, session: AuditSession
) -> None:
    previous = connection.execute(
        "SELECT target_id FROM audit_events "
        "WHERE action=? ORDER BY audit_id DESC LIMIT 1",
        (AuditAction.AUDIT_SESSION_START.value,),
    ).fetchone()
    if previous is not None:
        previous_id = str(previous[0])
        closed = connection.execute(
            "SELECT 1 FROM audit_events WHERE action=? AND target_id=? LIMIT 1",
            (AuditAction.AUDIT_SESSION_CLOSE.value, previous_id),
        ).fetchone()
        if closed is None:
            _fence_if_needed(store, connection, previous_id, "unclean_restart")
    store.append(_session_event(AuditAction.AUDIT_SESSION_START, session), connection=connection)


def close_session(store: AuditStore, session: AuditSession) -> None:
    """Durably mark a healthy process session as gracefully closed once."""
    with closing(open_runtime_database(store.path, actor=RuntimeActor.API)) as connection:
        with write_transaction(connection):
            exists = connection.execute(
                "SELECT 1 FROM audit_events WHERE action=? AND target_id=? LIMIT 1",
                (AuditAction.AUDIT_SESSION_CLOSE.value, session.session_id),
            ).fetchone()
            if exists is None:
                store.append(
                    _session_event(AuditAction.AUDIT_SESSION_CLOSE, session),
                    connection=connection,
                )


def append_with_recovery(
    store: AuditStore,
    event: AuditEvent,
    session: AuditSession,
    failure_code: str,
    connection: sqlite3.Connection | None = None,
) -> AuditRecord:
    if connection is not None:
        if not connection.in_transaction:
            raise AuditVerificationError("recovery append requires an active transaction")
        _fence_if_needed(store, connection, session.session_id, failure_code)
        return store.append(event, connection=connection)
    with closing(open_runtime_database(store.path, actor=RuntimeActor.API)) as owned:
        with write_transaction(owned):
            _fence_if_needed(store, owned, session.session_id, failure_code)
            return store.append(event, connection=owned)


def _fence_if_needed(
    store: AuditStore,
    connection: sqlite3.Connection,
    session_id: str,
    failure_code: str,
) -> None:
    exists = connection.execute(
        "SELECT 1 FROM audit_events WHERE action=? AND target_id=? LIMIT 1",
        (AuditAction.RECOVERY_FENCE.value, session_id),
    ).fetchone()
    if exists is not None:
        return
    from backend.app.features.audit.store import AuditEvent, utc_now

    store.append(
        AuditEvent(
            occurred_at=utc_now(), actor_id="audit-readiness",
            action=AuditAction.RECOVERY_FENCE, target_id=session_id,
            detail=recovery_detail(failure_code, utc_now()),
            actor_type=AuditActorType.SYSTEM,
            auth_mechanism=AuditAuthMechanism.INTERNAL,
        ),
        connection=connection,
    )


def _session_event(action: AuditAction, session: AuditSession) -> AuditEvent:
    from backend.app.features.audit.store import AuditEvent, utc_now

    return AuditEvent(
        occurred_at=utc_now(), actor_id="audit-readiness", action=action,
        target_id=session.session_id, detail=session_detail(action),
        actor_type=AuditActorType.SYSTEM, auth_mechanism=AuditAuthMechanism.INTERNAL,
    )


__all__ = ["AuditSession", "append_with_recovery", "close_session", "start_session"]
