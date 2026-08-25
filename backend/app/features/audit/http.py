"""HTTP boundary policy for fail-closed governed operations."""

from __future__ import annotations

import sqlite3
from threading import Lock
from typing import NoReturn

from fastapi import Request, Response, status

from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.sessions import (
    AuditSession,
    append_with_recovery,
    start_session,
)
from backend.app.features.audit.store import (
    AuditEvent,
    AuditStore,
    AuditVerificationError,
    VerificationCheckpoint,
    utc_now,
)


class AuditUnavailableError(RuntimeError):
    """A governed response cannot start because its audit commit failed."""


class AuditReadiness:
    """Mutable process readiness and one bounded degraded-interval fence."""

    __slots__ = ("_lock", "failure_code", "healthy", "session")

    def __init__(
        self,
        healthy: bool = True,
        failure_code: str | None = None,
        session: AuditSession | None = None,
    ) -> None:
        self.healthy = healthy
        self.failure_code = failure_code
        self.session = session
        self._lock = Lock()

    def degraded(self, failure_code: str) -> None:
        with self._lock:
            self.healthy = False
            self.failure_code = failure_code[:64]

    def current_failure(self) -> str | None:
        with self._lock:
            return self.failure_code

    def recovered(self) -> None:
        with self._lock:
            self.healthy = True
            self.failure_code = None

    def ensure_session(
        self, store: AuditStore, connection: sqlite3.Connection | None = None
    ) -> AuditSession:
        with self._lock:
            if self.session is None:
                self.session = (
                    start_session(store)
                    if connection is None
                    else start_session(store, connection)
                )
            return self.session


def audit_store(request: Request) -> AuditStore:
    existing = getattr(request.app.state, "audit_store", None)
    if isinstance(existing, AuditStore):
        return existing
    store = AuditStore()
    request.app.state.audit_store = store
    return store


def audit_readiness(request: Request) -> AuditReadiness:
    existing = getattr(request.app.state, "audit_readiness", None)
    if isinstance(existing, AuditReadiness):
        return existing
    readiness = AuditReadiness()
    request.app.state.audit_readiness = readiness
    return readiness


def _verify_incremental(request: Request, store: AuditStore) -> None:
    candidate = getattr(request.app.state, "audit_checkpoint", None)
    checkpoint = candidate if isinstance(candidate, VerificationCheckpoint) else None
    try:
        request.app.state.audit_checkpoint = store.verify(checkpoint)
    except AuditVerificationError as error:
        audit_readiness(request).degraded("incremental_verification")
        request.app.state.readiness = {
            "ready": False, "status": "degraded", "reason": "audit unavailable"
        }
        raise AuditUnavailableError from error


def append_governed(
    request: Request, *, actor_id: str, action: AuditAction, target_id: str
) -> None:
    """Commit one governed success before FastAPI starts its response."""
    readiness = audit_readiness(request)
    event = AuditEvent(
        occurred_at=utc_now(), actor_id=actor_id, action=action,
        target_id=target_id, detail=empty_detail(action),
    )
    failure_code = readiness.current_failure()
    try:
        store = audit_store(request)
        _verify_incremental(request, store)
        session = readiness.ensure_session(store)
        if failure_code is None:
            store.append(event)
        else:
            append_with_recovery(store, event, session, failure_code)
        readiness.recovered()
        request.app.state.readiness = {"ready": True, "status": "ready"}
    except (OSError, sqlite3.Error, EdgeDatabaseError) as error:
        refuse_unavailable(request, error)


def append_transactional(
    request: Request, connection: sqlite3.Connection, event: AuditEvent
) -> None:
    """Append on a governed mutation's existing transaction."""
    readiness = audit_readiness(request)
    failure_code = readiness.current_failure()
    try:
        store = audit_store(request)
        _verify_incremental(request, store)
        session = readiness.ensure_session(store, connection)
        if failure_code is None:
            store.append(event, connection=connection)
        else:
            append_with_recovery(
                store, event, session, failure_code, connection
            )
        readiness.recovered()
        request.app.state.readiness = {"ready": True, "status": "ready"}
    except (OSError, sqlite3.Error, EdgeDatabaseError) as error:
        refuse_unavailable(request, error)


def refuse_unavailable(
    request: Request, error: OSError | sqlite3.Error | EdgeDatabaseError
) -> NoReturn:
    audit_readiness(request).degraded(error.__class__.__name__)
    request.app.state.readiness = {
        "ready": False, "status": "degraded", "reason": "audit unavailable"
    }
    raise AuditUnavailableError from error


def audit_unavailable_handler(
    request: Request, error: AuditUnavailableError
) -> Response:
    del request, error
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


def note_fail_open_degradation(request: Request, failure_code: str) -> None:
    """Bound fail-open degradation to one in-memory interval summary."""
    readiness = audit_readiness(request)
    readiness.degraded(failure_code)
    request.app.state.readiness = {
        "ready": False, "status": "degraded", "reason": "audit unavailable"
    }


__all__ = [
    "AuditReadiness", "AuditUnavailableError", "append_governed",
    "append_transactional", "audit_readiness", "audit_store",
    "audit_unavailable_handler", "note_fail_open_degradation", "refuse_unavailable",
]
