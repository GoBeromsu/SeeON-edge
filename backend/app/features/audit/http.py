"""HTTP boundary policy for fail-closed governed operations."""

from __future__ import annotations

import sqlite3
from threading import Lock
from typing import NoReturn

from fastapi import Request, Response, status

from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.features.audit.catalog import AuditAction, parse_detail
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

    __slots__ = ("_lock", "failure_code", "healthy")

    def __init__(self, healthy: bool = True, failure_code: str | None = None) -> None:
        self.healthy = healthy
        self.failure_code = failure_code
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


def _recovery_event(failure_code: str) -> AuditEvent:
    return AuditEvent(
        occurred_at=utc_now(), actor_id="audit-readiness",
        action=AuditAction.RECOVERY_FENCE, target_id="degraded-interval",
        detail=parse_detail(
            AuditAction.RECOVERY_FENCE,
            {"failure_code": failure_code, "ended_at": utc_now()},
        ),
    )


def append_governed(
    request: Request, *, actor_id: str, action: AuditAction, target_id: str
) -> None:
    """Commit one governed success before FastAPI starts its response."""
    readiness = audit_readiness(request)
    event = AuditEvent(
        occurred_at=utc_now(), actor_id=actor_id, action=action,
        target_id=target_id, detail=parse_detail(action, {}),
    )
    failure_code = readiness.current_failure()
    events = (event,) if failure_code is None else (_recovery_event(failure_code), event)
    try:
        store = audit_store(request)
        _verify_incremental(request, store)
        store.append_batch(events)
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
        if failure_code is not None:
            store.append(_recovery_event(failure_code), connection=connection)
        store.append(event, connection=connection)
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
