"""HTTP boundary policy for fail-closed governed operations."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from fastapi import FastAPI, Request, Response, status

from backend.app.edge_db import DatabaseDriverError
from backend.app.edge_db.postgres import PostgresError
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import AuditMutation, PostgresAuditRuntime
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.shared.dashboard_credentials import DashboardCredentialsStoreError


class AuditUnavailableError(RuntimeError):
    """A governed response cannot start because its audit commit failed."""


_READY = {"ready": True, "status": "ready"}
_DEGRADED = {"ready": False, "status": "degraded", "reason": "audit unavailable"}
_READINESS_LOCK = Lock()


def publish_audit_readiness(
    app: FastAPI, runtime: PostgresAuditRuntime, *, boot: bool = False
) -> bool:
    """Publish runtime-derived readiness; outside boot, only heal audit-owned degradation."""
    with _READINESS_LOCK:
        eligible = runtime.snapshot().eligible_to_attempt
        if not eligible:
            app.state.readiness = dict(_DEGRADED)
        elif boot or getattr(app.state, "readiness", None) == _DEGRADED:
            app.state.readiness = dict(_READY)
        return eligible


def audit_runtime(request: Request) -> PostgresAuditRuntime:
    """Require the injected native runtime; never start a request-owned session."""
    runtime = getattr(request.app.state, "audit_runtime", None)
    if not isinstance(runtime, PostgresAuditRuntime):
        raise AuditUnavailableError("native audit runtime is not injected")
    return runtime


def mutation_audit(request: Request, event_factory: Callable[[], AuditEvent]) -> AuditMutation:
    return AuditMutation(audit_runtime(request), event_factory)


def append_governed(
    request: Request, *, actor_id: str, action: AuditAction, target_id: str
) -> None:
    """Commit one governed success before FastAPI starts its response."""
    runtime = audit_runtime(request)
    event = AuditEvent(
        occurred_at=utc_now(),
        actor_id=actor_id,
        action=action,
        target_id=target_id,
        detail=empty_detail(action),
    )
    runtime.append_owned(event)
    publish_audit_readiness(request.app, runtime)


def audit_unavailable_handler(request: Request, error: Exception) -> Response:
    runtime = getattr(request.app.state, "audit_runtime", None)
    if isinstance(
        error, (PostgresError, DatabaseDriverError, DashboardCredentialsStoreError)
    ) and isinstance(runtime, PostgresAuditRuntime):
        runtime.record_failure(error)
    if isinstance(runtime, PostgresAuditRuntime):
        publish_audit_readiness(request.app, runtime)
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


__all__ = [
    "AuditUnavailableError",
    "append_governed",
    "audit_runtime",
    "audit_unavailable_handler",
    "mutation_audit",
    "publish_audit_readiness",
]
