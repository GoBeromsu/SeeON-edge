"""Startup owner for the PostgreSQL audit runtime and its readiness publication."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI

from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.http import publish_audit_readiness
from backend.app.features.audit.postgres_runtime import (
    AuditRuntimeUnavailable,
    PostgresAuditRuntime,
)
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.postgres_root import PostgresRoot

_LOGGER = logging.getLogger(__name__)

# Admission needs verification younger than this; the tick keeps it fresh.
_MAXIMUM_SNAPSHOT_AGE_SEC = 30.0
_VERIFY_INTERVAL_SEC = 10.0
_VERIFY_JOIN_TIMEOUT_SEC = 5.0


@dataclass(slots=True)
class _OwnedAudit:
    runtime: PostgresAuditRuntime
    stop: threading.Event
    thread: threading.Thread | None = None


def _owned(app: FastAPI) -> _OwnedAudit | None:
    owned = getattr(app.state, "audit_owner", None)
    return owned if isinstance(owned, _OwnedAudit) else None


def _postgres_root(app: FastAPI) -> PostgresRoot:
    root = getattr(app.state, "postgres_root", None)
    if isinstance(root, PostgresRoot):
        return root
    raise RuntimeError("audit runtime requires the PostgreSQL root")


def _configured_runtime(app: FastAPI) -> PostgresAuditRuntime:
    runtime = getattr(app.state, "audit_runtime", None)
    if isinstance(runtime, PostgresAuditRuntime):
        return runtime
    raise RuntimeError("audit runtime is not configured")


def configure_audit_readiness(app: FastAPI, *, clock: Callable[[], float] = time.monotonic) -> bool:
    """Verify audit history and open this process session on the PostgreSQL root.

    An injected runtime wins and stays owned by its injector. Failure leaves
    the runtime degraded rather than refusing boot; missing wiring refuses.
    """
    injected = getattr(app.state, "audit_runtime", None)
    if isinstance(injected, PostgresAuditRuntime) and _owned(app) is None:
        return injected.snapshot().eligible_to_attempt
    root = _postgres_root(app)
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(root.database, root.authority),
        maximum_snapshot_age_sec=_MAXIMUM_SNAPSHOT_AGE_SEC,
        clock=clock,
    )
    try:
        runtime.verify_once()
        runtime.start_session_once()
    except (AuditRuntimeUnavailable, CommitOutcomeUnknown):
        _LOGGER.warning("audit startup verification failed; audit is degraded")
    app.state.audit_runtime = runtime
    app.state.audit_owner = _OwnedAudit(runtime, threading.Event())
    return runtime.snapshot().eligible_to_attempt


def verify_audit_runtime(app: FastAPI, runtime: PostgresAuditRuntime) -> bool:
    """Refresh verification, finish a deferred session start, then publish readiness."""
    try:
        runtime.verify_once()
        if not runtime.snapshot().session_established:
            runtime.start_session_once()
    except (AuditRuntimeUnavailable, CommitOutcomeUnknown):
        _LOGGER.warning("audit verification failed; audit is degraded")
    return publish_audit_readiness(app, runtime)


def start_audit_verification(
    app: FastAPI, *, verify_interval_sec: float = _VERIFY_INTERVAL_SEC
) -> None:
    """Publish boot readiness; an owned runtime also gets its periodic verifier."""
    runtime = _configured_runtime(app)
    publish_audit_readiness(app, runtime, boot=True)
    owned = _owned(app)
    if owned is None or owned.runtime is not runtime or owned.thread is not None:
        return

    def verify_until_stopped() -> None:
        while not owned.stop.wait(verify_interval_sec):
            verify_audit_runtime(app, runtime)

    owned.thread = threading.Thread(target=verify_until_stopped, name="audit-verifier", daemon=True)
    owned.thread.start()


def close_audit_session(app: FastAPI) -> bool:
    """Close a healthy owned session; degradation deliberately leaves an unclean marker.

    An injected runtime stays untouched for its injector to stop and close.
    """
    owned = _owned(app)
    if owned is None:
        return False
    owned.stop.set()
    if owned.thread is not None:
        owned.thread.join(timeout=_VERIFY_JOIN_TIMEOUT_SEC)
    runtime = owned.runtime
    healthy = runtime.snapshot().ready
    runtime.stop()
    closed = False
    if healthy:
        try:
            closed = runtime.close_session_once()
        except (AuditRuntimeUnavailable, CommitOutcomeUnknown):
            _LOGGER.warning("audit session close failed; next startup will fence it")
    if getattr(app.state, "audit_runtime", None) is runtime:
        delattr(app.state, "audit_runtime")
    delattr(app.state, "audit_owner")
    return closed


__all__ = [
    "close_audit_session",
    "configure_audit_readiness",
    "start_audit_verification",
    "verify_audit_runtime",
]
