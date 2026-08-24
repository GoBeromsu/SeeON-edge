"""Startup owner for bounded audit verification and readiness state."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from fastapi import FastAPI

from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.features.audit.http import AuditReadiness
from backend.app.features.audit.sessions import close_session, start_session
from backend.app.features.audit.store import AuditStore, AuditVerificationError

_LOGGER = logging.getLogger(__name__)


def close_audit_session(app: FastAPI) -> bool:
    """Close a healthy session; failure deliberately leaves an unclean restart marker."""
    store = getattr(app.state, "audit_store", None)
    readiness = getattr(app.state, "audit_readiness", None)
    if not isinstance(store, AuditStore):
        return False
    closed = False
    if (
        isinstance(readiness, AuditReadiness)
        and readiness.healthy
        and readiness.session is not None
    ):
        try:
            close_session(store, readiness.session)
            closed = True
        except (AuditVerificationError, OSError, sqlite3.Error, EdgeDatabaseError):
            _LOGGER.warning("audit session close failed; next startup will fence it")
    store.close_verifier()
    return closed


def configure_audit_readiness(app: FastAPI, database_path: Path) -> bool:
    """Verify schema-18 audit history and publish explicit process state."""
    store = AuditStore(database_path)
    app.state.audit_store = store
    try:
        app.state.audit_checkpoint = store.verify()
        session = start_session(store)
    except (AuditVerificationError, OSError, sqlite3.Error, EdgeDatabaseError) as error:
        app.state.audit_error = str(error)
        app.state.audit_readiness = AuditReadiness(
            healthy=False, failure_code="startup_verification"
        )
        return False
    app.state.audit_readiness = AuditReadiness(session=session)
    return True


__all__ = ["close_audit_session", "configure_audit_readiness"]
