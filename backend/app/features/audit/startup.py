"""Startup owner for bounded audit verification and readiness state."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from backend.app.features.audit.http import AuditReadiness
from backend.app.features.audit.store import AuditStore, AuditVerificationError


def configure_audit_readiness(app: FastAPI, database_path: Path) -> bool:
    """Verify schema-18 audit history and publish explicit process state."""
    store = AuditStore(database_path)
    app.state.audit_store = store
    try:
        app.state.audit_checkpoint = store.verify()
    except AuditVerificationError as error:
        app.state.audit_error = str(error)
        app.state.audit_readiness = AuditReadiness(
            healthy=False, failure_code="startup_verification"
        )
        return False
    app.state.audit_readiness = AuditReadiness()
    return True


__all__ = ["configure_audit_readiness"]
