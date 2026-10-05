"""Refresh the audit verification and deferred process session."""

from __future__ import annotations

import logging

from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.postgres_runtime import (
    AuditRuntimeUnavailable,
    PostgresAuditRuntime,
)

_LOGGER = logging.getLogger(__name__)


def verify_audit_runtime(runtime: PostgresAuditRuntime) -> None:
    """Refresh verification without owning HTTP readiness publication."""
    try:
        runtime.verify_once()
        if not runtime.snapshot().session_established:
            runtime.start_session_once()
    except (AuditRuntimeUnavailable, CommitOutcomeUnknown):
        _LOGGER.warning("audit verification failed; audit is degraded")



__all__ = ["verify_audit_runtime"]
