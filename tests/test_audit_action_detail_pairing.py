from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.store import (
    AuditEvent,
    AuditStore,
    AuditVerificationError,
    utc_now,
)


def test_action_detail_pairing_rejects_a_different_valid_action(tmp_path: Path) -> None:
    path = tmp_path / "wrong-detail.sqlite3"
    migrate_database(path)
    event = AuditEvent(
        utc_now(),
        "test",
        AuditAction.CONNECTION_SYNC,
        "camera-roster",
        empty_detail(AuditAction.TOPOLOGY_CONFIRM),
    )

    with pytest.raises(
        AuditVerificationError, match="audit action/detail variants do not match"
    ):
        AuditStore(path).append(event)

    with sqlite3.connect(path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='connection.sync'"
        ).fetchone()[0]
    assert count == 0
