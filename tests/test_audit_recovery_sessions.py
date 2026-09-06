from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.sessions import (
    append_with_recovery,
    close_session,
    start_session,
)
from backend.app.features.audit.store import AuditEvent, AuditStore, utc_now


def _database_path() -> Path:
    from backend.app.features.audit import store

    return store.EDGE_DATABASE_PATH


def _event(target_id: str) -> AuditEvent:
    return AuditEvent(
        occurred_at=utc_now(),
        actor_id="admin",
        action=AuditAction.AUDIT_LIST,
        target_id=target_id,
        detail=empty_detail(AuditAction.AUDIT_LIST),
    )


def test_concurrent_recovery_appends_one_durable_session_fence() -> None:
    # Given: one durable process session and concurrent callers released together.
    store = AuditStore(_database_path())
    session = start_session(store)
    barrier = Barrier(8)

    def recover(index: int) -> None:
        barrier.wait()
        append_with_recovery(store, _event(f"audit-{index}"), session, "SQLITE_FULL")

    # When: every caller observes the same degraded session.
    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(recover, range(8)))

    # Then: BEGIN IMMEDIATE and durable target identity permit one fence only.
    with sqlite3.connect(_database_path()) as connection:
        fences = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='audit.recovery-fence' AND target_id=?",
            (session.session_id,),
        ).fetchone()[0]
        operations = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='audit.list'"
        ).fetchone()[0]
    assert fences == 1
    assert operations == 8


def test_unclean_restart_is_fenced_once_and_clean_restart_adds_no_duplicate() -> None:
    # Given: a process session that stops before recovery or graceful close.
    store = AuditStore(_database_path())
    abandoned = start_session(store)

    # When: the next startup opens a session, then closes it cleanly, and starts again.
    recovered = AuditStore(_database_path())
    second = start_session(recovered)
    close_session(recovered, second)
    third_store = AuditStore(_database_path())
    third = start_session(third_store)

    # Then: the abandoned identity has one conservative fence and no restart duplicates it.
    with sqlite3.connect(_database_path()) as connection:
        abandoned_fences = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='audit.recovery-fence' AND target_id=?",
            (abandoned.session_id,),
        ).fetchone()[0]
        all_fences = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='audit.recovery-fence'"
        ).fetchone()[0]
    close_session(third_store, third)
    assert abandoned_fences == 1
    assert all_fences == 1


def test_repeated_recovery_for_same_session_never_duplicates_fence() -> None:
    # Given: one session whose first recovered operation writes its durable fence.
    store = AuditStore(_database_path())
    session = start_session(store)
    append_with_recovery(store, _event("first"), session, "SQLITE_FULL")

    # When: readiness/recovery is exercised repeatedly for the same interval identity.
    for index in range(20):
        append_with_recovery(store, _event(f"repeat-{index}"), session, "SQLITE_FULL")

    # Then: no per-call recovery backlog is created.
    with sqlite3.connect(_database_path()) as connection:
        fences = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='audit.recovery-fence' AND target_id=?",
            (session.session_id,),
        ).fetchone()[0]
    assert fences == 1
