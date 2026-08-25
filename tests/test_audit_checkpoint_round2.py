from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.features.audit.catalog import (
    AuditAction,
    AuditDetailError,
    empty_detail,
    parse_detail_json,
)
from backend.app.features.audit.store import (
    AuditEvent,
    AuditStore,
    AuditVerificationError,
    utc_now,
)


def _database_path() -> Path:
    from backend.app.features.audit import store

    return store.EDGE_DATABASE_PATH


def _event(action: AuditAction, target: str) -> AuditEvent:
    return AuditEvent(
        occurred_at=utc_now(), actor_id="admin", action=action,
        target_id=target, detail=empty_detail(action),
    )


def test_data_version_rejects_restored_schema_epoch_historical_bypass() -> None:
    # Given: a checkpoint observed through one verifier connection.
    audit = AuditStore(_database_path())
    first = audit.append(_event(AuditAction.CLIP_LIST, "first"))
    audit.append(_event(AuditAction.AUDIT_LIST, "anchor"))
    checkpoint = audit.verify()
    with sqlite3.connect(_database_path()) as writer:
        original_schema_version = writer.execute("PRAGMA schema_version").fetchone()[0]
        trigger_sql = writer.execute(
            "SELECT sql FROM sqlite_master WHERE name='audit_events_immutable_update'"
        ).fetchone()[0]
        writer.execute("DROP TRIGGER audit_events_immutable_update")
        writer.execute(
            "UPDATE audit_events SET actor_id='mutated' WHERE audit_id=?", (first.audit_id,)
        )
        writer.execute(trigger_sql)
        writer.execute(f"PRAGMA schema_version={original_schema_version}")

    # When/Then: a resettable schema epoch cannot hide the separate commit.
    observer = audit._verifier_connection
    assert observer is not None
    changed_version = observer.execute("PRAGMA data_version").fetchone()[0]
    observer.execute(f"PRAGMA data_version={checkpoint.data_version}")
    assert changed_version != checkpoint.data_version
    assert observer.execute("PRAGMA data_version").fetchone()[0] == changed_version
    with pytest.raises(AuditVerificationError, match="hash"):
        audit.verify(checkpoint)


def test_incremental_data_version_accounts_for_valid_tail_and_stays_bounded() -> None:
    # Given: an unchanged checkpoint observed by the retained verifier connection.
    audit = AuditStore(_database_path())
    audit.append_batch(tuple(_event(AuditAction.AUDIT_LIST, str(index)) for index in range(3)))
    checkpoint = audit.verify()
    observer = audit._verifier_connection
    assert observer is not None
    statements: list[str] = []
    observer.set_trace_callback(statements.append)

    # When: unchanged verification and then one legitimate audited commit are checked.
    unchanged = audit.verify(checkpoint)
    appended = audit.append(_event(AuditAction.AUDIT_DETAIL, "tail"))
    advanced = audit.verify(unchanged)

    # Then: the suffix accounts for the commit without OFFSET or an unbounded historical read.
    assert advanced.audit_id == appended.audit_id
    selects = [
        statement.upper()
        for statement in statements
        if "FROM AUDIT_EVENTS" in statement.upper()
    ]
    assert selects
    assert all("OFFSET" not in statement for statement in selects)
    assert any("LIMIT" in statement and "AUDIT_ID>" in statement for statement in selects)


def test_rolling_keyset_cycle_detects_older_non_anchor_corruption() -> None:
    # Given: enough history that one rolling page cannot reach row 1201.
    audit = AuditStore(_database_path())
    audit.append_batch(
        tuple(_event(AuditAction.AUDIT_LIST, str(index)) for index in range(1_502))
    )
    checkpoint = audit.verify()
    with sqlite3.connect(_database_path()) as writer:
        original_schema_version = writer.execute("PRAGMA schema_version").fetchone()[0]
        trigger_sql = writer.execute(
            "SELECT sql FROM sqlite_master WHERE name='audit_events_immutable_update'"
        ).fetchone()[0]
        writer.execute("DROP TRIGGER audit_events_immutable_update")
        writer.execute("UPDATE audit_events SET actor_id='mutated' WHERE audit_id=1201")
        writer.execute(trigger_sql)
        writer.execute(f"PRAGMA schema_version={original_schema_version}")
    audit.append(_event(AuditAction.AUDIT_DETAIL, "accounted-tail"))

    # When: a valid tail accounts for the data-version change, one page is rechecked.
    first_cycle = audit.verify(checkpoint)

    # Then: the next bounded keyset page reaches and rejects the older corruption.
    assert first_cycle.rolling_audit_id == 1_000
    with pytest.raises(AuditVerificationError, match="hash"):
        audit.verify(first_cycle)


def test_restart_checkpoint_always_full_verifies_on_new_observer() -> None:
    # Given: a checkpoint from a different retained observer identity.
    first_store = AuditStore(_database_path())
    first_store.append(_event(AuditAction.AUDIT_LIST, "first"))
    checkpoint = first_store.verify()
    second_store = AuditStore(_database_path())

    # When/Then: restart does not reuse connection-local data_version trust.
    restarted = second_store.verify(checkpoint)
    assert restarted.observer_id != checkpoint.observer_id
    assert restarted.audit_id == checkpoint.audit_id


def test_detail_version_is_exact_uncoerced_json_integer() -> None:
    # Given/When/Then: only JSON int 1 is accepted, never bool/float/string/null.
    assert parse_detail_json(AuditAction.CLIP_LIST, '{"version":1}').json == '{"version":1}'
    for encoded in (
        '{"version":true}', '{"version":false}', '{"version":1.0}',
        '{"version":1.00}', '{"version":"1"}', '{"version":null}',
        '{"version":2}',
    ):
        with pytest.raises(AuditDetailError, match="version"):
            parse_detail_json(AuditAction.CLIP_LIST, encoded)


def test_nested_detail_payload_does_not_coerce_version() -> None:
    encoded = '{"version":true,"safe":{"version":1}}'
    with pytest.raises(AuditDetailError):
        parse_detail_json(AuditAction.CLIP_LIST, encoded)
