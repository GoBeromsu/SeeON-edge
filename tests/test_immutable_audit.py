from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from fastapi import FastAPI

from backend.app.edge_db import RuntimeActor, open_runtime_database
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditDetailError,
    JsonValue,
    empty_detail,
    parse_detail_json,
    recovery_detail,
)
from backend.app.features.audit.startup import configure_audit_readiness
from backend.app.features.audit.store import (
    GENESIS_HASH,
    AuditEvent,
    AuditStore,
    AuditVerificationError,
)


def _database_path() -> Path:
    from backend.app.features.audit import store

    return store.EDGE_DATABASE_PATH


def _event(action: AuditAction = AuditAction.CLIP_LIST) -> AuditEvent:
    return AuditEvent(
        occurred_at="2026-08-24T00:00:00.000Z",
        actor_id="admin",
        action=action,
        target_id="clips",
        detail=empty_detail(action),
    )


def test_hash_chain_roundtrips_when_appending_registered_events() -> None:
    # Given: a fresh real schema-18 database
    audit = AuditStore(_database_path())

    # When: two governed operations append
    first = audit.append(_event())
    second = audit.append(_event(AuditAction.AUDIT_LIST))

    # Then: genesis, linkage, and a full verification agree
    assert first.previous_hash == GENESIS_HASH
    assert second.previous_hash == first.record_hash
    assert audit.verify().audit_id == second.audit_id


def test_immutable_triggers_refuse_update_and_delete() -> None:
    # Given: one committed audit event
    audit = AuditStore(_database_path())
    row = audit.append(_event())
    connection = open_runtime_database(_database_path(), actor=RuntimeActor.API)

    # When/Then: neither mutation form can alter history
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            "UPDATE audit_events SET action='mutant' WHERE audit_id=?", (row.audit_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute("DELETE FROM audit_events WHERE audit_id=?", (row.audit_id,))
    connection.close()


@pytest.mark.parametrize(
    ("statement", "value"),
    [
        ("UPDATE audit_events SET action=? WHERE audit_id=?", "mutant"),
        ("UPDATE audit_events SET detail_json=? WHERE audit_id=?", '{"safe":"value"}'),
        ("UPDATE audit_events SET previous_hash=? WHERE audit_id=?", "1" * 64),
        ("UPDATE audit_events SET record_hash=? WHERE audit_id=?", "f" * 64),
    ],
)
def test_verification_detects_corrupted_chain(statement: str, value: str) -> None:
    # Given: a valid event whose immutable trigger is removed by a migration connection
    audit = AuditStore(_database_path())
    row = audit.append(_event())
    connection = sqlite3.connect(_database_path())
    connection.execute("DROP TRIGGER audit_events_immutable_update")
    connection.execute(statement, (value, row.audit_id))
    connection.commit()
    connection.close()

    # When/Then: startup verification refuses the changed catalog value
    with pytest.raises(AuditVerificationError):
        audit.verify()


def test_restart_verification_marks_corrupted_schema18_audit_unready() -> None:
    # Given: a real schema-18 database whose immutable trigger contract was corrupted.
    connection = sqlite3.connect(_database_path())
    connection.execute("DROP TRIGGER audit_events_immutable_delete")
    connection.commit()
    connection.close()
    app = FastAPI()

    # When: the audit startup owner verifies the restarted database.
    healthy = configure_audit_readiness(app, _database_path())

    # Then: corruption is explicit degraded truth, never an empty healthy history.
    assert healthy is False
    assert app.state.audit_readiness.healthy is False
    assert "contract is invalid" in app.state.audit_error


def test_concurrent_appends_serialize_one_unbroken_chain() -> None:
    # Given: two writers released at the same deterministic barrier.
    audit = AuditStore(_database_path())
    barrier = Barrier(2)

    def append(action: AuditAction) -> int:
        barrier.wait()
        return audit.append(_event(action)).audit_id

    # When: SQLite serializes both BEGIN IMMEDIATE transactions.
    with ThreadPoolExecutor(max_workers=2) as executor:
        ids = tuple(
            executor.map(append, (AuditAction.CLIP_LIST, AuditAction.AUDIT_LIST))
        )

    # Then: both commits survive and incremental verification reaches the second row.
    assert set(ids) == {1, 2}
    assert audit.verify().audit_id == 2


def test_incremental_verification_resumes_after_checkpoint() -> None:
    # Given: a verified prefix and one later append.
    audit = AuditStore(_database_path())
    audit.append(_event())
    checkpoint = audit.verify()
    later = audit.append(_event(AuditAction.AUDIT_LIST))

    # When: verification resumes from the explicit checkpoint.
    resumed = audit.verify(checkpoint)

    # Then: only the new suffix advances the checkpoint.
    assert resumed.audit_id == later.audit_id
    assert resumed.record_hash == later.record_hash


@pytest.mark.parametrize(
    "detail",
    [
        {"nested": {"PassWord": "redacted"}},
        {"nested": {"safe": "session-token"}},
        {"MediaBytes": "00ff"},
        {"rawPose": [1, 2]},
    ],
)
def test_detail_parser_rejects_recursive_privacy_aliases(
    detail: dict[str, JsonValue],
) -> None:
    # Given/When/Then: privacy-bearing mixed-case keys or values never enter audit JSON
    with pytest.raises(AuditDetailError):
        parse_detail_json(
            AuditAction.CLIP_LIST, json.dumps({"version": 1, "nested": detail})
        )


def test_capacity_refusal_reads_only_count_before_history_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the production million-row threshold represented by a one-row boundary fixture.
    from backend.app.features.audit import verification as audit_module

    audit = AuditStore(_database_path())
    audit.append(_event())
    monkeypatch.setattr(audit_module, "MAX_AUDIT_ROWS", 1)
    connection = open_runtime_database(_database_path(), actor=RuntimeActor.API)
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    # When: startup verification reaches the refusal boundary.
    with pytest.raises(AuditVerificationError, match="one-million-row"):
        audit._verify(connection, None)
    connection.close()

    # Then: no audit row body or OFFSET query was executed.
    assert statements == ["SELECT COUNT(audit_id) FROM audit_events"]


def test_detail_parser_rejects_more_than_sixteen_kibibytes() -> None:
    # Given/When/Then: canonical UTF-8 detail is bounded before SQLite
    with pytest.raises(AuditDetailError):
        recovery_detail("x" * 17000, "2026-08-24T00:01:00.000Z")


def test_detail_parser_canonicalizes_safe_registered_detail() -> None:
    # Given: a registered reconciliation detail shape
    # When: keys arrive in a non-canonical order
    detail = recovery_detail("SQLITE_FULL", "2026-08-24T00:01:00.000Z")

    # Then: the machine-consumed JSON is deterministic
    assert detail.json == (
        '{"ended_at":"2026-08-24T00:01:00.000Z",'
        '"failure_code":"SQLITE_FULL","version":1}'
    )
