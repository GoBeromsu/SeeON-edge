from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.audit.catalog import AuditAction, AuditDetailError, empty_detail
from backend.app.features.audit.store import AuditEvent, AuditStore, AuditVerificationError
from backend.app.main import create_app, no_lifespan


def _database_path() -> Path:
    from backend.app.features.audit import store

    return store.EDGE_DATABASE_PATH


def _event(action: AuditAction, target_id: str) -> AuditEvent:
    return AuditEvent(
        occurred_at="2026-08-24T00:00:00.000Z",
        actor_id="admin",
        action=action,
        target_id=target_id,
        detail=empty_detail(action),
    )


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
    assert response.status_code == 204


def test_action_detail_catalog_is_exhaustive_and_versioned() -> None:
    # Given: the machine-consumed action-specific declaration catalog.
    from backend.app.features.audit.catalog import (
        ACTION_DETAIL_CATALOG,
        assert_catalog_complete,
        empty_detail,
    )

    # When/Then: every action has one v1 declaration and removing one is rejected.
    assert {declaration.action for declaration in ACTION_DETAIL_CATALOG} == set(AuditAction)
    assert all(declaration.version == 1 for declaration in ACTION_DETAIL_CATALOG)
    assert empty_detail(AuditAction.CLIP_LIST).json == '{"version":1}'
    with pytest.raises(AuditDetailError, match="catalog"):
        assert_catalog_complete(ACTION_DETAIL_CATALOG[:-1])


def test_incremental_verification_rechecks_history_after_trigger_epoch_change() -> None:
    # Given: a fully verified two-row chain and the canonical immutable trigger SQL.
    audit = AuditStore(_database_path())
    first = audit.append(_event(AuditAction.CLIP_LIST, "clips"))
    audit.append(_event(AuditAction.AUDIT_LIST, "audit"))
    checkpoint = audit.verify()
    with sqlite3.connect(_database_path()) as connection:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' "
            "AND name='audit_events_immutable_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER audit_events_immutable_update")
        connection.execute(
            "UPDATE audit_events SET actor_id='mutated-admin' WHERE audit_id=?",
            (first.audit_id,),
        )
        connection.execute(trigger_sql)

    # When/Then: incremental verification agrees with full verification and rejects history.
    with pytest.raises(AuditVerificationError, match="hash|contract"):
        audit.verify(checkpoint)


def test_full_verification_rejects_same_name_wrong_trigger_definition() -> None:
    # Given: all required trigger names but one non-canonical body.
    audit = AuditStore(_database_path())
    with sqlite3.connect(_database_path()) as connection:
        connection.execute("DROP TRIGGER audit_events_immutable_update")
        connection.execute(
            "CREATE TRIGGER audit_events_immutable_update "
            "BEFORE UPDATE ON audit_events BEGIN SELECT 1; END"
        )

    # When/Then: fresh full verification rejects names-only impersonation.
    with pytest.raises(AuditVerificationError, match="canonical|schema 18 contract"):
        audit.verify()


def test_incremental_verification_rejects_missing_trigger() -> None:
    # Given: a valid checkpoint whose immutable-delete trigger is then removed.
    audit = AuditStore(_database_path())
    checkpoint = audit.verify()
    with sqlite3.connect(_database_path()) as connection:
        connection.execute("DROP TRIGGER audit_events_immutable_delete")

    # When/Then: incremental verification refuses the incomplete contract.
    with pytest.raises(AuditVerificationError, match="canonical|schema 18 contract"):
        audit.verify(checkpoint)


def test_incremental_checkpoint_is_bound_to_database_file_identity(tmp_path: Path) -> None:
    # Given: one checkpoint and a separately valid schema-18 database.
    audit = AuditStore(_database_path())
    checkpoint = audit.verify()
    replacement = tmp_path / "replacement.sqlite3"
    bootstrap_database(replacement)

    # When: a valid-looking different file replaces the checkpointed path.
    os.replace(replacement, _database_path())

    # Then: stale history cannot be trusted solely because row ids/hashes look valid.
    with pytest.raises(AuditVerificationError, match="identity"):
        audit.verify(checkpoint)


def test_full_and_incremental_verification_agree_for_healthy_history() -> None:
    # Given: a verified prefix followed by one canonical event.
    audit = AuditStore(_database_path())
    audit.append(_event(AuditAction.AUDIT_LIST, "first"))
    checkpoint = audit.verify()
    audit.append(_event(AuditAction.AUDIT_DETAIL, "second"))

    # When/Then: both modes return the same healthy terminal checkpoint.
    assert audit.verify(checkpoint) == audit.verify()


def test_full_and_incremental_verification_agree_for_corrupted_history() -> None:
    # Given: a checkpoint whose anchor is changed behind a recreated canonical trigger.
    audit = AuditStore(_database_path())
    first = audit.append(_event(AuditAction.AUDIT_LIST, "first"))
    checkpoint = audit.verify()
    with sqlite3.connect(_database_path()) as connection:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='audit_events_immutable_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER audit_events_immutable_update")
        connection.execute(
            "UPDATE audit_events SET actor_id='changed' WHERE audit_id=?", (first.audit_id,)
        )
        connection.execute(trigger_sql)

    # When/Then: both bounded modes reject the same corrupted history.
    with pytest.raises(AuditVerificationError):
        audit.verify(checkpoint)
    with pytest.raises(AuditVerificationError):
        audit.verify()


def test_runtime_settings_success_appends_exactly_one_audit_row() -> None:
    # Given: an authenticated real schema-18 app.
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        with sqlite3.connect(_database_path()) as connection:
            before = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

        # When: the governed runtime setting mutates.
        response = client.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": True, "expected_version": 0},
        )

        with sqlite3.connect(_database_path()) as connection:
            after = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    # Then: state and exactly one audit event commit together.
    assert response.status_code == 200
    assert response.json() == {"clip_export_enabled": True, "version": 1}
    assert after - before == 1


def test_invalid_video_range_appends_no_success_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one ten-byte descriptor-backed clip.
    root = tmp_path / "clips"
    clip_dir = root / "clips" / "range-clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"0123456789")
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "range-clip",
                "camera_id": "camera-a",
                "event_ref": "event-a",
                "event_type": "fall",
                "started_at": "2026-08-24T00:00:00Z",
                "duration_s": 1.0,
                "codec": "h264",
                "path": "clips/range-clip",
                "video_available": True,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIP_STORE_DIR", str(root))
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        _login(client)
        with sqlite3.connect(_database_path()) as connection:
            before = connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
            ).fetchone()[0]

        # When: range preparation rejects a non-overlapping request.
        response = client.get("/api/v1/clips/range-clip/video", headers={"Range": "bytes=999-1000"})

        with sqlite3.connect(_database_path()) as connection:
            after = connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
            ).fetchone()[0]

    # Then: 416 is not recorded as successful access.
    assert (response.status_code, response.content) == (416, b"")
    assert after == before
