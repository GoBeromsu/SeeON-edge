from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction, AuditDetailError
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

pytest_plugins = ("tests_support.postgres_sandbox",)


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


def test_runtime_settings_success_appends_exactly_one_audit_row(
    postgres_product_sandbox, postgres_audit_runtime
) -> None:
    # Given: the native owner and verified audit runtime explicitly injected.
    sandbox = postgres_product_sandbox
    app = create_app(lifespan=no_lifespan)
    app.state.runtime_settings_store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    app.state.audit_runtime = postgres_audit_runtime
    app.state.dashboard_credentials_store = PostgresDashboardCredentialsStore(
        sandbox.database, sandbox.authority
    )
    with TestClient(app) as client:
        _login(client)
        before = sandbox.admin.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

        # When: the governed runtime setting mutates.
        response = client.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": True, "expected_version": 0},
        )

        after = sandbox.admin.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    # Then: state and exactly one audit event commit together.
    assert response.status_code == 200
    assert response.json() == {"clip_export_enabled": True, "version": 1}
    assert after - before == 1
    assert sandbox.admin.execute(
        "SELECT clip_export_enabled,runtime_settings_version FROM edge_site WHERE id=1"
    ).fetchone() == (1, 1)


def test_invalid_video_range_appends_no_success_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox,
    postgres_audit_runtime,
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
    sandbox = postgres_product_sandbox
    app.state.audit_runtime = postgres_audit_runtime
    app.state.dashboard_credentials_store = PostgresDashboardCredentialsStore(
        sandbox.database, sandbox.authority
    )
    with TestClient(app) as client:
        _login(client)
        before = sandbox.admin.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
        ).fetchone()[0]

        # When: range preparation rejects a non-overlapping request.
        response = client.get("/api/v1/clips/range-clip/video", headers={"Range": "bytes=999-1000"})

        after = sandbox.admin.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
        ).fetchone()[0]
        valid = client.get("/api/v1/clips/range-clip/video", headers={"Range": "bytes=0-2"})
        assert (valid.status_code, valid.content) == (206, b"012")
        assert (
            sandbox.admin.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='clip.play'"
            ).fetchone()[0]
            == after + 1
        )

    # Then: 416 is not recorded as successful access.
    assert (response.status_code, response.content) == (416, b"")
    assert after == before
