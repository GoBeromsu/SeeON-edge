"""With the PostgreSQL root configured, audit has one authority: PostgreSQL.

The real lifespan boots from deployment-style secret files. A schema-19
``edge.sqlite3`` sentinel stands where the legacy audit store would open it, so
a split-brain session row or read shows up as a changed or new file there.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction
from backend.app.main import create_app
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.sqlite_source import create_schema19_source

pytest_plugins = ("tests_support.postgres_sandbox", "tests_support.postgres_app_env")


def _redirect_edge_database(sentinel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every loaded backend and worker binding of the SQLite path at the sentinel."""
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"backend", "worker"}:
            continue
        if isinstance(getattr(module, "EDGE_DATABASE_PATH", None), Path):
            monkeypatch.setattr(module, "EDGE_DATABASE_PATH", sentinel)


def _directory_state(directory: Path) -> dict[str, tuple[str, int, int]]:
    return {
        entry.name: (
            hashlib.sha256(entry.read_bytes()).hexdigest(),
            entry.stat().st_mtime_ns,
            entry.stat().st_size,
        )
        for entry in sorted(directory.iterdir())
    }


def _postgres_rows(sandbox: ProductSandbox) -> list[tuple[int, str, str, str]]:
    return sandbox.admin.execute(
        "SELECT audit_id, action, target_id, record_hash FROM audit_events ORDER BY audit_id DESC"
    ).fetchall()


def test_owned_boot_keeps_audit_in_postgres_and_lists_it_over_http(
    postgres_app_env: ProductSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "sentinel-state" / "edge.sqlite3"
    create_schema19_source(sentinel)
    _redirect_edge_database(sentinel, monkeypatch)
    before = _directory_state(sentinel.parent)

    with TestClient(create_app()) as client:
        login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
        assert login.status_code == 204
        update = client.put(
            "/api/v1/runtime-settings", json={"clip_export_enabled": True, "expected_version": 0}
        )
        assert update.status_code == 200
        written = _postgres_rows(postgres_app_env)
        listed = client.get("/api/v1/audit", params={"limit": 100})

    assert listed.status_code == 200
    assert [
        (event["audit_id"], event["action"], event["target_id"], event["record_hash"])
        for event in listed.json()["events"]
    ] == written
    assert [action for _, action, _, _ in written] == [
        AuditAction.RUNTIME_SETTINGS_UPDATE,
        AuditAction.AUTH_LOGIN,
        AuditAction.AUDIT_SESSION_START,
    ]
    assert [action for _, action, _, _ in _postgres_rows(postgres_app_env)[:2]] == [
        AuditAction.AUDIT_SESSION_CLOSE,
        AuditAction.AUDIT_LIST,
    ]
    assert _directory_state(sentinel.parent) == before
