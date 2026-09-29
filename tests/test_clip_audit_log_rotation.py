"""JSONL clip audit rotation is absent; PostgreSQL audit_events is the live path."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def test_jsonl_audit_log_module_is_absent() -> None:
    assert not (
        Path(__file__).resolve().parents[1] / "backend/app/features/clips/audit_log.py"
    ).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.audit_log")


def test_clip_list_records_postgres_audit_without_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    monkeypatch.delenv("API_AUDIT_LOG", raising=False)
    monkeypatch.delenv("API_BACKEND_CLIP_EVENTS_URL", raising=False)
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    with TestClient(app) as client:
        login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
        assert login.status_code == 204
        listed = client.get("/api/v1/clips")
        audit = client.get("/api/v1/audit")
    assert listed.status_code == 200
    assert audit.status_code == 200
    actions = [event["action"] for event in audit.json()["events"]]
    assert "clip.list" in actions
    assert not (tmp_path / "audit.jsonl").exists()
