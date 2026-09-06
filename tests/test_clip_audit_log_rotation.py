"""JSONL clip audit rotation is absent; SQLite audit_events is the live path."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan


def test_jsonl_audit_log_module_is_absent() -> None:
    assert not (
        Path(__file__).resolve().parents[1] / "backend/app/features/clips/audit_log.py"
    ).exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.audit_log")


def test_clip_list_records_sqlite_audit_without_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    monkeypatch.delenv("API_AUDIT_LOG", raising=False)
    monkeypatch.delenv("API_BACKEND_CLIP_EVENTS_URL", raising=False)
    app = create_app(lifespan=no_lifespan)
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
