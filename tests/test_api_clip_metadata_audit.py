from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan


def _write_manifest(store_root: Path, clip_id: str) -> None:
    clip_dir = store_root / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_id,
        "camera_id": "camera-a",
        "event_ref": "event-a",
        "event_type": "fall",
        "started_at": "2026-08-09T00:00:00Z",
        "duration_s": 10.0,
        "codec": "",
        "path": None,
        "video_available": False,
        "finalized": True,
    }
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


@pytest.fixture(autouse=True)
def clip_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLIP_STORE_DIR", str(tmp_path / "clip-store"))
    monkeypatch.setenv("API_LABEL_STORE", str(tmp_path / "label-store"))
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.delenv("API_AUDIT_LOG", raising=False)
    monkeypatch.delenv("API_BACKEND_CLIP_EVENTS_URL", raising=False)
    monkeypatch.delenv("API_BACKEND_FACILITY_TOKEN", raising=False)
    monkeypatch.delenv("API_FACILITY_TOKEN", raising=False)
    return tmp_path


def test_successful_metadata_read_appends_clip_scoped_audit(clip_env: Path) -> None:
    clip_id = "clip-audit"
    _write_manifest(clip_env / "clip-store", clip_id)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get(f"/api/v1/clips/{clip_id}/metadata")

    assert response.status_code == 200
    with sqlite3.connect(clip_env / ".central-fixture" / "edge.sqlite3") as connection:
        rows = connection.execute(
            "SELECT actor_id,action,target_id FROM audit_events WHERE action='clip.detail'"
        ).fetchall()
    assert rows == [("admin", "clip.detail", clip_id)]


def test_missing_metadata_read_does_not_append_success_audit(clip_env: Path) -> None:
    with TestClient(create_app(lifespan=no_lifespan)) as client:
        _login(client)
        response = client.get("/api/v1/clips/missing/metadata")

    assert response.status_code == 404
    with sqlite3.connect(clip_env / ".central-fixture" / "edge.sqlite3") as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action='clip.detail'"
        ).fetchone()[0]
    assert count == 0


def test_unauthorized_metadata_read_does_not_append_success_audit(clip_env: Path) -> None:
    clip_id = "clip-audit"
    _write_manifest(clip_env / "clip-store", clip_id)

    with TestClient(create_app(lifespan=no_lifespan)) as client:
        response = client.get(f"/api/v1/clips/{clip_id}/metadata")

    assert response.status_code == 401
    with sqlite3.connect(clip_env / ".central-fixture" / "edge.sqlite3") as connection:
        count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    assert count == 0
