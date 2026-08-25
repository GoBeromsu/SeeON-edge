"""Schema-17 clip listing index is absent; compact listing is the HTTP path."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan


def test_clip_listing_index_module_is_absent() -> None:
    listing_index = Path(__file__).resolve().parents[1] / (
        "backend/app/features/clips/listing_index.py"
    )
    assert not listing_index.exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backend.app.features.clips.listing_index")


def test_http_listing_does_not_install_a_listing_index() -> None:
    app = create_app(lifespan=no_lifespan)
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        )
        assert login.status_code == 204
        response = client.get("/api/v1/clips", params={"limit": 48})
    assert response.status_code == 200
    assert not hasattr(app.state, "clip_listing_index")
    assert response.json()["pagination"]["next_cursor"] is None
