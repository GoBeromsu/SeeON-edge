"""Clip listing filter, facet and lifespan contracts on the PostgreSQL catalogue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.clips.catalog_indexer import API_CLIP_CATALOG_INTERVAL_SEC_ENV
from backend.app.features.clips.store import ClipStore
from backend.app.main import create_app
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_app_env import inject_sandbox_root
from tests_support.postgres_clip_app import index_clips
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox", "tests_support.postgres_app_env")


def _write_manifest(
    root: Path,
    clip_id: str,
    *,
    camera_id: str = "camera-a",
    event_type: str | None = "fall",
    event_ref: str = "event-ref",
) -> None:
    clip_dir = root / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, str | float | bool] = {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "event_ref": event_ref,
        "started_at": f"2026-08-09T00:00:{clip_id[-2:]}Z",
        "duration_s": 1.0,
        "codec": "h264",
        "video_available": False,
        "finalized": True,
    }
    if event_type is not None:
        payload["event_type"] = event_type
    _ = (clip_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


@pytest.mark.parametrize(
    "params",
    [{"event_type": "legacy"}, {"event_type": "mystery", "limit": 48}],
)
def test_event_type_filter_rejects_noncanonical_values(
    tmp_path: Path,
    params: dict[str, str | int],
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01")
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.clip_store = ClipStore(root)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips", params=params)

    assert response.status_code == 422


def test_listing_lifespan_can_enter_same_app_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    monkeypatch.setenv(API_CLIP_CATALOG_INTERVAL_SEC_ENV, "300")
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01")
    app = create_app()
    app.state.clip_store = ClipStore(root)
    inject_sandbox_root(app, postgres_product_sandbox, postgres_audit_runtime)

    pages: list[int] = []
    for _ in range(2):
        with TestClient(app) as client:
            _login(client)
            response = client.get("/api/v1/clips", params={"limit": 48})
            assert response.status_code == 200
            pages.append(len(response.json()["clips"]))

    assert pages == [1, 1]


def test_http_listing_exposes_canonical_event_facets(
    tmp_path: Path,
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> None:
    root = tmp_path / "clip-store"
    _write_manifest(root, "clip-01", event_type="fall")
    _write_manifest(root, "clip-02", event_type="bed-exit")
    _write_manifest(root, "clip-03", event_type=None, event_ref="legacy")
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.clip_store = ClipStore(root)
    index_clips(app)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/clips")
    assert response.status_code == 200
    assert response.json()["event_type_counts"] == {"bed-exit": 1, "fall": 1, "other": 1}
