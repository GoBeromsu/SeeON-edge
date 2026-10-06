from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.status.heartbeat_store import HeartbeatStore
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    return postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)


def _register(app: FastAPI, *camera_ids: str) -> None:
    for camera_id in camera_ids:
        app.state.camera_registry.create(
            camera_id=camera_id,
            label=camera_id,
            rtsp_url=f"rtsp://example/{camera_id}",
            space_id=None,
            status="online",
        )


def test_status_reports_online_and_never_seen_from_heartbeats(app: FastAPI) -> None:
    _register(app, "cam-a", "cam-b")
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1")
    app.state.heartbeat_store = store

    response = TestClient(app).get("/api/v1/status")

    assert response.status_code == 200
    body = response.json()
    assert body["cameras"]["cam-a"]["status"] == "online"
    assert body["cameras"]["cam-b"]["status"] == "never_seen"
    assert body["stale_after_sec"] == 90.0


def test_status_defaults_to_never_seen_without_heartbeats(app: FastAPI) -> None:
    _register(app, "cam-x")

    body = TestClient(app).get("/api/v1/status").json()

    assert body["cameras"]["cam-x"]["status"] == "never_seen"


def test_status_does_not_read_worker_runtime_state(app: FastAPI) -> None:
    # /api/v1/status must derive purely from the api-owned heartbeat store, never from a
    # worker runtime object (zero cross-boundary shared state).
    body = TestClient(app).get("/api/v1/status").json()
    assert body["cameras"] == {}
    assert body["stale_after_sec"] == HeartbeatStore().stale_after_sec
    assert body["runtime"] == {
        "facilities": {},
        "stale_after_sec": 15.0,
        "cameras": {},
        "worker": None,
        "device": None,
        "clip_recorder": None,
        "delivery_queue": None,
        "clip_export_applied": {"enabled": None, "version": None, "freshness": "unknown"},
    }
    assert body["runtime_settings"] == {"clip_export_enabled": False, "version": 0}
    assert not hasattr(app.state, "runtime")
