from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.status.heartbeat_store import HeartbeatStore
from backend.app.main import create_app, no_lifespan


def _registry(tmp_path: Path, *camera_ids: str) -> CameraRegistryStore:
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    for camera_id in camera_ids:
        store.create(
            camera_id=camera_id,
            label=camera_id,
            rtsp_url=f"rtsp://example/{camera_id}",
            space_id=None,
            status="online",
        )
    return store


def test_status_reports_online_and_never_seen_from_heartbeats(tmp_path: Path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = _registry(tmp_path, "cam-a", "cam-b")
    store = HeartbeatStore(stale_after_sec=90.0)
    store.record("cam-a", "fac-1")
    app.state.heartbeat_store = store

    response = TestClient(app).get("/api/v1/status")

    assert response.status_code == 200
    body = response.json()
    assert body["cameras"]["cam-a"]["status"] == "online"
    assert body["cameras"]["cam-b"]["status"] == "never_seen"
    assert body["stale_after_sec"] == 90.0


def test_status_defaults_to_never_seen_without_heartbeats(tmp_path: Path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = _registry(tmp_path, "cam-x")

    body = TestClient(app).get("/api/v1/status").json()

    assert body["cameras"]["cam-x"]["status"] == "never_seen"


def test_status_does_not_read_worker_runtime_state() -> None:
    # /api/v1/status must derive purely from the api-owned heartbeat store, never from a
    # worker runtime object (zero cross-boundary shared state).
    app = create_app(lifespan=no_lifespan)
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
    }
    assert not hasattr(app.state, "runtime")
