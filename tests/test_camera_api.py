from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


def test_camera_topology_api_binds_stable_refs_without_exposing_transport(
    tmp_path, monkeypatch
) -> None:
    # Given
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    # When
    with TestClient(app) as client:
        _login(client)
        floor = client.post(
            "/api/v1/cameras/topology/floors",
            json={"edge_ref": "floor-a", "name": "First", "order_index": 1},
        )
        room = client.post(
            "/api/v1/cameras/topology/rooms",
            json={
                "edge_ref": "room-a",
                "floor_edge_ref": "floor-a",
                "name": "101",
                "legacy_canonical_space_id": "a2222222-2222-4222-8222-222222222222",
            },
        )
        camera = client.post(
            "/api/v1/cameras",
            json={
                "label": "Bed camera",
                "rtsp_url": "rtsp://operator:private@10.0.0.9/live",
                "edge_ref": "camera-a",
                "room_edge_ref": "room-a",
            },
        )
        topology = client.get("/api/v1/cameras/topology")

    # Then
    assert floor.status_code == 201
    assert room.status_code == 201
    assert camera.status_code == 201
    assert camera.json()["edge_ref"] == "camera-a"
    assert camera.json()["room_edge_ref"] == "room-a"
    assert topology.status_code == 200
    body = topology.json()
    assert body["registry_version"] == 3
    assert body["readiness_error"] is None
    assert body["floors"][0]["rooms"][0]["cameras"] == [
        {"edge_ref": "camera-a", "label": "Bed camera"}
    ]
    serialized = topology.text.lower()
    assert "rtsp" not in serialized
    assert "private" not in serialized
    assert "10.0.0.9" not in serialized


def test_camera_topology_api_returns_typed_conflict_without_partial_write(
    tmp_path,
) -> None:
    # Given
    app = create_app(lifespan=no_lifespan)
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    # When
    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras/topology/rooms",
            json={"edge_ref": "room-a", "floor_edge_ref": "missing", "name": "101"},
        )

    # Then
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "MISSING_PARENT"
    assert store.topology_snapshot().registry_version == 0
