from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    return postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


def test_camera_topology_api_binds_stable_refs_without_exposing_transport(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")

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
                "rtsp_url": "rtsp://operator:private@camera.example/live",
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
    assert "camera.example" not in serialized


def test_camera_topology_api_returns_typed_conflict_without_partial_write(
    app: FastAPI,
) -> None:
    # Given
    store = app.state.camera_registry

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


def test_camera_patch_binds_explicit_edge_and_room_refs_in_one_registry_revision(
    app: FastAPI,
) -> None:
    # Given
    store = app.state.camera_registry
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="101")
    store.create(
        camera_id="camera-local",
        label="Legacy camera",
        rtsp_url="rtsp://camera/live",
        space_id=None,
        status="online",
    )

    # When
    with TestClient(app) as client:
        _login(client)
        response = client.patch(
            "/api/v1/cameras/camera-local",
            json={"edge_ref": "camera-a", "room_edge_ref": "room-a"},
        )

    # Then
    assert response.status_code == 200
    assert response.json()["edge_ref"] == "camera-a"
    assert response.json()["room_edge_ref"] == "room-a"
    snapshot = store.topology_snapshot()
    assert snapshot.registry_version == 4
    assert snapshot.floors[0].rooms[0].cameras[0].edge_ref == "camera-a"
    assert snapshot.dirty is not None
    assert snapshot.dirty.registry_version == 4


def test_camera_patch_invalid_rebind_rolls_back_record_binding_and_dirty_marker(
    app: FastAPI,
) -> None:
    # Given
    store = app.state.camera_registry
    store.create_floor(edge_ref="floor-a", name="First", order_index=1)
    store.create_room(edge_ref="room-a", floor_edge_ref="floor-a", name="101")
    store.create(
        camera_id="camera-local",
        label="Bound camera",
        rtsp_url="rtsp://camera/live",
        space_id=None,
        status="online",
        edge_ref="camera-a",
        room_edge_ref="room-a",
    )
    before_record = store.get("camera-local")
    before_topology = store.topology_snapshot()

    # When
    with TestClient(app) as client:
        _login(client)
        response = client.patch(
            "/api/v1/cameras/camera-local",
            json={"edge_ref": "camera-b", "room_edge_ref": "missing-room"},
        )

    # Then
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "MISSING_PARENT"
    assert store.get("camera-local") == before_record
    assert store.topology_snapshot() == before_topology
