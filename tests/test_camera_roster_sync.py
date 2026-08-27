from __future__ import annotations

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.roster_sync import sync_camera_roster
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import (
    API_CONNECTION_SETTINGS_PATH_ENV,
    ConnectionSettingsStore,
)
from backend.app.lifespan import apply_connection_settings
from backend.app.main import create_app, no_lifespan
from tests_support.compact_authority_db import prepare_compact_database


class _TopologyHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str | None, bytes]] = []

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.requests.append((self.path, self.headers.get("Authorization"), body))
        request = json.loads(body)
        snapshot_id = self.path.rsplit("/", maxsplit=1)[-1]
        response = json.dumps(
            {
                "schemaVersion": 1,
                "snapshotId": snapshot_id,
                "clientRevision": request["clientRevision"],
                "serverRevision": request["expectedServerRevision"] + 1,
                "result": {
                    "floors": {"created": 1, "updated": 0, "unchanged": 0},
                    "rooms": {"created": 1, "updated": 0, "unchanged": 0},
                    "cameras": {"created": 1, "updated": 0, "unchanged": 0},
                },
                "omissions": None,
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        _ = format, args
        return


@pytest.fixture(autouse=True)
def connection_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection.sqlite3"))
    yield


def _run_server(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _ready_app(tmp_path: Path, events_url: str, monkeypatch: pytest.MonkeyPatch):
    app = create_app(lifespan=no_lifespan)
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    app.state.camera_registry = store
    monkeypatch.setattr(
        ConnectionSettingsStore,
        "from_env",
        classmethod(lambda cls: cls(registry_path)),
    )
    monkeypatch.setenv("API_BACKEND_BASE_URL", events_url.removesuffix("/api/v1/events"))
    ConnectionSettingsStore.from_env().save(
        {
            "facility_code": "FAC-001",
            "client_installation_ref": "edge-unit-001",
            "facility_id": "11111111-1111-4111-8111-111111111111",
            "facility_token": "secret-token",
            "edge_installation_id": "c72bd9a7-3e04-47ba-a8cd-a56e54f98152",
            "enrollment_generation": 1,
        }
    )
    apply_connection_settings(app)
    store.create_floor(edge_ref="floor-1", name="First", order_index=1)
    store.create_room(edge_ref="room-101", floor_edge_ref="floor-1", name="101")
    store.create(
        camera_id="local-camera-id",
        label="Lobby",
        rtsp_url="rtsp://user:password@camera/private",
        space_id="legacy-space",
        status="online",
        edge_ref="camera-1",
        room_edge_ref="room-101",
    )
    return app, store


def test_sync_camera_roster_sends_complete_stable_topology_without_local_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    _TopologyHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TopologyHandler)
    thread = _run_server(server)
    try:
        app, store = _ready_app(
            tmp_path,
            f"http://127.0.0.1:{server.server_port}/api/v1/events",
            monkeypatch,
        )

        # When
        result = sync_camera_roster(app)

        # Then
        assert result.attempted is True
        assert result.status == "synced"
        assert len(_TopologyHandler.requests) == 1
        path, authorization, raw_body = _TopologyHandler.requests[0]
        body = json.loads(raw_body)
        assert path.startswith("/api/v1/edge/topology-snapshots/")
        assert authorization == "Bearer secret-token"
        assert body["floors"][0]["rooms"][0]["cameras"] == [
            {"edgeRef": "camera-1", "label": "Lobby"}
        ]
        assert "facility" not in raw_body.decode().lower()
        assert "rtsp" not in raw_body.decode().lower()
        assert "password" not in raw_body.decode()
        assert store.topology_snapshot().dirty is None
    finally:
        server.shutdown()
        thread.join(timeout=1)


def test_sync_camera_roster_fails_closed_for_unmapped_camera(tmp_path: Path) -> None:
    # Given
    app = create_app(lifespan=no_lifespan)
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    app.state.camera_registry = store
    store.create(
        camera_id="legacy-camera",
        label="Legacy",
        rtsp_url="rtsp://camera/private",
        space_id="legacy-space",
        status="online",
    )

    # When
    result = sync_camera_roster(app)

    # Then
    assert result.attempted is False
    assert result.status == "pending"
    assert result.error_class == "unconfigured"
    assert store.topology_snapshot().dirty is not None


def test_floor_crud_emits_one_event_driven_sync_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    app = create_app(lifespan=no_lifespan)
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    app.state.camera_registry = CameraRegistryStore(registry_path)
    calls: list[tuple[bool, bool]] = []
    from backend.app.features.cameras import router as router_module

    def capture_sync(_app, *, _force: bool = False, _refresh: bool = False) -> None:
        calls.append((_force, _refresh))

    monkeypatch.setattr(router_module, "sync_camera_roster", capture_sync)

    # When
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            ).status_code
            == 204
        )
        response = client.post(
            "/api/v1/cameras/topology/floors",
            json={"edge_ref": "floor-1", "name": "First", "order_index": 1},
        )

    # Then
    assert response.status_code == 201
    assert calls == [(True, True)]
