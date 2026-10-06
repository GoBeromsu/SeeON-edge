from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.cameras.dependencies import sync_camera_roster
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettingsStore,
)
from backend.app.lifespan import apply_connection_settings
from backend.app.main import create_app, no_lifespan
from backend.app.postgres_root import PostgresRoot, install_postgres_stores
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


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


def _run_server(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _postgres_app(sandbox: ProductSandbox):
    app = create_app(lifespan=no_lifespan)
    install_postgres_stores(app, PostgresRoot(sandbox.database, sandbox.authority))
    store = app.state.camera_registry
    assert isinstance(store, CameraRegistryStore)
    return app, store


def _ready_app(sandbox: ProductSandbox, base_url: str, monkeypatch: pytest.MonkeyPatch):
    app, store = _postgres_app(sandbox)
    monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, base_url)
    ConnectionSettingsStore(sandbox.database, sandbox.authority).save(
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
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    _TopologyHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TopologyHandler)
    thread = _run_server(server)
    try:
        app, store = _ready_app(
            postgres_product_sandbox,
            f"http://127.0.0.1:{server.server_port}",
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


def test_sync_camera_roster_fails_closed_for_unmapped_camera(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given
    app, store = _postgres_app(postgres_product_sandbox)
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
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    app, _store = _postgres_app(postgres_product_sandbox)
    app.state.audit_runtime = postgres_audit_runtime
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
