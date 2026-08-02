"""Camera roster sync to the external backend (story G004): payload shape,
per-camera/overall sync state, attempt-time backoff, CRUD/connection
triggers, and the explicit POST /connection/sync-cameras entrypoint.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.cameras import roster_sync as roster_sync_module
from backend.app.features.cameras.roster_sync import (
    BASE_BACKOFF_SEC,
    camera_sync_view,
    get_roster_sync_state,
    sync_camera_roster,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import (
    API_CONNECTION_SETTINGS_PATH_ENV,
    ConnectionSettingsStore,
)
from backend.app.main import create_app, no_lifespan
from backend.app.shared.backend_mapping import (
    API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV,
    CameraPushResult,
    RosterPushResult,
)

AUTH = {"Authorization": "Bearer relay-token"}


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "API_CAMERA_STORE",
        API_CONNECTION_SETTINGS_PATH_ENV,
        "API_EDGE_RELAY_TOKEN",
        "API_FACILITY_ID",
        "API_BACKEND_EDGE_CAMERAS_URL",
        "API_BACKEND_URL",
        "API_BACKEND_EVENTS_URL",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
        "EDGE_FACILITY_TOKEN",
        API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV,
        "ML_API_WORKER_PROBE_ORIGIN",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# Fixture helpers: local HTTP servers standing in for the external backend's
# PUT /v1/edge/cameras, mirroring the ThreadingHTTPServer/BaseHTTPRequestHandler
# pattern already used by tests/test_connection_api.py.
# --------------------------------------------------------------------------


class _RosterOKHandler(BaseHTTPRequestHandler):
    received_bodies: list[str] = []
    received_auth: list[str | None] = []

    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.__class__.received_bodies.append(self.rfile.read(length).decode("utf-8"))
        self.__class__.received_auth.append(self.headers.get("Authorization"))
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: str) -> None:
        return


class _RosterAuthFailHandler(BaseHTTPRequestHandler):
    def do_PUT(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(401)
        self.end_headers()

    def log_message(self, _format: str, *args: str) -> None:
        return


def _run_server(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _closed_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens here afterwards -> connection refused
    return port


def _configure_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    events_url: str,
    facility_token: str | None,
    facility_id: str = "facility-1",
) -> None:
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
    )
    ConnectionSettingsStore.from_env().save(
        {"events_url": events_url, "facility_token": facility_token, "facility_id": facility_id}
    )


def _app_with_registry(tmp_path: Path) -> tuple[object, CameraRegistryStore]:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = CameraRegistryStore(tmp_path / "cameras.json")
    app.state.camera_registry = store
    return app, store


# --------------------------------------------------------------------------
# Payload shape + success path
# --------------------------------------------------------------------------


def test_sync_camera_roster_builds_payload_from_registry_and_marks_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cam-1 has a space_id and gets pushed as its own single-object PUT;
    cam-2 has none and must never be sent (PUT /v1/edge/cameras requires
    spaceId) -- it stays pending, awaiting a space assignment, while cam-1
    still syncs successfully.
    """
    _RosterOKHandler.received_bodies = []
    _RosterOKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterOKHandler)
    thread = _run_server(server)
    try:
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        store.create(
            camera_id="cam-2", label="Hallway", rtsp_url="rtsp://b", space_id=None,
            status="online",
        )
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}/api/v1/edge/events",
            facility_token="tok-1",
        )

        result = sync_camera_roster(app)

        assert result.attempted is True
        assert result.status == "synced"
        assert result.error_class is None
        assert result.last_ok_at is not None
        assert result.camera_count == 2
        # Only cam-1 (space-assigned) is ever sent -- one single-object PUT.
        assert _RosterOKHandler.received_auth == ["Bearer tok-1"]
        assert len(_RosterOKHandler.received_bodies) == 1
        sent = json.loads(_RosterOKHandler.received_bodies[0])
        assert sent == {"edge_camera_ref": "cam-1", "label": "Lobby", "spaceId": "space-1"}
        assert "tok-1" not in _RosterOKHandler.received_bodies[0]

        synced_view = camera_sync_view(app, "cam-1")
        assert synced_view["status"] == "synced"
        assert synced_view["error_class"] is None
        assert synced_view["last_ok_at"] is not None

        pending_view = camera_sync_view(app, "cam-2")
        assert pending_view["status"] == "pending"
        assert pending_view["error_class"] is None
        assert pending_view["detail"]
        assert "공간" in pending_view["detail"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_sync_camera_roster_skips_cameras_without_space_id_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every camera lacks a space_id, nothing is ever pushed (zero PUT
    calls) and the overall result reads pending, not a false "synced".
    """
    _RosterOKHandler.received_bodies = []
    _RosterOKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterOKHandler)
    thread = _run_server(server)
    try:
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id=None,
            status="online",
        )
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}/api/v1/edge/events",
            facility_token="tok-1",
        )

        result = sync_camera_roster(app)

        assert result.attempted is True
        assert result.status == "pending"
        assert result.error_class is None
        assert result.camera_count == 1
        assert _RosterOKHandler.received_bodies == []

        view = camera_sync_view(app, "cam-1")
        assert view["status"] == "pending"
        assert view["error_class"] is None
        assert "공간" in view["detail"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_sync_camera_roster_partial_success_reports_per_camera_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One space-assigned camera succeeds and another is rejected (401) in
    the same sync attempt -- per-camera state must reflect each camera's own
    real outcome, not a single outcome copied onto every camera.
    """

    class _PartialHandler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if body.get("edge_camera_ref") == "cam-fail":
                self.send_response(401)
                self.end_headers()
                return
            payload = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *args: str) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _PartialHandler)
    thread = _run_server(server)
    try:
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-ok", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        store.create(
            camera_id="cam-fail", label="Hallway", rtsp_url="rtsp://b", space_id="space-2",
            status="online",
        )
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}/api/v1/edge/events",
            facility_token="tok-1",
        )

        result = sync_camera_roster(app)

        # Overall reflects that at least one eligible camera failed.
        assert result.attempted is True
        assert result.status == "failed"
        assert result.error_class == "auth"
        assert result.next_retry_at is not None

        ok_view = camera_sync_view(app, "cam-ok")
        assert ok_view["status"] == "synced"
        assert ok_view["error_class"] is None
        assert ok_view["last_ok_at"] is not None

        fail_view = camera_sync_view(app, "cam-fail")
        assert fail_view["status"] == "failed"
        assert fail_view["error_class"] == "auth"
        assert fail_view["detail"]
        assert "인증" in fail_view["detail"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_sync_camera_roster_auth_failure_records_korean_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterAuthFailHandler)
    thread = _run_server(server)
    try:
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}",
            facility_token="wrong-token",
        )

        result = sync_camera_roster(app)

        assert result.attempted is True
        assert result.status == "failed"
        assert result.error_class == "auth"
        assert result.detail
        assert "인증" in result.detail
        assert "wrong-token" not in result.detail

        view = camera_sync_view(app, "cam-1")
        assert view["status"] == "failed"
        assert view["error_class"] == "auth"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_sync_camera_roster_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _closed_port()
    app, store = _app_with_registry(tmp_path)
    store.create(
        camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1", status="online"
    )
    _configure_connection(
        monkeypatch, tmp_path, events_url=f"http://127.0.0.1:{port}", facility_token="tok-1"
    )

    result = sync_camera_roster(app)

    assert result.attempted is True
    assert result.status == "failed"
    assert result.error_class == "unreachable"


def test_sync_camera_roster_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _HangHandler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:  # noqa: N802
            import time

            time.sleep(2)

        def log_message(self, _format: str, *args: str) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _HangHandler)
    thread = _run_server(server)
    try:
        monkeypatch.setenv(API_BACKEND_CAMERA_MAPPING_TIMEOUT_SEC_ENV, "0.2")
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}",
            facility_token="tok-1",
        )

        result = sync_camera_roster(app)

        assert result.attempted is True
        assert result.status == "failed"
        assert result.error_class == "timeout"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_sync_camera_roster_unconfigured_is_disabled_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
    )
    app, store = _app_with_registry(tmp_path)
    store.create(
        camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1", status="online"
    )

    result = sync_camera_roster(app)

    assert result.attempted is False
    assert result.status == "disabled"
    assert result.error_class == "unconfigured"

    view = camera_sync_view(app, "cam-1")
    assert view["status"] == "disabled"
    assert view["error_class"] == "unconfigured"


# --------------------------------------------------------------------------
# Attempt-time backoff
# --------------------------------------------------------------------------


def test_sync_camera_roster_backoff_is_a_no_op_within_window_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _closed_port()
    app, store = _app_with_registry(tmp_path)
    store.create(
        camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1", status="online"
    )
    _configure_connection(
        monkeypatch, tmp_path, events_url=f"http://127.0.0.1:{port}", facility_token="tok-1"
    )

    first = sync_camera_roster(app, _now=1_000.0)
    assert first.attempted is True
    assert first.status == "failed"
    assert first.next_retry_at is not None

    within_window = sync_camera_roster(app, _now=1_000.0 + 1.0)
    assert within_window.attempted is False
    assert within_window.status == "failed"  # unchanged, no new attempt made

    after_window = sync_camera_roster(app, _now=1_000.0 + BASE_BACKOFF_SEC + 0.1)
    assert after_window.attempted is True  # backoff elapsed -> a real retry happened


# --------------------------------------------------------------------------
# Single-flight: concurrent triggers must never race a push
# --------------------------------------------------------------------------


def test_sync_camera_roster_single_flight_prevents_concurrent_pushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, store = _app_with_registry(tmp_path)
    store.create(
        camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1", status="online"
    )

    block = threading.Event()
    call_count: list[int] = []

    class _BlockingMapper:
        configured = True

        def put_roster(self, payload: list[dict[str, object]]) -> RosterPushResult:
            call_count.append(1)
            block.wait(timeout=5.0)
            return RosterPushResult(
                ok=True,
                error_class=None,
                status_code=200,
                cameras={"cam-1": CameraPushResult(ok=True, error_class=None, status_code=200)},
            )

    monkeypatch.setattr(roster_sync_module, "_build_mapper", lambda app: _BlockingMapper())

    first_result: list[object] = []
    thread = Thread(target=lambda: first_result.append(sync_camera_roster(app)))
    thread.start()
    try:
        for _ in range(500):  # wait until the first call has actually entered put_roster
            if call_count:
                break
            time.sleep(0.01)
        assert call_count == [1]

        second = sync_camera_roster(app)
        assert second.attempted is False  # single-flight: no second push started
        assert call_count == [1]
    finally:
        block.set()
        thread.join(timeout=5.0)

    assert first_result[0].attempted is True
    assert first_result[0].status == "synced"

    state = get_roster_sync_state(app)
    assert state.last_ok_at is not None
    assert state.consecutive_failures == 0


# --------------------------------------------------------------------------
# CRUD/connection-settings triggers (event-driven, best-effort)
# --------------------------------------------------------------------------


def test_camera_create_triggers_background_sync_without_breaking_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterOKHandler)
    thread = _run_server(server)
    try:
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}",
            facility_token="tok-1",
        )
        app, _store = _app_with_registry(tmp_path)
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/cameras",
                headers=AUTH,
                json={
                    "label": "Lobby",
                    "rtsp_url": "rtsp://camera/stream",
                    "space_id": "space-1",
                    "force_register": True,
                },
            )
            assert created.status_code == 201
            # CRUD responses never populate sync -- avoids racing the
            # fire-and-forget BackgroundTask against a pinned response shape.
            assert created.json()["sync"] is None

            listed = client.get("/api/v1/cameras", headers=AUTH).json()
        sync = listed["cameras"][0]["sync"]
        assert sync["status"] == "synced"
        assert sync["last_ok_at"] is not None
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_camera_create_trigger_is_best_effort_when_backend_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
    )
    app, _store = _app_with_registry(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera/stream",
                "space_id": "space-1",
                "force_register": True,
            },
        )
        # Unreachable/unconfigured backend must never fail the CRUD response.
        assert created.status_code == 201
        assert created.json()["sync"] is None

        listed = client.get("/api/v1/cameras", headers=AUTH).json()
    sync = listed["cameras"][0]["sync"]
    assert sync["status"] == "disabled"
    assert sync["error_class"] == "unconfigured"


def test_put_connection_triggers_roster_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RosterOKHandler.received_bodies = []
    _RosterOKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterOKHandler)
    thread = _run_server(server)
    try:
        monkeypatch.setenv(
            API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
        )
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        with TestClient(app) as client:
            put_response = client.put(
                "/api/v1/connection",
                headers=AUTH,
                json={
                    "events_url": f"http://127.0.0.1:{server.server_port}/api/v1/edge/events",
                    "facility_token": "tok-1",
                    "facility_id": "facility-1",
                },
            )
            assert put_response.status_code == 200

            listed = client.get("/api/v1/cameras", headers=AUTH).json()
        sync = listed["cameras"][0]["sync"]
        assert sync["status"] == "synced"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


# --------------------------------------------------------------------------
# POST /connection/sync-cameras
# --------------------------------------------------------------------------


def test_sync_cameras_endpoint_requires_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _store = _app_with_registry(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/v1/connection/sync-cameras")
    assert response.status_code == 401


def test_sync_cameras_endpoint_returns_fresh_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RosterOKHandler.received_bodies = []
    _RosterOKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterOKHandler)
    thread = _run_server(server)
    try:
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}",
            facility_token="tok-1",
        )
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        with TestClient(app) as client:
            response = client.post("/api/v1/connection/sync-cameras", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "synced"
        assert body["error_class"] is None
        assert body["last_ok_at"] is not None
        assert body["camera_count"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_sync_cameras_endpoint_unconfigured_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
    )
    app, _store = _app_with_registry(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/v1/connection/sync-cameras", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "disabled"
    assert body["error_class"] == "unconfigured"
    assert body["camera_count"] == 0


def test_sync_cameras_endpoint_never_exposes_the_facility_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RosterOKHandler.received_bodies = []
    _RosterOKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RosterOKHandler)
    thread = _run_server(server)
    try:
        _configure_connection(
            monkeypatch,
            tmp_path,
            events_url=f"http://127.0.0.1:{server.server_port}",
            facility_token="super-secret-token",
        )
        app, store = _app_with_registry(tmp_path)
        store.create(
            camera_id="cam-1", label="Lobby", rtsp_url="rtsp://a", space_id="space-1",
            status="online",
        )
        with TestClient(app) as client:
            response = client.post("/api/v1/connection/sync-cameras", headers=AUTH)
        assert response.status_code == 200
        assert "super-secret-token" not in response.text
        assert all(
            "super-secret-token" not in body for body in _RosterOKHandler.received_bodies
        )
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
