"""Dashboard HTTP API for the connection settings (story G003): GET/PUT
/connection and POST /connection/test.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.connection.store import (
    API_CONNECTION_SETTINGS_PATH_ENV,
    ConnectionSettingsStore,
)
from backend.app.lifespan import (
    API_BACKEND_CONFIG_URL_ENV,
    API_BACKEND_EVENTS_URL_ENV,
    API_EDGE_RELAY_TOKEN_ENV,
    API_FACILITY_ID_ENV,
    EDGE_FACILITY_TOKEN_ENV,
)
from backend.app.main import create_app, no_lifespan
from shared.events.edge_ingest_client import EdgeIngestClient

AUTH = {"Authorization": "Bearer relay-token"}
_TEST_TIMEOUT_ENV = "ML_API_CONNECTION_TEST_TIMEOUT_S"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        API_CONNECTION_SETTINGS_PATH_ENV,
        API_BACKEND_EVENTS_URL_ENV,
        API_BACKEND_CONFIG_URL_ENV,
        API_FACILITY_ID_ENV,
        EDGE_FACILITY_TOKEN_ENV,
        API_EDGE_RELAY_TOKEN_ENV,
        _TEST_TIMEOUT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, test_timeout_s: float = 0.3
) -> TestClient:
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
    )
    monkeypatch.setenv(_TEST_TIMEOUT_ENV, str(test_timeout_s))
    get_settings.cache_clear()
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    return TestClient(app)


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConnectionSettingsStore:
    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json")
    )
    return ConnectionSettingsStore.from_env()


# --------------------------------------------------------------------------
# GET /connection
# --------------------------------------------------------------------------


def test_get_connection_requires_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/v1/connection")
    assert response.status_code == 401


def test_get_connection_masks_token_and_reports_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, monkeypatch)
    store.save(
        {
            "events_url": "http://backend.example/api/v1/edge/events",
            "config_url": "http://backend.example/api/v1/edge/config",
            "facility_id": "facility-1",
            "facility_token": "super-secret-facility-token",
        }
    )
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/v1/connection", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["events_url"] == "http://backend.example/api/v1/edge/events"
    assert body["config_url"] == "http://backend.example/api/v1/edge/config"
    assert body["facility_id"] == "facility-1"
    assert body["facility_token_set"] is True
    assert body["facility_token_masked"] == "****oken"
    assert body["configured"] is True
    assert body["reachable"] is None
    assert body["last_ok_at"] is None
    assert "super-secret-facility-token" not in response.text


def test_get_connection_unconfigured_when_nothing_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/v1/connection", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is False
    assert body["facility_token_set"] is False
    assert body["facility_token_masked"] is None


def test_get_connection_heartbeat_relay_absent_state_reads_disabled_with_nulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # no_lifespan test apps never populate backend_heartbeat_relay_state.
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/v1/connection", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["heartbeat_relay"] == {
        "enabled": False,
        "last_success_at": None,
        "last_error_class": None,
        "detail": None,
    }


def test_get_connection_heartbeat_relay_reflects_state_and_maps_korean_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.features.status.backend_heartbeat_relay import HeartbeatRelayState

    client = _client(tmp_path, monkeypatch)
    client.app.state.backend_heartbeat_relay_task = object()  # loop "configured"
    client.app.state.backend_heartbeat_relay_state = HeartbeatRelayState(
        last_error_class="auth", last_success_at="2026-01-01T00:00:00.000Z"
    )

    response = client.get("/api/v1/connection", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["heartbeat_relay"] == {
        "enabled": True,
        "last_success_at": "2026-01-01T00:00:00.000Z",
        "last_error_class": "auth",
        "detail": "외부 백엔드 인증에 실패했습니다. 시설 토큰을 확인해 주세요.",
    }


def test_get_connection_heartbeat_relay_task_none_reads_disabled_even_with_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.features.status.backend_heartbeat_relay import HeartbeatRelayState

    client = _client(tmp_path, monkeypatch)
    client.app.state.backend_heartbeat_relay_task = None  # disabled via env kill-switch
    client.app.state.backend_heartbeat_relay_state = HeartbeatRelayState()

    response = client.get("/api/v1/connection", headers=AUTH)

    assert response.status_code == 200
    body = response.json()["heartbeat_relay"]
    assert body["enabled"] is False
    assert body["last_error_class"] is None
    assert body["detail"] is None


# --------------------------------------------------------------------------
# PUT /connection
# --------------------------------------------------------------------------


def test_put_connection_requires_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.put("/api/v1/connection", json={"events_url": "http://x.example/events"})
    assert response.status_code == 401


def test_put_connection_saves_and_relinks_the_running_app_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={
            "events_url": "http://backend.example/api/v1/edge/events",
            "facility_token": "tok-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["events_url"] == "http://backend.example/api/v1/edge/events"
    assert body["facility_token_set"] is True
    assert body["configured"] is True

    # Same app instance -- apply_connection_settings() must have rebuilt the
    # ingest client synchronously within the request, no restart required.
    ingest = client.app.state.backend_ingest_client
    assert isinstance(ingest, EdgeIngestClient)
    assert ingest.events_url == "http://backend.example/api/v1/edge/events"
    assert ingest.bearer_token == "tok-1"


def test_put_connection_explicit_null_clears_saved_value_back_to_env_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "env-seeded-token")
    client = _client(tmp_path, monkeypatch)

    saved = client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={"events_url": "http://backend.example/events", "facility_token": "explicit-token"},
    )
    assert saved.json()["facility_token_masked"] == "****oken"

    cleared = client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={"facility_token": None},
    )
    assert cleared.status_code == 200
    # env seeds the gap once the saved value is explicitly cleared.
    assert cleared.json()["facility_token_masked"] == "****oken"  # last 4 of env-seeded-token too
    assert cleared.json()["facility_token_set"] is True


def test_put_connection_omitted_field_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={"events_url": "http://backend.example/events", "facility_id": "facility-1"},
    )

    response = client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={"events_url": "http://backend-2.example/events"},
    )

    assert response.status_code == 200
    assert response.json()["facility_id"] == "facility-1"
    assert response.json()["events_url"] == "http://backend-2.example/events"


def test_put_connection_rejects_invalid_url_with_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={"events_url": "not-a-url"},
    )
    assert response.status_code == 422
    assert "events_url" in response.text


def test_put_connection_rejects_non_http_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/v1/connection",
        headers=AUTH,
        json={"config_url": "ftp://backend.example/config"},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# POST /connection/test
# --------------------------------------------------------------------------


class _OKHandler(BaseHTTPRequestHandler):
    received_auth: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        self.__class__.received_auth.append(self.headers.get("Authorization"))
        body = json.dumps({"configVersion": 1, "cameras": []}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: str) -> None:
        return


class _AuthFailHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(401)
        self.end_headers()

    def log_message(self, _format: str, *args: str) -> None:
        return


class _HangHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        time.sleep(5)  # much longer than the small test timeout

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


def test_connection_test_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _OKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    thread = _run_server(server)
    try:
        client = _client(tmp_path, monkeypatch)
        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
                "facility_token": "tok-1",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["error_class"] is None
        assert body["probed_url"] == f"http://127.0.0.1:{server.server_port}/facility-1"
        assert _OKHandler.received_auth == ["Bearer tok-1"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_auth_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthFailHandler)
    thread = _run_server(server)
    try:
        client = _client(tmp_path, monkeypatch)
        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
                "facility_token": "wrong-secret-token",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error_class"] == "auth"
        assert "wrong-secret-token" not in response.text
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_unreachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port = _closed_port()
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/connection/test",
        headers=AUTH,
        json={"config_url": f"http://127.0.0.1:{port}", "facility_id": "facility-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error_class"] == "unreachable"


def test_connection_test_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HangHandler)
    thread = _run_server(server)
    try:
        client = _client(tmp_path, monkeypatch, test_timeout_s=0.2)
        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error_class"] == "timeout"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_unconfigured_with_empty_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/v1/connection/test", headers=AUTH, json={})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error_class"] == "unconfigured"
    assert body["probed_url"] is None


def test_connection_test_requires_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/v1/connection/test", json={})
    assert response.status_code == 401


def test_connection_test_body_override_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    thread = _run_server(server)
    try:
        store = _store(tmp_path, monkeypatch)
        store.save({"config_url": "http://saved.example/config", "facility_id": "saved-facility"})
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
            },
        )
        assert response.json()["ok"] is True

        # The probe body must not have touched the saved settings.
        unchanged = store.load()
        assert unchanged.config_url == "http://saved.example/config"
        assert unchanged.facility_id == "saved-facility"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_overridden_config_url_without_token_blocks_stored_token_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored facility_token must never be sent to a config_url the caller
    switched to without also supplying a token -- no outbound request at all.
    """
    hits: list[str | None] = []

    class _RecordingHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hits.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *args: str) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = _run_server(server)
    try:
        store = _store(tmp_path, monkeypatch)
        store.save(
            {
                "config_url": "http://saved.example/config",
                "facility_id": "facility-1",
                "facility_token": "super-secret-stored-token",
            }
        )
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={"config_url": f"http://127.0.0.1:{server.server_port}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error_class"] == "auth"
        assert "저장된 토큰을 사용할 수 없습니다" in body["detail"]
        assert body["probed_url"] is None
        assert "super-secret-stored-token" not in response.text
        assert hits == []  # no outbound request was made at all
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_overridden_config_url_with_explicit_token_probes_with_body_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _OKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    thread = _run_server(server)
    try:
        store = _store(tmp_path, monkeypatch)
        store.save(
            {
                "config_url": "http://saved.example/config",
                "facility_id": "facility-1",
                "facility_token": "stored-token",
            }
        )
        client = _client(tmp_path, monkeypatch)

        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_token": "body-token",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert _OKHandler.received_auth == ["Bearer body-token"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_same_config_url_override_keeps_stored_token_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _OKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    thread = _run_server(server)
    try:
        saved_config_url = f"http://127.0.0.1:{server.server_port}"
        store = _store(tmp_path, monkeypatch)
        store.save(
            {
                "config_url": saved_config_url,
                "facility_id": "facility-1",
                "facility_token": "stored-token",
            }
        )
        client = _client(tmp_path, monkeypatch)

        # Same URL, just with a trailing slash -- still "unchanged" after rstrip.
        response = client.post(
            "/api/v1/connection/test",
            headers=AUTH,
            json={"config_url": f"{saved_config_url}/"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert _OKHandler.received_auth == ["Bearer stored-token"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_malformed_body_url_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/connection/test",
        headers=AUTH,
        json={"config_url": "not-a-url", "facility_id": "facility-1"},
    )
    assert response.status_code == 422
