from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests_support.connection_api import (
    AuthFailHandler,
    HangHandler,
    OKHandler,
    closed_port,
    connection_client,
    connection_store,
    login,
    response_json,
    run_server,
)
from tests_support.connection_api import clear_env as clear_env


def test_connection_test_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    OKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), OKHandler)
    thread = run_server(server)
    try:
        client = connection_client(tmp_path, monkeypatch)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
                "facility_token": "tok-1",
            },
        )
        assert response.status_code == 200
        body = response_json(response)
        assert body["ok"] is True
        assert body["error_class"] is None
        assert body["probed_url"] == f"http://127.0.0.1:{server.server_port}/facility-1"
        assert OKHandler.received_auth == ["Bearer tok-1"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_auth_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), AuthFailHandler)
    thread = run_server(server)
    try:
        client = connection_client(tmp_path, monkeypatch)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
                "facility_token": "wrong-secret-token",
            },
        )
        assert response.status_code == 200
        body = response_json(response)
        assert body["ok"] is False
        assert body["error_class"] == "auth"
        assert "wrong-secret-token" not in response.text
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_unreachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port = closed_port()
    client = connection_client(tmp_path, monkeypatch)
    login(client)
    response = client.post(
        "/api/v1/connection/test",
        json={"config_url": f"http://127.0.0.1:{port}", "facility_id": "facility-1"},
    )
    assert response.status_code == 200
    body = response_json(response)
    assert body["ok"] is False
    assert body["error_class"] == "unreachable"


def test_connection_test_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), HangHandler)
    thread = run_server(server)
    try:
        client = connection_client(tmp_path, monkeypatch, test_timeout_s=0.2)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
            },
        )
        assert response.status_code == 200
        body = response_json(response)
        assert body["ok"] is False
        assert body["error_class"] == "timeout"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_unconfigured_with_empty_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = connection_client(tmp_path, monkeypatch)
    login(client)
    response = client.post("/api/v1/connection/test", json={})
    assert response.status_code == 200
    body = response_json(response)
    assert body["ok"] is False
    assert body["error_class"] == "unconfigured"
    assert body["probed_url"] is None


def test_connection_test_requires_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = connection_client(tmp_path, monkeypatch)
    response = client.post("/api/v1/connection/test", json={})
    assert response.status_code == 401


def test_connection_test_body_override_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), OKHandler)
    thread = run_server(server)
    try:
        store = connection_store(tmp_path, monkeypatch)
        _ = store.save(
            {
                "config_url": "http://saved.example/config",
                "facility_id": "saved-facility",
            }
        )
        client = connection_client(tmp_path, monkeypatch)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_id": "facility-1",
            },
        )
        assert response_json(response)["ok"] is True
        unchanged = store.load()
        assert unchanged.config_url == "http://saved.example/config"
        assert unchanged.facility_id == "saved-facility"
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_malformed_body_url_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = connection_client(tmp_path, monkeypatch)
    login(client)
    response = client.post(
        "/api/v1/connection/test",
        json={"config_url": "not-a-url", "facility_id": "facility-1"},
    )
    assert response.status_code == 422
