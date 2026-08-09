from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typing_extensions import override

from tests_support.connection_api import (
    OKHandler,
    connection_client,
    connection_store,
    login,
    response_json,
    run_server,
)
from tests_support.connection_api import clear_env as clear_env


def test_overridden_config_url_without_token_blocks_stored_token_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits: list[str | None] = []

    class RecordingHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hits.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        @override
        def log_message(self, format: str, *args: str) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
    thread = run_server(server)
    try:
        store = connection_store(tmp_path, monkeypatch)
        _ = store.save(
            {
                "config_url": "http://saved.example/config",
                "facility_id": "facility-1",
                "facility_token": "super-secret-stored-token",
            }
        )
        client = connection_client(tmp_path, monkeypatch)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={"config_url": f"http://127.0.0.1:{server.server_port}"},
        )
        assert response.status_code == 200
        body = response_json(response)
        assert body["ok"] is False
        assert body["error_class"] == "auth"
        detail = body["detail"]
        assert isinstance(detail, str)
        assert "저장된 토큰을 사용할 수 없습니다" in detail
        assert body["probed_url"] is None
        assert "super-secret-stored-token" not in response.text
        assert hits == []
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_overridden_config_url_with_explicit_token_probes_with_body_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    OKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), OKHandler)
    thread = run_server(server)
    try:
        store = connection_store(tmp_path, monkeypatch)
        _ = store.save(
            {
                "config_url": "http://saved.example/config",
                "facility_id": "facility-1",
                "facility_token": "stored-token",
            }
        )
        client = connection_client(tmp_path, monkeypatch)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={
                "config_url": f"http://127.0.0.1:{server.server_port}",
                "facility_token": "body-token",
            },
        )
        assert response.status_code == 200
        assert response_json(response)["ok"] is True
        assert OKHandler.received_auth == ["Bearer body-token"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_same_config_url_override_keeps_stored_token_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    OKHandler.received_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), OKHandler)
    thread = run_server(server)
    try:
        saved_config_url = f"http://127.0.0.1:{server.server_port}"
        store = connection_store(tmp_path, monkeypatch)
        _ = store.save(
            {
                "config_url": saved_config_url,
                "facility_id": "facility-1",
                "facility_token": "stored-token",
            }
        )
        client = connection_client(tmp_path, monkeypatch)
        login(client)
        response = client.post(
            "/api/v1/connection/test",
            json={"config_url": f"{saved_config_url}/"},
        )
        assert response.status_code == 200
        assert response_json(response)["ok"] is True
        assert OKHandler.received_auth == ["Bearer stored-token"]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
