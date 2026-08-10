from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from backend.app.features.connection.store import API_BACKEND_BASE_URL_ENV
from tests_support.connection_api import (
    EnrollmentVerifyHandler,
    connection_client,
    login,
    response_json,
    run_server,
)
from tests_support.connection_api import clear_env as clear_env

_PAYLOAD = {
    "facility_code": "NH-7H2K9M4QXP",
    "facility_token": "eft_v1.test.secret",
    "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
}


def test_connection_test_verifies_without_persisting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EnrollmentVerifyHandler.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), EnrollmentVerifyHandler)
    thread = run_server(server)
    try:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}")
        client = connection_client(tmp_path, monkeypatch)
        login(client)

        response = client.post("/api/v1/connection/test", json=_PAYLOAD)

        assert response.status_code == 200
        body = response_json(response)
        assert body["ok"] is True
        assert body["facility_id"] == "87d79f24-b32f-49a3-b534-19f0af7d9135"
        assert client.get("/api/v1/connection").json()["enrolled"] is False
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_connection_test_requires_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = connection_client(tmp_path, monkeypatch).post(
        "/api/v1/connection/test", json=_PAYLOAD
    )
    assert response.status_code == 401


def test_connection_test_rejects_legacy_url_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = connection_client(tmp_path, monkeypatch)
    login(client)
    response = client.post(
        "/api/v1/connection/test",
        json={**_PAYLOAD, "config_url": "http://attacker.invalid"},
    )
    assert response.status_code == 422
