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


def test_connection_test_classifies_rejected_token_without_leaking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EnrollmentVerifyHandler.reset()
    EnrollmentVerifyHandler.response_status = 403
    server = ThreadingHTTPServer(("127.0.0.1", 0), EnrollmentVerifyHandler)
    thread = run_server(server)
    try:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}")
        client = connection_client(tmp_path, monkeypatch)
        login(client)

        response = client.post(
            "/api/v1/connection/test",
            json={
                "facility_code": "NH-7H2K9M4QXP",
                "facility_token": "revoked-secret-token",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            },
        )

        assert response.status_code == 200
        assert response_json(response)["error_class"] == "auth"
        assert "revoked-secret-token" not in response.text
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
