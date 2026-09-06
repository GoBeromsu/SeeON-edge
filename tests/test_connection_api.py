"""Dashboard GET/PUT API tests for connection settings."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from backend.app.features.connection.store import API_BACKEND_BASE_URL_ENV
from tests_support.connection_api import (
    EnrollmentVerifyHandler,
    response_json,
    run_server,
)
from tests_support.connection_api import connection_client as _client
from tests_support.connection_api import connection_store as _store
from tests_support.connection_api import login as _login

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
    _ = store.save(
        {
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
            "facility_token": "super-secret-facility-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 3,
        }
    )
    client = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.get("/api/v1/connection")

    assert response.status_code == 200
    body = response_json(response)
    assert body["facility_code"] == "NH-7H2K9M4QXP"
    assert body["client_installation_ref"] == "aa83ea3f-6e5f-4f45-a401-fb36c38835b6"
    assert body["facility_id"] == "87d79f24-b32f-49a3-b534-19f0af7d9135"
    assert body["edge_installation_id"] == "d17e0eb8-cb81-4d8e-a427-dfe690518f2b"
    assert body["enrollment_generation"] == 3
    assert body["facility_token_set"] is True
    assert body["facility_token_masked"] == "****oken"
    assert body["enrolled"] is True
    assert body["reachable"] is None
    assert body["last_ok_at"] is None
    assert "super-secret-facility-token" not in response.text


def test_get_connection_unconfigured_when_nothing_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    _login(client)
    response = client.get("/api/v1/connection")
    assert response.status_code == 200
    body = response_json(response)
    assert body["configured"] is False
    assert body["facility_token_set"] is False
    assert body["facility_token_masked"] is None


def test_get_connection_heartbeat_relay_absent_state_reads_disabled_with_nulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # no_lifespan test apps never populate backend_heartbeat_relay_state.
    client = _client(tmp_path, monkeypatch)
    _login(client)
    response = client.get("/api/v1/connection")
    assert response.status_code == 200
    assert response_json(response)["heartbeat_relay"] == {
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
    _login(client)
    client.app.state.backend_heartbeat_relay_task = object()  # loop "configured"
    client.app.state.backend_heartbeat_relay_state = HeartbeatRelayState(
        last_error_class="auth", last_success_at="2026-01-01T00:00:00.000Z"
    )

    response = client.get("/api/v1/connection")

    assert response.status_code == 200
    assert response_json(response)["heartbeat_relay"] == {
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
    _login(client)
    client.app.state.backend_heartbeat_relay_task = None  # disabled via env kill-switch
    client.app.state.backend_heartbeat_relay_state = HeartbeatRelayState()

    response = client.get("/api/v1/connection")

    assert response.status_code == 200
    body = response_json(response)["heartbeat_relay"]
    assert isinstance(body, dict)
    assert body["enabled"] is False
    assert body["last_error_class"] is None
    assert body["detail"] is None


# --------------------------------------------------------------------------
# PUT /connection enrollment
# --------------------------------------------------------------------------


def test_put_connection_requires_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.put(
        "/api/v1/connection",
        json={
            "facility_code": "NH-7H2K9M4QXP",
            "facility_token": "eft_v1.token.secret",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
        },
    )
    assert response.status_code == 401


def test_put_connection_verifies_persists_and_publishes_one_bundle_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EnrollmentVerifyHandler.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), EnrollmentVerifyHandler)
    thread = run_server(server)
    try:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}")
        client = _client(tmp_path, monkeypatch)
        _login(client)

        response = client.put(
            "/api/v1/connection",
            json={
                "facility_code": "NH-7H2K9M4QXP",
                "facility_token": "eft_v1.token.secret",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            },
        )

        assert response.status_code == 200
        body = response_json(response)
        assert body["facility_id"] == "87d79f24-b32f-49a3-b534-19f0af7d9135"
        assert body["enrollment_generation"] == 3
        assert body["facility_token_masked"] == "****cret"
        assert "eft_v1.token.secret" not in response.text
        assert EnrollmentVerifyHandler.received_paths == ["/api/v1/edge/enrollments/verify"]
        assert EnrollmentVerifyHandler.received_auth == ["Bearer eft_v1.token.secret"]
        assert EnrollmentVerifyHandler.received_bodies == [
            {
                "schemaVersion": 1,
                "facilityCode": "NH-7H2K9M4QXP",
                "clientInstallationRef": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            }
        ]
        bundle = client.app.state.backend_client_bundle
        assert bundle.enrollment_generation == 3
        assert bundle.facility_id == "87d79f24-b32f-49a3-b534-19f0af7d9135"
        assert bundle.ingest_client.bearer_token == "eft_v1.token.secret"
        assert bundle.evidence_client.bearer_token == "eft_v1.token.secret"
        assert not hasattr(bundle, "camera_mapper")
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_put_connection_failed_verification_keeps_prior_enrollment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    EnrollmentVerifyHandler.reset()
    EnrollmentVerifyHandler.response_status = 403
    server = ThreadingHTTPServer(("127.0.0.1", 0), EnrollmentVerifyHandler)
    thread = run_server(server)
    try:
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}")
        store = _store(tmp_path, monkeypatch)
        _ = store.save(
            {
                "facility_code": "NH-7H2K9M4QXP",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
                "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
                "facility_token": "prior-token",
                "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
                "enrollment_generation": 2,
            }
        )
        client = _client(tmp_path, monkeypatch)
        _login(client)

        response = client.put(
            "/api/v1/connection",
            json={
                "facility_code": "NH-7H2K9M4QXP",
                "facility_token": "revoked-token",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            },
        )

        assert response.status_code == 403
        assert "revoked-token" not in response.text
        assert store.load().facility_token == "prior-token"
        assert store.load().enrollment_generation == 2
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
