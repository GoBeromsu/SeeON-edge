"""Dashboard GET/PUT API tests for connection settings."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from backend.app.lifespan import EDGE_FACILITY_TOKEN_ENV
from shared.events.edge_ingest_client import EdgeIngestClient
from tests_support.connection_api import clear_env as clear_env
from tests_support.connection_api import connection_client as _client
from tests_support.connection_api import connection_store as _store
from tests_support.connection_api import login as _login
from tests_support.connection_api import response_json

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
            "events_url": "http://backend.example/api/v1/edge/events",
            "config_url": "http://backend.example/api/v1/edge/config",
            "facility_id": "facility-1",
            "facility_token": "super-secret-facility-token",
        }
    )
    client = _client(tmp_path, monkeypatch)
    _login(client)

    response = client.get("/api/v1/connection")

    assert response.status_code == 200
    body = response_json(response)
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
    _login(client)

    response = client.put(
        "/api/v1/connection",
        json={
            "events_url": "http://backend.example/api/v1/edge/events",
            "facility_token": "tok-1",
        },
    )

    assert response.status_code == 200
    body = response_json(response)
    assert body["events_url"] == "http://backend.example/api/v1/edge/events"
    assert body["facility_token_set"] is True
    assert body["configured"] is True

    # Same app instance -- apply_connection_settings() must have rebuilt the
    # ingest client synchronously within the request, no restart required.
    ingest = cast(EdgeIngestClient, client.app.state.backend_ingest_client)
    assert isinstance(ingest, EdgeIngestClient)
    assert ingest.events_url == "http://backend.example/api/v1/edge/events"
    assert ingest.bearer_token == "tok-1"


def test_put_connection_explicit_null_does_not_restore_env_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EDGE_FACILITY_TOKEN_ENV, "env-seeded-token")
    client = _client(tmp_path, monkeypatch)
    _login(client)
    saved = client.put(
        "/api/v1/connection",
        json={
            "events_url": "http://backend.example/events",
            "facility_token": "explicit-token",
        },
    )
    saved_body = response_json(saved)
    assert saved_body["facility_token_masked"] == "****oken"

    cleared = client.put(
        "/api/v1/connection",
        json={"facility_token": None},
    )
    cleared_body = response_json(cleared)
    assert cleared.status_code == 200
    assert cleared_body["facility_token_masked"] is None
    assert cleared_body["facility_token_set"] is False


def test_put_connection_omitted_field_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    _login(client)
    _ = client.put(
        "/api/v1/connection",
        json={"events_url": "http://backend.example/events", "facility_id": "facility-1"},
    )

    response = client.put(
        "/api/v1/connection",
        json={"events_url": "http://backend-2.example/events"},
    )

    assert response.status_code == 200
    response_body = response_json(response)
    assert response_body["facility_id"] == "facility-1"
    assert response_body["events_url"] == "http://backend-2.example/events"


def test_put_connection_rejects_invalid_url_with_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    _login(client)
    response = client.put(
        "/api/v1/connection",
        json={"events_url": "not-a-url"},
    )
    assert response.status_code == 422
    assert "events_url" in response.text


def test_put_connection_rejects_non_http_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    _login(client)
    response = client.put(
        "/api/v1/connection",
        json={"config_url": "ftp://backend.example/config"},
    )
    assert response.status_code == 422
