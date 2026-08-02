"""Runtime relink: apply_connection_settings() rebuilds backend ingest/evidence
clients from ConnectionSettingsStore without a process restart (story G002).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
    apply_connection_settings,
)
from backend.app.main import create_app, no_lifespan
from shared.events.edge_ingest_client import BackendEvidenceClient, EdgeIngestClient

_ALERT_PAYLOAD = {
    "event_type": "bed-exit",
    "probability": 0.87,
    "detected_at": "2026-06-25T12:00:00.000Z",
    "camera_id": "camera-1",
    "facility_id": "facility-1",
    "evidence": {"domain": "night-bed-exit", "clip_id": "clip-123"},
}


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        API_CONNECTION_SETTINGS_PATH_ENV,
        API_BACKEND_EVENTS_URL_ENV,
        API_BACKEND_CONFIG_URL_ENV,
        API_FACILITY_ID_ENV,
        EDGE_FACILITY_TOKEN_ENV,
        API_EDGE_RELAY_TOKEN_ENV,
        "API_CAMERA_INVENTORY",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConnectionSettingsStore:
    monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection_settings.json"))
    return ConnectionSettingsStore.from_env()


def test_apply_connection_settings_rebuilds_clients_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _settings_store(tmp_path, monkeypatch)

    with TestClient(create_app()) as client:
        app = client.app
        # No env, no saved file yet -- boot leaves both clients unset.
        assert getattr(app.state, "backend_ingest_client", None) is None
        assert getattr(app.state, "backend_evidence_client", None) is None

        store.save(
            {
                "events_url": "http://backend.example/api/v1/edge/events",
                "facility_token": "tok-1",
            }
        )
        apply_connection_settings(app)  # same app instance -- no restart

        ingest = app.state.backend_ingest_client
        evidence = app.state.backend_evidence_client
        assert isinstance(ingest, EdgeIngestClient)
        assert isinstance(evidence, BackendEvidenceClient)
        assert ingest.events_url == "http://backend.example/api/v1/edge/events"
        assert ingest.bearer_token == "tok-1"
        assert evidence.events_url == "http://backend.example/api/v1/edge/events"
        assert evidence.bearer_token == "tok-1"


def test_apply_connection_settings_relinks_to_a_new_saved_events_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _settings_store(tmp_path, monkeypatch)

    with TestClient(create_app()) as client:
        app = client.app
        store.save({"events_url": "http://backend-1.example/api/v1/edge/events"})
        apply_connection_settings(app)
        first_ingest = app.state.backend_ingest_client
        first_evidence = app.state.backend_evidence_client

        store.save({"events_url": "http://backend-2.example/api/v1/edge/events"})
        apply_connection_settings(app)
        second_ingest = app.state.backend_ingest_client
        second_evidence = app.state.backend_evidence_client

        assert second_ingest is not first_ingest
        assert second_evidence is not first_evidence
        assert second_ingest.events_url == "http://backend-2.example/api/v1/edge/events"
        assert second_evidence.events_url == "http://backend-2.example/api/v1/edge/events"


def test_apply_connection_settings_clears_clients_and_relay_returns_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _settings_store(tmp_path, monkeypatch)
    store.save(
        {"events_url": "http://backend.example/api/v1/edge/events", "facility_token": "tok-1"}
    )

    # no_lifespan pattern from tests/test_api_ingest_relay.py: wire the app
    # state a boot would normally set, without running the real lifespan.
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {
        "camera-1": {"camera_id": "camera-1", "facility_id": "facility-1", "resident_id": None},
    }
    apply_connection_settings(app)
    assert isinstance(app.state.backend_ingest_client, EdgeIngestClient)
    assert isinstance(app.state.backend_evidence_client, BackendEvidenceClient)

    store.save({"events_url": None})
    apply_connection_settings(app)

    assert getattr(app.state, "backend_ingest_client", None) is None
    assert getattr(app.state, "backend_evidence_client", None) is None

    response = TestClient(app).post(
        "/api/v1/relay/alerts",
        json=_ALERT_PAYLOAD,
        headers={"X-Edge-Relay-Token": "relay-token"},
    )
    assert response.status_code == 503


def test_boot_time_fixture_injection_still_survives_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _settings_store(tmp_path, monkeypatch)
    store.save({"events_url": "http://backend.example/api/v1/edge/events"})

    sentinel = object()
    app = create_app()
    app.state.backend_ingest_client = sentinel  # assigned before lifespan runs

    with TestClient(app) as client:
        # The hasattr guard in _configure_backend_ingest() must keep the
        # pre-assigned fixture client intact at boot, even though a saved
        # connection-settings file is present and would otherwise rebuild it.
        assert client.app.state.backend_ingest_client is sentinel
