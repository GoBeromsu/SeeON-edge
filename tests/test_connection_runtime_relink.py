"""Runtime relink: apply_connection_settings() rebuilds backend ingest/evidence
clients from ConnectionSettingsStore without a process restart (story G002).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import lifespan as lifespan_module
from backend.app.features.connection import store as connection_store_module
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    API_BACKEND_CONFIG_URL_ENV,
    API_BACKEND_EVENTS_URL_ENV,
    API_CONNECTION_SETTINGS_PATH_ENV,
    ConnectionSettingsStore,
)
from backend.app.lifespan import (
    API_EDGE_RELAY_TOKEN_ENV,
    apply_connection_settings,
)
from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledWorkerConfig


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        API_CONNECTION_SETTINGS_PATH_ENV,
        API_BACKEND_EVENTS_URL_ENV,
        API_BACKEND_CONFIG_URL_ENV,
        "API_FACILITY_ID",
        "EDGE_FACILITY_TOKEN",
        API_EDGE_RELAY_TOKEN_ENV,
        "API_CAMERA_INVENTORY",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings_store(monkeypatch: pytest.MonkeyPatch) -> ConnectionSettingsStore:
    monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "http://backend.example")
    central_database = connection_store_module.EDGE_DATABASE_PATH
    monkeypatch.setattr(
        ConnectionSettingsStore,
        "from_env",
        classmethod(lambda cls: cls(central_database)),
    )
    store = ConnectionSettingsStore.from_env()
    assert store.path == central_database
    assert store.path.name == "edge.sqlite3"
    return store


def test_boot_time_fixture_injection_still_survives_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _settings_store(monkeypatch)
    store.save(
        {
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
            "facility_token": "persisted-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 1,
        }
    )

    sentinel = object()
    app = create_app()
    app.state.backend_ingest_client = sentinel  # assigned before lifespan runs

    with TestClient(app):
        # The hasattr guard in _configure_backend_ingest() must keep the
        # pre-assigned fixture client intact at boot, even though a saved
        # connection-settings file is present and would otherwise rebuild it.
        assert app.state.backend_ingest_client is sentinel


def test_complete_enrollment_restores_one_generation_bundle_on_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _settings_store(monkeypatch)
    store.save(
        {
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
            "facility_token": "persisted-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 4,
        }
    )
    app = create_app()
    with TestClient(app):
        bundle = app.state.backend_client_bundle
        assert bundle.facility_id == "87d79f24-b32f-49a3-b534-19f0af7d9135"
        assert bundle.enrollment_generation == 4
        assert bundle.ingest_client.bearer_token == "persisted-token"
        assert bundle.evidence_client.bearer_token == "persisted-token"
        assert not hasattr(bundle, "camera_mapper")


def test_relink_publishes_a_whole_new_generation_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _settings_store(monkeypatch)
    app = create_app(lifespan=no_lifespan)
    for generation, token in ((1, "token-one"), (2, "token-two")):
        store.save(
            {
                "facility_code": "NH-7H2K9M4QXP",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
                "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
                "facility_token": token,
                "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
                "enrollment_generation": generation,
            }
        )
        apply_connection_settings(app)
        bundle = app.state.backend_client_bundle
        assert bundle.enrollment_generation == generation
        assert {
            bundle.ingest_client.bearer_token,
            bundle.evidence_client.bearer_token,
        } == {token}
        assert not hasattr(bundle, "camera_mapper")


def test_config_refresh_discards_result_when_relink_changes_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _settings_store(monkeypatch)
    app = create_app(lifespan=no_lifespan)

    def save_generation(generation: int, token: str) -> None:
        _ = store.save(
            {
                "facility_code": "NH-7H2K9M4QXP",
                "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
                "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
                "facility_token": token,
                "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
                "enrollment_generation": generation,
            }
        )
        apply_connection_settings(app)

    save_generation(1, "token-one")

    def relink_during_fetch(_bundle: object, restart_epoch: int) -> PulledWorkerConfig:
        save_generation(2, "token-two")
        return PulledWorkerConfig(
            config_version=7,
            restart_epoch=restart_epoch,
            night_window=None,
            cameras=(),
        )

    monkeypatch.setattr(lifespan_module, "_fetch_backend_config", relink_during_fetch)

    assert lifespan_module.refresh_backend_config(app) is False
    assert getattr(app.state, "pulled_config", None) is None
    assert app.state.backend_client_bundle.enrollment_generation == 2


def test_corrupt_enrollment_store_fails_closed_despite_identity_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _settings_store(monkeypatch)
    store.path.write_bytes(b"not-a-sqlite-database")
    app = create_app(lifespan=no_lifespan)

    apply_connection_settings(app)

    assert getattr(app.state, "backend_client_bundle", None) is None
    assert app.state.backend_configured is False
