from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import final

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.runtime_settings.store import RuntimeSettingsStore
from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledWorkerConfig
from worker.runtime.config import (
    BackendWorkerConfigPayload,
    JsonObject,
    LiveClipExportPolicy,
    WorkerConfigLkgStore,
    pull_worker_config_poll,
)

DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}
RELAY_HEADERS = {"X-Edge-Relay-Token": "relay-token"}


@pytest.fixture(autouse=True)
def _migrated_compact_database(tmp_path: Path) -> None:
    migrate_database(tmp_path / "catalog.sqlite3")


def _login(client: TestClient) -> None:
    assert client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN).status_code == 204


def _app(tmp_path: Path):
    app = create_app(lifespan=no_lifespan)
    database = tmp_path / "catalog.sqlite3"
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(database)
    app.state.runtime_settings_store = RuntimeSettingsStore(database)
    return app


def test_fresh_store_defaults_clip_export_off_at_version_zero(tmp_path: Path) -> None:
    setting = RuntimeSettingsStore(tmp_path / "catalog.sqlite3").get()

    assert setting.clip_export_enabled is False
    assert setting.version == 0


def test_store_persists_changes_and_only_advances_version_when_value_changes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalog.sqlite3"
    store = RuntimeSettingsStore(database)

    enabled = store.set_clip_export_enabled(True)
    unchanged = store.set_clip_export_enabled(True)
    disabled = store.set_clip_export_enabled(False)
    reopened = RuntimeSettingsStore(database).get()

    assert (enabled.clip_export_enabled, enabled.version) == (True, 1)
    assert unchanged == enabled
    assert (disabled.clip_export_enabled, disabled.version) == (False, 2)
    assert reopened == disabled


def test_runtime_settings_api_requires_dashboard_auth_and_round_trips(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        denied_get = client.get("/api/v1/runtime-settings")
        denied_put = client.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": True, "expected_version": 0},
        )
        _login(client)
        initial = client.get("/api/v1/runtime-settings")
        saved = client.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": True, "expected_version": 0},
        )
        fetched = client.get("/api/v1/runtime-settings")

    assert denied_get.status_code == 401
    assert denied_put.status_code == 401
    assert initial.json() == {"clip_export_enabled": False, "version": 0}
    assert saved.json() == {"clip_export_enabled": True, "version": 1}
    assert fetched.json() == saved.json()


def test_runtime_settings_put_requires_strict_boolean_and_expected_version(
    tmp_path: Path,
) -> None:
    with TestClient(_app(tmp_path)) as client:
        _login(client)
        malformed = [
            client.put(
                "/api/v1/runtime-settings",
                json={"clip_export_enabled": value, "expected_version": 0},
            )
            for value in (1, 0, "true", "false", None)
        ]
        missing_version = client.put("/api/v1/runtime-settings", json={"clip_export_enabled": True})

    assert all(response.status_code == 422 for response in malformed)
    assert missing_version.status_code == 422


def test_two_clients_use_optimistic_concurrency_and_noop_does_not_advance_version(
    tmp_path: Path,
) -> None:
    with TestClient(_app(tmp_path)) as first, TestClient(_app(tmp_path)) as second:
        _login(first)
        _login(second)
        first_version = first.get("/api/v1/runtime-settings").json()["version"]
        second_version = second.get("/api/v1/runtime-settings").json()["version"]

        saved = first.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": True, "expected_version": first_version},
        )
        conflict = second.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": False, "expected_version": second_version},
        )
        noop = first.put(
            "/api/v1/runtime-settings",
            json={"clip_export_enabled": True, "expected_version": saved.json()["version"]},
        )

    assert saved.json() == {"clip_export_enabled": True, "version": 1}
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == {
        "code": "runtime_settings_version_conflict",
        "current": {"clip_export_enabled": True, "version": 1},
    }
    assert noop.status_code == 200
    assert noop.json() == saved.json()


def test_worker_config_projects_effective_runtime_setting_without_restart_change(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    app.state.restart_epoch = 7
    app.state.config_version = 11
    app.state.pulled_config = PulledWorkerConfig(
        config_version=11,
        restart_epoch=7,
        night_window=None,
        cameras=(),
    )
    app.state.runtime_settings_store.set_clip_export_enabled(True)

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)

    assert response.status_code == 200
    assert response.json()["clip_export_enabled"] is True
    assert response.json()["clip_export_version"] == 1
    assert response.json()["config_version"] == 11
    assert response.json()["restart_epoch"] == 7


def test_old_worker_payload_and_old_lkg_default_clip_export_off(tmp_path: Path) -> None:
    raw: JsonObject = {
        "registry_version": 1,
        "config_version": 1,
        "restart_epoch": 0,
        "cameras": [],
    }
    parsed = BackendWorkerConfigPayload.model_validate(raw)
    worker = parsed.to_worker_config("http://relay.test", "relay-token")
    store = WorkerConfigLkgStore(tmp_path / "worker-state.sqlite3")
    assert store.save(raw, parsed.directive)
    stored = store.load()
    assert stored is not None
    reparsed = BackendWorkerConfigPayload.model_validate(stored.payload)

    assert worker.clip_export_enabled is False
    assert worker.clip_export_version == 0
    assert reparsed.clip_export_enabled is False
    assert reparsed.clip_export_version == 0


@pytest.mark.parametrize("value", (1, 0, "true", "false", None))
def test_worker_config_rejects_non_boolean_clip_export_values(value: object) -> None:
    with pytest.raises(ValidationError):
        BackendWorkerConfigPayload.model_validate(
            {
                "registry_version": 1,
                "config_version": 1,
                "restart_epoch": 0,
                "clip_export_enabled": value,
                "clip_export_version": 1,
                "cameras": [],
            }
        )


def test_config_poll_carries_live_policy_without_changing_restart_directive() -> None:
    payload = {
        "registry_version": 3,
        "config_version": 9,
        "restart_epoch": 2,
        "clip_export_enabled": True,
        "clip_export_version": 4,
        "cameras": [],
    }

    @final
    class Response:
        status: int = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exception_type, exception, traceback

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    polled = pull_worker_config_poll(
        "http://relay.test",
        "relay-token",
        urlopen=lambda _request, _timeout: Response(),
    )

    assert polled is not None
    assert polled.clip_export_enabled is True
    assert polled.clip_export_version == 4
    assert (polled.restart_config.config_version, polled.restart_config.restart_epoch) == (9, 2)


def test_live_policy_applies_new_versions_and_ignores_stale_snapshots() -> None:
    policy = LiveClipExportPolicy()

    assert policy.apply(enabled=True, version=2) is True
    assert policy.apply(enabled=False, version=1) is False
    assert policy.enabled() is True
    assert policy.version == 2
    assert policy.apply(enabled=False, version=3) is True
    assert policy.enabled() is False


def test_new_worker_payload_threads_runtime_setting_and_version() -> None:
    parsed = BackendWorkerConfigPayload.model_validate(
        {
            "registry_version": 1,
            "config_version": 1,
            "restart_epoch": 0,
            "clip_export_enabled": True,
            "clip_export_version": 4,
            "cameras": [],
        }
    )

    worker = parsed.to_worker_config("http://relay.test", "relay-token")

    assert worker.clip_export_enabled is True
    assert worker.clip_export_version == 4


def test_status_exposes_effective_runtime_setting_and_version(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.state.runtime_settings_store.set_clip_export_enabled(True)

    response = TestClient(app).get("/api/v1/status")

    assert response.status_code == 200
    assert response.json()["runtime_settings"] == {
        "clip_export_enabled": True,
        "version": 1,
    }
    assert response.json()["runtime"]["clip_export_applied"] == {
        "enabled": None,
        "version": None,
        "freshness": "unknown",
    }
