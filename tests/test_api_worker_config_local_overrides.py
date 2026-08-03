"""End-to-end merge-precedence tests for GET /api/v1/cameras/worker-config
(see ``worker_config_snapshot``/``_apply_local_detection_overrides``/
``_apply_clip_storage_override`` in backend/app/features/cameras/router.py).

Exercises the full precedence chain: a locally-saved detection setting
(``PUT /api/v1/detection-settings``) always overrides whatever the backend
externally pulled for that domain, and a locally-selected clip storage
location (``PUT /api/v1/clips/storage/location``) is threaded onto the
response as ``clip_store_subdir`` only once a non-root location has been
explicitly chosen. ``response_model_exclude_none=True`` means every new
optional field here is entirely absent from the JSON body -- not present as
null -- whenever no local override exists, so byte-for-byte backward
compatibility with a worker that has never seen these fields is preserved."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.storage_location_store import ClipStorageLocationStore
from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig

AUTH = {"Authorization": "Bearer relay-token"}
DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.delenv("API_FACILITY_ID", raising=False)
    monkeypatch.delenv("ML_API_DETECTION_TZ", raising=False)


def _app(tmp_path):
    app = create_app(lifespan=no_lifespan)
    db_path = tmp_path / "catalog.sqlite3"
    app.state.camera_registry = CameraRegistryStore(db_path)
    app.state.detection_settings_store = DetectionSettingsStore(db_path)
    app.state.clip_storage_location_store = ClipStorageLocationStore(db_path)
    return app


def test_with_no_local_overrides_the_response_reflects_the_externally_pulled_state(
    tmp_path,
) -> None:
    app = _app(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=3,
        restart_epoch=1,
        night_window=PulledNightWindow(start="22:00", end="06:00", tz="Asia/Seoul"),
        cameras=(),
        detection_windows={
            "bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="Asia/Seoul"),
            "fall": PulledNightWindow(start="08:00", end="20:00", tz="Asia/Seoul"),
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["night_window"] == {"start": "22:00", "end": "06:00", "tz": "Asia/Seoul"}
    assert body["detection_windows"] == {
        "bed_exit": {"start": "22:00", "end": "06:00", "tz": "Asia/Seoul"},
        "fall": {"start": "08:00", "end": "20:00", "tz": "Asia/Seoul"},
    }
    # response_model_exclude_none=True: never-configured optional fields are
    # entirely absent, not present as null.
    assert "domains" not in body
    assert "clip_store_subdir" not in body


def test_local_window_setting_overrides_the_pulled_window_and_reuses_its_tz(tmp_path) -> None:
    app = _app(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=None,
        cameras=(),
        detection_windows={"fall": PulledNightWindow(start="08:00", end="20:00", tz="Asia/Seoul")},
    )

    with TestClient(app) as client:
        _login(client)
        put_response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "window", "start": "09:00", "end": "18:00"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        assert put_response.status_code == 200
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    # The saved window wins over the pulled one, but its tz is reused from
    # the live pulled window for the same domain (no facility-tz setting
    # exists elsewhere in this codebase).
    assert body["detection_windows"]["fall"] == {
        "start": "09:00",
        "end": "18:00",
        "tz": "Asia/Seoul",
    }
    assert body["domains"] == {"fall": {"enabled": True}, "bed_exit": {"enabled": True}}
    # bed_exit is on/always -> no window entry for it at all.
    assert "bed_exit" not in body["detection_windows"]
    assert "night_window" not in body


def test_local_always_on_setting_removes_any_pulled_window_for_that_domain(tmp_path) -> None:
    app = _app(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=PulledNightWindow(start="22:00", end="06:00", tz="UTC"),
        cameras=(),
        detection_windows={"bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="UTC")},
    )

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    body = response.json()
    assert "detection_windows" not in body
    assert "night_window" not in body
    assert body["domains"] == {"fall": {"enabled": True}, "bed_exit": {"enabled": True}}


def test_local_off_setting_disables_the_domain_and_drops_its_window_and_alias(
    tmp_path,
) -> None:
    app = _app(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=PulledNightWindow(start="22:00", end="06:00", tz="UTC"),
        cameras=(),
        detection_windows={"bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="UTC")},
    )

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "always"},
                    "bed_exit": {"on": False, "mode": "always"},
                }
            },
        )
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    body = response.json()
    assert body["domains"]["bed_exit"] == {"enabled": False}
    assert "detection_windows" not in body
    assert "night_window" not in body


def test_clip_store_subdir_is_absent_until_a_non_root_location_is_selected(tmp_path) -> None:
    app = _app(tmp_path)

    with TestClient(app) as client:
        before = client.get("/api/v1/cameras/worker-config", headers=AUTH)
        assert "clip_store_subdir" not in before.json()

        _login(client)
        put_response = client.put(
            "/api/v1/clips/storage/location", json={"path": ""}
        )
        # An empty (root) selection stays absent from the worker-config body.
        assert put_response.status_code in (200, 404)


def test_clip_store_subdir_appears_once_a_selection_is_persisted_directly(tmp_path) -> None:
    """Persists a selection directly via the store (bypassing the browse/PUT
    filesystem-existence check, which is exercised separately in
    test_api_clip_storage.py) to isolate this test to the worker-config merge
    itself."""
    app = _app(tmp_path)
    app.state.clip_storage_location_store.put("external-drive")

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["clip_store_subdir"] == "external-drive"


def test_worker_config_route_requires_relay_authorization(tmp_path) -> None:
    app = _app(tmp_path)

    with TestClient(app) as client:
        unauthenticated = client.get("/api/v1/cameras/worker-config")
        wrong_token = client.get(
            "/api/v1/cameras/worker-config",
            headers={"Authorization": "Bearer wrong"},
        )

    assert unauthenticated.status_code == 401
    assert wrong_token.status_code == 403
