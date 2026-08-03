"""API-level tests for GET/PUT /api/v1/detection-settings (see
backend/app/features/detection_settings/router.py).

Covers dashboard-auth enforcement, request-body validation (HH:MM format,
window-required-when-mode-is-window, start != end), PUT normalizing stray
start/end away for mode=always, persistence round-tripping through GET, and
the fallback chain a fresh (never-saved) domain uses: live pulled
detection_windows/night_window first, then on=true/mode=always."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.features.detection_settings.store import DetectionSettingsStore
from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig

DASHBOARD_LOGIN = {"username": "admin", "password": "admin"}

_DEFAULT_DOMAINS = {
    "fall": {"on": True, "mode": "always", "start": None, "end": None},
    "bed_exit": {"on": True, "mode": "always", "start": None, "end": None},
}


def _login(client: TestClient) -> None:
    response = client.post("/api/v1/auth/session", json=DASHBOARD_LOGIN)
    assert response.status_code == 204


def _app_with_store(tmp_path):
    app = create_app(lifespan=no_lifespan)
    app.state.detection_settings_store = DetectionSettingsStore(tmp_path / "catalog.sqlite3")
    return app


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ML_API_DETECTION_TZ", raising=False)


def test_get_falls_back_to_on_true_mode_always_when_nothing_pulled_or_stored(tmp_path) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        _login(client)
        response = client.get("/api/v1/detection-settings")

    assert response.status_code == 200
    assert response.json() == {"domains": _DEFAULT_DOMAINS}


def test_get_falls_back_to_the_live_pulled_detection_window_when_nothing_stored(
    tmp_path,
) -> None:
    app = _app_with_store(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=None,
        cameras=(),
        detection_windows={
            "fall": PulledNightWindow(start="08:00", end="20:00", tz="UTC"),
            "bed_exit": PulledNightWindow(start="22:00", end="06:00", tz="UTC"),
        },
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/detection-settings")

    assert response.status_code == 200
    assert response.json()["domains"]["fall"] == {
        "on": True,
        "mode": "window",
        "start": "08:00",
        "end": "20:00",
    }
    assert response.json()["domains"]["bed_exit"] == {
        "on": True,
        "mode": "window",
        "start": "22:00",
        "end": "06:00",
    }


def test_get_falls_back_to_the_deprecated_night_window_alias_for_bed_exit(tmp_path) -> None:
    app = _app_with_store(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=PulledNightWindow(start="21:00", end="05:00", tz="UTC"),
        cameras=(),
        detection_windows={},
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/detection-settings")

    assert response.json()["domains"]["bed_exit"] == {
        "on": True,
        "mode": "window",
        "start": "21:00",
        "end": "05:00",
    }
    # fall has no pulled window at all -> ambient default.
    assert response.json()["domains"]["fall"] == _DEFAULT_DOMAINS["fall"]


def test_put_persists_and_a_subsequent_get_reflects_exactly_what_was_saved(tmp_path) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        _login(client)
        put_response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "window", "start": "08:00", "end": "20:00"},
                    "bed_exit": {"on": False, "mode": "always"},
                }
            },
        )
        get_response = client.get("/api/v1/detection-settings")

    assert put_response.status_code == 200
    expected = {
        "domains": {
            "fall": {"on": True, "mode": "window", "start": "08:00", "end": "20:00"},
            "bed_exit": {"on": False, "mode": "always", "start": None, "end": None},
        }
    }
    assert put_response.json() == expected
    assert get_response.json() == expected


def test_put_normalizes_stray_start_end_to_null_when_mode_is_always(tmp_path) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        _login(client)
        response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {
                        "on": True,
                        "mode": "always",
                        "start": "08:00",
                        "end": "20:00",
                    },
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )

    assert response.status_code == 200
    assert response.json()["domains"]["fall"] == {
        "on": True,
        "mode": "always",
        "start": None,
        "end": None,
    }


def test_put_once_saved_overrides_the_live_pulled_fallback_on_a_later_get(tmp_path) -> None:
    app = _app_with_store(tmp_path)
    app.state.pulled_config = PulledWorkerConfig(
        config_version=1,
        restart_epoch=1,
        night_window=None,
        cameras=(),
        detection_windows={"fall": PulledNightWindow(start="08:00", end="20:00", tz="UTC")},
    )

    with TestClient(app) as client:
        _login(client)
        client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": False, "mode": "always"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )
        response = client.get("/api/v1/detection-settings")

    assert response.json()["domains"]["fall"] == {
        "on": False,
        "mode": "always",
        "start": None,
        "end": None,
    }


@pytest.mark.parametrize(
    "domain_payload",
    [
        {"on": True, "mode": "window", "start": "8:00", "end": "20:00"},
        {"on": True, "mode": "window", "start": "08:00", "end": "24:00"},
        {"on": True, "mode": "window", "start": "aa:bb", "end": "20:00"},
    ],
)
def test_put_rejects_malformed_hhmm_times(tmp_path, domain_payload: dict[str, object]) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        _login(client)
        response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": domain_payload,
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )

    assert response.status_code == 422


def test_put_requires_start_and_end_when_mode_is_window(tmp_path) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        _login(client)
        response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "window"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )

    assert response.status_code == 422


def test_put_rejects_equal_start_and_end(tmp_path) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        _login(client)
        response = client.put(
            "/api/v1/detection-settings",
            json={
                "domains": {
                    "fall": {"on": True, "mode": "window", "start": "08:00", "end": "08:00"},
                    "bed_exit": {"on": True, "mode": "always"},
                }
            },
        )

    assert response.status_code == 422


def test_detection_settings_routes_require_a_dashboard_session(tmp_path) -> None:
    with TestClient(_app_with_store(tmp_path)) as client:
        get_response = client.get("/api/v1/detection-settings")
        put_response = client.put(
            "/api/v1/detection-settings",
            json={"domains": _DEFAULT_DOMAINS},
        )

    assert get_response.status_code == 401
    assert put_response.status_code == 401
