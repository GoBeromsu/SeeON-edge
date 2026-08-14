from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Self

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.lifespan import BACKEND_CONFIG_SHUTDOWN_WAIT_SEC, refresh_backend_config
from backend.app.main import create_app
from contracts.worker_config import CONFIG_VERSION_KEY, RESTART_EPOCH_KEY, PulledNightWindow


class FakeBackendIngestClient:
    def __init__(self) -> None:
        self.heartbeats = 0

    def send_alert(self, **kwargs) -> bool:
        return True

    def send_heartbeat(self) -> bool:
        self.heartbeats += 1
        return True


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "API_EDGE_RELAY_TOKEN",
        "API_CAMERA_INVENTORY",
        "API_FACILITY_ID",
        "API_BACKEND_CONFIG_URL",
        "API_BACKEND_EVENTS_URL",
        "EDGE_FACILITY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _backend_config() -> dict[str, object]:
    return {
        "configVersion": 7,
        "nightWindow": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"},
        "cameras": [
            {
                "id": "cam-pulled-1",
                "spaceId": "room-101",
                "label": "Room 101",
                "rtspUrl": "rtsp://camera/101",
                "online": True,
            },
            {
                "id": "cam-pulled-2",
                "spaceId": "room-102",
                "label": "Room 102",
                "rtspUrl": None,
                "online": False,
            },
        ],
    }


def _set_pull_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path | None = None) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "http://backend:3000/api/v1/ml-config/")
    # facility_id is DB-only; seed connection_settings so ml-config pull runs.
    # A backend_client_bundle (and therefore refresh_backend_config) is only
    # published once every enrollment field is present -- not just the
    # ml-config pull's own config_url/facility_id/facility_token -- so a
    # complete enrollment row is required here even though this module's
    # tests only exercise the config-pull half of that bundle.
    from backend.app.features.connection.store import (
        API_CONNECTION_SETTINGS_PATH_ENV,
        ConnectionSettingsStore,
    )

    path = (tmp_path or Path("/tmp")) / "connection-settings-pull.sqlite3"
    if tmp_path is None:
        import tempfile

        path = Path(tempfile.mkdtemp()) / "connection-settings-pull.sqlite3"
    monkeypatch.setenv(API_CONNECTION_SETTINGS_PATH_ENV, str(path))
    ConnectionSettingsStore(path).save(
        {
            "events_url": "http://backend:3000/api/v1/events",
            "config_url": "http://backend:3000/api/v1/ml-config/",
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "facility-pulled",
            "facility_token": "facility-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 1,
        }
    )


def _dashboard_camera_registry(tmp_path: Path) -> CameraRegistryStore:
    """A dashboard camera registry with one registered camera (issue #33: the
    registry, not a backend ml-config pull, is the sole roster source), so
    tests whose real subject is config refresh / detection windows / token
    auth -- not the roster itself -- get an available (non-empty) worker
    config from `/relay/config` (require_available=True) rather than a 503.
    """
    store = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="dashboard-camera",
        label="Dashboard Camera",
        rtsp_url="rtsp://dashboard/stream",
        space_id=None,
        status="online",
    )
    return store


def test_backend_config_pull_applies_metadata_not_camera_roster(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ml-config pull keeps detection windows/config_version only.

    Pulled cameras must not become a local inventory authority; worker-config
    stays registry-only (empty when registry is empty).
    """
    captured: list[tuple[str, str | None, float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        captured.append((request.full_url, request.get_header("Authorization"), timeout))
        return FakeHTTPResponse(_backend_config())

    _set_pull_env(monkeypatch, tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
        app = client.app
        assert captured == [
            (
                "http://backend:3000/api/v1/ml-config/facility-pulled",
                "Bearer facility-token",
                0.5,
            )
        ]
        assert app.state.config_version == 7
        assert not hasattr(app.state, "camera_inventory")

        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    assert worker_config.json() == {
        "registry_version": 0,
        "clip_export_enabled": False,
        "clip_export_version": 0,
        "cameras": [],
        "config_version": 7,
        "restart_epoch": 0,
        "night_window": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"},
        "detection_windows": {"bed_exit": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"}},
    }
    assert relay_config.status_code == 503


def test_backend_detection_windows_present_ignores_legacy_night_window_entirely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Plan-file payload-level rule: once ``detectionWindows`` is present at
    all, it is the sole authority for every domain and legacy ``nightWindow``
    is ignored entirely -- even for domains the map doesn't mention. This is
    what makes an operator clearing bed_exit's window in the dashboard stick,
    instead of a stale ``nightWindow`` resurrecting it.

    A dashboard camera is registered so /relay/config (require_available=True)
    stays available; the pulled camera in the fake backend payload is there
    only to exercise the pull parser and must not seed the roster (issue #33)."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            {
                "configVersion": 7,
                # Legacy alias present alongside the map: must be ignored.
                "nightWindow": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"},
                "detectionWindows": {
                    "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
                },
                "cameras": [
                    {
                        "id": "cam-pulled-1",
                        "spaceId": "room-101",
                        "label": "Room 101",
                        "rtspUrl": "rtsp://camera/101",
                        "online": True,
                    }
                ],
            }
        )

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["detection_windows"] == {
        "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
    }
    # bed_exit is not in the map, so the map's authority means ALWAYS (24/7)
    # for bed_exit -- NOT the legacy nightWindow value. The response omits
    # night_window entirely when it's None (response_model_exclude_none).
    assert "bed_exit" not in body["detection_windows"]
    assert "night_window" not in body
    # The pulled camera must never seed the roster (issue #33).
    assert body["cameras"] == [
        {
            "camera_id": "dashboard-camera",
            "rtsp_url": "rtsp://dashboard/stream",
        }
    ]


def test_backend_detection_windows_absent_still_uses_legacy_night_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``detectionWindows`` is absent entirely, the legacy single
    ``nightWindow`` field still applies to bed_exit (compat fallback).

    A dashboard camera is registered so /relay/config (require_available=True)
    stays available (see issue #33: the pulled camera in the fake backend
    payload must not seed the roster)."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(_backend_config())

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    assert response.json()["detection_windows"] == {
        "bed_exit": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"}
    }


def test_backend_config_pull_survives_invalid_window_and_never_populates_cameras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed window value for one domain (start == end here) must not
    crash the whole pull: it fails open to ALWAYS for that domain (logged to
    stderr) while other domains' windows still populate. The pulled camera
    must never seed the worker-config roster (issue #33) -- a registered
    dashboard camera is the only thing that appears in ``cameras``."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            {
                "configVersion": 9,
                "detectionWindows": {
                    "bed_exit": {"start": "09:00", "end": "09:00", "tz": "UTC"},
                    "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
                },
                "cameras": [
                    {
                        "id": "cam-pulled-1",
                        "spaceId": "room-101",
                        "label": "Room 101",
                        "rtspUrl": "rtsp://camera/101",
                        "online": True,
                    }
                ],
            }
        )

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        assert client.app.state.pulled_config is not None
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["config_version"] == 9
    assert body["cameras"] == [
        {
            "camera_id": "dashboard-camera",
            "rtsp_url": "rtsp://dashboard/stream",
        }
    ]
    assert "bed_exit" not in body["detection_windows"]
    assert body["detection_windows"]["fall"] == {"start": "22:00", "end": "05:00", "tz": "UTC"}


def test_backend_config_pull_failure_does_not_create_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("boom")

    _set_pull_env(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "API_CAMERA_INVENTORY",
        json.dumps(
            [
                {
                    "camera_id": "cam-env",
                    "facility_id": "facility-env",
                    "resident_id": "resident-env",
                }
            ]
        ),
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
        assert client.app.state.pulled_config is None
        assert not hasattr(client.app.state, "camera_inventory")


def test_config_and_restart_require_token_and_restart_reflects_live_epoch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_pull_env(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout: FakeHTTPResponse(_backend_config()),
    )
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        assert client.get("/api/v1/relay/config").status_code == 401
        assert (
            client.get(
                "/api/v1/relay/config",
                headers={"X-Edge-Relay-Token": "wrong"},
            ).status_code
            == 403
        )
        assert client.post("/api/v1/relay/restart").status_code == 401
        assert (
            client.post(
                "/api/v1/relay/restart",
                headers={"X-Edge-Relay-Token": "wrong"},
            ).status_code
            == 403
        )

        restart = client.post(
            "/api/v1/relay/restart",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert restart.status_code == 202
    assert restart.json() == {RESTART_EPOCH_KEY: 1}
    assert config.json()[RESTART_EPOCH_KEY] == 1
    assert config.json()[CONFIG_VERSION_KEY] == 7


def test_heartbeat_config_version_surfaces_in_status_and_remains_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    for camera_id in ("cam-1", "cam-2"):
        store.create(
            camera_id=camera_id,
            label=camera_id,
            rtsp_url=f"rtsp://example/{camera_id}",
            space_id=None,
            status="online",
        )

    with TestClient(create_app()) as client:
        client.app.state.camera_registry = store
        client.app.state.backend_ingest_client = FakeBackendIngestClient()
        with_version = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "cam-1", "facility_id": "local", "config_version": 7},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        without_version = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "cam-2", "facility_id": "local"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        status = client.get("/api/v1/status")

    assert with_version.status_code == 202
    assert without_version.status_code == 202
    assert status.json()["cameras"]["cam-1"]["config_version"] == 7
    assert status.json()["cameras"]["cam-2"]["config_version"] is None


def test_config_returns_503_when_backend_config_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Relay token + env inventory present, but NO backend pull configured, so
    # app.state.pulled_config stays None. /config MUST signal UNAVAILABLE (503)
    # rather than emit an empty 200 that the worker would persist as LKG,
    # clobbering a valid last-known-good config.
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv(
        "API_CAMERA_INVENTORY",
        json.dumps([{"camera_id": "cam-1", "facility_id": "facility-1"}]),
    )

    with TestClient(create_app()) as client:
        assert client.app.state.pulled_config is None
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 503


def test_config_refresh_reflects_backend_change_without_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configs = [
        _backend_config(),  # boot -> config_version 7
        {  # first GET re-pull -> changed backend config, no restart
            "configVersion": 8,
            "nightWindow": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
            "cameras": [
                {
                    "id": "cam-pulled-1",
                    "spaceId": "room-101",
                    "label": "Room 101",
                    "rtspUrl": "rtsp://camera/101",
                    "online": True,
                }
            ],
        },
    ]
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        payload = configs[min(calls["n"], len(configs) - 1)]
        calls["n"] += 1
        return FakeHTTPResponse(payload)

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        assert client.app.state.config_version == 7
        initial = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        assert initial.json()["config_version"] == 7
        assert calls["n"] == 1

        assert refresh_backend_config(client.app) is True
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    assert response.json()["config_version"] == 8
    assert response.json()["night_window"] == {
        "start": "22:00",
        "end": "05:00",
        "tz": "Asia/Seoul",
    }


def test_config_refresh_preserves_last_good_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeHTTPResponse(_backend_config())  # boot success -> version 7
        raise urllib.error.URLError("backend down")  # refresh fails

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        assert client.app.state.config_version == 7
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    # A transient refresh failure preserves the last-good config (200, not 503/blanked).
    assert response.status_code == 200
    assert response.json()["config_version"] == 7


def test_config_refresh_rejects_partial_roster_and_preserves_last_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = [
        _backend_config(),
        {
            "configVersion": 8,
            "cameras": [
                {
                    "id": "cam-pulled-1",
                    "spaceId": "room-101",
                    "label": "Room 101",
                    "rtspUrl": "rtsp://camera/101",
                    "online": True,
                },
                {"id": "malformed-camera"},
            ],
        },
    ]
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        payload = configs[calls["count"]]
        calls["count"] += 1
        return FakeHTTPResponse(payload)

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
        app = client.app
        previous_roster = app.state.backend_roster.copy()
        previous_config = app.state.pulled_config

        assert refresh_backend_config(app) is False
        assert app.state.pulled_config is previous_config
        assert app.state.pulled_config.config_version == 7
        assert not hasattr(app.state, "camera_inventory")
        assert app.state.backend_roster == {
            "config_version": 7,
            "received_at": previous_roster["received_at"],
            "stale": True,
        }


def test_shutdown_waits_for_inflight_backend_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeHTTPResponse(_backend_config())
        refresh_started.set()
        assert release_refresh.wait(timeout=3)
        return FakeHTTPResponse(_backend_config())

    _set_pull_env(monkeypatch)
    monkeypatch.setenv("API_BACKEND_CONFIG_REFRESH_SEC", "1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    release_timer: threading.Timer | None = None
    started_at = time.monotonic()

    try:
        with TestClient(app):
            assert refresh_started.wait(timeout=2)
            release_timer = threading.Timer(0.05, release_refresh.set)
            release_timer.start()
    finally:
        release_refresh.set()
        if release_timer is not None:
            release_timer.join(timeout=1)

    assert time.monotonic() - started_at >= 0.05
    assert calls["count"] == 2
    assert app.state.backend_config_refresh_task is None


def test_shutdown_bounds_late_refresh_and_discards_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    refresh_finished = threading.Event()
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeHTTPResponse(_backend_config())
        refresh_started.set()
        assert release_refresh.wait(timeout=5)
        refresh_finished.set()
        return FakeHTTPResponse(_backend_config() | {"configVersion": 8})

    _set_pull_env(monkeypatch)
    monkeypatch.setenv("API_BACKEND_CONFIG_REFRESH_SEC", "1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()

    try:
        with TestClient(app):
            assert refresh_started.wait(timeout=2)
            previous_config = app.state.pulled_config
            shutdown_started_at = time.monotonic()

        assert time.monotonic() - shutdown_started_at < BACKEND_CONFIG_SHUTDOWN_WAIT_SEC + 0.5
        assert app.state.backend_config_refresh_task is None
        assert app.state.pulled_config == previous_config
        assert app.state.config_version == 7
        assert not hasattr(app.state, "camera_inventory")

        release_refresh.set()
        assert refresh_finished.wait(timeout=2)
        assert app.state.pulled_config == previous_config
        assert app.state.config_version == 7
        assert not hasattr(app.state, "camera_inventory")
    finally:
        release_refresh.set()


def test_successful_refresh_resumes_roster_sync_without_per_camera_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A backend connectivity recovery (refresh_backend_config transitioning
    unreachable -> reachable) resumes pending roster-sync work via
    ``resume_camera_roster_after_connectivity`` -> ``topology_retry_coordinator
    .trigger(force=True, refresh=True)`` -- not a per-camera mapping PUT any
    more (there is no ``_map_backend`` retry path left to exercise; see
    BLOCKER 2 in the merge notes). That trigger only ever reaches the backend
    once every registered camera has an explicit floor/room reference; this
    camera has neither, so the coordinator's readiness check stops the
    snapshot before any outbound call.

    Pinning the two BLOCKER-2 properties this test actually protects:
    (2) unmapped camera retained rather than dropped -- refresh leaves the
    pending record exactly as it was (still local-id-addressed, still
    unmapped), it does not get spuriously mapped or removed; and (1)
    canonical id resolution -- the worker-config projection falls back to the
    record's own local id while it is unmapped.
    """
    from backend.app.features.cameras.store import CameraRegistryStore
    from backend.app.features.connection.store import ConnectionSettingsStore

    _set_pull_env(monkeypatch, tmp_path)
    # Roster sync resolves its client off the connection store's events_url
    # (via backend_client_bundle), same as ml-config pull. _set_pull_env only
    # seeds config_url; seed events_url here too so a client is constructible
    # (even though, per the docstring above, no call ends up going out).
    ConnectionSettingsStore(tmp_path / "connection-settings-pull.sqlite3").save(
        {"events_url": "http://backend/api/v1/events"}
    )

    mapping_calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        if request.full_url == "http://backend/api/v1/edge/cameras":
            mapping_calls.append(json.loads(request.data.decode("utf-8")))
            return FakeHTTPResponse({"cameraId": "backend-cam-9"})
        return FakeHTTPResponse(_backend_config())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
        store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
        store.create(
            camera_id="local-uuid-9",
            label="Room 9",
            rtsp_url="rtsp://camera/9",
            space_id="space-9",
            status="online",
            backend_camera_id=None,
            mapping_pending=True,
        )
        client.app.state.camera_registry = store

        assert refresh_backend_config(client.app) is True

        record = store.get("local-uuid-9")
        assert record is not None
        # (2) Retained, not dropped: refresh does not touch this record --
        # no legacy per-camera mapping endpoint exists any more to call.
        assert record["backend_camera_id"] is None
        assert record["mapping_pending"] is True
        assert mapping_calls == []

        # (1) Canonical id resolution: unmapped falls back to the local id.
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        assert worker_config.status_code == 200
        camera = worker_config.json()["cameras"][0]
        assert camera["camera_id"] == "local-uuid-9"


def test_backend_camera_mapper_has_no_env_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: ``BackendCameraMapper`` must not grow a ``from_env()``
    (or similarly named) constructor again. Facility identity -- including the
    mapper's bearer token -- is DB-only (``ConnectionSettingsStore``,
    dashboard-entered); the only construction site is ``lifespan.py``, which
    passes DB-sourced values explicitly. An env-seeded constructor previously
    existed here, was dead in production, and was removed because it was the
    only way EDGE_FACILITY_TOKEN (or its legacy aliases) could reintroduce a
    token through the environment/compose/Git."""
    from backend.app.shared.backend_mapping import BackendCameraMapper

    assert not hasattr(BackendCameraMapper, "from_env")

    for name in (
        "EDGE_FACILITY_TOKEN",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
    ):
        monkeypatch.setenv(name, "should-be-ignored")

    # Constructing directly with no token (as an unenrolled edge would be,
    # since nothing DB-sourced is passed) must not somehow pick up the env
    # values set above -- there is no code path left that could, but this
    # pins the observable behavior.
    mapper = BackendCameraMapper(endpoint="http://backend:8080/api/v1/edge/cameras", token=None)
    assert mapper.configured is False
    assert mapper.token is None


def test_backend_detection_windows_populate_per_domain_map_and_bed_exit_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The backend's ``detectionWindows`` (domain -> window|null) becomes
    ``PulledWorkerConfig.detection_windows``; a null entry for a domain is
    dropped rather than stored, and "bed_exit" also populates the deprecated
    ``night_window`` alias for old workers.

    A dashboard camera is registered so /relay/config (require_available=True)
    stays available; the pulled camera in the fake backend payload must not
    seed the roster (issue #33)."""

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            {
                "configVersion": 9,
                "detectionWindows": {
                    "bed_exit": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"},
                    "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
                    "wander": None,
                },
                "cameras": [
                    {
                        "id": "cam-1",
                        "spaceId": "room-1",
                        "label": "Room 1",
                        "rtspUrl": "rtsp://camera/1",
                        "online": True,
                    }
                ],
            }
        )

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = create_app()
    app.state.camera_registry = _dashboard_camera_registry(tmp_path)

    with TestClient(app) as client:
        pulled = client.app.state.pulled_config
        assert pulled.detection_windows == {
            "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="Asia/Seoul"),
            "fall": PulledNightWindow(start="22:00", end="05:00", tz="UTC"),
        }
        assert pulled.night_window == PulledNightWindow(start="21:00", end="06:00", tz="Asia/Seoul")

        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["night_window"] == {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"}
    assert body["detection_windows"] == {
        "bed_exit": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"},
        "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
    }
    # The pulled camera ("cam-1") must never seed the roster (issue #33).
    assert body["cameras"] == [
        {
            "camera_id": "dashboard-camera",
            "rtsp_url": "rtsp://dashboard/stream",
        }
    ]


def test_backend_detection_windows_absent_falls_back_to_legacy_night_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the backend has not rolled out ``detectionWindows`` yet, the
    legacy single ``nightWindow`` field still maps to "bed_exit"."""

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(_backend_config())

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
        pulled = client.app.state.pulled_config
        assert pulled.detection_windows == {
            "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="Asia/Seoul"),
        }
