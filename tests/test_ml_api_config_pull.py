from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Self

import pytest
from fastapi.testclient import TestClient

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


def _set_pull_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_FACILITY_ID", "facility-pulled")
    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "http://backend:3000/api/v1/ml-config/")


def test_backend_config_pull_seeds_inventory_and_worker_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str | None, float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        captured.append(
            (request.full_url, request.get_header("Authorization"), timeout)
        )
        return FakeHTTPResponse(_backend_config())

    _set_pull_env(monkeypatch)
    monkeypatch.setenv("EDGE_FACILITY_TOKEN", "facility-token")
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
        assert app.state.camera_inventory == {
            "cam-pulled-1": {
                "camera_id": "cam-pulled-1",
                "facility_id": "facility-pulled",
                "resident_id": None,
            },
            "cam-pulled-2": {
                "camera_id": "cam-pulled-2",
                "facility_id": "facility-pulled",
                "resident_id": None,
            },
        }

        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "registry_version": 0,
        "cameras": [
            {
                "camera_id": "cam-pulled-1",
                "facility_id": "facility-pulled",
                "rtsp_url": "rtsp://camera/101",
            }
        ],
        "config_version": 7,
        "restart_epoch": 0,
        "night_window": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"},
        "detection_windows": {
            "bed_exit": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"}
        },
    }


def test_backend_detection_windows_present_ignores_legacy_night_window_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan-file payload-level rule: once ``detectionWindows`` is present at
    all, it is the sole authority for every domain and legacy ``nightWindow``
    is ignored entirely -- even for domains the map doesn't mention. This is
    what makes an operator clearing bed_exit's window in the dashboard stick,
    instead of a stale ``nightWindow`` resurrecting it."""

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

    with TestClient(create_app()) as client:
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


def test_backend_detection_windows_absent_still_uses_legacy_night_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``detectionWindows`` is absent entirely, the legacy single
    ``nightWindow`` field still applies to bed_exit (compat fallback)."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(_backend_config())

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
        response = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    assert response.json()["detection_windows"] == {
        "bed_exit": {"start": "21:00", "end": "06:00", "tz": "Asia/Seoul"}
    }


def test_backend_config_pull_survives_invalid_window_and_still_populates_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed window value for one domain (start == end here) must not
    crash the whole pull: it fails open to ALWAYS for that domain (logged to
    stderr) while cameras and other domains' windows still populate."""

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

    with TestClient(create_app()) as client:
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
            "camera_id": "cam-pulled-1",
            "facility_id": "facility-pulled",
            "rtsp_url": "rtsp://camera/101",
        }
    ]
    assert "bed_exit" not in body["detection_windows"]
    assert body["detection_windows"]["fall"] == {"start": "22:00", "end": "05:00", "tz": "UTC"}


def test_backend_config_pull_failure_keeps_env_inventory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError("boom")

    _set_pull_env(monkeypatch)
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
        assert client.app.state.camera_inventory == {
            "cam-env": {
                "camera_id": "cam-env",
                "facility_id": "facility-env",
                "resident_id": "resident-env",
            }
        }


def test_config_and_restart_require_token_and_restart_reflects_live_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_pull_env(monkeypatch)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout: FakeHTTPResponse(_backend_config()),
    )

    with TestClient(create_app()) as client:
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
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv(
        "API_CAMERA_INVENTORY",
        json.dumps(
            [
                {"camera_id": "cam-1", "facility_id": "facility-1"},
                {"camera_id": "cam-2", "facility_id": "facility-1"},
            ]
        ),
    )

    with TestClient(create_app()) as client:
        client.app.state.backend_ingest_client = FakeBackendIngestClient()
        with_version = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "cam-1", "facility_id": "facility-1", "config_version": 7},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        without_version = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "cam-2", "facility_id": "facility-1"},
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
    monkeypatch: pytest.MonkeyPatch,
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

    with TestClient(create_app()) as client:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeHTTPResponse(_backend_config())  # boot success -> version 7
        raise urllib.error.URLError("backend down")  # refresh fails

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(create_app()) as client:
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
        previous_inventory = app.state.camera_inventory.copy()

        assert refresh_backend_config(app) is False
        assert app.state.pulled_config.config_version == 7
        assert app.state.camera_inventory == previous_inventory
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
            previous_inventory = app.state.camera_inventory.copy()
            shutdown_started_at = time.monotonic()

        assert time.monotonic() - shutdown_started_at < BACKEND_CONFIG_SHUTDOWN_WAIT_SEC + 0.5
        assert app.state.backend_config_refresh_task is None
        assert app.state.pulled_config == previous_config
        assert app.state.config_version == 7
        assert app.state.camera_inventory == previous_inventory

        release_refresh.set()
        assert refresh_finished.wait(timeout=2)
        assert app.state.pulled_config == previous_config
        assert app.state.config_version == 7
        assert app.state.camera_inventory == previous_inventory
    finally:
        release_refresh.set()


def test_successful_refresh_retries_pending_backend_mappings(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A reachable backend converges mapping_pending registry records to their
    canonical backend camera id via the refresh owner."""
    from backend.app.features.cameras.store import CameraRegistryStore
    from backend.app.shared.backend_mapping import BackendCameraMapper, MappingResult

    _set_pull_env(monkeypatch)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: FakeHTTPResponse(_backend_config())
    )

    class FakeMapper(BackendCameraMapper):
        def __init__(self) -> None:
            super().__init__(endpoint="http://backend/api/v1/edge/cameras", token="tok")
            self.calls: list[dict[str, str]] = []

        def put_mapping(self, *, edge_camera_ref: str, label: str, space_id: str) -> MappingResult:
            self.calls.append(
                {"edge_camera_ref": edge_camera_ref, "label": label, "space_id": space_id}
            )
            return MappingResult(
                backend_camera_id="backend-cam-9", pending=False, reachable=True
            )

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
        mapper = FakeMapper()
        client.app.state.backend_camera_mapper = mapper

        assert refresh_backend_config(client.app) is True

        record = store.get("local-uuid-9")
        assert record is not None
        assert record["backend_camera_id"] == "backend-cam-9"
        assert record["mapping_pending"] is False
        assert mapper.calls == [
            {"edge_camera_ref": "local-uuid-9", "label": "Room 9", "space_id": "space-9"}
        ]


def test_backend_camera_mapper_accepts_canonical_edge_facility_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EDGE_FACILITY_TOKEN (the name lifespan + compose use) must configure the
    mapper; the legacy aliases stay accepted."""
    from backend.app.shared.backend_mapping import BackendCameraMapper

    for name in (
        "EDGE_FACILITY_TOKEN",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("API_BACKEND_EVENTS_URL", "http://backend:8080/api/v1/events")
    assert BackendCameraMapper.from_env().configured is False

    monkeypatch.setenv("EDGE_FACILITY_TOKEN", "facility-token")
    mapper = BackendCameraMapper.from_env()
    assert mapper.configured is True
    assert mapper.token == "facility-token"
    assert mapper.endpoint == "http://backend:8080/api/v1/edge/cameras"


def test_backend_detection_windows_populate_per_domain_map_and_bed_exit_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend's ``detectionWindows`` (domain -> window|null) becomes
    ``PulledWorkerConfig.detection_windows``; a null entry for a domain is
    dropped rather than stored, and "bed_exit" also populates the deprecated
    ``night_window`` alias for old workers."""

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

    with TestClient(create_app()) as client:
        pulled = client.app.state.pulled_config
        assert pulled.detection_windows == {
            "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="Asia/Seoul"),
            "fall": PulledNightWindow(start="22:00", end="05:00", tz="UTC"),
        }
        assert pulled.night_window == PulledNightWindow(
            start="21:00", end="06:00", tz="Asia/Seoul"
        )

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
