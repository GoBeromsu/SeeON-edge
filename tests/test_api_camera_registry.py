from __future__ import annotations

import json
import time
import urllib.error
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Self, TypedDict

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.cameras.router import _authorize_worker
from backend.app.features.cameras.store import CameraRegistryStore, public_camera
from backend.app.features.status.heartbeat_store import get_heartbeat_store
from backend.app.lifespan import apply_connection_settings, refresh_backend_config
from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledCameraConfig, PulledNightWindow, PulledWorkerConfig
from worker.adapters.decode.cpu_av.adapter import CpuAvAdapter
from worker.adapters.decode.nvdec_cuvid.adapter import NvdecCuvidAdapter
from worker.runtime.config import (
    CameraRuntimeConfig,
    JsonObject,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
)
from worker.runtime.ingest_composition import decoder_for

AUTH = {"Authorization": "Bearer relay-token"}
REPO_ROOT = Path(__file__).resolve().parents[1]

# Dashboard auth now always resolves to a session store (persisted file > env
# > the built-in admin/admin default, see backend/app/shared/dashboard_auth.py).
# A worker relay bearer token is still valid for the dedicated worker-config
# and relay/config routes (a separate auth mechanism, see _authorize_worker in
# backend/app/features/cameras/router.py), but every other /cameras route now
# requires a real dashboard session cookie -- these tests log in as the
# zero-config default before touching any dashboard-only route.


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


class CapturedBackendCall(TypedDict):
    url: str
    method: str
    authorization: str | None
    facility_id: str | None
    body: dict[str, object]
    timeout: float


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "API_EDGE_RELAY_TOKEN",
        "API_FACILITY_ID",
        "API_BACKEND_EDGE_CAMERAS_URL",
        "API_BACKEND_URL",
        "API_BACKEND_EVENTS_URL",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
        "ML_EDGE_VERSION",
        "ML_WORKER_STATE_DIR",
        "RELAY_TOKEN",
        "RELAY_URL",
        "ML_API_WORKER_PROBE_ORIGIN",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_camera_registry_crud_masks_rtsp_versions_and_worker_config_auth(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("API_BACKEND_EDGE_CAMERAS_URL", "http://backend/api/v1/edge/cameras")
    monkeypatch.setenv("API_FACILITY_TOKEN", "facility-token")
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")
    monkeypatch.setenv(
        "API_CONNECTION_SETTINGS_PATH", str(tmp_path / "connection-settings.sqlite3")
    )
    from backend.app.features.connection.store import ConnectionSettingsStore

    # Roster sync is driven by TopologyClient off ConnectionSettingsStore (the
    # dashboard-entered store, not the env-only API_FACILITY_TOKEN above), so
    # this saved row is what makes the coordinator's client_provider() resolve
    # a principal at all -- see BLOCKER 2 in the merge notes for why the sync
    # still stops at "pending" without ever reaching the backend.
    ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3").save(
        {
            "events_url": "http://backend/api/v1/events",
            "facility_id": "facility-1",
            "facility_token": "facility-token",
        }
    )
    captured: list[CapturedBackendCall] = []
    probe_calls: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        if request.full_url == "http://worker.local:8090/probe":
            probe_calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "relay_token": request.headers.get("X-edge-relay-token"),
                    "body": json.loads(request.data.decode("utf-8")),
                    "timeout": timeout,
                }
            )
            return FakeHTTPResponse({"ok": False, "error_class": "timeout"})
        decoded_body = json.loads(request.data.decode("utf-8"))
        assert isinstance(decoded_body, dict)
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "authorization": request.headers.get("Authorization"),
                "facility_id": request.headers.get("X-facility-id"),
                "body": {str(key): value for key, value in decoded_body.items()},
                "timeout": timeout,
            }
        )
        return FakeHTTPResponse({"cameraId": "backend-camera-1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://user:secret@camera.local:8554/live",
                "space_id": "space-1",
                # The fake worker probe below always reports failure; force
                # registration so this test can still exercise the full CRUD
                # flow (masking, versioning, delete) against a real record.
                "force_register": True,
            },
        )
        assert created.status_code == 201
        camera = created.json()
        assert camera["label"] == "Lobby"
        assert camera["rtsp_url_masked"] == "rtsp://***:***@redacted-camera:8554/live"
        camera_json = json.dumps(camera)
        assert "secret" not in camera_json
        assert "camera.local" not in camera_json
        # Creation is unmapped-by-construction now: there is no synchronous
        # per-camera PUT any more (that path was replaced by the async
        # topology-snapshot roster sync triggered below), so the record is
        # addressed by its own local id and retained rather than dropped
        # while mapping is pending -- see BLOCKER 2 in the merge notes.
        assert camera["backend_camera_id"] is None
        assert isinstance(camera["id"], str) and camera["id"]
        assert camera["status"] == "offline"

        listed = client.get("/api/v1/cameras", headers=AUTH).json()
        assert listed["registry_version"] == 1
        assert len(listed["cameras"]) == 1
        listed_camera = listed["cameras"][0]
        # The POST response never populates `sync` (avoids a BackgroundTask
        # race against this pinned-shape assertion); GET does. Roster sync now
        # publishes a complete topology snapshot (see BLOCKER 2 in the merge
        # notes), not a per-camera PUT -- and a snapshot can't go out until
        # every camera has an explicit floor/room reference, which this
        # camera (created with only a bare space_id) does not have. So the
        # coordinator stops at "pending" with that readiness detail instead
        # of ever reaching the backend; this is "unmapped camera retained
        # rather than dropped" made concrete -- the record stays listed and
        # addressable by its local id while sync legitimately cannot proceed.
        sync = listed_camera["sync"]
        assert sync["status"] == "pending"
        assert sync["error_class"] is None
        assert sync["detail"] == "모든 카메라에 명시적인 층/방/카메라 참조를 배정해야 합니다."
        assert sync["last_ok_at"] is None
        assert {**listed_camera, "sync": None} == camera

        assert client.get("/api/v1/cameras/worker-config").status_code == 401
        assert (
            client.get(
                "/api/v1/cameras/worker-config",
                headers={"X-Edge-Relay-Token": "wrong"},
            ).status_code
            == 403
        )
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"Authorization": "Bearer relay-token"},
        )
        assert worker_config.status_code == 200
        assert worker_config.json() == {
            "registry_version": 1,
            "clip_export_enabled": False,
            "clip_export_version": 0,
            "cameras": [
                {
                    "camera_id": camera["id"],
                    "space_id": "space-1",
                    "rtsp_url": "rtsp://user:secret@camera.local:8554/live",
                }
            ],
        }

        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        assert relay_config.status_code == 200
        assert relay_config.json() == worker_config.json()

        patched = client.patch(
            f"/api/v1/cameras/{camera['id']}",
            headers=AUTH,
            json={"label": "Lobby North"},
        )
        assert patched.status_code == 200
        assert client.get("/api/v1/cameras", headers=AUTH).json()["registry_version"] == 2

        tested = client.post(f"/api/v1/cameras/{camera['id']}/test", headers=AUTH)
        assert tested.status_code == 200
        assert tested.json() == {"ok": False, "error_class": "timeout"}

        deleted = client.delete(f"/api/v1/cameras/{camera['id']}", headers=AUTH)
        assert deleted.status_code == 204
        after_delete = client.get("/api/v1/cameras", headers=AUTH).json()
        # v1 create, v2 patch, v3 POST /test now persists its probe result
        # (Task 3), v4 delete.
        assert after_delete == {"registry_version": 4, "cameras": []}

    # No per-camera PUT to /api/v1/edge/cameras ever went out: roster sync now
    # only pushes a complete topology snapshot (a different endpoint, driven
    # by TopologyClient, not BackendCameraMapper), and that snapshot itself
    # never left the coordinator because this camera has no floor/room
    # reference -- see the "pending" sync assertion above. `captured` here
    # only records calls to the legacy edge-cameras URL, so it stays empty.
    assert captured == []
    assert probe_calls == [
        {
            "url": "http://worker.local:8090/probe",
            "method": "POST",
            "relay_token": "relay-token",
            "body": {"rtsp_url": "rtsp://user:secret@camera.local:8554/live"},
            "timeout": 5.0,
        },
        {
            "url": "http://worker.local:8090/probe",
            "method": "POST",
            "relay_token": "relay-token",
            "body": {"rtsp_url": "rtsp://user:secret@camera.local:8554/live"},
            "timeout": 5.0,
        },
    ]


def test_list_cameras_survives_conflict_paused_topology_sync(tmp_path) -> None:
    """Regression test for the 2026-08-12 prod incident: while topology roster
    sync is paused with ``pause_reason=CONFLICT`` (a 409 from the backend's
    topology-snapshots endpoint), ``current_retry_result`` reports
    ``error_class="conflict"`` (see backend/app/features/connection/
    topology_retry_result.py). The ``CameraSyncStatus`` response model's
    ``error_class`` literal used to omit ``"conflict"``, so GET /api/v1/cameras
    500'd with a pydantic validation error (``Input should be 'unreachable',
    'timeout', 'auth' or 'unconfigured'``) for every camera any time roster
    sync was conflict-paused.
    """
    from backend.app.features.cameras.edge_topology_sync_state import (
        EdgeTopologySyncStateStore,
        TopologyPauseReason,
    )
    from backend.app.features.cameras.topology_client import TopologyPaused
    from backend.app.features.connection.topology_retry_coordinator import (
        TopologyRetryCoordinator,
    )
    from contracts.edge_provisioning_v1 import MachinePrincipal

    registry_path = tmp_path / "catalog.sqlite3"
    registry = CameraRegistryStore(registry_path)
    registry.create_floor(edge_ref="floor-1", name="1F", order_index=1)
    registry.create_room(edge_ref="room-101", floor_edge_ref="floor-1", name="101")
    registry.create(
        label="Lobby",
        rtsp_url="rtsp://user:secret@camera.local:8554/live",
        space_id=None,
        status="online",
        edge_ref="camera-1",
        room_edge_ref="room-101",
    )

    class _ConflictClient:
        principal = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)

        def put(self, pending):
            return TopologyPaused(TopologyPauseReason.CONFLICT, 409)

        def refresh_server_revision(self) -> int | None:
            return None

        def confirm(self, snapshot_id, confirmation):
            raise AssertionError("not exercised by this test")

    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(registry_path), lambda: _ConflictClient()
    )
    # Drive the coordinator directly (mirrors tests/test_topology_retry_coordinator.py)
    # so the persisted state actually has pause_reason=CONFLICT before GET /cameras.
    paused = coordinator.trigger(force=True, now_epoch=100.0)
    assert paused.status == "failed"
    assert paused.error_class == "conflict"

    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = registry
    app.state.topology_retry_coordinator = coordinator

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert len(body["cameras"]) == 1
    sync = body["cameras"][0]["sync"]
    assert sync["status"] == "failed"
    assert sync["error_class"] == "conflict"


def test_create_camera_is_store_only_and_never_credentialed_from_env(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera creation must work off dashboard state alone, and the environment
    must not be able to supply the facility token.

    The env below deliberately carries every legacy facility-token alias plus a
    facility id. None of them may reach an outbound request: identity is owned by
    ``ConnectionSettingsStore`` (dashboard-entered), which here is intentionally
    left *unenrolled*. So creation must still succeed -- an operator adds cameras
    before enrollment completes -- while nothing is pushed anywhere.

    The camera keeps ``backend_camera_id is None`` and is addressed by its local
    id. The Edge never self-assigns a backend id: cameras reach the Hub through
    the topology snapshot flow (``preview``/``confirm``), which is what publishes
    ``edge_ref``. Persisting the Hub-assigned id back onto the Edge record is a
    tracked follow-up, not a create-time PUT -- the per-camera
    ``PUT /v1/edge/cameras`` path this test used to assert was replaced by
    complete-topology snapshots (see ``backend/app/shared/backend_mapping.py``).
    """
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv(
        "API_CONNECTION_SETTINGS_PATH", str(tmp_path / "connection-settings.sqlite3")
    )
    for alias in (
        "EDGE_FACILITY_TOKEN",
        "API_FACILITY_TOKEN",
        "API_BACKEND_FACILITY_TOKEN",
        "API_EDGE_FACILITY_TOKEN",
    ):
        monkeypatch.setenv(alias, "token-from-env-must-be-ignored")
    monkeypatch.setenv("API_FACILITY_ID", "facility-from-env-must-be-ignored")
    monkeypatch.setenv("API_BACKEND_EDGE_CAMERAS_URL", "http://backend/api/v1/edge/cameras")

    from backend.app.features.connection.store import ConnectionSettingsStore

    # Unenrolled on purpose: no facility_id, no facility_token in the DB.
    unenrolled = ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3").load()
    assert unenrolled.facility_token is None
    assert unenrolled.facility_id is None

    outbound: list[str] = []

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        outbound.append(f"{request.get_method()} {request.full_url}")
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://user:secret@camera.local:8554/live",
                "space_id": "space-1",
                "force_register": True,
            },
        )
        assert created.status_code == 201
        camera = created.json()
        assert camera["backend_camera_id"] is None
        assert camera["id"] and camera["id"] != "facility-from-env-must-be-ignored"
        assert camera["rtsp_url_masked"] == "rtsp://***:***@redacted-camera:8554/live"

    # The env tokens above configured nothing, so no camera push was attempted.
    assert not [call for call in outbound if "/v1/edge/cameras" in call]


def test_worker_config_uses_registry_first_and_metadata_from_backend_pull(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.config_version = 42
    app.state.restart_epoch = 5
    app.state.pulled_config = PulledWorkerConfig(
        config_version=42,
        restart_epoch=5,
        night_window=PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        cameras=(
            PulledCameraConfig(
                camera_id="pulled-camera",
                space_id="space-pulled",
                label="Pulled",
                rtsp_url="rtsp://pulled/stream",
                online=True,
            ),
        ),
    )
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    expected = {
        "registry_version": 1,
        "clip_export_enabled": False,
        "clip_export_version": 0,
        "cameras": [
            {
                "camera_id": "camera-1",
                "space_id": "space-1",
                "rtsp_url": "rtsp://camera/stream",
            }
        ],
        "config_version": 42,
        "restart_epoch": 5,
        "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
    }
    assert worker_config.json() == expected
    assert relay_config.status_code == 200
    assert relay_config.json() == expected


def test_worker_config_emits_detection_windows_alongside_legacy_night_window(tmp_path) -> None:
    """Per-domain ``detection_windows`` (issue #24) is emitted alongside the
    deprecated single ``night_window`` field so old workers keep working
    while new ones can read the full per-domain map."""
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.config_version = 42
    app.state.restart_epoch = 5
    app.state.pulled_config = PulledWorkerConfig(
        config_version=42,
        restart_epoch=5,
        night_window=PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        cameras=(
            PulledCameraConfig(
                camera_id="pulled-camera",
                space_id="space-pulled",
                label="Pulled",
                rtsp_url="rtsp://pulled/stream",
                online=True,
            ),
        ),
        detection_windows={
            "bed_exit": PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
            "fall": PulledNightWindow(start="22:00", end="05:00", tz="Asia/Seoul"),
        },
    )
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    body = worker_config.json()
    assert body["night_window"] == {"start": "21:00", "end": "06:00", "tz": "UTC"}
    assert body["detection_windows"] == {
        "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
        "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
    }


def test_authorize_worker_accepts_non_ascii_relay_token_without_crashing() -> None:
    """A non-ASCII relay token (e.g. a Korean value in API_EDGE_RELAY_TOKEN)
    must not crash the constant-time compare in `_authorize_worker` with a
    TypeError -- hmac.compare_digest rejects non-ASCII `str` arguments, so
    the comparison must encode to UTF-8 bytes first (see issue #23's
    compare_digest sweep).

    Exercised as a direct call against `_authorize_worker` rather than over
    HTTP: an HTTP header is a byte-oriented channel, and this repo's actual
    worker HTTP client (stdlib `http.client`/`urllib.request`, see
    worker/__main__.py, worker/runtime/worker.py,
    worker/runtime/config/config_pull.py) encodes a `str` header value via
    Latin-1 *client-side* -- a genuine Korean/CJK token would raise
    UnicodeEncodeError in the worker itself before a request is ever sent,
    never reaching this comparison. Calling `_authorize_worker` directly
    isolates the actual fix (the compare_digest call) from that unrelated,
    non-reachable HTTP-header-encoding concern.
    """
    state = SimpleNamespace(edge_relay_token="중계-토큰")
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    _authorize_worker(request, "중계-토큰")  # must not raise

    with pytest.raises(HTTPException) as exc_info:
        _authorize_worker(request, "wrong-token")
    assert exc_info.value.status_code == 403


def test_worker_config_emits_default_camera_fps_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_CAMERA_FPS", "15")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["fps"] == 15.0
    assert camera["camera_id"] == "camera-1"


def test_worker_config_emits_default_frame_stride_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_FRAME_STRIDE", "3")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["frame_stride"] == 3
    assert camera["camera_id"] == "camera-1"


def test_worker_config_omits_frame_stride_when_unset_or_invalid(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ML_DEFAULT_FRAME_STRIDE", raising=False)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert "frame_stride" not in camera

    monkeypatch.setenv("ML_DEFAULT_FRAME_STRIDE", "not-a-number")
    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert "frame_stride" not in camera


def test_worker_config_surfaces_empty_roster_when_registry_empty(tmp_path) -> None:
    """An empty dashboard camera registry must surface as an empty worker-config
    roster -- never silently substitute app.state.pulled_config's legacy
    backend-pulled camera list (see issue #33: that inbound path is deprecated
    per AGENTS.md ANTI-PATTERNS, and an empty registry must be a visible,
    diagnosable state rather than one indistinguishable from a legacy-sourced
    roster). Non-camera fields (config_version, restart_epoch, night_window)
    still legitimately flow from pulled_config -- only the camera fallback is
    removed.

    /cameras/worker-config (require_available=False) returns 200 with an
    empty cameras list; /relay/config (require_available=True, see
    backend/app/features/relay/router.py) treats the now-genuinely-empty
    roster as a still-unavailable config and returns 503 -- both are visible,
    diagnosable states, unlike the old legacy-sourced 200."""
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.config_version = 42
    app.state.restart_epoch = 5
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=0,
        night_window=PulledNightWindow(start="21:00", end="06:00", tz="UTC"),
        cameras=(
            PulledCameraConfig(
                camera_id="pulled-camera",
                space_id="space-pulled",
                label="Pulled",
                rtsp_url="rtsp://pulled/stream",
                online=True,
            ),
            PulledCameraConfig(
                camera_id="pulled-without-rtsp",
                space_id="space-empty",
                label="Empty",
                rtsp_url=None,
                online=False,
            ),
        ),
    )
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        relay_config = client.get(
            "/api/v1/relay/config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    expected = {
        "registry_version": 0,
        "clip_export_enabled": False,
        "clip_export_version": 0,
        "cameras": [],
        "config_version": 42,
        "restart_epoch": 5,
        "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
    }
    assert worker_config.status_code == 200
    assert worker_config.json() == expected
    # relay/config requires an available roster (require_available=True); an
    # empty registry is now a genuinely empty roster rather than a
    # legacy-sourced one, so it correctly surfaces as 503 (see
    # test_config_returns_503_when_backend_config_unavailable in
    # tests/test_ml_api_config_pull.py for the analogous no-pull case).
    assert relay_config.status_code == 503


def test_example_camera_registry_seed_is_loadable_and_sanitized() -> None:
    """``cameras.example.json`` is a docs-only reference of the pre-SQLite
    JSON registry file shape (see ``backend/app/cameras.example.json``); it
    is not read by ``CameraRegistryStore`` itself anymore (issue #35 moved
    that store's storage to the shared ``catalog.sqlite3`` database), so this
    test parses it directly with ``json.loads`` and exercises the same
    ``public_camera`` sanitization the store's readers use."""
    seed_path = REPO_ROOT / "backend" / "app" / "cameras.example.json"
    snapshot = json.loads(seed_path.read_text(encoding="utf-8"))

    assert snapshot["registry_version"] == 1
    cameras = snapshot["cameras"]
    assert isinstance(cameras, list)
    assert len(cameras) == 1
    loaded_record = cameras[0]
    assert isinstance(loaded_record, dict)
    record = {str(key): value for key, value in loaded_record.items()}
    assert record["id"] == "example-room-camera"
    assert record["rtsp_url"] == "rtsp://camera.example.invalid/trackID=1"

    public = public_camera(record)
    assert public == {
        "id": "example-room-camera",
        "label": "Example Room Camera",
        "rtsp_url_masked": "rtsp://redacted-camera/trackID=1",
        "space_id": "example-room",
        "backend_camera_id": None,
        # 기사님이 현장에서 클라우드 연동 여부를 확인해야 하므로 응답에 싣는다.
        "mapping_pending": False,
        "status": "unknown",
        "decode_backend": None,
        "fps": None,
        "floor": None,
        "created_at": "2026-01-01T00:00:00.000Z",
        "never_connected": None,
        "last_ok_at": None,
        "last_probed_at": None,
    }

    serialized = json.dumps(snapshot, sort_keys=True).lower()
    for forbidden in ("10.10.", "@", "admin", "password", "token"):
        assert forbidden not in serialized


def test_system_reports_backend_state_and_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ML_EDGE_VERSION", "2026.07.06")
    # "configured" now comes from an applied backend_client_bundle, which
    # requires a complete ConnectionSettingsStore enrollment row -- API_BACKEND_URL
    # alone (the old env-only signal) no longer has any authority over it.
    from backend.app.features.connection.store import (
        API_CONNECTION_SETTINGS_PATH_ENV,
        ConnectionSettingsStore,
    )

    monkeypatch.setenv(
        API_CONNECTION_SETTINGS_PATH_ENV, str(tmp_path / "connection-settings.sqlite3")
    )
    ConnectionSettingsStore.from_env().save(
        {
            "events_url": "http://backend/api/v1/events",
            "config_url": "http://backend/api/v1/ml-config",
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "87d79f24-b32f-49a3-b534-19f0af7d9135",
            "facility_token": "facility-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 1,
        }
    )

    app = create_app(lifespan=no_lifespan)
    apply_connection_settings(app)
    app.state.backend_reachable = True
    app.state.backend_last_ok_at = "2026-07-06T00:00:00.000Z"

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/system")

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == {
        "configured": True,
        "reachable": True,
        "last_ok_at": "2026-07-06T00:00:00.000Z",
    }
    assert body["version"] == "2026.07.06"
    assert body["image_digests"] == {"ml_api": None, "ml_worker": None}
    assert set(body["storage"]["clip_store"]) == {"total_bytes", "used_bytes", "used_pct"}
    assert body["updated_at"].endswith("Z")


# test_config_pull_persists_lkg_and_falls_back (edge): LKG persist/fallback round-trip is
# superseded by tests/test_worker_config_pull_lkg.py (whole-file, worker-side pull/LKG/YAML
# coverage) and tests/test_worker_config_lifecycle.py
# ::test_fresh_pull_uses_auth_and_replaces_lkg_atomically /
# ::test_unreachable_backend_uses_stale_lkg_without_zeroing_cameras. The one assertion
# neither file makes is the fps/enabled_domains pull-mapping, ported below.
# test_restart_check_detects_registry_version_change (edge): superseded by
# tests/test_worker_restart_directive.py::test_restart_check_polls_immediately_on_first_call,
# ::test_restart_check_polls_again_once_interval_elapses, and
# ::test_tracker_observe_advances_current_on_strictly_greater_candidate.
def test_worker_config_pull_maps_fps_and_enabled_domains_from_relay_payload(
    tmp_path: Path,
) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.sqlite3")
    payload: JsonObject = {
        "registry_version": 9,
        "config_version": 9,
        "restart_epoch": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "rtsp_url": "rtsp://camera/stream",
                "fps": 4,
                "domains": ["fall"],
            }
        ],
    }

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-token",
        store=store,
        urlopen=lambda _request, _timeout: FakeHTTPResponse(payload),
    )

    assert snapshot is not None
    assert snapshot.config.cameras[0].fps == 4
    assert snapshot.config.domains.enabled_domains == ("fall",)


def test_worker_config_pull_maps_frame_stride_from_relay_payload(
    tmp_path: Path,
) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.sqlite3")
    payload: JsonObject = {
        "registry_version": 9,
        "config_version": 9,
        "restart_epoch": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "rtsp_url": "rtsp://camera/stream",
                "fps": 15,
                "frame_stride": 3,
            }
        ],
    }

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-token",
        store=store,
        urlopen=lambda _request, _timeout: FakeHTTPResponse(payload),
    )

    assert snapshot is not None
    assert snapshot.config.cameras[0].frame_stride == 3


def test_worker_config_pull_defaults_frame_stride_to_one_when_absent(
    tmp_path: Path,
) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.sqlite3")
    payload: JsonObject = {
        "registry_version": 9,
        "config_version": 9,
        "restart_epoch": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "rtsp_url": "rtsp://camera/stream",
                "fps": 15,
            }
        ],
    }

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-token",
        store=store,
        urlopen=lambda _request, _timeout: FakeHTTPResponse(payload),
    )

    assert snapshot is not None
    assert snapshot.config.cameras[0].frame_stride == 1


def test_patch_pending_camera_preserves_local_id_and_stays_retained(
    tmp_path,
) -> None:
    """PATCH no longer performs a synchronous backend mapping: there is no
    ``_map_backend`` call any more (that per-camera resolution was replaced
    by the async topology-snapshot roster sync -- see BLOCKER 2 in the merge
    notes), and ``UpdateCameraRequest`` doesn't even accept
    ``backend_camera_id``/``mapping_pending`` as patchable fields. So this
    test instead pins the two properties BLOCKER 2 actually cares about:

    (1) canonical id resolution -- with ``backend_camera_id`` still unset,
        the worker-config/relay-config projection (``worker_config_snapshot``
        in ``backend/app/features/cameras/router.py``) resolves the
        camera's canonical id to its own local id, exactly as it does before
        any PATCH.
    (2) unmapped camera retained rather than dropped -- patching an
        unmapped camera does not remove it from the registry or swap its
        address; existing clip manifests that refer to the local id stay
        valid, and the record is still listed with ``backend_camera_id`` still
        unset (nothing mapped it). ``mapping_pending`` itself is always False
        at create time now (``create_camera`` never sets it True -- see
        BLOCKER 2's follow-up note on whether the async roster sync should
        persist a Hub-assigned id back onto the record at all), so it is not
        a useful signal here and this test does not assert it.
    """
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    with TestClient(app) as client:
        _login(client)
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera/stream",
                "space_id": "space-1",
                # No worker probe origin is configured in this test, so the
                # probe fails closed; force registration to exercise the
                # unmapped-camera PATCH flow this test is actually about.
                "force_register": True,
            },
        )
        assert created.status_code == 201
        local_id = created.json()["id"]
        assert store.get(local_id)["backend_camera_id"] is None

        response = client.patch(
            f"/api/v1/cameras/{local_id}",
            headers=AUTH,
            json={"label": "Lobby North"},
        )
        assert response.status_code == 200

        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.json()["id"] == local_id
    assert response.json()["label"] == "Lobby North"
    # (2) Retained, not dropped: still addressable by its own local id, still
    # unmapped -- nothing in this flow assigns a backend id.
    assert response.json()["backend_camera_id"] is None
    assert store.get(local_id) is not None
    assert store.get(local_id)["backend_camera_id"] is None

    # (1) Canonical id resolution: unmapped means the projection falls back
    # to the local id (worker_config_snapshot's
    # `str(record.get("backend_camera_id") or record.get("id", ""))`).
    assert worker_config.status_code == 200
    assert worker_config.json()["cameras"] == [
        {
            "camera_id": local_id,
            "space_id": "space-1",
            "rtsp_url": "rtsp://camera/stream",
        }
    ]


def test_patch_camera_sets_decode_backend_and_worker_config_emits_it(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"decode_backend": "NVDEC"},
        )
        assert patched.status_code == 200
        assert patched.json()["decode_backend"] == "nvdec"

        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["decode_backend"] == "nvdec"
    assert camera["camera_id"] == "camera-1"

    # Issue #42: the value must not just round-trip through the backend --
    # it has to actually reach the composed per-camera ingest loop's decoder
    # selection, not be dropped once it lands on CameraRuntimeConfig.
    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-token",
        store=WorkerConfigLkgStore(tmp_path / "worker-config-pull.sqlite3"),
        urlopen=lambda _request, _timeout: FakeHTTPResponse(worker_config.json()),
    )
    assert snapshot is not None
    runtime_camera = snapshot.config.cameras[0]
    assert isinstance(runtime_camera, CameraRuntimeConfig)
    assert runtime_camera.decode_backend == "nvdec"
    decoder = decoder_for("nvdec", runtime_camera.decode_backend)
    assert isinstance(decoder, NvdecCuvidAdapter)
    assert not isinstance(decoder, CpuAvAdapter)


def test_patch_camera_rejects_invalid_decode_backend(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"decode_backend": "gstreamer"},
        )

    assert patched.status_code == 400


def test_create_camera_rejects_invalid_decode_backend(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera.local/live",
                "decode_backend": "gstreamer",
            },
        )

    assert created.status_code == 400


def test_worker_config_emits_default_decode_backend_when_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_DECODE_BACKEND", "cpu")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["decode_backend"] == "cpu"
    assert camera["camera_id"] == "camera-1"


def test_worker_config_prefers_record_decode_backend_over_env_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_DECODE_BACKEND", "cpu")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        decode_backend="nvdec",
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["decode_backend"] == "nvdec"


def test_patch_camera_sets_fps_override_and_worker_config_emits_it(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"fps": 12},
        )
        assert patched.status_code == 200
        assert patched.json()["fps"] == 12.0

        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["fps"] == 12.0
    assert camera["camera_id"] == "camera-1"


def test_patch_camera_clears_fps_override_with_explicit_null(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        fps=12.0,
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"fps": None},
        )

    assert patched.status_code == 200
    assert patched.json()["fps"] is None
    assert store.get("camera-1")["fps"] is None


def test_patch_camera_rejects_invalid_fps(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"fps": 0},
        )

    assert patched.status_code == 400


def test_create_camera_rejects_invalid_fps(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera.local/live",
                "fps": -1,
            },
        )

    assert created.status_code == 400


def test_worker_config_prefers_record_fps_over_env_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_DEFAULT_CAMERA_FPS", "8")
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        fps=20.0,
    )

    with TestClient(app) as client:
        _login(client)
        worker_config = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert worker_config.status_code == 200
    camera = worker_config.json()["cameras"][0]
    assert camera["fps"] == 20.0


def test_create_camera_sets_floor(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        created = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera.local/live",
                "floor": 2,
                "force_register": True,
            },
        )

    assert created.status_code == 201
    assert created.json()["floor"] == 2


def test_create_camera_rejects_a_floor_outside_the_fixed_catalog(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        rejected = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby",
                "rtsp_url": "rtsp://camera.local/live",
                "floor": 99,
                "force_register": True,
            },
        )

    assert rejected.status_code == 400


def test_patch_camera_sets_floor(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"floor": 3},
        )

    assert patched.status_code == 200
    assert patched.json()["floor"] == 3
    assert store.get("camera-1")["floor"] == 3


def test_patch_camera_clears_floor_with_explicit_null(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        floor=3,
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"floor": None},
        )

    assert patched.status_code == 200
    assert patched.json()["floor"] is None
    assert store.get("camera-1")["floor"] is None


def test_patch_camera_rejects_a_floor_outside_the_fixed_catalog(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        floor=3,
    )

    with TestClient(app) as client:
        _login(client)
        patched = client.patch(
            "/api/v1/cameras/camera-1",
            headers=AUTH,
            json={"floor": 0},
        )

    assert patched.status_code == 400
    # Rejected write must not clobber the previously-stored valid floor.
    assert store.get("camera-1")["floor"] == 3


def test_list_cameras_user_set_floor_survives_roster_sync(tmp_path) -> None:
    """Precedence decision (issue #85): a user-set ``floor`` override must
    survive a space-sync roster re-pull untouched, even when that roster
    carries its own (different) ``floor_name`` for the same camera -- see
    the ``floor`` field doc-comment on CameraResponse and public_camera().
    """
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="camera-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-101",
        status="online",
        backend_camera_id="backend-1",
        floor=3,
    )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-101",
                label="Lobby",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    # The roster sync's floor_name lands as usual...
    assert camera["floor_name"] == "1층"
    # ...but the locally user-set floor is untouched by it.
    assert camera["floor"] == 3


def test_list_cameras_includes_backend_only_roster_camera(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-101",
                label="Room 101",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["cameras"] == [
        {
            "id": "backend-1",
            "label": "Room 101",
            "rtsp_url_masked": "rtsp://***",
            "space_id": "space-101",
            "backend_camera_id": "backend-1",
            "mapping_pending": True,
            "status": "unknown",
            "decode_backend": None,
            "fps": None,
            "floor": None,
            "created_at": "2026-07-10T00:00:00.000Z",
            "space_name": "101호",
            "floor_name": "1층",
            "last_heartbeat_at": None,
            "heartbeat_age_sec": None,
            "never_connected": None,
            "last_ok_at": None,
            "last_probed_at": None,
            "sync": None,
            "bed_zone": None,
        }
    ]


def test_list_cameras_includes_backend_only_roster_camera_without_created_at(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-no-created-at",
                space_id="space-101",
                label="Room 101",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at=None,
            ),
        ),
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    assert response.json()["cameras"] == [
        {
            "id": "backend-no-created-at",
            "label": "Room 101",
            "rtsp_url_masked": "rtsp://***",
            "space_id": "space-101",
            "backend_camera_id": "backend-no-created-at",
            "mapping_pending": True,
            "status": "unknown",
            "decode_backend": None,
            "fps": None,
            "floor": None,
            "created_at": None,
            "space_name": "101호",
            "floor_name": "1층",
            "last_heartbeat_at": None,
            "heartbeat_age_sec": None,
            "never_connected": None,
            "last_ok_at": None,
            "last_probed_at": None,
            "sync": None,
            "bed_zone": None,
        }
    ]


def test_list_cameras_includes_local_only_camera_with_null_roster_names(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="local-1",
        label="Local camera",
        rtsp_url="rtsp://local/stream",
        space_id="local-space",
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["id"] == "local-1"
    assert camera["space_name"] is None
    assert camera["floor_name"] is None
    assert camera["mapping_pending"] is False


def test_list_cameras_joins_local_transport_with_backend_roster_metadata(tmp_path) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="local-1",
        label="Local label",
        rtsp_url="rtsp://user:secret@local/stream",
        space_id="local-space",
        status="online",
        backend_camera_id="backend-1",
        decode_backend="nvdec",
    )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="backend-space",
                label="Backend label",
                rtsp_url="rtsp://backend/stream",
                online=False,
                space_name="Backend room",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )
    # GET /cameras now derives status from live heartbeat freshness (Task 1),
    # not the frozen registry snapshot, so this join test must record one for
    # the record's canonical id (backend_camera_id) to see "online".
    get_heartbeat_store(app).record("backend-1", "backend-space")

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["id"] == "local-1"
    assert camera["label"] == "Backend label"
    assert camera["space_id"] == "backend-space"
    assert camera["created_at"] == "2026-07-10T00:00:00.000Z"
    assert camera["space_name"] == "Backend room"
    assert camera["floor_name"] == "2F"
    assert camera["rtsp_url_masked"] == "rtsp://***:***@redacted-camera/stream"
    assert camera["decode_backend"] == "nvdec"
    assert camera["status"] == "online"


def test_list_cameras_joins_unmapped_local_camera_by_unambiguous_space(tmp_path) -> None:
    """A local camera without backend_camera_id joins the sole same-space roster row.

    Local registrations predating the backend mapping carry a space_id only;
    cameras never move between rooms (spec R17), so a single roster camera in
    the same space is the same physical camera and must not appear twice.
    """
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="local-1",
        label="Local label",
        rtsp_url="rtsp://user:secret@local/stream",
        space_id="space-1",
        status="online",
        backend_camera_id=None,
        decode_backend="nvdec",
    )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-1",
                label="Backend label",
                rtsp_url=None,
                online=True,
                space_name="205호",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
            PulledCameraConfig(
                camera_id="backend-2",
                space_id="space-2",
                label="Other room",
                rtsp_url=None,
                online=True,
                space_name="202호",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    joined = [c for c in cameras if c["id"] == "local-1"]
    assert len(joined) == 1
    assert joined[0]["space_name"] == "205호"
    assert joined[0]["backend_camera_id"] == "backend-1"
    # The joined roster row must not also appear as a backend-only duplicate.
    assert not any(c["id"] == "backend-1" for c in cameras)
    # The unrelated roster camera still appears backend-only.
    assert any(c["id"] == "backend-2" and c["status"] == "unknown" for c in cameras)


@pytest.mark.parametrize("unmapped_first", [True, False])
def test_space_fallback_never_steals_explicitly_mapped_roster_row(
    tmp_path, unmapped_first: bool
) -> None:
    """An unmapped local sibling must not consume a roster row owned by an
    explicitly mapped local camera, regardless of registry iteration order."""
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    order = (
        ("local-unmapped", "local-mapped") if unmapped_first else ("local-mapped", "local-unmapped")
    )
    for camera_id in order:
        store.create(
            camera_id=camera_id,
            label=camera_id,
            # Distinct per-camera paths: this test targets space-based roster
            # matching ambiguity, not stream dedup, so a shared rtsp_url would
            # incorrectly trigger DuplicateCameraError during setup.
            rtsp_url=f"rtsp://local/stream-{camera_id}",
            space_id="space-1",
            status="online",
            backend_camera_id="backend-1" if camera_id == "local-mapped" else None,
            decode_backend="nvdec",
        )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="backend-1",
                space_id="space-1",
                label="Backend label",
                rtsp_url=None,
                online=True,
                space_name="205호",
                floor_name="2F",
                created_at="2026-07-10T00:00:00.000Z",
            ),
        ),
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = {c["id"]: c for c in response.json()["cameras"]}
    # The explicit mapping owns the roster metadata...
    assert cameras["local-mapped"]["space_name"] == "205호"
    assert cameras["local-mapped"]["backend_camera_id"] == "backend-1"
    # ...and the unmapped sibling never steals it.
    assert cameras["local-unmapped"]["space_name"] is None
    assert cameras["local-unmapped"]["backend_camera_id"] is None
    backend_ids = [c.get("backend_camera_id") for c in cameras.values()]
    assert backend_ids.count("backend-1") == 1


@pytest.mark.parametrize(
    ("local_ids", "backend_ids"),
    (
        (("local-1", "local-2"), ("backend-1",)),
        (("local-2", "local-1"), ("backend-1",)),
        (("local-1",), ("backend-1", "backend-2")),
        (("local-1",), ("backend-2", "backend-1")),
    ),
)
def test_list_cameras_does_not_join_ambiguous_space(
    tmp_path, local_ids: tuple[str, ...], backend_ids: tuple[str, ...]
) -> None:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    for camera_id in local_ids:
        store.create(
            camera_id=camera_id,
            label=f"Local {camera_id}",
            rtsp_url=f"rtsp://{camera_id}/stream",
            space_id="space-1",
            status="online",
        )
    app.state.pulled_config = PulledWorkerConfig(
        config_version=9,
        restart_epoch=0,
        night_window=None,
        cameras=tuple(
            PulledCameraConfig(
                camera_id=camera_id,
                space_id="space-1",
                label=f"Backend {camera_id}",
                rtsp_url=None,
                online=True,
                space_name="101호",
                floor_name="1층",
                created_at="2026-07-10T00:00:00.000Z",
            )
            for camera_id in backend_ids
        ),
    )

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    local_cameras = [camera for camera in cameras if camera["id"] in local_ids]
    assert len(local_cameras) == len(local_ids)
    assert all(camera["backend_camera_id"] is None for camera in local_cameras)
    assert all(camera["space_name"] is None for camera in local_cameras)
    assert {camera["id"] for camera in cameras} == {*local_ids, *backend_ids}


def test_list_cameras_reflects_room_name_after_roster_refresh(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [
        {
            "configVersion": 1,
            "cameras": [
                {
                    "id": "backend-1",
                    "spaceId": "space-1",
                    "label": "Room 101",
                    "rtspUrl": None,
                    "online": True,
                    "spaceName": "101호",
                    "floorName": "1층",
                    "createdAt": "2026-07-10T00:00:00.000Z",
                }
            ],
        },
        {
            "configVersion": 2,
            "cameras": [
                {
                    "id": "backend-1",
                    "spaceId": "space-1",
                    "label": "Room 101",
                    "rtspUrl": None,
                    "online": True,
                    "spaceName": "새 101호",
                    "floorName": "1층",
                    "createdAt": "2026-07-10T01:00:00.000Z",
                }
            ],
        },
    ]
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        payload = configs[calls["count"]]
        calls["count"] += 1
        return FakeHTTPResponse(payload)

    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "http://backend/ml-config")
    monkeypatch.setenv(
        "API_CONNECTION_SETTINGS_PATH", str(tmp_path / "connection-settings.sqlite3")
    )
    from backend.app.features.connection.store import ConnectionSettingsStore

    # refresh_backend_config only runs once a backend_client_bundle exists, which
    # requires the full enrollment row -- not just config_url/facility_id.
    ConnectionSettingsStore(tmp_path / "connection-settings.sqlite3").save(
        {
            "events_url": "http://backend/api/v1/events",
            "config_url": "http://backend/ml-config",
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "facility-1",
            "facility_token": "facility-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 1,
        }
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    apply_connection_settings(app)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    assert refresh_backend_config(app) is True
    with TestClient(app) as client:
        _login(client)
        assert (
            client.get("/api/v1/cameras", headers=AUTH).json()["cameras"][0]["space_name"]
            == "101호"
        )

        assert refresh_backend_config(app) is True
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.json()["cameras"][0]["space_name"] == "새 101호"


def test_roster_refresh_failure_preserves_last_good_and_marks_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(url: str, timeout: float) -> FakeHTTPResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeHTTPResponse(
                {
                    "configVersion": 9,
                    "cameras": [
                        {
                            "id": "backend-1",
                            "spaceId": "space-1",
                            "label": "Room 101",
                            "rtspUrl": None,
                            "online": True,
                            "createdAt": "2026-07-10T00:00:00.000Z",
                        }
                    ],
                }
            )
        raise urllib.error.URLError("offline")

    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "http://backend/ml-config")
    from backend.app.features.connection.store import ConnectionSettingsStore

    conn_path = tmp_path / "connection-settings.sqlite3"
    monkeypatch.setenv("API_CONNECTION_SETTINGS_PATH", str(conn_path))
    # refresh_backend_config only runs once a backend_client_bundle exists, which
    # requires the full enrollment row -- not just config_url/facility_id.
    ConnectionSettingsStore(conn_path).save(
        {
            "events_url": "http://backend/api/v1/events",
            "config_url": "http://backend/ml-config",
            "facility_code": "NH-7H2K9M4QXP",
            "client_installation_ref": "aa83ea3f-6e5f-4f45-a401-fb36c38835b6",
            "facility_id": "facility-1",
            "facility_token": "facility-token",
            "edge_installation_id": "d17e0eb8-cb81-4d8e-a427-dfe690518f2b",
            "enrollment_generation": 1,
        }
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    apply_connection_settings(app)

    assert refresh_backend_config(app) is True
    received_at = app.state.backend_roster["received_at"]
    assert refresh_backend_config(app) is False
    assert app.state.pulled_config.config_version == 9
    assert app.state.backend_roster == {
        "config_version": 9,
        "received_at": received_at,
        "stale": True,
    }


def test_list_cameras_status_reflects_heartbeat_freshness(tmp_path) -> None:
    """GET /cameras derives status from live heartbeat freshness (Task 1), not
    the frozen registry snapshot -- covers never-seen, stale, and fresh cases
    without a real sleep by directly stamping received_at in the past."""
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    for camera_id in ("never-seen", "stale", "fresh"):
        store.create(
            camera_id=camera_id,
            label=camera_id,
            rtsp_url=f"rtsp://local/{camera_id}",
            space_id=None,
            # The frozen registry status says "online" for all three; the
            # heartbeat join must override this, never trust it.
            status="online",
        )
    heartbeats = get_heartbeat_store(app)
    now = time.time()
    heartbeats.record("stale", "facility-1", received_at=now - 1000.0)
    heartbeats.record("fresh", "facility-1", received_at=now)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)

    assert response.status_code == 200
    cameras = {c["id"]: c for c in response.json()["cameras"]}

    assert cameras["never-seen"]["status"] == "offline"
    assert cameras["never-seen"]["last_heartbeat_at"] is None
    assert cameras["never-seen"]["heartbeat_age_sec"] is None

    assert cameras["stale"]["status"] == "offline"
    assert cameras["stale"]["last_heartbeat_at"] == pytest.approx(now - 1000.0)
    assert cameras["stale"]["heartbeat_age_sec"] > 90.0

    assert cameras["fresh"]["status"] == "online"
    assert cameras["fresh"]["last_heartbeat_at"] == pytest.approx(now)
    assert cameras["fresh"]["heartbeat_age_sec"] < 90.0


def test_list_cameras_status_matches_heartbeat_under_either_local_or_backend_id(
    tmp_path,
) -> None:
    """Regression for the heartbeat key-mismatch bug: relay_heartbeat records
    under whichever raw id the worker sends (local registry id OR backend
    id -- see _camera_binding_from_registry, which accepts either), but a
    camera can be registered locally and later backend-mapped to a distinct
    backend_camera_id while the worker keeps heartbeating under the old local
    id. GET /cameras must still resolve "online" by trying both ids, not
    just the canonical one -- and must NOT fall back to matching an
    unrelated id."""
    now = time.time()

    def _make_app() -> tuple[object, CameraRegistryStore]:
        app = create_app(lifespan=no_lifespan)
        app.state.edge_relay_token = "relay-token"
        store = app.state.camera_registry = CameraRegistryStore(
            tmp_path / f"cameras-{uuid.uuid4()}.json"
        )
        store.create(
            camera_id="loc-12",
            label="mapped-camera",
            rtsp_url="rtsp://local/mapped-camera",
            space_id=None,
            status="online",
            backend_camera_id="be-77",
        )
        return app, store

    # Case 1: worker still heartbeats under the old local id -- must resolve
    # online via the local-id fallback, even though backend_camera_id is the
    # canonical id GET /cameras otherwise prefers.
    app, _ = _make_app()
    get_heartbeat_store(app).record("loc-12", "facility-1", received_at=now)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)
    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["status"] == "online"
    assert camera["heartbeat_age_sec"] is not None

    # Case 2 (mirror): worker heartbeats under the canonical backend id --
    # must also resolve online. Fresh app so no heartbeat from case 1 leaks
    # in and masks a real failure here.
    app, _ = _make_app()
    get_heartbeat_store(app).record("be-77", "facility-1", received_at=now)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)
    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["status"] == "online"
    assert camera["heartbeat_age_sec"] is not None

    # Case 3 (negative): a heartbeat recorded under a third, unrelated id
    # must NOT match -- proves the dual-key lookup isn't match-anything.
    # Fresh app so neither of the above heartbeats is present to (correctly
    # or incorrectly) satisfy the lookup.
    app, _ = _make_app()
    get_heartbeat_store(app).record("unrelated-camera", "facility-1", received_at=now)
    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras", headers=AUTH)
    assert response.status_code == 200
    camera = response.json()["cameras"][0]
    assert camera["status"] == "offline"


def test_create_camera_rejects_duplicate_rtsp_url(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same physical stream registered twice (default port vs explicit :554,
    which normalize_stream_identity elides for rtsp://) must 409, not silently
    create a second registry row for the same camera."""
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": True, "width": 1920, "height": 1080})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        first = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={"label": "Lobby", "rtsp_url": "rtsp://cam.local/stream"},
        )
        assert first.status_code == 201
        second = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={"label": "Lobby copy", "rtsp_url": "rtsp://cam.local:554/stream"},
        )

    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["error"] == "duplicate_camera"
    assert detail["existing_camera_id"] == first.json()["id"]
    assert detail["existing_label"] == "Lobby"


def test_create_camera_rejects_duplicate_ignoring_credentials(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rotating a camera's password (e.g. default admin/admin -> a real
    password) must not spawn a zombie duplicate registration."""
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        first = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={"label": "Lobby", "rtsp_url": "rtsp://admin:admin@cam.local/stream"},
        )
        assert first.status_code == 201
        second = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Lobby rotated creds",
                "rtsp_url": "rtsp://admin:newpass@cam.local/stream",
            },
        )

    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "duplicate_camera"


def test_create_camera_allows_dahua_subtype_variants_as_distinct_streams(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dahua encodes main-vs-sub stream selection in the query string
    (?subtype=0 vs ?subtype=1 on the same path); these are physically
    distinct streams and must both be allowed."""
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        main = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Main stream",
                "rtsp_url": "rtsp://cam.local/cam/realmonitor?channel=1&subtype=0",
            },
        )
        sub = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Sub stream",
                "rtsp_url": "rtsp://cam.local/cam/realmonitor?channel=1&subtype=1",
            },
        )

    assert main.status_code == 201
    assert sub.status_code == 201
    assert main.json()["id"] != sub.json()["id"]


def test_create_camera_persists_on_probe_failure_without_force_register(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """등록은 저장이고, probe는 상태 표시일 뿐이다.

    예전에는 probe 실패를 422로 거절했는데, 그러면 최초 등록이 구조적으로
    막혔다: probe는 worker의 ``/probe``에 위임되는데 worker는 카메라가 한 대
    이상일 때만 부팅하므로, 첫 카메라를 넣으려면 아직 뜨지 않은 worker의
    판정을 통과해야 했다. 죽은 카메라는 offline/never_connected로 목록에
    남고, 연결 여부는 worker의 첫 heartbeat이 확정한다.
    """
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "auth"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={"label": "Dead camera", "rtsp_url": "rtsp://dead.local/stream"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "offline"
    assert body["never_connected"] is True
    assert body["last_ok_at"] is None

    snapshot = store.snapshot()
    assert snapshot["registry_version"] == 1
    assert [camera["label"] for camera in snapshot["cameras"]] == ["Dead camera"]


def test_create_camera_force_register_persists_despite_probe_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "timeout"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            headers=AUTH,
            json={
                "label": "Known offline camera",
                "rtsp_url": "rtsp://offline.local/stream",
                "force_register": True,
            },
        )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "offline"
    # The probe-history fields must round-trip through the public API too,
    # not just the internal store record (frontend Camera type reads these
    # directly off the GET/POST /cameras response body).
    assert body["never_connected"] is True
    assert body["last_ok_at"] is None
    assert body["last_probed_at"] is not None
    record = store.get(body["id"])
    assert record is not None
    assert record["never_connected"] is True
    assert record["last_ok_at"] is None
    assert record["last_probed_at"] is not None


def test_patch_camera_rtsp_url_rejects_duplicate(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="cam-a",
        label="A",
        rtsp_url="rtsp://a.local/stream",
        space_id=None,
        status="online",
    )
    store.create(
        camera_id="cam-b",
        label="B",
        rtsp_url="rtsp://b.local/stream",
        space_id=None,
        status="online",
    )

    with TestClient(app) as client:
        _login(client)
        response = client.patch(
            "/api/v1/cameras/cam-b",
            headers=AUTH,
            json={"rtsp_url": "rtsp://a.local/stream"},
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "duplicate_camera"
    assert detail["existing_camera_id"] == "cam-a"


def test_test_camera_persists_probe_result(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def fake_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": True, "width": 640, "height": 480})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id="cam-a",
        label="A",
        rtsp_url="rtsp://a.local/stream",
        space_id=None,
        status="offline",
        never_connected=True,
    )

    with TestClient(app) as client:
        _login(client)
        response = client.post("/api/v1/cameras/cam-a/test", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"ok": True, "width": 640, "height": 480}
    record = store.get("cam-a")
    assert record is not None
    assert record["last_probed_at"] is not None
    assert record["last_ok_at"] is not None
    assert record["never_connected"] is False

    # The persisted result must also surface through GET /cameras, not just
    # the internal store record.
    with TestClient(app) as client:
        _login(client)
        listed = client.get("/api/v1/cameras", headers=AUTH).json()
    camera = listed["cameras"][0]
    assert camera["never_connected"] is False
    assert camera["last_ok_at"] is not None
    assert camera["last_probed_at"] is not None


def _register_camera_for_probe(client: TestClient, label: str, rtsp_url: str) -> dict[str, object]:
    """probe 분류를 검사하기 위한 카메라 한 대를 등록한다.

    등록 자체는 probe 결과와 무관하므로(#147/#159로 게이트가 제거됐다) 어떤
    probe 응답이 오든 201이다.
    """
    response = client.post(
        "/api/v1/cameras", headers=AUTH, json={"label": label, "rtsp_url": rtsp_url}
    )
    assert response.status_code == 201
    return response.json()


def test_probe_reports_unavailable_when_worker_cannot_be_reached(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker에 닿지 못한 것을 "디코드 실패"라고 단정하면 안 된다 (이슈 #151).

    예전에는 연결 거부/DNS 실패까지 전부 ``error_class="decode"``로 뭉갰다.
    그래서 현장에서 RTSP 비밀번호가 틀렸을 때 대시보드가 "영상 스트림을
    디코드하지 못했다"고 표시했고, 원인을 찾는 데 시간이 걸렸다. worker가
    검사해서 실패한 것과 worker에 닿지 못해 검사 자체를 못 한 것은 다른
    축이다.
    """
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def refused_urlopen(request, timeout: float) -> FakeHTTPResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refused_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        camera = _register_camera_for_probe(client, "Unreachable", "rtsp://cam.local/s")
        tested = client.post(f"/api/v1/cameras/{camera['id']}/test", headers=AUTH)

    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is False
    assert body["probe_unavailable"] is True
    # 검사를 못 했으므로 어떤 분류도 단정하지 않는다.
    assert "error_class" not in body


def test_probe_reports_unavailable_when_worker_request_times_out(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker로 보낸 HTTP 요청의 timeout은 RTSP timeout이 아니다 (이슈 #151).

    RTSP가 timeout하면 worker가 살아서 payload로 ``error_class="timeout"``을
    알려준다. 여기까지 왔다는 건 worker가 제때 답을 못 했다는 뜻이므로,
    카메라 탓으로 돌리면 안 된다.
    """
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def timing_out_urlopen(request, timeout: float) -> FakeHTTPResponse:
        raise TimeoutError

    monkeypatch.setattr("urllib.request.urlopen", timing_out_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        camera = _register_camera_for_probe(client, "Slow worker", "rtsp://cam.local/s")
        tested = client.post(f"/api/v1/cameras/{camera['id']}/test", headers=AUTH)

    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is False
    assert body["probe_unavailable"] is True
    assert "error_class" not in body


def test_probe_reports_unavailable_when_probe_origin_is_unset(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """probe origin이 비어 있으면 요청을 보낼 주소조차 없다 (이슈 #151)."""
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")

    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        camera = _register_camera_for_probe(client, "No probe origin", "rtsp://cam.local/s")
        tested = client.post(f"/api/v1/cameras/{camera['id']}/test", headers=AUTH)

    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is False
    assert body["probe_unavailable"] is True
    assert "error_class" not in body


def test_probe_keeps_auth_classification_when_worker_answers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker가 살아 있을 때의 auth/timeout/decode 분류는 그대로 유지한다.

    이슈 #151의 원래 증상이 이것이다: RTSP 401을 "디코드 실패"로 오진했다.
    worker가 응답한 이상 그 판정은 신뢰할 수 있으므로 그대로 실어 보낸다.
    """
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "http://worker.local:8090")

    def unauthorized_urlopen(request, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse({"ok": False, "error_class": "auth"})

    monkeypatch.setattr("urllib.request.urlopen", unauthorized_urlopen)
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")

    with TestClient(app) as client:
        _login(client)
        camera = _register_camera_for_probe(client, "Bad password", "rtsp://cam.local/s")
        tested = client.post(f"/api/v1/cameras/{camera['id']}/test", headers=AUTH)

    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is False
    assert body["error_class"] == "auth"
    # worker가 실제로 검사했으므로 "검사 불가"가 아니다.
    assert "probe_unavailable" not in body
