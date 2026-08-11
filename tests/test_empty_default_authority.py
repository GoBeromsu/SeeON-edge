"""Empty-default authority contracts (Wave B: retire env inventory + env facility).

Locks: registry-only cameras, connection DB-only facility_id, worker pull without
facility stamp, relay without env facility gates, status never_seen from registry,
and production grep-gates against reintroduced inventory/facility env authority.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettingsStore,
)
from backend.app.features.status.heartbeat_store import HeartbeatStore
from backend.app.lifespan import API_FACILITY_ID_ENV, EDGE_FACILITY_TOKEN_ENV
from backend.app.main import create_app, no_lifespan
from worker.runtime.config.pull_models import BackendWorkerConfigPayload

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (
    REPO_ROOT / "backend",
    REPO_ROOT / "worker",
    REPO_ROOT / "shared",
)


def _client_with_registry(tmp_path: Path, *, camera_id: str = "cam-1") -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    store.create(
        camera_id=camera_id,
        label="Room",
        rtsp_url="rtsp://example/stream",
        space_id=None,
        status="online",
        backend_camera_id=camera_id,
    )
    app.state.camera_registry = store
    return TestClient(app)


class TestNoCameraInventoryAuthority:
    def test_lifespan_ignores_api_camera_inventory_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(
            "API_CAMERA_INVENTORY",
            '[{"camera_id":"ghost","facility_id":"fac"}]',
        )
        monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
        monkeypatch.setenv(
            "API_CONNECTION_SETTINGS_PATH",
            str(tmp_path / "connection.sqlite3"),
        )
        with TestClient(create_app()) as client:
            assert not hasattr(client.app.state, "camera_inventory")
            body = client.get("/api/v1/status").json()
            assert "ghost" not in body["cameras"]


class TestConnectionFacilityDbOnly:
    def test_env_facility_id_does_not_seed_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_FACILITY_ID_ENV, "facility-from-env")
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://api.example.com")
        monkeypatch.delenv(EDGE_FACILITY_TOKEN_ENV, raising=False)
        store = ConnectionSettingsStore(tmp_path / "connection.sqlite3")
        settings = store.load()
        assert settings.facility_id is None
        assert settings.events_url is not None

    def test_configured_requires_facility_id_and_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "API_CONNECTION_SETTINGS_PATH", str(tmp_path / "connection.sqlite3")
        )
        monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://api.example.com")
        store = ConnectionSettingsStore.from_env()
        store.save({"events_url": "https://api.example.com/api/v1/events"})
        app = create_app(lifespan=no_lifespan)
        client = TestClient(app)
        login = client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        )
        assert login.status_code == 204
        body = client.get("/api/v1/connection").json()
        assert body["configured"] is False
        store.save({"facility_id": "fac-1", "facility_token": "tok-secret"})
        body = client.get("/api/v1/connection").json()
        assert body["configured"] is True


class TestWorkerConfigPullLocalFacility:
    def test_camera_id_and_rtsp_enough_defaults_facility_local(self) -> None:
        payload = BackendWorkerConfigPayload.model_validate(
            {
                "config_version": 1,
                "cameras": [
                    {"camera_id": "cam-only", "rtsp_url": "rtsp://cam/1"},
                ],
            }
        )
        config = payload.to_worker_config("http://ml-api:8000", "relay-token")
        assert len(config.cameras) == 1
        assert config.cameras[0].facility_id == "local"

    def test_worker_config_snapshot_omits_facility_stamp(
        self, tmp_path: Path
    ) -> None:
        client = _client_with_registry(tmp_path)
        response = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
        assert response.status_code == 200
        cameras = response.json()["cameras"]
        assert len(cameras) == 1
        assert "facility_id" not in cameras[0]
        assert cameras[0]["camera_id"] == "cam-1"
        assert cameras[0]["rtsp_url"] == "rtsp://example/stream"


class TestRelayRegistryOnly:
    def test_alert_accepts_registry_camera_without_env_facility(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("API_FACILITY_ID", raising=False)
        client = _client_with_registry(tmp_path)
        # No backend client: local accept still OK; binding must not 403.
        response = client.post(
            "/api/v1/relay/alerts",
            headers={"X-Edge-Relay-Token": "relay-token"},
            json={
                "event_type": "fall",
                "probability": 0.9,
                "detected_at": "2026-06-25T12:00:00.000Z",
                "camera_id": "cam-1",
                "facility_id": "local",
            },
        )
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"

    def test_alert_unknown_camera_without_inventory_rescue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("API_FACILITY_ID", raising=False)
        app = create_app(lifespan=no_lifespan)
        app.state.edge_relay_token = "relay-token"
        app.state.camera_registry = CameraRegistryStore(tmp_path / "empty.sqlite3")
        # Leftover inventory must not rescue unknown cameras.
        app.state.camera_inventory = {
            "ghost": {"camera_id": "ghost", "facility_id": "fac"}
        }
        client = TestClient(app)
        response = client.post(
            "/api/v1/relay/alerts",
            headers={"X-Edge-Relay-Token": "relay-token"},
            json={
                "event_type": "fall",
                "probability": 0.9,
                "detected_at": "2026-06-25T12:00:00.000Z",
                "camera_id": "ghost",
                "facility_id": "fac",
            },
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "unknown camera"

    def test_runtime_status_ignores_env_facility_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_FACILITY_ID", "facility-configured")
        app = create_app(lifespan=no_lifespan)
        app.state.edge_relay_token = "relay-token"
        client = TestClient(app)
        response = client.post(
            "/api/v1/relay/runtime-status",
            headers={"X-Edge-Relay-Token": "relay-token"},
            json={
                "facility_id": "facility-other",
                "generation": None,
                "seq": 0,
                "cameras": [],
                "clip_recorder": {
                    "available": True,
                    "dropped_frames": 0,
                    "dropped_events": 0,
                    "failed_writes": 0,
                    "finalized_clips": 0,
                    "video_unavailable_clips": 0,
                    "active_clips": 0,
                    "encoder": "libx264",
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["accepted"] is True


class TestStatusFromRegistry:
    def test_never_seen_comes_from_registry_not_inventory(
        self, tmp_path: Path
    ) -> None:
        app = create_app(lifespan=no_lifespan)
        store = CameraRegistryStore(tmp_path / "catalog.sqlite3")
        store.create(
            camera_id="reg-a",
            label="A",
            rtsp_url="rtsp://a",
            space_id=None,
            status="online",
        )
        store.create(
            camera_id="reg-b",
            label="B",
            rtsp_url="rtsp://b",
            space_id=None,
            status="online",
        )
        app.state.camera_registry = store
        app.state.camera_inventory = {
            "inv-ghost": {"camera_id": "inv-ghost", "facility_id": "fac"}
        }
        beats = HeartbeatStore(stale_after_sec=90.0)
        beats.record("reg-a", "local")
        app.state.heartbeat_store = beats

        body = TestClient(app).get("/api/v1/status").json()
        assert body["cameras"]["reg-a"]["status"] == "online"
        assert body["cameras"]["reg-b"]["status"] == "never_seen"
        assert "inv-ghost" not in body["cameras"]


class TestLkgEmptyPull:
    def test_successful_empty_cameras_pull_becomes_current_lkg(
        self, tmp_path: Path
    ) -> None:
        from worker.runtime.config.config_pull import load_worker_config_from_relay
        from worker.runtime.config.lkg_store import WorkerConfigLkgStore

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self) -> bytes:
                return b'{"config_version": 3, "cameras": []}'

        store = WorkerConfigLkgStore(database_path=tmp_path / "worker-state.sqlite3")
        # Seed a non-empty older LKG so empty pull must replace it as current.
        store.save(
            {
                "config_version": 2,
                "cameras": [
                    {
                        "camera_id": "stale-cam",
                        "rtsp_url": "rtsp://stale",
                        "facility_id": "old",
                    }
                ],
            },
            __import__(
                "worker.runtime.config.restart", fromlist=["RestartDirective"]
            ).RestartDirective(generation=0, version=2),
        )

        def urlopen(request, timeout=5.0):  # noqa: ARG001
            return _Resp()

        snap = load_worker_config_from_relay(
            "http://ml-api:8000",
            "relay-token",
            store=store,
            urlopen=urlopen,
        )
        assert snap is not None
        assert snap.config.cameras == ()
        current = store.load()
        assert current is not None
        assert current.payload.get("cameras") == []
        assert current.directive.version == 3


class TestProductionGrepGates:
    """Fail if production code reintroduces retired authorities."""

    FORBIDDEN_SUBSTRINGS = (
        "camera_inventory",
        "API_CAMERA_INVENTORY",
        "def _facility_id",
    )
    FORBIDDEN_ENV_READ_PATTERNS = (
        'os.environ.get("API_FACILITY_ID"',
        "os.environ.get(API_FACILITY_ID_ENV",
        'os.environ["API_FACILITY_ID"]',
        "os.environ[API_FACILITY_ID_ENV",
    )

    def test_no_forbidden_production_authority(self) -> None:
        offenders: list[str] = []
        for root in PRODUCTION_ROOTS:
            for path in root.rglob("*.py"):
                if path.name == "test_empty_default_authority.py":
                    continue
                text = path.read_text(encoding="utf-8")
                rel = path.relative_to(REPO_ROOT)
                for needle in self.FORBIDDEN_SUBSTRINGS:
                    if needle in text:
                        offenders.append(f"{rel}: {needle}")
                for needle in self.FORBIDDEN_ENV_READ_PATTERNS:
                    if needle in text:
                        offenders.append(f"{rel}: {needle}")
                # AST pass: catch attribute writes to camera_inventory on app.state
                try:
                    tree = ast.parse(text, filename=str(path))
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and node.attr == "camera_inventory":
                        offenders.append(f"{rel}: attribute camera_inventory")
        assert offenders == [], "retired authority still present:\n" + "\n".join(offenders)
