from __future__ import annotations

import json
from typing import Any, Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig

# Worker relay auth
RELAY_HEADER_NAME: Final = "X-Edge-Relay-Token"
RELAY_TOKEN: Final = "relay-token"
RELAY_HEADERS = {RELAY_HEADER_NAME: RELAY_TOKEN}


class _FakeRegistryStore:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot


class _FakeBedZoneStore:
    def get_all(self) -> dict[str, Any]:
        return {}


class _FakeRuntimeSettingsStore:
    def __init__(self, *, enabled: bool, version: int) -> None:
        self._enabled = enabled
        self._version = version

    def get(self) -> Any:
        class _Setting:
            clip_export_enabled: bool = self._enabled
            version: int = self._version

        return _Setting()


class _FakeDetectionSettingsStore:
    class _Setting:
        def __init__(self, on: bool, mode: str, start: str | None, end: str | None) -> None:
            self.on = on
            self.mode = mode
            self.start = start
            self.end = end

        def as_dict(self) -> dict[str, object]:
            return {"on": self.on, "mode": self.mode, "start": self.start, "end": self.end}

    def __init__(self, domains: dict[str, dict[str, object]]) -> None:
        self._domains = {
            name: _FakeDetectionSettingsStore._Setting(
                on=bool(entry.get("on", False)),
                mode=str(entry.get("mode", "always")),
                start=(None if entry.get("start") is None else str(entry["start"])),
                end=(None if entry.get("end") is None else str(entry["end"])),
            )
            for name, entry in domains.items()
        }

    def get_all(self) -> dict[str, Any]:
        return self._domains


class _FakeDetectionPolicyBundle:
    def __init__(self, *, content_sha256: str, payload: dict[str, object]) -> None:
        self.content_sha256 = content_sha256
        self._payload = payload

    def as_dict(self) -> dict[str, object]:
        return self._payload


class _FakeDetectionPolicyStore:
    def __init__(self, *, generation: int, bundle: _FakeDetectionPolicyBundle | None) -> None:
        self._generation = generation
        self._bundle = bundle

    def generation(self, _facility_id: str | None) -> int:
        return self._generation

    def resolve_bundle(
        self, _facility_id: str | None, _cameras: tuple[object, ...]
    ) -> _FakeDetectionPolicyBundle:
        assert self._bundle is not None, "resolve_bundle called without a configured bundle"
        return self._bundle

    def acknowledge_applied(self, _facility_id: str) -> None:  # pragma: no cover - not exercised
        return


def _app() -> FastAPI:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = RELAY_TOKEN
    # Stable pulled baseline seen by the route
    app.state.pulled_config = PulledWorkerConfig(
        config_version=7,
        restart_epoch=2,
        night_window=None,
        cameras=(),
        detection_windows={},
    )
    app.state.config_version = 7
    app.state.restart_epoch = 2
    return app


def _patch_minimal_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry_snapshot: dict[str, Any],
    runtime_enabled: bool = False,
    runtime_version: int = 0,
) -> None:
    import backend.app.features.cameras.router as cameras_router

    # Camera registry and bed zones
    monkeypatch.setattr(
        cameras_router,
        "_store",
        lambda app: _FakeRegistryStore(registry_snapshot),
        raising=True,
    )
    monkeypatch.setattr(cameras_router, "_bed_zone_store", lambda app: _FakeBedZoneStore())
    # Clip storage location store (empty selection by default -> key absent)
    class _FakeClipStorageLocationStore:
        def get(self) -> str:
            return ""

    monkeypatch.setattr(
        cameras_router,
        "_clip_storage_location_store",
        lambda app: _FakeClipStorageLocationStore(),
        raising=True,
    )
    # No local detection overrides by default
    monkeypatch.setattr(
        cameras_router, "_detection_settings_store", lambda app: _FakeDetectionSettingsStore({})
    )
    # Connection facility_id lookup (used only when policies are present)
    monkeypatch.setattr(
        cameras_router,
        "get_connection_settings_store",
        lambda app: type("S", (), {"load": lambda self: type("X", (), {"facility_id": "facility-1"})()})(),
        raising=True,
    )
    # Runtime export setting
    monkeypatch.setattr(
        cameras_router,
        "get_runtime_settings_store",
        lambda app: _FakeRuntimeSettingsStore(enabled=runtime_enabled, version=runtime_version),
        raising=True,
    )
    # No numeric policies by default
    monkeypatch.setattr(
        cameras_router,
        "_detection_policy_store",
        lambda app: _FakeDetectionPolicyStore(generation=0, bundle=None),
        raising=True,
    )


def _read_fixture(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _dump_minified(obj: dict[str, object]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_byte_snapshot_no_policies_includes_unmapped_camera_and_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the exact worker-config body when no policies or local overrides exist.

    Covers:
    - unmapped camera stays included (logs only)
    - no detection_policies key
    - runtime export defaults are present (false/version 0)
    """
    app = _app()
    registry = {
        "registry_version": 5,
        "cameras": [
            {
                "id": "local-1",
                "label": "Cam A",
                "rtsp_url": "rtsp://camera.invalid/a",
                "backend_camera_id": None,
                "mapping_pending": False,
                "space_id": "space-101",
                "decode_backend": "auto",
            },
            {
                "id": "local-2",
                "label": "Cam B",
                "rtsp_url": "rtsp://camera.invalid/b",
                "backend_camera_id": "hub-2",
                "mapping_pending": False,
                "space_id": "space-101",
                "decode_backend": "cpu",
            },
        ],
    }
    _patch_minimal_dependencies(monkeypatch, registry_snapshot=registry, runtime_enabled=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
    assert response.status_code == 200

    expected_obj = {
        "registry_version": 5,
        "cameras": [
            {
                "camera_id": "local-1",
                "space_id": "space-101",
                "rtsp_url": "rtsp://camera.invalid/a",
                "decode_backend": "auto",
            },
            {
                "camera_id": "hub-2",
                "space_id": "space-101",
                "rtsp_url": "rtsp://camera.invalid/b",
                "decode_backend": "cpu",
            },
        ],
        "config_version": 7,
        "restart_epoch": 2,
        "clip_export_enabled": False,
        "clip_export_version": 0,
    }
    assert response.content == _dump_minified(expected_obj)


def test_byte_snapshot_with_policies_threads_facility_and_scales_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the exact body when a numeric policy bundle is active.

    Covers:
    - detection_policies present as a closed object
    - facility_id added to each camera
    - config_version scaled (base*1e9 + hashpart), restart_epoch incremented by generation
    """
    app = _app()
    registry = {
        "registry_version": 3,
        "cameras": [
            {
                "id": "local-9",
                "label": "Cam X",
                "rtsp_url": "rtsp://camera.invalid/x",
                "backend_camera_id": "hub-x",
                "mapping_pending": False,
                "space_id": "space-201",
            }
        ],
    }
    _patch_minimal_dependencies(monkeypatch, registry_snapshot=registry, runtime_enabled=False)
    # Patch detection policy store into the router module
    import backend.app.features.cameras.router as cameras_router

    bundle = _FakeDetectionPolicyBundle(
        content_sha256="00000042deadbeefcafebabe0000000000000000000000000000000000000042",
        payload={"module_id": "fall", "schema_id": "fall.policy", "values": {"threshold": 0.7}},
    )
    monkeypatch.setattr(
        cameras_router,
        "_detection_policy_store",
        lambda app: _FakeDetectionPolicyStore(generation=1, bundle=bundle),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
    assert response.status_code == 200

    expected_obj = {
        "registry_version": 3,
        "cameras": [
            {
                "camera_id": "hub-x",
                "facility_id": "facility-1",
                "space_id": "space-201",
                "rtsp_url": "rtsp://camera.invalid/x",
            }
        ],
        "config_version": 7 * 1_000_000_000 + 0x42,
        "restart_epoch": 3,
        "detection_policies": {"module_id": "fall", "schema_id": "fall.policy", "values": {"threshold": 0.7}},
        "clip_export_enabled": False,
        "clip_export_version": 0,
    }
    assert response.content == _dump_minified(expected_obj)


def test_byte_snapshot_with_local_detection_overrides_applies_and_sets_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins merge precedence of local detection overrides over pulled windows.

    Covers:
    - domains map present with enabled flags
    - detection_windows updated with local window and tz reused from pulled
    - night_window mirrors bed_exit window when present
    """
    app = _app()
    # Pulled live window for fall domain carries Asia/Seoul tz that must be reused
    app.state.pulled_config = PulledWorkerConfig(
        config_version=3,
        restart_epoch=1,
        night_window=None,
        cameras=(),
        detection_windows={"fall": PulledNightWindow(start="08:00", end="20:00", tz="Asia/Seoul")},
    )
    app.state.config_version = 3
    app.state.restart_epoch = 1
    registry = {"registry_version": 9, "cameras": []}
    _patch_minimal_dependencies(monkeypatch, registry_snapshot=registry, runtime_enabled=False)
    # Monkeypatch local detection settings
    import backend.app.features.cameras.router as cameras_router

    settings = _FakeDetectionSettingsStore(
        {
            "fall": {"on": True, "mode": "window", "start": "09:00", "end": "18:00"},
            "bed_exit": {"on": True, "mode": "always"},
        }
    )
    monkeypatch.setattr(cameras_router, "_detection_settings_store", lambda app: settings)

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
    assert response.status_code == 200

    expected_obj = {
        "registry_version": 9,
        "cameras": [],
        "config_version": 3
        + 1
        + (int(
            __import__("hashlib")
            .sha256(
                json.dumps(
                    {
                        "domains": {"bed_exit": {"enabled": True}, "fall": {"enabled": True}},
                        "detection_windows": {
                            "fall": {"start": "09:00", "end": "18:00", "tz": "Asia/Seoul"}
                        },
                        "night_window": None,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            )
            .hexdigest()[:8],
            16,
        )
        % 1_000_000),
        "restart_epoch": 1,
        "detection_windows": {"fall": {"start": "09:00", "end": "18:00", "tz": "Asia/Seoul"}},
        "domains": {"fall": {"enabled": True}, "bed_exit": {"enabled": True}},
        "clip_export_enabled": False,
        "clip_export_version": 0,
    }
    assert response.content == _dump_minified(expected_obj)


def test_byte_snapshot_threads_runtime_export_setting_without_restart_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins runtime export settings added without changing restart directive fields."""
    app = _app()
    registry = {"registry_version": 11, "cameras": []}
    _patch_minimal_dependencies(
        monkeypatch, registry_snapshot=registry, runtime_enabled=True, runtime_version=2
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/cameras/worker-config", headers=RELAY_HEADERS)
    assert response.status_code == 200

    expected_obj = {
        "registry_version": 11,
        "cameras": [],
        "config_version": 7,
        "restart_epoch": 2,
        "clip_export_enabled": True,
        "clip_export_version": 2,
    }
    assert response.content == _dump_minified(expected_obj)

