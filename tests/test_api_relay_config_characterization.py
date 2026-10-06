from __future__ import annotations

import json
from typing import Any, Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.main import create_app, no_lifespan
from contracts.worker_config import PulledWorkerConfig

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


class _FakeDetectionPolicyStore:
    def generation(self, _facility_id: str | None) -> int:
        return 0


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
    app: FastAPI,
    registry_snapshot: dict[str, Any],
    runtime_enabled: bool = False,
    runtime_version: int = 0,
) -> None:
    import backend.app.features.cameras.router as cameras_router

    class _FakeConnSettingsStore:
        def load(self) -> Any:
            class _Loaded:
                facility_id: str = "facility-1"

            return _Loaded()

    # Camera registry and bed zones
    app.state.camera_registry = _FakeRegistryStore(registry_snapshot)
    monkeypatch.setattr(
        cameras_router,
        "_store",
        lambda app: _FakeRegistryStore(registry_snapshot),
        raising=True,
    )
    # Relay route reads app.state bed zone store; seed a stub there
    class _FakeBedZoneStoreState:
        def get_all(self) -> dict[str, Any]:
            return {}

    app.state.bed_zone_store = _FakeBedZoneStoreState()
    monkeypatch.setattr(cameras_router, "_bed_zone_store", lambda app: _FakeBedZoneStore())
    # Clip storage location store (empty selection by default -> key absent)
    class _FakeClipStorageLocationStore:
        def get(self) -> str:
            return ""

    app.state.clip_storage_location_store = _FakeClipStorageLocationStore()
    monkeypatch.setattr(
        cameras_router,
        "_clip_storage_location_store",
        lambda app: _FakeClipStorageLocationStore(),
        raising=True,
    )
    # No local detection overrides by default
    class _FakeDetectionSettingsStoreState:
        def __init__(self) -> None:
            self._inner = _FakeDetectionSettingsStore({})

        def get_all(self) -> dict[str, Any]:
            return self._inner.get_all()

    app.state.detection_settings_store = _FakeDetectionSettingsStoreState()
    monkeypatch.setattr(
        cameras_router, "_detection_settings_store", lambda app: _FakeDetectionSettingsStore({})
    )
    # Connection facility_id lookup (used only when policies are present)
    monkeypatch.setattr(
        cameras_router,
        "get_connection_settings_store",
        lambda app: _FakeConnSettingsStore(),
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
    class _FakePolicyStoreState:
        def generation(self, _facility_id: str | None) -> int:
            return 0

    app.state.detection_policy_store = _FakePolicyStoreState()
    monkeypatch.setattr(
        cameras_router,
        "_detection_policy_store",
        lambda app: _FakeDetectionPolicyStore(),
        raising=True,
    )


def _dump_minified(obj: dict[str, object]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_relay_config_byte_snapshot_no_policies_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
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
    _patch_minimal_dependencies(monkeypatch, app=app, registry_snapshot=registry, runtime_enabled=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/relay/config", headers=RELAY_HEADERS)
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

