from __future__ import annotations

import json
import stat
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Self, final

import pytest

from worker.runtime.config import (
    ConfigSource,
    JsonObject,
    RestartDirective,
    WorkerConfig,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
    resolve_startup_config,
)


@final
class FakeResponse:
    def __init__(self, payload: JsonObject, status: int = 200) -> None:
        self._payload: JsonObject = payload
        self.status: int = status

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_fresh_pull_uses_auth_and_replaces_lkg_atomically(tmp_path: Path) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    captured: list[tuple[str, str | None, float]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        captured.append(
            (
                request.full_url,
                request.get_header("X-edge-relay-token"),
                timeout,
            )
        )
        return FakeResponse(_payload(registry_version=9, config_version=12, restart_epoch=4))

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        timeout_sec=0.25,
        store=store,
        urlopen=fake_urlopen,
    )

    assert snapshot is not None
    assert snapshot.source is ConfigSource.PULLED
    assert snapshot.stale is False
    assert snapshot.registry_version == 9
    assert snapshot.directive == RestartDirective(generation=4, version=12)
    assert snapshot.config.cameras[0].inference_rtsp_url == "rtsp://user:camera-pass@camera/live"
    assert captured == [
        (
            "http://ml-api:8000/api/v1/cameras/worker-config",
            "relay-secret",
            0.25,
        )
    ]
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_unreachable_backend_uses_stale_lkg_without_zeroing_cameras(tmp_path: Path) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    fresh = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=3, config_version=7, restart_epoch=2)
        ),
    )
    assert fresh is not None

    def offline(_request: urllib.request.Request, _timeout: float) -> FakeResponse:
        raise urllib.error.URLError("offline")

    stale = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=offline,
    )

    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.stale is True
    assert stale.registry_version == 3
    assert tuple(camera.camera_id for camera in stale.config.cameras) == ("camera-1",)


def test_malformed_pull_keeps_prior_lkg_and_redacts_secrets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    fresh = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=3, config_version=7, restart_epoch=2)
        ),
    )
    assert fresh is not None
    malformed = _payload(registry_version=4, config_version=8, restart_epoch=2)
    malformed["cameras"] = [
        {
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "rtsp_url": "rtsp://user:leaked-camera-password@camera/live",
            "fps": "invalid-fps",
        }
    ]

    stale = load_worker_config_from_relay(
        "http://ml-api:8000",
        "leaked-relay-token",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(malformed),
    )

    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.directive == RestartDirective(generation=2, version=7)
    error = capsys.readouterr().err
    assert "leaked-camera-password" not in error
    assert "leaked-relay-token" not in error
    assert "camera/live" not in error


def test_concurrent_lkg_writes_preserve_newest_directive(tmp_path: Path) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    barrier = threading.Barrier(3)

    def save(version: int) -> bool:
        _ = barrier.wait()
        return store.save(
            _payload(registry_version=version, config_version=version, restart_epoch=1),
            RestartDirective(generation=1, version=version),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(save, 8)
        newer = executor.submit(save, 9)
        _ = barrier.wait()
        _ = older.result()
        _ = newer.result()

    stored = store.load()
    assert stored is not None
    assert stored.directive == RestartDirective(generation=1, version=9)
    assert stored.payload["registry_version"] == 9
    assert json.loads(store.path.read_text(encoding="utf-8"))["payload"][
        "registry_version"
    ] == 9


def test_race_loss_with_healthy_stored_lkg_returns_lkg_snapshot(tmp_path: Path) -> None:
    """A fresh pull that validates but loses the revision race to a strictly
    newer, still-healthy on-disk LKG must fall back to that LKG unchanged
    (issue #34, race-loss branch with a healthy stored LKG)."""
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    newer_payload = _payload(registry_version=9, config_version=9, restart_epoch=1)
    assert store.save(newer_payload, RestartDirective(generation=1, version=9))

    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(
            _payload(registry_version=5, config_version=5, restart_epoch=1)
        ),
    )

    assert snapshot is not None
    assert snapshot.source is ConfigSource.LKG
    assert snapshot.stale is True
    assert snapshot.registry_version == 9
    assert snapshot.directive == RestartDirective(generation=1, version=9)
    # The healthy LKG that won the race is left in place.
    stored = store.load()
    assert stored is not None
    assert stored.registry_version == 9


def test_race_loss_with_corrupt_stored_lkg_deletes_lkg_and_returns_fresh(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fresh pull that validates but loses the revision race to a strictly
    newer on-disk LKG that no longer re-validates must not leave the corrupt
    LKG in place -- it would keep winning the race forever. Instead, delete
    the corrupt LKG, log loudly, and return the fresh (race-losing) snapshot
    since a parseable-but-older config beats an unparseable newer one
    (issue #34, decided design option 3)."""
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")
    corrupt_payload = _payload(registry_version=9, config_version=9, restart_epoch=1)
    corrupt_payload["cameras"] = [
        {
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "rtsp_url": "rtsp://user:camera-pass@camera/live",
            "fps": "invalid-fps",
        }
    ]
    assert store.save(corrupt_payload, RestartDirective(generation=1, version=9))
    assert store.path.exists()

    fresh_payload = _payload(registry_version=5, config_version=5, restart_epoch=1)
    snapshot = load_worker_config_from_relay(
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=lambda _request, _timeout: FakeResponse(fresh_payload),
    )

    assert snapshot is not None
    assert snapshot.source is ConfigSource.PULLED
    assert snapshot.stale is False
    assert snapshot.registry_version == 5
    assert snapshot.directive == RestartDirective(generation=1, version=5)
    # The corrupt LKG must be deleted, not left behind to keep winning the
    # revision race against every future legitimate pull.
    assert not store.path.exists()
    error = capsys.readouterr().err
    assert "WARNING" in error
    assert str(store.path) in error


def test_offline_without_lkg_falls_back_to_yaml(tmp_path: Path) -> None:
    store = WorkerConfigLkgStore(tmp_path / "worker-config.json")

    def offline(_request: urllib.request.Request, _timeout: float) -> FakeResponse:
        raise TimeoutError

    snapshot = resolve_startup_config(
        _yaml_config(),
        "http://ml-api:8000",
        "relay-secret",
        store=store,
        urlopen=offline,
    )

    assert snapshot.source is ConfigSource.YAML
    assert snapshot.stale is True
    assert snapshot.config.cameras[0].camera_id == "yaml-camera"
    assert snapshot.directive == RestartDirective(generation=0, version=0)


def _payload(
    *, registry_version: int, config_version: int, restart_epoch: int
) -> JsonObject:
    return {
        "registry_version": registry_version,
        "config_version": config_version,
        "restart_epoch": restart_epoch,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://user:camera-pass@camera/live",
                "fps": 7.5,
                "domains": ["fall"],
            }
        ],
    }


def _yaml_config() -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "relay": {"url": "http://ml-api:8000", "token": "relay-secret"},
            "cameras": [
                {
                    "camera_id": "yaml-camera",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://yaml/camera",
                }
            ],
        }
    )
