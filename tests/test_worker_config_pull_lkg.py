from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from contracts.worker_config import PulledCameraConfig, PulledWorkerConfig
from worker.runtime.config import (
    BackendWorkerConfigPayload,
    CameraRuntimeConfig,
    CameraStreamsConfig,
    ConfigSource,
    NightWindowConfig,
    RelayConfig,
    RestartDirective,
    WorkerConfig,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
    pull_worker_config,
    resolve_effective_config,
)


def test_to_worker_config_threads_pulled_detection_windows_into_domains_config() -> None:
    """pull_models + config_resolver thread pulled detection_windows into the
    resolved WorkerConfig for every domain, not just bed_exit (issue #24)."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": {"start": "21:00", "end": "06:00", "tz": "UTC"},
                "fall": {"start": "22:00", "end": "05:00", "tz": "Asia/Seoul"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows == {
        "bed_exit": NightWindowConfig(start="21:00", end="06:00", tz="UTC"),
        "fall": NightWindowConfig(start="22:00", end="05:00", tz="Asia/Seoul"),
    }
    assert worker_config.domains.resolved_detection_window(
        "bed_exit"
    ) == NightWindowConfig(start="21:00", end="06:00", tz="UTC")


def test_to_worker_config_still_accepts_legacy_night_window_payload_field() -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "night_window": {"start": "21:00", "end": "06:00", "tz": "UTC"},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows == {
        "bed_exit": NightWindowConfig(start="21:00", end="06:00", tz="UTC"),
    }


def test_to_worker_config_drops_start_equal_end_window_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A degenerate start == end window fails open to ALWAYS (dropped from
    detection_windows, logged loudly) rather than being threaded through as
    a window that DetectionWindow.contains would treat as matching nothing."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": {"start": "09:00", "end": "09:00", "tz": "UTC"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    # An empty detection_windows map is normalized to None by to_worker_config
    # (falsy dict -> None), matching DomainsConfig's own default.
    assert worker_config.domains.detection_windows is None
    assert worker_config.domains.resolved_detection_window("bed_exit") is None
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_to_worker_config_drops_invalid_timezone_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "fall": {"start": "22:00", "end": "05:00", "tz": "Not/A_Zone"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows is None
    err = capsys.readouterr().err
    assert "fall" in err


def test_to_pulled_config_drops_malformed_hhmm_and_falls_open(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": {"start": "25:00", "end": "17:00", "tz": "UTC"},
            },
        }
    )

    pulled = payload.to_pulled_config()

    assert pulled.detection_windows == {}
    assert pulled.night_window is None
    err = capsys.readouterr().err
    assert "bed_exit" in err


def test_to_worker_config_drops_explicit_null_domain_entry_without_crashing_payload() -> None:
    """A stray ``null`` for one domain (e.g. a hand-edited LKG file, or a
    version-skewed ml-api) must not fail pydantic validation for the whole
    payload -- it's dropped at parse time (ALWAYS for that domain) exactly
    like the contracts and lifespan.py boundaries, and the rest of the
    payload (other domains, cameras) still parses normally."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                }
            ],
            "detection_windows": {
                "bed_exit": None,
                "fall": {"start": "22:00", "end": "05:00", "tz": "UTC"},
            },
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.domains.detection_windows == {
        "fall": NightWindowConfig(start="22:00", end="05:00", tz="UTC"),
    }
    assert worker_config.domains.resolved_detection_window("bed_exit") is None


def test_pull_worker_config_returns_none_on_urllib_error() -> None:
    # The worker transport (http_transport.stdlib_urlopen) is http.client-based,
    # not urllib.request.urlopen, so the fake transport is injected via the
    # `urlopen` parameter rather than monkeypatched globally.
    def _raise(request: urllib.request.Request, timeout: float) -> object:
        raise urllib.error.URLError("offline")

    assert (
        pull_worker_config("http://ml-api:8000", "token", timeout_sec=0.01, urlopen=_raise)
        is None
    )


def test_resolve_effective_config_pull_ok_overrides_rtsp_and_versions() -> None:
    yaml_config = _yaml_config()
    pulled = _pulled(config_version=11, restart_epoch=5, rtsp_url="rtsp://pulled/sub")

    effective, config_version, restart_epoch, source = resolve_effective_config(
        yaml_config,
        pulled,
    )

    assert effective.cameras[0].streams is not None
    assert effective.cameras[0].streams.sub == "rtsp://pulled/sub"
    assert effective.cameras[1].rtsp_url == "rtsp://yaml/legacy"
    assert config_version == 11
    assert restart_epoch == 5
    assert source == "pulled"


def test_resolve_effective_config_pull_fail_lkg_present_uses_lkg_values() -> None:
    yaml_config = _yaml_config()
    lkg = _pulled(config_version=13, restart_epoch=8, rtsp_url="rtsp://lkg/sub")

    effective, config_version, restart_epoch, source = resolve_effective_config(
        yaml_config,
        lkg,
        source="lkg",
    )

    assert effective.cameras[0].streams is not None
    assert effective.cameras[0].streams.sub == "rtsp://lkg/sub"
    assert config_version == 13
    assert restart_epoch == 8
    assert source == "lkg"


def test_resolve_effective_config_pull_fail_no_lkg_uses_yaml_unchanged() -> None:
    yaml_config = _yaml_config()

    effective, config_version, restart_epoch, source = resolve_effective_config(yaml_config, None)

    assert effective is yaml_config
    assert effective.cameras[0].streams is not None
    assert effective.cameras[0].streams.sub == "rtsp://yaml/sub"
    assert config_version == 0
    assert restart_epoch == 0
    assert source == "yaml"


def _yaml_config() -> WorkerConfig:
    return WorkerConfig(
        relay=RelayConfig(url="http://ml-api:8000", token="token"),
        cameras=(
            CameraRuntimeConfig(
                camera_id="camera-1",
                facility_id="facility-1",
                streams=CameraStreamsConfig(sub="rtsp://yaml/sub", main="rtsp://yaml/main"),
            ),
            CameraRuntimeConfig(
                camera_id="camera-2",
                facility_id="facility-1",
                rtsp_url="rtsp://yaml/legacy",
            ),
        ),
    )


def _pulled(config_version: int, restart_epoch: int, rtsp_url: str) -> PulledWorkerConfig:
    return PulledWorkerConfig(
        config_version=config_version,
        restart_epoch=restart_epoch,
        night_window=None,
        cameras=(
            PulledCameraConfig(
                camera_id="camera-1",
                space_id="space-1",
                label="Room 1",
                rtsp_url=rtsp_url,
                online=True,
            ),
        ),
    )


def test_unavailable_pull_returns_none_and_preserves_existing_lkg(tmp_path) -> None:
    # Regression: when ml-api has no backend config it returns 503, so the pull
    # MUST return None and the worker MUST keep its existing LKG (not overwrite
    # it with an empty placeholder). config_pull.load_worker_config_from_relay
    # only persists the LKG (WorkerConfigLkgStore's config_current/
    # config_history tables in worker-state.sqlite3) on a successful,
    # validated pull.
    assert (
        pull_worker_config("http://ml-api:8000", "token", timeout_sec=0.01, urlopen=_raise_503)
        is None
    )

    store = WorkerConfigLkgStore(tmp_path / "worker-config.sqlite3")
    good_payload = {
        "registry_version": 5,
        "config_version": 5,
        "restart_epoch": 1,
        "cameras": [
            {
                "camera_id": "camera-1",
                "facility_id": "facility-1",
                "rtsp_url": "rtsp://lkg/good",
            }
        ],
    }

    def _respond_good(request: urllib.request.Request, timeout: float) -> object:
        return _FakeResponse(good_payload)

    fresh = load_worker_config_from_relay(
        "http://ml-api:8000",
        "token",
        store=store,
        urlopen=_respond_good,
    )
    assert fresh is not None
    assert fresh.source is ConfigSource.PULLED

    stale = load_worker_config_from_relay(
        "http://ml-api:8000",
        "token",
        store=store,
        urlopen=_raise_503,
    )

    # Existing LKG is intact: an unavailable pull never clobbers last-known-good.
    assert stale is not None
    assert stale.source is ConfigSource.LKG
    assert stale.registry_version == 5
    assert stale.directive == RestartDirective(generation=1, version=5)


def _raise_503(request: urllib.request.Request, timeout: float) -> object:
    raise urllib.error.HTTPError(
        "http://ml-api:8000/api/v1/cameras/worker-config", 503, "unavailable", {}, None
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status = 200

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")
