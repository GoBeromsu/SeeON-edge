from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from worker.runtime.config import (
    BackendWorkerConfigPayload,
    ConfigSource,
    NightWindowConfig,
    RestartDirective,
    WorkerConfigLkgStore,
    load_worker_config_from_relay,
    pull_worker_config,
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


def test_to_worker_config_threads_bed_zone_polygon_into_camera_runtime_config() -> None:
    """The persisted bed-zone polygon (see the bed-zone recognize endpoint)
    is pulled down as part of the worker config and must survive the
    ``_CameraPayload`` -> ``CameraRuntimeConfig`` conversion unchanged, so
    ``WorkerRuntime._build_camera`` can seed ``SceneState.persisted_bed_regions``
    from it (issue: on-demand bed-zone recognition)."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 5,
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-1/stream",
                    "bed_zone_polygon": [[1, 2], [9, 2], [9, 8], [1, 8]],
                    "bed_zone_image_width": 640,
                    "bed_zone_image_height": 480,
                },
                {
                    "camera_id": "camera-2",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://camera-2/stream",
                },
            ],
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    cameras = {camera.camera_id: camera for camera in worker_config.cameras}
    assert cameras["camera-1"].bed_zone_polygon == ((1, 2), (9, 2), (9, 8), (1, 8))
    assert cameras["camera-1"].bed_zone_image_width == 640
    assert cameras["camera-1"].bed_zone_image_height == 480
    assert cameras["camera-2"].bed_zone_polygon is None
    assert cameras["camera-2"].bed_zone_image_width is None
    assert cameras["camera-2"].bed_zone_image_height is None


def test_to_worker_config_with_empty_camera_list_boots_with_an_empty_roster() -> None:
    """Issue #150: a fresh install with zero registered cameras must still
    resolve to a bootable ``WorkerConfig`` -- ``to_worker_config`` used to
    raise ``WorkerConfigError("worker config must include at least one
    camera")`` here, which ``config_pull.py`` swallowed as a malformed pull,
    so the worker could never boot before its first camera existed."""
    payload = BackendWorkerConfigPayload.model_validate({"config_version": 1, "cameras": []})

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.cameras == ()


def test_to_worker_config_with_every_camera_missing_rtsp_url_boots_with_an_empty_roster() -> None:
    """Same as the empty-list case above, but for the more realistic
    mid-onboarding shape: cameras exist in the dashboard but none has an RTSP
    URL yet, so every entry is dropped by the ``camera.rtsp_url is not None``
    filter and the resolved roster is empty rather than raising."""
    payload = BackendWorkerConfigPayload.model_validate(
        {
            "config_version": 1,
            "cameras": [
                {"camera_id": "camera-1", "facility_id": "facility-1", "rtsp_url": None},
            ],
        }
    )

    worker_config = payload.to_worker_config("http://relay.test", "relay-token")

    assert worker_config.cameras == ()


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


def test_load_on_fresh_central_edge_db_returns_none_not_migration_error(tmp_path) -> None:
    # Regression: on the production central edge DB path (edge.sqlite3),
    # open_connection routes through backend.app.edge_db.open_runtime_database,
    # which raises MigrationRequiredError (an EdgeDatabaseError, not a
    # sqlite3.Error) when the file does not exist yet. An unprovisioned
    # first boot must degrade to "no LKG" (None) so `--check-config`'s static
    # path exits 0 without touching disk -- it must NOT propagate and crash.
    store = WorkerConfigLkgStore(tmp_path / "edge.sqlite3")
    assert not (tmp_path / "edge.sqlite3").exists()

    assert store.load() is None
    # Read-only: reporting "no cache" must not provision the central DB.
    assert not (tmp_path / "edge.sqlite3").exists()


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
