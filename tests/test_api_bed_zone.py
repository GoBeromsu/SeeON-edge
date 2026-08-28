"""Tests for POST /api/v1/cameras/{camera_id}/bed-zone/recognize -- the
dashboard-session-authed proxy to the worker's one-shot bed segmentation
route (see worker/pipeline/output/_mjpeg_http.py's
POST /overlay/{camera_id}/bed-zone/recognize), and its persistence into
BedZoneStore (surfaced back out via GET /cameras' bed_zone field and
GET /cameras/worker-config's bed_zone_polygon/... fields).
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import NoReturn, Self

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.cameras.bed_zone_store import BedZoneStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import LifespanFactory, create_app, no_lifespan

NO_LIFESPAN: LifespanFactory = no_lifespan
RECOGNIZE_PATH = "/api/v1/cameras/camera-1/bed-zone/recognize"


@pytest.fixture(autouse=True)
def _migrated_compact_database(tmp_path: Path) -> None:
    bootstrap_database(tmp_path / "catalog.sqlite3")


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


class FakeUpstreamResponse:
    status: int = 200
    headers: dict[str, str] = {"Content-Type": "application/json"}

    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
        return self._body

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()


def _http_error(code: int, payload: dict[str, object] | None = None) -> urllib.error.HTTPError:
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    return urllib.error.HTTPError(
        "http://worker.local:8090/overlay/camera-1/bed-zone/recognize",
        code,
        "upstream status",
        {},
        io.BytesIO(body),
    )


@pytest.fixture(autouse=True)
def bed_zone_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", "http://worker.local:8090")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _app(tmp_path: Path):
    app = create_app(lifespan=NO_LIFESPAN)
    registry = CameraRegistryStore(tmp_path / "catalog.sqlite3")
    registry.create(
        camera_id="camera-1",
        label="Bed camera",
        rtsp_url="rtsp://camera.invalid/live",
        space_id=None,
        status="unknown",
    )
    app.state.camera_registry = registry
    app.state.bed_zone_store = BedZoneStore(tmp_path / "catalog.sqlite3")
    return app


def test_recognize_success_persists_and_returns_the_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUpstreamResponse:
        calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "headers": dict(request.headers),
            }
        )
        return FakeUpstreamResponse(
            {"polygon": [[1, 2], [9, 2], [9, 8], [1, 8]], "image_width": 640, "image_height": 480}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(RECOGNIZE_PATH)

    assert response.status_code == 200
    body = response.json()
    assert body["bed_zone"]["polygon"] == [[1, 2], [9, 2], [9, 8], [1, 8]]
    assert body["bed_zone"]["image_width"] == 640
    assert body["bed_zone"]["image_height"] == 480
    assert isinstance(body["bed_zone"]["recognized_at"], str) and body["bed_zone"]["recognized_at"]
    # Relay token stays server-side; browser response must not disclose it.
    assert "relay-token" not in response.text
    assert "X-Edge-Relay-Token" not in response.headers
    assert calls == [
        {
            "url": "http://worker.local:8090/overlay/camera-1/bed-zone/recognize",
            "method": "POST",
            "timeout": 8.0,
            "headers": {"X-edge-relay-token": "relay-token"},
        }
    ]


def test_recognize_persists_across_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUpstreamResponse:
        del request, timeout
        return FakeUpstreamResponse(
            {"polygon": [[1, 2], [9, 2], [9, 8], [1, 8]], "image_width": 640, "image_height": 480}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = _app(tmp_path)

    with TestClient(app) as client:
        _login(client)
        client.post(RECOGNIZE_PATH)

    store: BedZoneStore = app.state.bed_zone_store
    saved = store.get("camera-1")
    assert saved is not None
    assert saved.polygon == ((1, 2), (9, 2), (9, 8), (1, 8))


def test_recognize_bed_not_found_maps_to_a_structured_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise _http_error(404, {"error_class": "bed_not_found"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(RECOGNIZE_PATH)

    assert response.status_code == 422
    assert response.json()["detail"] == {"error_class": "bed_not_found"}


def test_recognize_bed_not_found_does_not_persist_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise _http_error(404, {"error_class": "bed_not_found"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    app = _app(tmp_path)

    with TestClient(app) as client:
        _login(client)
        client.post(RECOGNIZE_PATH)

    store: BedZoneStore = app.state.bed_zone_store
    assert store.get("camera-1") is None


@pytest.mark.parametrize(
    "raise_error",
    [
        lambda: _http_error(404),  # unstructured 404 (e.g. unknown camera at the worker)
        lambda: _http_error(503),
        lambda: urllib.error.URLError("connection refused"),
        lambda: TimeoutError("timed out"),
        lambda: OSError("connection reset"),
    ],
    ids=["plain_404", "worker_503", "connection_refused", "bare_timeout_error", "bare_os_error"],
)
def test_recognize_reports_worker_unavailability_as_503(
    raise_error, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise raise_error()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(RECOGNIZE_PATH)

    assert response.status_code == 503


class _ReadFailsUpstreamResponse:
    status: int = 200
    headers: dict[str, str] = {"Content-Type": "application/json"}

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
        raise self._error

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "make_error",
    [lambda: TimeoutError("timed out"), lambda: OSError("connection reset")],
    ids=["timeout_error", "os_error"],
)
def test_recognize_reports_a_read_timeout_as_503_not_500(
    make_error, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TimeoutError (or other OSError) raised while reading the response
    body -- e.g. the connection was accepted but the worker stalled mid
    response -- must map to the same clean 503 as an upfront connect/read
    timeout, not surface as an unhandled 500."""
    upstream = _ReadFailsUpstreamResponse(make_error())

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _ReadFailsUpstreamResponse:
        del request, timeout
        return upstream

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(RECOGNIZE_PATH)

    assert response.status_code == 503
    assert upstream.closed is True


def test_recognize_rejects_a_malformed_upstream_payload_as_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUpstreamResponse:
        del request, timeout
        return FakeUpstreamResponse({"polygon": [[1, 2]], "image_width": 640, "image_height": 480})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(RECOGNIZE_PATH)

    assert response.status_code == 503


def test_recognize_requires_a_dashboard_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> NoReturn:
        del request, timeout
        raise AssertionError("upstream must not be called without a dashboard session")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(_app(tmp_path)) as client:
        response = client.post(RECOGNIZE_PATH)

    assert response.status_code == 401


def test_get_cameras_includes_a_null_bed_zone_before_recognition(tmp_path: Path) -> None:
    app = _app(tmp_path)

    with TestClient(app) as client:
        _login(client)
        response = client.get("/api/v1/cameras")

    assert response.status_code == 200
    cameras = response.json()["cameras"]
    assert len(cameras) == 1
    assert cameras[0]["bed_zone"] is None


def test_get_cameras_includes_the_bed_zone_after_recognition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUpstreamResponse:
        del request, timeout
        return FakeUpstreamResponse(
            {"polygon": [[1, 2], [9, 2], [9, 8], [1, 8]], "image_width": 640, "image_height": 480}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with TestClient(app) as client:
        _login(client)
        assert client.post(RECOGNIZE_PATH).status_code == 200

        response = client.get("/api/v1/cameras")

    cameras = response.json()["cameras"]
    assert len(cameras) == 1
    assert cameras[0]["bed_zone"] == {
        "polygon": [[1, 2], [9, 2], [9, 8], [1, 8]],
        "image_width": 640,
        "image_height": 480,
        "recognized_at": cameras[0]["bed_zone"]["recognized_at"],
    }


def test_deleting_a_camera_also_deletes_its_bed_zone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    bed_zone_store: BedZoneStore = app.state.bed_zone_store
    bed_zone_store.put(
        "camera-1",
        polygon=[[1, 2], [9, 2], [9, 8], [1, 8]],
        image_width=640,
        image_height=480,
        recognized_at="2026-08-01T00:00:00.000Z",
    )
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")

    with TestClient(app) as client:
        _login(client)
        deleted = client.delete("/api/v1/cameras/camera-1")

    assert deleted.status_code == 204
    assert bed_zone_store.get("camera-1") is None


def test_worker_config_snapshot_includes_bed_zone_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    app = _app(tmp_path)
    registry: CameraRegistryStore = app.state.camera_registry
    registry.create(
        camera_id="camera-2",
        label="Room 2",
        rtsp_url="rtsp://camera.local/2",
        space_id=None,
        status="online",
    )
    bed_zone_store: BedZoneStore = app.state.bed_zone_store
    bed_zone_store.put(
        "camera-1",
        polygon=[[1, 2], [9, 2], [9, 8], [1, 8]],
        image_width=640,
        image_height=480,
        recognized_at="2026-08-01T00:00:00.000Z",
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 200
    cameras_by_id = {camera["camera_id"]: camera for camera in response.json()["cameras"]}
    assert cameras_by_id["camera-1"]["bed_zone_polygon"] == [[1, 2], [9, 2], [9, 8], [1, 8]]
    assert cameras_by_id["camera-1"]["bed_zone_image_width"] == 640
    assert cameras_by_id["camera-1"]["bed_zone_image_height"] == 480
    assert "bed_zone_polygon" not in cameras_by_id["camera-2"]
