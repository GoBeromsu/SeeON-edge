"""Focused canonical multi-region bed-zone API coverage."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from types import TracebackType
from typing import NoReturn, Self

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)
RECOGNIZE_PATH = "/api/v1/cameras/camera-1/bed-zone/recognize"
SAVE_PATH = "/api/v1/cameras/camera-1/bed-zone"


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("API_EDGE_RELAY_TOKEN", "relay-token")
    monkeypatch.setenv("ML_API_WORKER_STREAM_ORIGIN", "http://worker.local:8090")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _login(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
        ).status_code
        == 204
    )


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.camera_registry.create(
        camera_id="camera-1",
        label="Bed camera",
        rtsp_url="rtsp://camera.invalid/live",
        space_id=None,
        status="unknown",
    )
    return app


class FakeUpstreamResponse:
    status = 200
    headers: dict[str, str] = {"Content-Type": "application/json"}

    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode()
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        del size
        return self.body

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


def _candidate() -> dict[str, object]:
    return {
        "regions": [
            {
                "id": "bed-a",
                "polygon": [[1, 2], [9, 2], [9, 8], [1, 8]],
                "origin": "model",
            },
            {
                "id": "bed-b",
                "polygon": [[20, 20], [30, 20], [30, 30], [20, 30]],
                "origin": "model",
            },
        ],
        "image_width": 640,
        "image_height": 480,
    }


def test_recognition_forwards_threshold_but_does_not_save_or_bump_registry(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUpstreamResponse:
        calls.append(
            {
                "body": json.loads(request.data or b"null"),
                "headers": dict(request.headers),
                "timeout": timeout,
            }
        )
        return FakeUpstreamResponse(_candidate())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    before = app.state.camera_registry.snapshot()["registry_version"]
    with TestClient(app) as client:
        _login(client)
        response = client.post(RECOGNIZE_PATH, json={"confidence": 0.4})
    assert response.status_code == 200
    assert response.json()["bed_zone"]["regions"] == _candidate()["regions"]
    assert response.json()["bed_zone"]["recognized_at"]
    assert app.state.bed_zone_store.get("camera-1") is None
    assert app.state.camera_registry.snapshot()["registry_version"] == before
    assert calls == [
        {
            "body": {"confidence": 0.4},
            "headers": {
                "X-edge-relay-token": "relay-token",
                "Content-type": "application/json",
            },
            "timeout": 25.0,
        }
    ]


def test_recognition_uses_default_confidence_when_body_is_omitted(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies: list[object] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeUpstreamResponse:
        del timeout
        bodies.append(json.loads(request.data or b"null"))
        return FakeUpstreamResponse(_candidate())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with TestClient(app) as client:
        _login(client)
        assert client.post(RECOGNIZE_PATH).status_code == 200
    assert bodies == [{"confidence": 0.15}]


@pytest.mark.parametrize("confidence", [0.049, 0.951, "NaN", "Infinity"])
def test_recognition_rejects_invalid_confidence(
    confidence: object, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("invalid request reached worker"),
    )
    with TestClient(app) as client:
        _login(client)
        assert client.post(RECOGNIZE_PATH, json={"confidence": confidence}).status_code == 422


def test_explicit_save_roundtrips_two_regions_and_bumps_once(app: FastAPI) -> None:
    before = app.state.camera_registry.snapshot()["registry_version"]
    with TestClient(app) as client:
        _login(client)
        saved = client.put(SAVE_PATH, json=_candidate())
        listed = client.get("/api/v1/cameras")
        worker = client.get(
            "/api/v1/cameras/worker-config",
            headers={"X-Edge-Relay-Token": "relay-token"},
        )
    assert saved.status_code == 200
    assert listed.json()["cameras"][0]["bed_zone"] == saved.json()["bed_zone"]
    worker_camera = worker.json()["cameras"][0]
    assert worker_camera["bed_zone_regions"] == _candidate()["regions"]
    assert "bed_zone_polygon" not in worker_camera
    assert app.state.camera_registry.snapshot()["registry_version"] == before + 1


def test_empty_regions_explicitly_clear_the_saved_zone(app: FastAPI) -> None:
    with TestClient(app) as client:
        _login(client)
        assert client.put(SAVE_PATH, json=_candidate()).status_code == 200
        before_clear = app.state.camera_registry.snapshot()["registry_version"]
        cleared = client.put(
            SAVE_PATH,
            json={"regions": [], "image_width": 640, "image_height": 480},
        )
        listed = client.get("/api/v1/cameras")
    assert cleared.status_code == 200
    assert cleared.json()["bed_zone"] is None
    assert listed.json()["cameras"][0]["bed_zone"] is None
    assert app.state.bed_zone_store.get("camera-1") is None
    assert app.state.camera_registry.snapshot()["registry_version"] == before_clear + 1


@pytest.mark.parametrize(
    "regions",
    [
        [
            {"id": "same", "polygon": [[1, 1], [8, 1], [8, 8]], "origin": "manual"},
            {"id": "same", "polygon": [[2, 2], [9, 2], [9, 9]], "origin": "model"},
        ],
        [{"id": "line", "polygon": [[1, 1], [2, 2], [3, 3]], "origin": "manual"}],
        [
            {
                "id": "cross",
                "polygon": [[1, 1], [9, 9], [1, 9], [9, 1]],
                "origin": "manual",
            }
        ],
        [{"id": "outside", "polygon": [[1, 1], [641, 1], [1, 8]], "origin": "manual"}],
    ],
    ids=["duplicate_ids", "degenerate", "self_intersecting", "out_of_bounds"],
)
def test_save_rejects_invalid_regions(regions: list[dict[str, object]], app: FastAPI) -> None:
    with TestClient(app) as client:
        _login(client)
        response = client.put(
            SAVE_PATH,
            json={"regions": regions, "image_width": 640, "image_height": 480},
        )
    assert response.status_code == 422
    assert app.state.bed_zone_store.get("camera-1") is None


def test_save_rejects_regions_whose_compact_utf8_encoding_exceeds_4096_bytes(
    app: FastAPI,
) -> None:
    scale = 10**12
    quarter = scale // 4
    polygon = [
        [1, 1],
        [quarter, 1],
        [2 * quarter, 1],
        [3 * quarter, 1],
        [scale - 1, 1],
        [scale - 1, quarter],
        [scale - 1, 2 * quarter],
        [scale - 1, 3 * quarter],
        [scale - 1, scale - 1],
        [3 * quarter, scale - 1],
        [2 * quarter, scale - 1],
        [quarter, scale - 1],
        [1, scale - 1],
        [1, 3 * quarter],
        [1, 2 * quarter],
        [1, quarter],
    ]
    regions = [
        {"id": f"{index}-" + "😀" * 62, "polygon": polygon, "origin": "manual"}
        for index in range(8)
    ]
    with TestClient(app) as client:
        _login(client)
        response = client.put(
            SAVE_PATH,
            json={"regions": regions, "image_width": scale, "image_height": scale},
        )
    assert response.status_code == 422
    assert "4096" in response.text
    assert app.state.bed_zone_store.get("camera-1") is None


@pytest.mark.parametrize("method,path", [("post", RECOGNIZE_PATH), ("put", SAVE_PATH)])
def test_bed_zone_routes_require_auth(method: str, path: str, app: FastAPI) -> None:
    payload = None if method == "post" else _candidate()
    with TestClient(app) as client:
        response = getattr(client, method)(path, json=payload)
    assert response.status_code == 401


def test_bed_zone_routes_report_unknown_camera_without_calling_worker(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_urlopen(*args, **kwargs) -> NoReturn:
        raise AssertionError("unknown camera reached worker")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)
    with TestClient(app) as client:
        _login(client)
        assert client.post(RECOGNIZE_PATH.replace("camera-1", "missing")).status_code == 404
        assert (
            client.put(SAVE_PATH.replace("camera-1", "missing"), json=_candidate()).status_code
            == 404
        )
