"""API admission rejects unsafe RTSP destinations before store/probe."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from shared.rtsp_url_policy import ALLOW_LOCAL_RTSP_ENV, ALLOW_PRIVATE_RTSP_ENV
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> FastAPI:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.edge_relay_token = "worker-secret"
    return app


def test_create_camera_rejects_loopback_and_metadata_urls(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    with TestClient(app) as client:
        _login(client)
        loopback = client.post(
            "/api/v1/cameras",
            json={"label": "bad", "rtsp_url": "rtsp://127.0.0.1:8554/live"},
        )
        metadata = client.post(
            "/api/v1/cameras",
            json={"label": "meta", "rtsp_url": "rtsp://169.254.169.254/latest"},
        )
        http_scheme = client.post(
            "/api/v1/cameras",
            json={"label": "http", "rtsp_url": "http://camera.example/live"},
        )
        ok = client.post(
            "/api/v1/cameras",
            json={"label": "ok", "rtsp_url": "rtsp://camera.example/live"},
        )

    assert loopback.status_code == 400
    assert metadata.status_code == 400
    assert http_scheme.status_code == 400
    assert ok.status_code == 201
    assert "rtsp_url" not in ok.json()
    assert ok.json()["rtsp_url_masked"].startswith("rtsp://")


def test_local_allowance_admits_loopback_fixture_url(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.setenv(ALLOW_LOCAL_RTSP_ENV, "1")
    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "fixture", "rtsp_url": "rtsp://127.0.0.1:8554/live"},
        )
    assert response.status_code == 201


def test_patch_camera_rejects_private_destination_without_allowance(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    store = app.state.camera_registry
    store.create(
        camera_id="cam-1",
        label="cam",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    with TestClient(app) as client:
        _login(client)
        rejected = client.patch(
            "/api/v1/cameras/cam-1",
            json={"rtsp_url": "rtsp://10.0.0.9/live"},
        )
        accepted = client.patch(
            "/api/v1/cameras/cam-1",
            json={"rtsp_url": "rtsps://camera.example/secure"},
        )
    assert rejected.status_code == 400
    assert accepted.status_code == 200


def test_private_allowance_admits_facility_lan_url(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "1")
    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "lan", "rtsp_url": "rtsp://10.0.0.9/live"},
        )
    assert response.status_code == 201


def test_private_destination_rejected_when_private_flag_is_zero(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "0")
    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "lan", "rtsp_url": "rtsp://10.0.0.9/live"},
        )
    assert response.status_code == 400


def test_create_camera_rejects_hostname_that_resolves_to_metadata(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)

    import shared.rtsp_url_policy as policy

    monkeypatch.setattr(
        policy,
        "resolve_host_a_aaaa",
        lambda _host: ("169.254.169.254",),
    )
    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "rebinding", "rtsp_url": "rtsp://cam.example/live"},
        )
    assert response.status_code == 400
    assert "metadata" in response.json()["detail"]


def test_create_camera_rejects_hostname_that_resolves_to_private_without_allowance(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)

    import shared.rtsp_url_policy as policy

    monkeypatch.setattr(policy, "resolve_host_a_aaaa", lambda _host: ("10.0.0.9",))
    with TestClient(app) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "lan-dns", "rtsp_url": "rtsp://cam.example/live"},
        )
    assert response.status_code == 400
    assert "private" in response.json()["detail"]
