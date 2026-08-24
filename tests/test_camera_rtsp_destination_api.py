"""API admission rejects unsafe RTSP destinations before store/probe."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.main import create_app, no_lifespan
from shared.rtsp_url_policy import ALLOW_LOCAL_RTSP_ENV, ALLOW_PRIVATE_RTSP_ENV
from tests_support.compact_authority_db import prepare_compact_database


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session",
        json={"username": "admin", "password": "admin"},
    )
    assert response.status_code == 204


def _app(tmp_path):
    app = create_app(lifespan=no_lifespan)
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    app.state.camera_registry = CameraRegistryStore(registry_path)
    app.state.edge_relay_token = "worker-secret"
    return app


def test_create_camera_rejects_loopback_and_metadata_urls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    with TestClient(_app(tmp_path)) as client:
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


def test_local_allowance_admits_loopback_fixture_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.setenv(ALLOW_LOCAL_RTSP_ENV, "1")
    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "fixture", "rtsp_url": "rtsp://127.0.0.1:8554/live"},
        )
    assert response.status_code == 201


def test_patch_camera_rejects_private_destination_without_allowance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    app = _app(tmp_path)
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


def test_private_allowance_admits_facility_lan_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "1")
    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "lan", "rtsp_url": "rtsp://10.0.0.9/live"},
        )
    assert response.status_code == 201


def test_private_destination_rejected_when_private_flag_is_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.setenv(ALLOW_PRIVATE_RTSP_ENV, "0")
    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "lan", "rtsp_url": "rtsp://10.0.0.9/live"},
        )
    assert response.status_code == 400


def test_create_camera_rejects_hostname_that_resolves_to_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)

    import shared.rtsp_url_policy as policy

    monkeypatch.setattr(
        policy,
        "resolve_host_a_aaaa",
        lambda _host: ("169.254.169.254",),
    )
    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "rebinding", "rtsp_url": "rtsp://cam.example/live"},
        )
    assert response.status_code == 400
    assert "metadata" in response.json()["detail"]


def test_create_camera_rejects_hostname_that_resolves_to_private_without_allowance(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ML_API_WORKER_PROBE_ORIGIN", "")
    monkeypatch.delenv(ALLOW_LOCAL_RTSP_ENV, raising=False)
    monkeypatch.delenv(ALLOW_PRIVATE_RTSP_ENV, raising=False)

    import shared.rtsp_url_policy as policy

    monkeypatch.setattr(policy, "resolve_host_a_aaaa", lambda _host: ("10.0.0.9",))
    with TestClient(_app(tmp_path)) as client:
        _login(client)
        response = client.post(
            "/api/v1/cameras",
            json={"label": "lan-dns", "rtsp_url": "rtsp://cam.example/live"},
        )
    assert response.status_code == 400
    assert "private" in response.json()["detail"]
