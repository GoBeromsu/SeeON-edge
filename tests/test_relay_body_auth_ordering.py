"""Relay body bounds and auth run before expensive JSON parse work."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.relay import router as relay_router
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)


def _free_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture
def app(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
) -> FastAPI:
    app = postgres_api_app(postgres_product_sandbox, postgres_audit_runtime)
    app.state.edge_relay_token = "worker-secret"
    return app


@pytest.fixture
def app_with_camera(app: FastAPI) -> FastAPI:
    app.state.camera_registry.create(
        camera_id="cam-1",
        label="cam",
        rtsp_url="rtsp://camera.example/live",
        space_id=None,
        status="offline",
    )
    return app


def test_oversized_content_length_is_rejected_before_body_parse(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            headers={
                "X-Edge-Relay-Token": "worker-secret",
                "Content-Type": "application/json",
                "Content-Length": str(relay_router.MAX_RELAY_REQUEST_BODY_BYTES + 1),
            },
            content=b"{}",
        )
    assert response.status_code == 413


def test_missing_relay_token_is_rejected_without_accepting_payload(app: FastAPI) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json={
                "event_type": "fall",
                "probability": 0.9,
                "detected_at": "2026-08-14T00:00:00Z",
                "camera_id": "cam-1",
                "facility_id": "fac-1",
            },
        )
    assert response.status_code == 401


def test_authorized_small_heartbeat_is_accepted(app_with_camera: FastAPI) -> None:
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={"X-Edge-Relay-Token": "worker-secret"},
            json={"camera_id": "cam-1", "facility_id": "fac-1"},
        )
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_authorize_relay_non_ascii_token_compares_constant_time_without_typeerror() -> None:
    # hmac.compare_digest raises TypeError on non-ASCII str; the auth path must
    # encode both sides to UTF-8 bytes first. A matching non-ASCII token
    # authorizes; a mismatch is 403, never a 500. Exercised as a direct call
    # (an HTTP header is a Latin-1 channel and could never carry a CJK token),
    # matching test_api_camera_registry's _authorize_worker convention.
    from backend.app.features.relay.auth import authorize_relay

    state = SimpleNamespace(edge_relay_token="중계-토큰")
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    authorize_relay(request, "중계-토큰")  # must not raise

    with pytest.raises(HTTPException) as exc_info:
        authorize_relay(request, "wrong-token")
    assert exc_info.value.status_code == 403


def _oversized_chunks(total_bytes: int, *, chunk: int = 512) -> Iterator[bytes]:
    """Stream ``total_bytes`` of body with no Content-Length (chunked)."""
    sent = 0
    while sent < total_bytes:
        step = min(chunk, total_bytes - sent)
        sent += step
        yield b"a" * step


def test_chunked_oversized_body_without_content_length_is_rejected(
    app_with_camera: FastAPI,
) -> None:
    # httpx streams a generator body as Transfer-Encoding: chunked with no
    # Content-Length, so the cheap header pre-check cannot catch it -- only the
    # BoundedBodyRoute streaming bound can. Body far exceeds the 4 KiB heartbeat
    # cap and must never be fully buffered for the Pydantic parse.
    over = relay_router.MAX_RELAY_HEARTBEAT_BODY_BYTES + 4096
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={
                "X-Edge-Relay-Token": "worker-secret",
                "Content-Type": "application/json",
            },
            content=_oversized_chunks(over),
        )
    assert response.status_code == 413


def test_chunked_body_without_content_length_is_accepted(app_with_camera: FastAPI) -> None:
    body = json.dumps({"camera_id": "cam-1", "facility_id": "fac-1"}).encode("utf-8")

    def _stream() -> Iterator[bytes]:
        # Two chunks, no Content-Length: proves the bounded read caches the body
        # so the Pydantic model still parses the reassembled payload.
        yield body[: len(body) // 2]
        yield body[len(body) // 2 :]

    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={
                "X-Edge-Relay-Token": "worker-secret",
                "Content-Type": "application/json",
            },
            content=_stream(),
        )
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_unauthorized_within_limit_body_is_rejected_before_pydantic_parse(
    app_with_camera: FastAPI,
) -> None:
    # A within-limit body is read fine, then the auth dependency rejects the
    # missing token (401) before the payload is validated -- auth-before-parse.
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={"Content-Type": "application/json"},
            json={"camera_id": "cam-1", "facility_id": "fac-1"},
        )
    assert response.status_code == 401


def test_unauthorized_oversized_chunked_body_is_rejected_at_transport_bound(
    app_with_camera: FastAPI,
) -> None:
    # The body bound lives at the route boundary, so an oversized chunked body is
    # rejected (413) before it is fully buffered -- an unauthenticated caller
    # cannot force the server to buffer megabytes just to reach the 401.
    over = relay_router.MAX_RELAY_HEARTBEAT_BODY_BYTES + 4096
    with TestClient(app_with_camera) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            headers={"Content-Type": "application/json"},
            content=_oversized_chunks(over),
        )
    assert response.status_code == 413


class _StartupSignalServer(uvicorn.Server):
    """uvicorn server that signals an event once its sockets are listening."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.listening = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            self.listening.set()


class _LiveApp:
    """A relay app served by real uvicorn so bodies traverse a real socket."""

    def __init__(self, app: FastAPI) -> None:
        self.port = _free_tcp_port()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = _StartupSignalServer(config)
        self._thread = threading.Thread(
            target=self._server.run, daemon=True, name="relay-bounded-read"
        )
        self._thread.start()
        if not self._server.listening.wait(timeout=10.0):
            pytest.fail("timed out waiting for relay uvicorn startup")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


def test_real_uvicorn_no_content_length_oversized_is_rejected(app_with_camera: FastAPI) -> None:
    server = _LiveApp(app_with_camera)
    try:
        over = relay_router.MAX_RELAY_HEARTBEAT_BODY_BYTES + 4096
        with httpx.Client(base_url=server.base_url) as client:
            request = client.build_request(
                "POST",
                "/api/v1/relay/heartbeat",
                headers={
                    "X-Edge-Relay-Token": "worker-secret",
                    "Content-Type": "application/json",
                },
                content=_oversized_chunks(over),
            )
            assert "content-length" not in request.headers
            assert request.headers.get("transfer-encoding") == "chunked"
            response = client.send(request)
    finally:
        server.stop()
    assert response.status_code == 413


def test_real_uvicorn_no_content_length_within_limit_is_accepted(app_with_camera: FastAPI) -> None:
    server = _LiveApp(app_with_camera)
    body = json.dumps({"camera_id": "cam-1", "facility_id": "fac-1"}).encode("utf-8")

    def _stream() -> Iterator[bytes]:
        yield body[:3]
        yield body[3:]

    try:
        with httpx.Client(base_url=server.base_url) as client:
            request = client.build_request(
                "POST",
                "/api/v1/relay/heartbeat",
                headers={
                    "X-Edge-Relay-Token": "worker-secret",
                    "Content-Type": "application/json",
                },
                content=_stream(),
            )
            assert "content-length" not in request.headers
            response = client.send(request)
    finally:
        server.stop()
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
