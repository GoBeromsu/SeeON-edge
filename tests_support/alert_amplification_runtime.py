"""Shared diagnostic runtime helpers: served Hub fixture + real relay client.

Both live entirely on loopback with no credentials, no RTSP, and no media. The
relay runs on the PostgreSQL product sandbox; register next to the sandbox
plugin: ``pytest_plugins = ("tests_support.postgres_sandbox",)``.
"""

from __future__ import annotations

import socket
import threading
from contextlib import closing
from typing import Any

import uvicorn
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from shared.events.edge_ingest_client import EdgeIngestClient
from tests_support.local_backend_fixture import LocalBackendFixture
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import RELAY_TOKEN, relay_postgres_app

CAMERA_ID = "room-camera"
FACILITY_ID = "facility-1"


def free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _StartupSignalServer(uvicorn.Server):
    """uvicorn server that signals an event once its sockets are listening."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.listening = threading.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            self.listening.set()


class ServedFixture:
    """Runs the contract-exact Hub fixture over real loopback HTTP."""

    def __init__(self, *, faulty_event_identity: bool = False) -> None:
        self.fixture = LocalBackendFixture(faulty_event_identity=faulty_event_identity)
        self.port = free_port()
        self._server = _StartupSignalServer(
            uvicorn.Config(
                self.fixture.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
                lifespan="off",
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> ServedFixture:
        self._thread.start()
        if not self._server.listening.wait(timeout=10.0):
            raise RuntimeError("fixture Hub did not start")
        return self

    def __exit__(self, *_args: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def hub_client(origin: str) -> EdgeIngestClient:
    """The real Hub ingest client pointed at ``origin``."""

    # Loopback http needs no insecure opt-in under the product's own hub policy.
    return EdgeIngestClient(
        events_url=f"{origin}/api/v1/events",
        bearer_token="fixture-token",
        camera_id=CAMERA_ID,
        timeout_sec=5.0,
    )


def relay_client(
    origin: str,
    sandbox: ProductSandbox,
    audit_runtime: PostgresAuditRuntime,
    *,
    ingest_client: Any = None,
) -> TestClient:
    """Real ml-api app on the PostgreSQL product root, delivering to ``origin``.

    ``ingest_client`` replaces the default ``hub_client(origin)``; the relay
    commits the incident and its outbox row before any Hub delivery.
    """

    app = relay_postgres_app(
        sandbox,
        audit_runtime,
        client=hub_client(origin) if ingest_client is None else ingest_client,
        camera_id=CAMERA_ID,
        backend_camera_id=CAMERA_ID,
        rtsp_url=f"rtsp://role-gateway:8554/{CAMERA_ID}",
    )
    return TestClient(app)


def deliver_alert(
    client: TestClient,
    edge_event_id: str,
    *,
    detected_at: str = "2026-08-16T00:00:00.000Z",
) -> str:
    """POST one relay alert and return the accepted backend event id (B)."""

    response = client.post(
        "/api/v1/relay/alerts",
        json={
            "edge_event_id": edge_event_id,
            "event_type": "fall",
            "probability": 0.91,
            "detected_at": detected_at,
            "camera_id": CAMERA_ID,
            "facility_id": FACILITY_ID,
        },
        headers={"X-Edge-Relay-Token": RELAY_TOKEN},
    )
    assert response.status_code == 202, response.text
    return str(response.json()["event_id"])


__all__ = [
    "CAMERA_ID",
    "FACILITY_ID",
    "RELAY_TOKEN",
    "ServedFixture",
    "deliver_alert",
    "free_port",
    "hub_client",
    "relay_client",
]
