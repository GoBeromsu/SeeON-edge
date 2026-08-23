"""Shared diagnostic runtime helpers: served Hub fixture + real relay client.

Both live entirely on loopback with no credentials, no RTSP, and no media.
"""

from __future__ import annotations

import socket
import tempfile
import threading
from contextlib import closing
from pathlib import Path

import uvicorn
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.evidence.relay_projection import RelayEvidenceProjection
from backend.app.main import create_app, no_lifespan
from shared.events.edge_ingest_client import EdgeIngestClient
from tests_support.local_backend_fixture import LocalBackendFixture

RELAY_TOKEN = "relay-token"
CAMERA_ID = "room-camera"
FACILITY_ID = "facility-1"


def free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServedFixture:
    """Runs the contract-exact Hub fixture over real loopback HTTP."""

    def __init__(self, *, faulty_event_identity: bool = False) -> None:
        self.fixture = LocalBackendFixture(faulty_event_identity=faulty_event_identity)
        self.port = free_port()
        self._server = uvicorn.Server(
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
        waiter = threading.Event()
        for _ in range(200):
            if self._server.started:
                return self
            waiter.wait(0.05)
        raise RuntimeError("fixture Hub did not start")

    def __exit__(self, *_args: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def relay_client(origin: str, tmp_path: Path, *, database: Path | None = None) -> TestClient:
    """Real ml-api app wired to a real EdgeIngestClient against ``origin``."""

    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = RELAY_TOKEN
    registry = CameraRegistryStore(Path(tempfile.mkdtemp()) / "registry.sqlite3")
    registry.create(
        camera_id=CAMERA_ID,
        label=CAMERA_ID,
        rtsp_url=f"rtsp://role-gateway:8554/{CAMERA_ID}",
        space_id=None,
        status="online",
        backend_camera_id=CAMERA_ID,
    )
    app.state.camera_registry = registry
    app.state.catalog_store = CatalogStore.open(tmp_path / "relay-catalog.sqlite3")
    if database is not None:
        app.state.relay_evidence_projection = RelayEvidenceProjection(database)
    # Loopback http needs no insecure opt-in under the product's own hub policy.
    app.state.backend_ingest_client = EdgeIngestClient(
        events_url=f"{origin}/api/v1/events",
        bearer_token="fixture-token",
        camera_id=CAMERA_ID,
        timeout_sec=5.0,
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
    "relay_client",
]
