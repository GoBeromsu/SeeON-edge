from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.main import create_app, no_lifespan
from shared.events.evidence_export_client import RelayEvidenceClient
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure

TOKEN = "relay-token"
EVENT_ID = "00000000-0000-4000-8000-000000000020"
ATTACHMENT = {
    "edge_event_id": EVENT_ID,
    "snapshot_id": "snapshot-1",
    "sha256": "a" * 64,
    "media_reference": "snapshots/camera-1/snapshot-1.jpg",
    "size_bytes": 42,
    "mime_type": "image/jpeg",
}
DISPOSITION = {
    "edge_event_id": EVENT_ID,
    "snapshot_id": "snapshot-missing",
    "disposition": "UNAVAILABLE",
    "reason": "camera offline",
}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = TOKEN
    registry = CameraRegistryStore(tmp_path / "registry.sqlite3")
    registry.create(
        camera_id="camera-1",
        label="camera-1",
        rtsp_url="rtsp://example/camera-1",
        space_id=None,
        status="online",
        backend_camera_id="camera-1",
    )
    app.state.camera_registry = registry
    app.state.catalog_store = CatalogStore.open(tmp_path / "catalog.sqlite3")
    with TestClient(app) as test_client:
        yield test_client
    app.state.catalog_store.close()


def _post(client: TestClient, path: str, payload: dict[str, object]):
    return client.post(path, json=payload, headers={"X-Edge-Relay-Token": TOKEN})


def test_snapshot_attachment_is_idempotent_and_rebinding_conflicts(client: TestClient) -> None:
    path = "/api/v1/relay/snapshot-attachments"

    assert _post(client, path, ATTACHMENT).status_code == 202
    assert _post(client, path, ATTACHMENT).status_code == 202

    rebound = {**ATTACHMENT, "sha256": "b" * 64}
    conflict = _post(client, path, rebound)
    invalid = _post(client, path, {**ATTACHMENT, "sha256": "not-a-hash"})

    assert conflict.status_code == 409
    assert "content identity" in conflict.json()["detail"]
    assert invalid.status_code == 422
    assert len(client.app.state.catalog_store.records("snapshots")) == 1


def test_snapshot_disposition_is_durable_and_never_changes_referenced_event(
    client: TestClient,
) -> None:
    event = {
        "edge_event_id": EVENT_ID,
        "event_type": "bed-exit",
        "probability": 0.8,
        "detected_at": "2026-08-21T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
    }
    assert _post(client, "/api/v1/relay/alerts", event).status_code == 202
    before = client.app.state.catalog_store.records("events")

    response = _post(client, "/api/v1/relay/snapshot-dispositions", DISPOSITION)

    assert response.status_code == 202
    assert client.app.state.catalog_store.records("events") == before
    audit = client.app.state.catalog_store.records("audit")
    assert audit == [{"action": "snapshot_disposition", **DISPOSITION}]


def test_snapshot_attachment_rejects_inline_media_payload(client: TestClient) -> None:
    response = _post(
        client,
        "/api/v1/relay/snapshot-attachments",
        {**ATTACHMENT, "media_bytes_base64": "aGVsbG8="},
    )

    assert response.status_code == 422
    assert client.app.state.catalog_store.records("snapshots") == []


class _OutcomeHandler(BaseHTTPRequestHandler):
    responses: ClassVar[list[tuple[int, bytes]]] = []

    def do_POST(self) -> None:
        status, body = type(self).responses.pop(0)
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def outcome_server() -> Iterator[str]:
    _OutcomeHandler.responses = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OutcomeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_client_preserves_conflict_and_validation_outcomes(outcome_server: str) -> None:
    _OutcomeHandler.responses = [
        (409, json.dumps({"detail": "content identity conflict"}).encode()),
        (422, json.dumps({"detail": "invalid attachment"}).encode()),
    ]
    relay = RelayEvidenceClient(outcome_server, TOKEN)

    conflict = relay.send_snapshot_attachment(ATTACHMENT)
    invalid = relay.send_snapshot_attachment(ATTACHMENT)

    assert conflict == DeliveryFailure(
        DeliveryDisposition.PERMANENT, "HTTP_409", status_code=409
    )
    assert invalid == DeliveryFailure(
        DeliveryDisposition.PERMANENT, "HTTP_422", status_code=422
    )
