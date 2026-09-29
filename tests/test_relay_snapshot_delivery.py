"""Snapshot companion routes on the PostgreSQL product root.

The oracles read committed ``artifacts``, ``incidents`` and ``audit_events``
rows through the sandbox admin connection.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from shared.events.evidence_export_client import RelayEvidenceClient
from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import RELAY_TOKEN, artifact_count, relay_postgres_app

pytest_plugins = ("tests_support.postgres_sandbox",)

TOKEN = RELAY_TOKEN
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
def client(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> Iterator[TestClient]:
    app = relay_postgres_app(postgres_product_sandbox, postgres_audit_runtime)
    with TestClient(app) as test_client:
        yield test_client


def _post(client: TestClient, path: str, payload: Mapping[str, object]):
    return client.post(path, json=payload, headers={"X-Edge-Relay-Token": TOKEN})


def _audit_count(sandbox: ProductSandbox, action: str) -> int:
    row = sandbox.admin.execute(
        "SELECT count(*) FROM audit_events WHERE action = %s", (action,)
    ).fetchone()
    assert row is not None
    return int(row[0])


def _incident(sandbox: ProductSandbox) -> tuple[object, ...] | None:
    return sandbox.admin.execute(
        "SELECT edge_event_id, event_type, revision, review_version "
        "FROM incidents WHERE edge_event_id = %s",
        (EVENT_ID,),
    ).fetchone()


def test_snapshot_attachment_is_idempotent_and_rebinding_conflicts(
    client: TestClient, postgres_product_sandbox: ProductSandbox
) -> None:
    sandbox = postgres_product_sandbox
    path = "/api/v1/relay/snapshot-attachments"
    event = {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.8,
        "detected_at": "2026-08-21T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
    }
    assert _post(client, "/api/v1/relay/alerts", event).status_code == 202

    assert _post(client, path, ATTACHMENT).status_code == 202
    assert _post(client, path, ATTACHMENT).status_code == 202

    rebound = {**ATTACHMENT, "sha256": "b" * 64}
    conflict = _post(client, path, rebound)
    invalid = _post(client, path, {**ATTACHMENT, "sha256": "not-a-hash"})

    assert conflict.status_code == 409
    assert "content identity" in conflict.json()["detail"]
    assert invalid.status_code == 422
    assert sandbox.admin.execute(
        "SELECT artifact_id, content_sha256 FROM artifacts WHERE kind = 'SNAPSHOT'"
    ).fetchall() == [("snapshot-1", "a" * 64)]
    assert _audit_count(sandbox, "relay.snapshot-attachment") == 2


def test_snapshot_disposition_is_durable_and_never_changes_referenced_event(
    client: TestClient, postgres_product_sandbox: ProductSandbox
) -> None:
    sandbox = postgres_product_sandbox
    event = {
        "edge_event_id": EVENT_ID,
        "event_type": "bed-exit",
        "probability": 0.8,
        "detected_at": "2026-08-21T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
    }
    assert _post(client, "/api/v1/relay/alerts", event).status_code == 202
    before = _incident(sandbox)

    response = _post(client, "/api/v1/relay/snapshot-dispositions", DISPOSITION)

    assert response.status_code == 202
    assert before is not None
    assert _incident(sandbox) == before
    assert sandbox.admin.execute(
        "SELECT state, reason FROM artifacts WHERE incident_id = ("
        "SELECT incident_id FROM incidents WHERE edge_event_id = %s) AND kind = 'SNAPSHOT'",
        (EVENT_ID,),
    ).fetchone() == ("UNAVAILABLE", "UNAVAILABLE:camera offline")


def test_snapshot_disposition_route_commits_canonical_action_and_detail(
    client: TestClient, postgres_product_sandbox: ProductSandbox
) -> None:
    event = {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.8,
        "detected_at": "2026-08-21T00:00:00Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
    }
    assert _post(client, "/api/v1/relay/alerts", event).status_code == 202

    response = _post(client, "/api/v1/relay/snapshot-dispositions", DISPOSITION)

    assert response.status_code == 202
    rows = postgres_product_sandbox.admin.execute(
        "SELECT action, target_id, actor_type, auth_mechanism, detail_json "
        "FROM audit_events WHERE target_id = 'snapshot-missing'"
    ).fetchall()
    assert rows == [
        (
            "relay.snapshot-disposition",
            "snapshot-missing",
            "service",
            "relay_token",
            '{"version":1}',
        )
    ]


def test_snapshot_attachment_rejects_inline_media_payload(
    client: TestClient, postgres_product_sandbox: ProductSandbox
) -> None:
    response = _post(
        client,
        "/api/v1/relay/snapshot-attachments",
        {**ATTACHMENT, "media_bytes_base64": "aGVsbG8="},
    )

    assert response.status_code == 422
    assert artifact_count(postgres_product_sandbox) == 0
    assert _audit_count(postgres_product_sandbox, "relay.snapshot-attachment") == 0


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

    assert conflict == DeliveryFailure(DeliveryDisposition.PERMANENT, "HTTP_409", status_code=409)
    assert invalid == DeliveryFailure(DeliveryDisposition.PERMANENT, "HTTP_422", status_code=422)
