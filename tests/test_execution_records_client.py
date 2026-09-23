"""HTTP client tests for Worker -> Backend execution-record batches."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from shared.events.evidence_export_contract import DeliveryDisposition
from shared.events.execution_records import (
    RELAY_EXECUTION_RECORDS_PATH,
    WireBatch,
    WireBatchReceipt,
    WireProvenance,
    WireRecord,
)
from shared.events.execution_records_client import ExecutionRecordsClient

_PROVENANCE = WireProvenance(
    worker_build_revision="abc123",
    worker_image_digest="sha256:deadbeef",
    model_digest="model-1",
    calibration_digest="cal-1",
    preprocessing_identity="pose-bbox56/v1",
    config_digest="cfg-1",
    policy_identity="fall.policy:2",
)


def _record(seq: int = 0) -> WireRecord:
    return WireRecord(
        record_kind="sdk.frame",
        camera_id="cam-1",
        worker_boot_id="boot-1",
        source_generation=0,
        stream_epoch=1,
        producer="sdk",
        producer_sequence=seq,
        observed_at_ns=1_000 + seq,
        time_quality="monotonic",
        causal_unit_id="cam-1:boot-1:1:frame:0",
        outcome="accepted",
        payload={"seq": seq},
    )


def _batch() -> WireBatch:
    return WireBatch("cam-1", "boot-1", _PROVENANCE, (_record(),))


class _Handler(BaseHTTPRequestHandler):
    status = 200
    requests: list[tuple[str, dict[str, str | None], bytes]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.requests.append((self.path, dict(self.headers.items()), body))
        payload = {
            "batch_id": json.loads(body)["batch_id"],
            "accepted": 1,
            "duplicates": 0,
            "rejected": [],
            "storage_state": "committed",
            "committed_at_ns": 7,
        }
        encoded = json.dumps(payload).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.status < 400:
            self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def _run(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_client_posts_wire_batch_with_relay_token() -> None:
    _Handler.requests = []
    _Handler.status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = _run(server)
    try:
        client = ExecutionRecordsClient(f"http://127.0.0.1:{server.server_port}", "relay-token")
        batch = _batch()
        result = client.post_batch(batch)
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
    assert isinstance(result, WireBatchReceipt)
    assert result.batch_id == batch.batch_id
    path, headers, body = _Handler.requests[0]
    assert path == f"/api/v1{RELAY_EXECUTION_RECORDS_PATH}"
    assert headers["X-Edge-Relay-Token"] == "relay-token"
    assert json.loads(body)["batch_id"] == batch.batch_id


@pytest.mark.parametrize("status", (401, 403))
def test_auth_failures_are_retry(status: int) -> None:
    _Handler.requests = []
    _Handler.status = status
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = _run(server)
    try:
        client = ExecutionRecordsClient(f"http://127.0.0.1:{server.server_port}", "token")
        result = client.post_batch(_batch())
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
    assert result.disposition is DeliveryDisposition.RETRY


@pytest.mark.parametrize("status", (404, 405))
def test_missing_route_is_compatibility(status: int) -> None:
    _Handler.requests = []
    _Handler.status = status
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = _run(server)
    try:
        client = ExecutionRecordsClient(f"http://127.0.0.1:{server.server_port}", "token")
        result = client.post_batch(_batch())
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
    assert result.disposition is DeliveryDisposition.COMPATIBILITY


def test_unprocessable_body_is_permanent() -> None:
    _Handler.requests = []
    _Handler.status = 422
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = _run(server)
    try:
        client = ExecutionRecordsClient(f"http://127.0.0.1:{server.server_port}", "token")
        result = client.post_batch(_batch())
    finally:
        server.shutdown()
        thread.join(timeout=1.0)
    assert result.disposition is DeliveryDisposition.PERMANENT
