from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from contracts.event import MutableEventPayload
from shared.events.edge_ingest_client import EdgeIngestClient
from shared.events.evidence_export_contract import EventReceipt


class _RecordingHandler(BaseHTTPRequestHandler):
    received: list[tuple[str, dict[str, str | None], MutableEventPayload]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        self.__class__.received.append(
            (
                self.path,
                {
                    "Content-Type": self.headers.get("Content-Type"),
                    "Authorization": self.headers.get("Authorization"),
                    "X-" + "Ingest-Key-Id": self.headers.get("X-" + "Ingest-Key-Id"),
                    "X-" + "Ingest-Timestamp": self.headers.get("X-" + "Ingest-Timestamp"),
                    "X-" + "Signature": self.headers.get("X-" + "Signature"),
                },
                payload,
            )
        )
        self.send_response(201)
        self.end_headers()

    def log_message(self, _format: str, *args: str) -> None:
        return


def test_edge_ingest_client_posts_event_api_alert_and_heartbeat_without_auth_headers() -> None:
    _RecordingHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )
    try:
        assert client.send_heartbeat() is True
        assert (
            client.send_alert(
                event_type="fall",
                detected_at="2026-06-23T12:00:00.000Z",
                probability=0.91,
                clip_id="clip-123",
            )
            is True
        )

        assert _RecordingHandler.received == [
            (
                "/events/heartbeat",
                {
                    "Content-Type": "application/json",
                    "Authorization": None,
                    "X-" + "Ingest-Key-Id": None,
                    "X-" + "Ingest-Timestamp": None,
                    "X-" + "Signature": None,
                },
                {"camera_id": "camera-1"},
            ),
            (
                "/events",
                {
                    "Content-Type": "application/json",
                    "Authorization": None,
                    "X-" + "Ingest-Key-Id": None,
                    "X-" + "Ingest-Timestamp": None,
                    "X-" + "Signature": None,
                },
                {
                    "camera_id": "camera-1",
                    "type": "fall",
                    "detected_at": "2026-06-23T12:00:00.000Z",
                    "confidence": 0.91,
                    "clip_id": "clip-123",
                },
            ),
        ]
        assert client.failure_count == 0
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_edge_ingest_client_sends_bearer_token_when_configured() -> None:
    _RecordingHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.2,
        bearer_token="edge-facility-token",
    )
    try:
        assert client.send_heartbeat() is True
        assert client.send_alert(
            event_type="fall",
            detected_at="2026-06-23T12:00:00.000Z",
            probability=0.91,
        )

        assert [headers["Authorization"] for _, headers, _ in _RecordingHandler.received] == [
            "Bearer edge-facility-token",
            "Bearer edge-facility-token",
        ]
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_edge_ingest_client_omits_authorization_header_when_token_blank() -> None:
    _RecordingHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.2,
        bearer_token="   ",
    )
    try:
        assert client.send_heartbeat() is True

        assert _RecordingHandler.received[0][1]["Authorization"] is None
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_edge_ingest_client_omits_missing_clip_id_for_backward_compatibility() -> None:
    _RecordingHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )
    try:
        assert client.send_alert(
            event_type="fall",
            detected_at="2026-06-23T12:00:00.000Z",
            probability=0.91,
        )

        assert _RecordingHandler.received[0][2] == {
            "camera_id": "camera-1",
            "type": "fall",
            "detected_at": "2026-06-23T12:00:00.000Z",
            "confidence": 0.91,
        }
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_edge_ingest_client_counts_backend_failure() -> None:
    class _RejectingHandler(_RecordingHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(500)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), _RejectingHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )
    try:
        assert client.send_heartbeat() is False
        assert client.failure_count == 1
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_edge_ingest_client_does_not_emit_detection_lost() -> None:
    _RecordingHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.2,
    )
    try:
        client.emit(
            {
                "event_type": "detection-lost",
                "detected_at": "2026-06-23T12:00:00.000Z",
                "probability": 0.91,
            }
        )

        assert _RecordingHandler.received == []
        assert client.failure_count == 0
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def test_edge_ingest_client_send_alert_receipt_calls_on_accepted_with_wall_clock_time() -> None:
    # Given: a real worker sends an idempotent alert with edge_event_id set,
    # which is the send_alert_receipt -> BackendEvidenceClient.send_event_payload
    # code path (distinct from the edge_event_id-less send_alert path exercised
    # by the tests above).
    class _ReceiptHandler(_RecordingHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            self.__class__.received.append((self.path, {}, payload))
            response = json.dumps(
                {
                    "status": "accepted",
                    "edge_event_id": payload["edge_event_id"],
                    "event_id": "event-1",
                }
            ).encode("utf-8")
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    _ReceiptHandler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReceiptHandler)
    thread = _run_server(server)
    client = EdgeIngestClient(
        events_url=f"http://127.0.0.1:{server.server_port}/events",
        camera_id="camera-1",
        timeout_sec=0.5,
    )
    accepted_values: list[float] = []
    try:
        # When: real bounded_request runs end to end against a live HTTP
        # response (a stub that never calls on_response would hide the
        # `time` module/`time.time` callable bug this test guards against).
        before = time.time()
        result = client.send_alert_receipt(
            edge_event_id="00000000-0000-4000-8000-000000000099",
            event_type="fall",
            detected_at="2026-06-23T12:00:00.000Z",
            probability=0.91,
            on_accepted=accepted_values.append,
        )
        after = time.time()

        # Then: it completes without TypeError, and on_accepted receives a
        # single wall-clock timestamp bounded by the surrounding calls.
        assert result == EventReceipt("accepted", "00000000-0000-4000-8000-000000000099", "event-1")
        assert len(accepted_values) == 1
        assert isinstance(accepted_values[0], float)
        assert before <= accepted_values[0] <= after
    finally:
        server.shutdown()
        thread.join(timeout=1.0)


def _run_server(server: ThreadingHTTPServer) -> Thread:
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def test_edge_ingest_client_skips_snapshot_put_for_a_local_accept_receipt(monkeypatch) -> None:
    """accepted_local has no upstream id; there is no `{event_id}/snapshot` to PUT to."""
    from shared.events import edge_ingest_client as module

    class _LocalAcceptBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def send_event_payload(self, payload, edge_event_id, on_accepted=None):  # noqa: ANN001
            del payload
            return EventReceipt("accepted_local", edge_event_id, "")

    puts: list[str] = []
    monkeypatch.setattr(module, "BackendEvidenceClient", _LocalAcceptBackend)
    monkeypatch.setattr(EdgeIngestClient, "_put_snapshot", lambda self, url, data: puts.append(url))
    client = EdgeIngestClient(events_url="http://relay/events", camera_id="camera-1")

    result = client.send_alert_receipt(
        edge_event_id="edge-1",
        event_type="fall",
        detected_at="2026-06-23T12:00:00.000Z",
        probability=0.9,
        snapshot_bytes=b"jpeg",
    )

    assert isinstance(result, EventReceipt)
    assert result.status == "accepted_local"
    assert puts == []
