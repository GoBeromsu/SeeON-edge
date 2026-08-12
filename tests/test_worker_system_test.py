from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pytest

from worker.pipeline.output.evidence.evidence_outbox import EdgeEventId, EvidenceOutbox
from worker.system_test import (
    SYSTEM_TEST_GATE_ENV,
    SYSTEM_TEST_GATE_VALUE,
    SystemTestConfig,
    SystemTestConfigurationError,
    SystemTestDisabledError,
    SystemTestEmitter,
    SystemTestStatus,
    require_system_test_gate,
)

EVENT_ID = "00000000-0000-4000-8000-000000000099"
VALIDATION_RUN_ID = "0197f671-3a31-7a6c-a6e4-83ed412de80f"


class SystemTestContractHandler(BaseHTTPRequestHandler):
    responses: ClassVar[list[tuple[int, bytes]]] = []
    requests: ClassVar[list[tuple[str, bytes, str | None]]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).requests.append(
            (self.path, body, self.headers.get("X-Edge-Relay-Token"))
        )
        status, payload = type(self).responses.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: str | int) -> None:
        return


@pytest.fixture
def system_test_server():
    SystemTestContractHandler.responses = []
    SystemTestContractHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), SystemTestContractHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _receipt() -> bytes:
    return json.dumps(
        {
            "status": "accepted",
            "edge_event_id": EVENT_ID,
            "event_id": "backend-system-test",
        },
        separators=(",", ":"),
    ).encode()


def _emitter(tmp_path: Path, server_url: str) -> SystemTestEmitter:
    return SystemTestEmitter(
        SystemTestConfig(
            database_path=tmp_path / "worker-state.sqlite3",
            relay_url=server_url,
            relay_token="local-relay-token",
        ),
        clock=lambda: datetime(2026, 8, 12, 1, 2, 3, 456000, tzinfo=UTC),
        event_id_factory=lambda: UUID(EVENT_ID),
    )


def test_gate_is_disabled_by_default_and_rejects_unknown_values() -> None:
    with pytest.raises(SystemTestDisabledError):
        require_system_test_gate({})
    with pytest.raises(SystemTestConfigurationError):
        require_system_test_gate({SYSTEM_TEST_GATE_ENV: "true"})

    require_system_test_gate({SYSTEM_TEST_GATE_ENV: SYSTEM_TEST_GATE_VALUE})


def test_emit_uses_real_sender_http_and_media_free_schema(
    tmp_path: Path,
    system_test_server: str,
) -> None:
    # Given: a deterministic local relay contract and explicit operator invocation.
    SystemTestContractHandler.responses = [(202, _receipt())]
    emitter = _emitter(tmp_path, system_test_server)

    # When: one SYSTEM_TEST is emitted.
    outcome = emitter.emit(UUID(VALIDATION_RUN_ID))

    # Then: the real outbox/sender/client path ACKs one exact privacy-safe payload.
    assert outcome.status is SystemTestStatus.ACKED
    assert outcome.edge_event_id == EVENT_ID
    assert outcome.correlation_id == EVENT_ID
    assert outcome.backend_event_id == "backend-system-test"
    assert len(SystemTestContractHandler.requests) == 1
    path, raw_body, token = SystemTestContractHandler.requests[0]
    assert path == "/api/v1/relay/system-tests"
    assert token == "local-relay-token"
    payload = json.loads(raw_body)
    assert payload == {
        "attempt_ordinal": 1,
        "detected_at": "2026-08-12T01:02:03.456Z",
        "edge_event_id": EVENT_ID,
        "label": "SYSTEM TEST - NOT A RESIDENT ALERT",
        "source": "SYSTEM_TEST",
        "test_mode": "SYSTEM_TEST",
        "type": "SYSTEM_TEST",
        "validation_run_id": VALIDATION_RUN_ID,
    }
    forbidden = {
        "camera_id",
        "facility_id",
        "room_id",
        "resident_id",
        "person_id",
        "bed_id",
        "snapshot",
        "clip_id",
        "media",
        "evidence",
        "audit",
        "rtsp_url",
        "probability",
        "confidence",
    }
    assert forbidden.isdisjoint(payload)
    with EvidenceOutbox.open(tmp_path / "worker-state.sqlite3") as outbox:
        assert outbox.event_delivery_state(EdgeEventId(EVENT_ID)) == "ACKED"


def test_exact_replay_reuses_payload_and_event_id_without_outbox_reset(
    tmp_path: Path,
    system_test_server: str,
) -> None:
    # Given: one accepted invocation and one accepted backend dedupe replay.
    SystemTestContractHandler.responses = [(202, _receipt()), (200, _receipt())]
    emitter = _emitter(tmp_path, system_test_server)
    first = emitter.emit(UUID(VALIDATION_RUN_ID))

    # When: the operator explicitly replays the terminal row.
    replay = emitter.replay(EdgeEventId(EVENT_ID))

    # Then: both receipts correlate and durable ACK state/attempt count remain terminal.
    assert first.backend_event_id == replay.backend_event_id == "backend-system-test"
    assert replay.status is SystemTestStatus.REPLAY_ACKED
    bodies = [json.loads(request[1]) for request in SystemTestContractHandler.requests]
    assert [body["edge_event_id"] for body in bodies] == [EVENT_ID, EVENT_ID]
    assert [body["attempt_ordinal"] for body in bodies] == [1, 2]
    assert [{k: v for k, v in body.items() if k != "attempt_ordinal"} for body in bodies] == [
        {k: v for k, v in bodies[0].items() if k != "attempt_ordinal"},
        {k: v for k, v in bodies[0].items() if k != "attempt_ordinal"},
    ]
    with EvidenceOutbox.open(tmp_path / "worker-state.sqlite3") as outbox:
        assert outbox.event_attempt_count(EdgeEventId(EVENT_ID)) == 1
        assert outbox.event_delivery_state(EdgeEventId(EVENT_ID)) == "ACKED"


def test_invalid_auth_is_classified_without_outbox_or_event_mutation(
    tmp_path: Path,
    system_test_server: str,
) -> None:
    # Given: local ml-api reports an actual backend 403 diagnostic response.
    SystemTestContractHandler.responses = [
        (
            200,
            json.dumps(
                {"disposition": "RETRY", "code": "HTTP_403", "status_code": 403},
                separators=(",", ":"),
            ).encode(),
        )
    ]
    emitter = _emitter(tmp_path, system_test_server)

    # When: the payload-free auth diagnostic is invoked.
    outcome = emitter.classify_invalid_auth()

    # Then: classification is typed and no event/outbox row exists.
    assert outcome.status is SystemTestStatus.AUTH_CLASSIFIED
    assert outcome.error_code == "HTTP_403"
    assert outcome.edge_event_id is None
    assert SystemTestContractHandler.requests == [
        ("/api/v1/relay/system-tests/auth-check", b"{}", "local-relay-token")
    ]
    with EvidenceOutbox.open(tmp_path / "worker-state.sqlite3") as outbox:
        assert outbox.pending_count() == 0
