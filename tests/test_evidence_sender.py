from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimLease,
    ClipId,
    ClipLocalState,
    ClipOutcome,
    EdgeEventId,
    EvidenceOutbox,
    EvidenceReasonCode,
    StagedEvent,
)
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep

EVENT_A = EdgeEventId("00000000-0000-4000-8000-000000000001")
EVENT_B = EdgeEventId("00000000-0000-4000-8000-000000000002")
CLIP = ClipId("clip-coalesced")


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.capability_result: BackendCapabilities | DeliveryFailure = BackendCapabilities(
            event_idempotency=1, clip_export=1
        )
        self.event_results: list[EventReceipt | DeliveryFailure] = []
        self.clip_results: list[ClipReceipt | DeliveryFailure] = []

    def probe_capabilities(self, camera_id: str) -> BackendCapabilities | DeliveryFailure:
        self.calls.append(f"capability:{camera_id}")
        return self.capability_result

    def send_event(self, payload_json: str, edge_event_id: EdgeEventId):
        payload = json.loads(payload_json)
        self.calls.append(f"event:{payload['edge_event_id']}")
        if self.event_results:
            return self.event_results.pop(0)
        return EventReceipt(
            status="accepted",
            edge_event_id=edge_event_id,
            event_id=f"backend-{edge_event_id}",
        )

    def send_clip(self, claim):
        self.calls.append(f"clip:{claim.clip_id}")
        if self.clip_results:
            return self.clip_results.pop(0)
        return ClipReceipt(
            clip_id=claim.clip_id,
            state="READY",
            state_version=claim.state_version,
            sha256=claim.sha256,
            size_bytes=claim.size_bytes,
        )


def _payload(event_id: EdgeEventId, *, camera_id: str = "camera-1") -> str:
    return json.dumps(
        {
            "edge_event_id": event_id,
            "event_type": "fall",
            "probability": 0.91,
            "detected_at": "2026-07-16T00:00:00Z",
            "camera_id": camera_id,
            "facility_id": "facility-1",
            "evidence": {"clip_id": CLIP},
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _stage_clip(database: Path) -> None:
    with EvidenceOutbox.open(database) as outbox:
        for index, event_id in enumerate((EVENT_A, EVENT_B), start=1):
            outbox.stage(
                StagedEvent(
                    edge_event_id=event_id,
                    detected_at="2026-07-16T00:00:00Z",
                    payload_json=_payload(event_id),
                    queued_at=float(index),
                )
            )
            outbox.bind_clip(event_id, CLIP)
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=CLIP,
                local_state=ClipLocalState.VERIFIED,
                manifest_path="clips/clip-coalesced/manifest.json",
                state_version=2,
                media_relpath="clips/clip-coalesced/clip.mp4",
                sha256="a" * 64,
                size_bytes=123,
                mime_type="video/mp4",
                codec="h264",
                duration_ms=1000,
                clip_start_at="2026-07-16T00:00:00Z",
                clip_end_at="2026-07-16T00:00:01Z",
                finalized_at="2026-07-16T00:00:02Z",
            )
        )


def _stage_unavailable_clip(database: Path) -> None:
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=EVENT_A,
                detected_at="2026-07-16T00:00:00Z",
                payload_json=_payload(EVENT_A),
                queued_at=1.0,
            )
        )
        outbox.bind_clip(EVENT_A, CLIP)
        outbox.record_clip_outcome(
            ClipOutcome(
                clip_id=CLIP,
                local_state=ClipLocalState.UNAVAILABLE,
                manifest_path="clips/clip-coalesced/manifest.json",
                state_version=2,
                unavailable_reason=EvidenceReasonCode.ENCODER_FAILED,
            )
        )


def _sender(database: Path, transport: FakeTransport) -> EvidenceSender:
    return EvidenceSender(
        database,
        SenderConfig(
            relay_url="http://127.0.0.1:1",
            relay_token="test-token",
            probe_camera_id="camera-1",
            enabled=True,
            lease_seconds=10.0,
        ),
        transport=transport,
        clock=lambda: 100.0,
        random_value=lambda: 0.5,
        owner="sender-a",
    )


def test_compatibility_probe_waits_for_reprobe_and_recovers_without_restart(
    tmp_path: Path,
) -> None:
    # Given: the relay is initially incompatible and durable evidence is waiting.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    transport.capability_result = DeliveryFailure(
        DeliveryDisposition.COMPATIBILITY,
        "HTTP_404",
        status_code=404,
    )
    now = [100.0]
    sender = EvidenceSender(
        database,
        SenderConfig(
            relay_url="http://127.0.0.1:1",
            relay_token="test-token",
            probe_camera_id="camera-1",
            enabled=True,
        ),
        transport=transport,
        clock=lambda: now[0],
        owner="sender-a",
    )
    assert sender.run_once() is SenderStep.CAPABILITY_BLOCKED
    transport.capability_result = BackendCapabilities(event_idempotency=1, clip_export=1)

    # When: turns run before and at the bounded re-probe deadline.
    now[0] = 159.0
    before_deadline = sender.run_once()
    now[0] = 160.0
    at_deadline = sender.run_once()

    # Then: no intermediate request is made, and delivery resumes without restart.
    assert before_deadline is SenderStep.CAPABILITY_BLOCKED
    assert at_deadline is SenderStep.EVENT_ACKED
    assert [call for call in transport.calls if call.startswith("capability:")] == [
        "capability:camera-1",
        "capability:camera-1",
    ]


def test_sender_acks_every_event_before_one_coalesced_clip(tmp_path: Path) -> None:
    # Given: two ordered events reference one verified clip.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    sender = _sender(database, transport)

    # When: one durable item is processed per sender turn.
    steps = (sender.run_once(), sender.run_once(), sender.run_once())

    # Then: all event receipts precede the single immutable clip receipt.
    assert steps == (SenderStep.EVENT_ACKED, SenderStep.EVENT_ACKED, SenderStep.CLIP_ACKED)
    assert [call for call in transport.calls if not call.startswith("capability:")] == [
        f"event:{EVENT_A}",
        f"event:{EVENT_B}",
        f"clip:{CLIP}",
    ]
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.pending_count() == 0
        assert outbox.clip_publish_state(CLIP) == "PUBLISHED"
        assert outbox.is_clip_held(CLIP) is False


def test_response_loss_retries_same_event_and_converges_after_restart(tmp_path: Path) -> None:
    # Given: the backend committed, but the first response was lost.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    transport.event_results = [
        DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK"),
        EventReceipt("accepted", EVENT_A, "backend-event-a"),
    ]
    sender = _sender(database, transport)

    # When: the retry deadline passes and a fresh sender instance takes over.
    first = sender.run_once()
    restarted = EvidenceSender(
        database,
        sender.config,
        transport=transport,
        clock=lambda: 101.0,
        random_value=lambda: 0.0,
        owner="sender-b",
    )
    second = restarted.run_once()

    # Then: the same stable ID is acknowledged once in durable state.
    assert first is SenderStep.RETRY_SCHEDULED
    assert second is SenderStep.EVENT_ACKED
    assert [call for call in transport.calls if call.startswith("event:")] == [
        f"event:{EVENT_A}",
        f"event:{EVENT_A}",
    ]


@pytest.mark.parametrize("status", (400, 401, 403, 413, 415, 422))
def test_permanent_http_classes_dead_letter_and_hold_media(tmp_path: Path, status: int) -> None:
    # Given: one event receives a contract-defined permanent response.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    transport.event_results = [
        DeliveryFailure(DeliveryDisposition.PERMANENT, f"HTTP_{status}", status_code=status)
    ]

    # When: the sender handles the response.
    step = _sender(database, transport).run_once()

    # Then: it does not retry and the associated evidence remains held.
    assert step is SenderStep.PERMANENT_FAILURE
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(EVENT_A) == "PERMANENT"
        assert outbox.is_clip_held(CLIP) is True


def test_clip_capability_absence_still_acks_events_and_holds_media(tmp_path: Path) -> None:
    # Given: the backend proves event idempotency but not clip export.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    transport.capability_result = BackendCapabilities(event_idempotency=1, clip_export=0)

    # When: sender turns drain the durable events before reaching the held clip.
    sender = _sender(database, transport)
    steps = (sender.run_once(), sender.run_once(), sender.run_once())

    # Then: every event has a receipt while clip egress stays blocked and held.
    assert steps == (
        SenderStep.EVENT_ACKED,
        SenderStep.EVENT_ACKED,
        SenderStep.CAPABILITY_BLOCKED,
    )
    assert len([call for call in transport.calls if call.startswith("event:")]) == 2
    assert not any(call.startswith("clip:") for call in transport.calls)
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(EVENT_A) == "ACKED"
        assert outbox.is_clip_held(CLIP) is True


def test_clip_capability_zero_releases_old_event_compatibility_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    transport.event_results = [
        DeliveryFailure(DeliveryDisposition.COMPATIBILITY, "HTTP_404", status_code=404)
    ]
    sender = _sender(database, transport)
    assert sender.run_once() is SenderStep.PERMANENT_FAILURE
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(EVENT_A) == "COMPATIBILITY"

    transport.capability_result = BackendCapabilities(event_idempotency=1, clip_export=0)
    step = sender.run_once()

    assert step is SenderStep.EVENT_ACKED
    assert [call for call in transport.calls if call.startswith("event:")] == [
        f"event:{EVENT_A}",
        f"event:{EVENT_A}",
    ]
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(EVENT_A) == "ACKED"
        assert outbox.is_clip_held(CLIP) is True


def test_operator_sender_uses_real_transport_and_ack_transition(tmp_path: Path) -> None:
    # Given: one media-free operator event is staged in the production outbox.
    database = tmp_path / "evidence.sqlite3"
    event_id = EdgeEventId("00000000-0000-4000-8000-000000000099")
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=event_id,
                detected_at="2026-07-16T00:00:00Z",
                payload_json=json.dumps(
                    {
                        "edge_event_id": event_id,
                        "type": "SYSTEM_TEST",
                        "source": "SYSTEM_TEST",
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                queued_at=1.0,
                operator_only=True,
            )
        )
        assert outbox.mark_ready(event_id)
    transport = FakeTransport()
    sender = _sender(database, transport)

    # When: the explicit operator seam drives one sender turn.
    step = sender.run_operator_once(event_id)

    # Then: the real send_event/receipt/ACK path runs without a capability camera probe.
    assert step is SenderStep.EVENT_ACKED
    assert transport.calls == [f"event:{event_id}"]
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(event_id) == "ACKED"
        assert outbox.event_backend_event_id(event_id) == f"backend-{event_id}"


def test_operator_retry_and_ack_reuse_the_same_event_identity(tmp_path: Path) -> None:
    # Given: one explicit operator event whose first HTTP attempt is retryable.
    database = tmp_path / "evidence.sqlite3"
    event_id = EdgeEventId("00000000-0000-4000-8000-000000000098")
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=event_id,
                detected_at="2026-07-16T00:00:00Z",
                payload_json=_payload(event_id),
                queued_at=100.0,
                operator_only=True,
            )
        )
        assert outbox.mark_ready(event_id)
    transport = FakeTransport()
    transport.event_results = [
        DeliveryFailure(DeliveryDisposition.RETRY, "HTTP_503", status_code=503),
        EventReceipt("accepted", event_id, "backend-system-test"),
    ]
    first = _sender(database, transport).run_operator_once(event_id)

    # When: an operator explicitly retries after the durable due time.
    second = EvidenceSender(
        database,
        _sender(database, transport).config,
        transport=transport,
        clock=lambda: 101.0,
        random_value=lambda: 0.0,
        owner="operator-retry",
    ).run_operator_once(event_id)

    # Then: existing retry/ACK semantics keep one immutable event identity.
    assert (first, second) == (SenderStep.RETRY_SCHEDULED, SenderStep.EVENT_ACKED)
    assert [call for call in transport.calls if call.startswith("event:")] == [
        f"event:{event_id}",
        f"event:{event_id}",
    ]
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_attempt_count(event_id) == 2
        assert outbox.event_backend_event_id(event_id) == "backend-system-test"


def test_operator_exact_replay_sends_acked_payload_without_resetting_outbox(
    tmp_path: Path,
) -> None:
    # Given: a SYSTEM_TEST row already ACKed through the operator sender.
    database = tmp_path / "evidence.sqlite3"
    event_id = EdgeEventId("00000000-0000-4000-8000-000000000097")
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=event_id,
                detected_at="2026-07-16T00:00:00Z",
                payload_json=_payload(event_id),
                queued_at=100.0,
                operator_only=True,
            )
        )
        assert outbox.mark_ready(event_id)
    transport = FakeTransport()
    transport.event_results = [
        EventReceipt("accepted", event_id, "backend-system-test"),
        EventReceipt("accepted", event_id, "backend-system-test"),
    ]
    sender = _sender(database, transport)
    assert sender.run_operator_once(event_id) is SenderStep.EVENT_ACKED

    # When: exact replay is explicitly requested after ACK.
    replay = sender.replay_operator(event_id)

    # Then: transport receives the same ID/payload and terminal outbox state is untouched.
    assert replay == EventReceipt("accepted", event_id, "backend-system-test")
    assert [call for call in transport.calls if call.startswith("event:")] == [
        f"event:{event_id}",
        f"event:{event_id}",
    ]
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(event_id) == "ACKED"
        assert outbox.event_attempt_count(event_id) == 1


def test_disabled_sender_never_opens_network_or_claims(tmp_path: Path) -> None:
    # Given: phase-one export remains disabled by default.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    config = SenderConfig(
        relay_url="http://127.0.0.1:1",
        relay_token="test-token",
        probe_camera_id="camera-1",
    )

    # When: a sender turn executes with the default configuration.
    step = EvidenceSender(database, config, transport=transport).run_once()

    # Then: durable evidence is untouched and no egress occurs.
    assert step is SenderStep.DISABLED
    assert transport.calls == []
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_attempt_count(EVENT_A) == 0


def test_response_loss_converges_when_backend_reports_expired_after_retention(
    tmp_path: Path,
) -> None:
    # Given: events were acknowledged and the original clip response was lost.
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    sender = _sender(database, transport)
    assert sender.run_once() is SenderStep.EVENT_ACKED
    assert sender.run_once() is SenderStep.EVENT_ACKED
    transport.clip_results = [DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK")]
    assert sender.run_once() is SenderStep.RETRY_SCHEDULED

    # When: after retention, the backend returns the terminal current version.
    transport.clip_results = [
        ClipReceipt(
            clip_id=CLIP,
            state="EXPIRED",
            state_version=3,
            sha256="a" * 64,
            size_bytes=123,
        )
    ]
    restarted = EvidenceSender(
        database,
        sender.config,
        transport=transport,
        clock=lambda: 101.0,
        random_value=lambda: 0.0,
        owner="sender-after-retention",
    )
    step = restarted.run_once()

    # Then: EXPIRED is acknowledged, persisted, and releases the local hold.
    assert step is SenderStep.CLIP_ACKED
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_publish_state(CLIP) == "PUBLISHED"
        assert outbox.clip_remote_state(CLIP) == "EXPIRED"
        assert outbox.is_clip_held(CLIP) is False


@pytest.mark.parametrize(
    "receipt",
    (
        ClipReceipt(CLIP, "EXPIRED", 1, "a" * 64, 123),
        ClipReceipt(CLIP, "EXPIRED", 3, "b" * 64, 123),
        ClipReceipt(CLIP, "EXPIRED", 3, "a" * 64, 124),
    ),
)
def test_sender_rejects_expired_receipt_without_matching_immutable_facts(
    tmp_path: Path,
    receipt: ClipReceipt,
) -> None:
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    sender = _sender(database, transport)
    assert sender.run_once() is SenderStep.EVENT_ACKED
    assert sender.run_once() is SenderStep.EVENT_ACKED
    transport.clip_results = [receipt]

    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_publish_state(CLIP) == "WAITING"
        assert outbox.is_clip_held(CLIP) is True


def test_sender_rejects_expired_receipt_for_unavailable_clip(tmp_path: Path) -> None:
    database = tmp_path / "evidence.sqlite3"
    _stage_unavailable_clip(database)
    transport = FakeTransport()
    sender = _sender(database, transport)
    assert sender.run_once() is SenderStep.EVENT_ACKED
    transport.clip_results = [ClipReceipt(CLIP, "EXPIRED", 3, None, None)]

    assert sender.run_once() is SenderStep.RETRY_SCHEDULED
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_publish_state(CLIP) == "WAITING"
        assert outbox.is_clip_held(CLIP) is True


@pytest.mark.parametrize(
    "result",
    (
        EventReceipt("accepted", EVENT_A, "backend-event-a"),
        DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK"),
        DeliveryFailure(DeliveryDisposition.PERMANENT, "HTTP_422"),
    ),
)
def test_stale_event_sender_never_reports_a_cas_transition(
    tmp_path: Path,
    result: EventReceipt | DeliveryFailure,
) -> None:
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)

    class _EventRaceTransport(FakeTransport):
        def send_event(self, payload_json: str, edge_event_id: EdgeEventId):
            del payload_json, edge_event_id
            with EvidenceOutbox.open(database) as outbox:
                winner = outbox.claim(ClaimLease("sender-b", 111.0, 10.0))
                assert winner is not None
                assert outbox.acknowledge(winner, backend_event_id="winner")
            return result

    step = _sender(database, _EventRaceTransport()).run_once()

    assert step is SenderStep.LEASE_LOST
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.event_delivery_state(EVENT_A) == "ACKED"
        assert outbox.event_attempt_count(EVENT_A) == 2


@pytest.mark.parametrize(
    "result",
    (
        ClipReceipt(CLIP, "READY", 2, "a" * 64, 123),
        DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK"),
        DeliveryFailure(DeliveryDisposition.PERMANENT, "HTTP_422"),
    ),
)
def test_stale_clip_sender_never_reports_a_cas_transition(
    tmp_path: Path,
    result: ClipReceipt | DeliveryFailure,
) -> None:
    database = tmp_path / "evidence.sqlite3"
    _stage_clip(database)
    transport = FakeTransport()
    sender = _sender(database, transport)
    assert sender.run_once() is SenderStep.EVENT_ACKED
    assert sender.run_once() is SenderStep.EVENT_ACKED

    class _ClipRaceTransport(FakeTransport):
        def send_clip(self, claim):
            with EvidenceOutbox.open(database) as outbox:
                winner = outbox.claim_clip(ClaimLease("sender-b", 111.0, 10.0))
                assert winner is not None
                assert outbox.acknowledge_clip(
                    winner,
                    acknowledged_at=111.0,
                    remote_state="READY",
                )
            return result

    step = _sender(database, _ClipRaceTransport()).run_once()

    assert step is SenderStep.LEASE_LOST
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_publish_state(CLIP) == "PUBLISHED"
        assert outbox.clip_remote_state(CLIP) == "READY"
        assert outbox.is_clip_held(CLIP) is False
