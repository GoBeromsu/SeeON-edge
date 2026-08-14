from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import worker.pipeline.output.evidence.evidence_runtime as runtime_module
from shared.events.evidence_export_contract import (
    BackendCapabilities,
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.pipeline.output.evidence.evidence_manifest import unavailable_manifest
from worker.pipeline.output.evidence.evidence_outbox import (
    ClaimedClip,
    ClipId,
    ClipLocalState,
    EdgeEventId,
    EvidenceOutbox,
    EvidenceReasonCode,
    StagedEvent,
)
from worker.pipeline.output.evidence.evidence_runtime import EvidenceExportRuntime
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep

EVENT_ID = EdgeEventId("00000000-0000-4000-8000-000000000099")
CLIP_ID = ClipId("clip-live-finalize")
FINALIZED_AT = datetime(2026, 7, 16, 1, 2, 4, tzinfo=UTC)


class LiveTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.probed = threading.Event()
        self.clip_attempted = threading.Event()
        self.clip_results: list[ClipReceipt | DeliveryFailure] = []

    def probe_capabilities(
        self,
        camera_id: str,
    ) -> BackendCapabilities | DeliveryFailure:
        self.calls.append(f"capability:{camera_id}")
        self.probed.set()
        return BackendCapabilities(event_idempotency=1, clip_export=1)

    def send_event(
        self,
        payload_json: str,
        edge_event_id: EdgeEventId,
    ) -> EventReceipt | DeliveryFailure:
        payload = json.loads(payload_json)
        self.calls.append(f"event:{payload['edge_event_id']}")
        return EventReceipt("accepted", edge_event_id, "backend-live-event")

    def send_clip(self, claim: ClaimedClip) -> ClipReceipt | DeliveryFailure:
        self.calls.append(f"clip:{claim.clip_id}")
        self.clip_attempted.set()
        if self.clip_results:
            return self.clip_results.pop(0)
        return ClipReceipt(
            clip_id=claim.clip_id,
            state="UNAVAILABLE",
            state_version=claim.state_version,
            sha256=claim.sha256,
            size_bytes=claim.size_bytes,
        )


def _sender(
    database: Path,
    transport: LiveTransport,
    *,
    now: float = 100.0,
    owner: str = "live-sender",
) -> EvidenceSender:
    return EvidenceSender(
        database,
        SenderConfig(
            relay_url="http://127.0.0.1:1",
            relay_token="test-token",
            probe_camera_id="camera-1",
            lease_seconds=10.0,
        ),
        transport=transport,
        clip_export_enabled=lambda: True,
        clock=lambda: now,
        random_value=lambda: 1.0,
        owner=owner,
    )


def _stage_awaiting_clip(database: Path) -> None:
    payload = json.dumps(
        {
            "edge_event_id": EVENT_ID,
            "event_type": "fall",
            "probability": 0.9,
            "detected_at": "2026-07-16T01:02:03Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "evidence": {"clip_id": CLIP_ID},
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    with EvidenceOutbox.open(database) as outbox:
        outbox.stage(
            StagedEvent(
                edge_event_id=EVENT_ID,
                detected_at="2026-07-16T01:02:03Z",
                payload_json=payload,
                queued_at=1.0,
            )
        )
        outbox.bind_clip(EVENT_ID, CLIP_ID)


def _finalize_unavailable(store: Path) -> None:
    clip_dir = store / "clips" / CLIP_ID
    clip_dir.mkdir(parents=True)
    manifest = unavailable_manifest(
        clip_id=CLIP_ID,
        camera_id="camera-1",
        event_refs=(EVENT_ID,),
        clip_start_at=FINALIZED_AT - timedelta(seconds=1),
        clip_end_at=FINALIZED_AT,
        finalized_at=FINALIZED_AT,
        reason_code=EvidenceReasonCode.NO_FRAMES,
    )
    (clip_dir / "manifest.json").write_text(
        manifest.model_dump_json(),
        encoding="utf-8",
    )


def test_live_finalize_moves_acked_event_from_awaiting_to_clip_ack(
    tmp_path: Path,
) -> None:
    # Given: an event is durable and bound to a clip that has not finalized yet.
    store = tmp_path / "store"
    database = tmp_path / "evidence.sqlite3"
    transport = LiveTransport()
    sender = _sender(database, transport)
    runtime = EvidenceExportRuntime(store, database, sender)
    runtime.initialize_under_lock()
    _stage_awaiting_clip(database)

    # When: the event is acknowledged before the recorder commits its manifest.
    before_finalize = (sender.run_once(), sender.run_once())

    # Then: the clip is not sent while durable state is still awaiting finalization.
    assert before_finalize == (SenderStep.EVENT_ACKED, SenderStep.IDLE)
    assert [call for call in transport.calls if call.startswith("clip:")] == []
    with EvidenceOutbox.open(database) as outbox:
        outcome = outbox.clip_outcome(CLIP_ID)
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.AWAITING_FINALIZE

    # When: the running sender is woken by the recorder's exact clip notification.
    _finalize_unavailable(store)
    runtime.start_sender()
    try:
        runtime.notify_clip_finalized(CLIP_ID)
        assert transport.clip_attempted.wait(timeout=1.0)
    finally:
        runtime.stop_sender()

    # Then: the live process reconciles and acknowledges the clip without a restart.
    assert [call for call in transport.calls if call.startswith("clip:")] == [f"clip:{CLIP_ID}"]
    with EvidenceOutbox.open(database) as outbox:
        assert outbox.clip_publish_state(CLIP_ID) == "PUBLISHED"


def test_live_finalize_survives_clip_response_loss_and_sender_restart(
    tmp_path: Path,
) -> None:
    # Given: a live finalized clip whose first backend response is lost.
    store = tmp_path / "store"
    database = tmp_path / "evidence.sqlite3"
    transport = LiveTransport()
    sender = _sender(database, transport)
    runtime = EvidenceExportRuntime(store, database, sender)
    runtime.initialize_under_lock()
    _stage_awaiting_clip(database)
    assert sender.run_once() is SenderStep.EVENT_ACKED
    transport.clip_results.append(DeliveryFailure(DeliveryDisposition.RETRY, "NETWORK"))
    _finalize_unavailable(store)
    runtime.notify_clip_finalized(CLIP_ID)
    runtime.start_sender()
    try:
        assert transport.clip_attempted.wait(timeout=1.0)
    finally:
        runtime.stop_sender()
    assert [call for call in transport.calls if call.startswith("clip:")] == [f"clip:{CLIP_ID}"]
    transport.clip_attempted.clear()

    # When: a fresh runtime and sender take over after the retry deadline.
    restarted = _sender(database, transport, now=101.0, owner="restarted-sender")
    restarted_runtime = EvidenceExportRuntime(
        store,
        database,
        restarted,
    )
    restarted_runtime.initialize_under_lock()
    restarted_runtime.start_sender()
    try:
        assert transport.clip_attempted.wait(timeout=1.0)
    finally:
        restarted_runtime.stop_sender()

    # Then: the same durable clip is retried and reaches its terminal ACK once.
    assert [call for call in transport.calls if call.startswith("clip:")] == [
        f"clip:{CLIP_ID}",
        f"clip:{CLIP_ID}",
    ]


def test_transient_live_reconcile_failure_is_retried_by_sender_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the first exact-clip reconciliation attempt loses local storage access.
    store = tmp_path / "store"
    database = tmp_path / "evidence.sqlite3"
    transport = LiveTransport()
    sender = _sender(database, transport)
    runtime = EvidenceExportRuntime(store, database, sender)
    runtime.initialize_under_lock()
    _stage_awaiting_clip(database)
    _finalize_unavailable(store)
    real_reconcile = runtime_module.reconcile_finalized_clip
    attempts = 0

    def flaky_reconcile(
        store_dir: Path,
        clip_id: ClipId,
        outbox: EvidenceOutbox,
    ) -> ClipLocalState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary outbox outage")
        return real_reconcile(store_dir, clip_id, outbox)

    monkeypatch.setattr(runtime_module, "reconcile_finalized_clip", flaky_reconcile)

    # When: notification fails once and the already-running sender loop retries it.
    runtime.notify_clip_finalized(CLIP_ID)
    runtime.start_sender()
    try:
        assert transport.probed.wait(timeout=1.0)
    finally:
        runtime.stop_sender()

    # Then: the durable pending notification is not missed.
    with EvidenceOutbox.open(database) as outbox:
        outcome = outbox.clip_outcome(CLIP_ID)
    assert attempts >= 2
    assert outcome is not None
    assert outcome.local_state is ClipLocalState.UNAVAILABLE


def test_finalize_notification_defers_descriptor_work_to_sender_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: descriptor reconciliation would be slow or temporarily unavailable.
    store = tmp_path / "store"
    database = tmp_path / "evidence.sqlite3"
    runtime = EvidenceExportRuntime(
        database_path=database, store_dir=store, sender=_sender(database, LiveTransport())
    )
    runtime.initialize_under_lock()
    called = threading.Event()
    release = threading.Event()
    second_returned = threading.Event()

    def blocked_reconcile(
        store_dir: Path,
        clip_id: ClipId,
        outbox: EvidenceOutbox,
    ) -> ClipLocalState:
        del store_dir, clip_id, outbox
        called.set()
        assert release.wait(timeout=1.0)
        return ClipLocalState.UNAVAILABLE

    monkeypatch.setattr(runtime_module, "reconcile_finalized_clip", blocked_reconcile)

    # When: the recorder notifies while no sender work is active.
    runtime.notify_clip_finalized(CLIP_ID)
    assert not called.is_set()

    # And: a second callback arrives while the sender is blocked in descriptor work.
    runtime.start_sender()
    assert called.wait(timeout=1.0)
    notifier = threading.Thread(
        target=lambda: (
            runtime.notify_clip_finalized(ClipId("clip-second")),
            second_returned.set(),
        ),
        daemon=True,
    )
    notifier.start()
    try:
        # Then: callbacks only queue and wake; they never wait on descriptor I/O.
        assert second_returned.wait(timeout=0.5)
    finally:
        release.set()
        notifier.join(timeout=1.0)
        runtime.stop_sender()
