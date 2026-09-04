from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from shared.events.delivery_queue import ClipEntry, DeliveryQueue, EventEntry
from shared.events.evidence_export_contract import (
    ClipReceipt,
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
from worker.pipeline.output.evidence.clip_publication import ClipPublicationMetadata, ClipPublisher
from worker.pipeline.output.evidence.evidence_outbox_types import EdgeEventId, EvidenceReasonCode
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep


class _Transport:
    def __init__(self, result: object) -> None:
        self.result = result
        self.claims: list[object] = []

    def send_clip(self, claim: object) -> object:
        self.claims.append(claim)
        return self.result

    def send_event(self, payload_json: str, edge_event_id: str) -> object:
        raise AssertionError("no event delivery expected")

    def send_snapshot_attachment(self, payload: dict[str, object]) -> object:
        raise AssertionError("no snapshot delivery expected")

    def send_snapshot_disposition(self, payload: dict[str, object]) -> object:
        raise AssertionError("no snapshot delivery expected")


def _entry() -> ClipEntry:
    return ClipEntry(
        clip_id="clip-1",
        event_ids=("00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"),
        camera_id="camera-1",
        facility_id="facility-1",
        local_state="VERIFIED",
        state_version=2,
        media_reference="clips/clip-1/clip.mp4",
        sha256="a" * 64,
        size_bytes=10,
        mime_type="video/mp4",
        codec="h264",
        duration_ms=1000,
        clip_start_at="2026-01-01T00:00:00Z",
        clip_end_at="2026-01-01T00:00:01Z",
        finalized_at="2026-01-01T00:00:02Z",
        unavailable_reason=None,
    )


def _sender(
    directory: Path,
    transport: _Transport,
    *,
    enabled: bool = True,
    flow_sealed_sidecar_directory: Path | None = None,
) -> EvidenceSender:
    return EvidenceSender(
        directory,
        SenderConfig("http://relay", "token", "camera-1"),
        transport=transport,
        clip_export_enabled=lambda: enabled,
        flow_sealed_sidecar_directory=flow_sealed_sidecar_directory,
    )


def test_published_clip_enqueues_all_contributors_and_delivers_receipt(tmp_path: Path) -> None:
    store = tmp_path / "store"
    queue_directory = tmp_path / "queue"
    reservation = ClipIdAllocator(store).reserve("camera-1")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    published = ClipPublisher(store, delivery_queue_directory=queue_directory).publish_unavailable(
        reservation,
        ClipPublicationMetadata(
            camera_id="camera-1",
            event_refs=(
                EdgeEventId("00000000-0000-4000-8000-000000000001"),
                EdgeEventId("00000000-0000-4000-8000-000000000002"),
            ),
            event_type="fall",
            clip_start_at=start,
            clip_end_at=start + timedelta(seconds=1),
            finalized_at=start + timedelta(seconds=2),
            started_at=start,
            detected_at=start,
            duration_s=1,
            encoder="test",
            facility_id="facility-1",
        ),
        EvidenceReasonCode.NO_FRAMES,
    )
    entries = tuple(DeliveryQueue(queue_directory).entries())
    assert len(entries) == 1
    assert entries[0]["kind"] == "CLIP"
    assert entries[0]["event_ids"] == [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ]

    transport = _Transport(ClipReceipt(str(published.clip_id), "UNAVAILABLE", 2, None, None))
    assert _sender(queue_directory, transport).run_once() is SenderStep.CLIP_ACKED
    assert not tuple(DeliveryQueue(queue_directory).entries())
    assert len(transport.claims) == 1


def test_clip_delivery_disabled_retains_entry_without_sending(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_entry()).accepted
    transport = _Transport(ClipReceipt("clip-1", "READY", 2, "a" * 64, 10))
    assert _sender(tmp_path, transport, enabled=False).run_once() is SenderStep.RETRY_SCHEDULED
    assert len(tuple(queue.entries())) == 1
    assert not transport.claims


def test_staged_event_is_handed_to_transport(tmp_path: Path) -> None:
    class _EventTransport(_Transport):
        def __init__(self) -> None:
            super().__init__(ClipReceipt("unused", "READY", 2, None, None))
            self.events: list[str] = []

        def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt:
            assert '"edge_event_id":"event-a"' in payload_json
            self.events.append(edge_event_id)
            return EventReceipt("accepted", edge_event_id, "backend-event")

    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(
        EventEntry(
            edge_event_id="event-a",
            event_type="fall",
            detected_at="2026-01-01T00:00:00Z",
            camera_id="camera-1",
            facility_id="facility-1",
            decision_trace=b"{}",
            values=b'{"edge_event_id":"event-a"}',
        )
    ).accepted
    transport = _EventTransport()

    assert _sender(tmp_path, transport).run_once() is SenderStep.EVENT_ACKED
    assert transport.events == ["event-a"]
    assert not tuple(DeliveryQueue(tmp_path).entries())


def test_clip_mapping_refusal_retries_but_bad_clip_dead_letters(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path)
    assert queue.try_admit(_entry()).accepted
    mapping_missing = _Transport(
        DeliveryFailure(DeliveryDisposition.PERMANENT, "CAMERA_MAPPING_MISSING", 409)
    )
    assert _sender(tmp_path, mapping_missing).run_once() is SenderStep.RETRY_SCHEDULED
    assert len(tuple(queue.entries())) == 1
    assert not queue.dead_letter_directory.exists()

    refused = _Transport(DeliveryFailure(DeliveryDisposition.PERMANENT, "INVALID", 400))
    assert _sender(tmp_path, refused).run_once() is SenderStep.RETRY_SCHEDULED
    assert not tuple(queue.entries())
    assert len(tuple(queue.dead_letter_directory.iterdir())) == 1

    assert queue.try_admit(_entry()).accepted
    retrying = _Transport(DeliveryFailure(DeliveryDisposition.RETRY, "UNAVAILABLE", 503))
    assert _sender(tmp_path, retrying).run_once() is SenderStep.RETRY_SCHEDULED
    assert len(tuple(queue.entries())) == 1


def test_clip_receipt_removes_flow_sealed_state(tmp_path: Path) -> None:
    queue = DeliveryQueue(tmp_path / "queue")
    assert queue.try_admit(_entry()).accepted
    sidecars = tmp_path / "flow-sealed"
    sidecars.mkdir()
    (sidecars / "clip-1.json").write_text("{}", encoding="utf-8")

    receipt = _Transport(ClipReceipt("clip-1", "READY", 2, "a" * 64, 10))
    assert (
        _sender(
            tmp_path / "queue",
            receipt,
            flow_sealed_sidecar_directory=sidecars,
        ).run_once()
        is SenderStep.CLIP_ACKED
    )
    assert not (sidecars / "clip-1.json").exists()


def test_clip_entry_survives_queue_restart(tmp_path: Path) -> None:
    assert DeliveryQueue(tmp_path).try_admit(_entry()).accepted
    reopened = DeliveryQueue(tmp_path)
    assert [entry["clip_id"] for entry in reopened.entries()] == ["clip-1"]
