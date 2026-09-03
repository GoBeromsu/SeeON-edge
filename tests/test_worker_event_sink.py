from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from contracts.event import EventEvidence
from contracts.frame import Frame
from worker.interfaces.output import EventSink
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
from worker.pipeline.output.evidence.clip_publication import ClipPublicationMetadata, ClipPublisher
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_outbox_types import EdgeEventId, EvidenceReasonCode
from worker.pipeline.output.evidence.snapshot_store import (
    SnapshotLimits,
    SnapshotStore,
    StoredSnapshot,
)
from worker.types import FramePacket
from worker.types.business_event import BusinessEvent

RUNTIME_MANIFEST_SHA256 = "b" * 64


@dataclass(slots=True)
class _RecordingStager:
    staged: list[WorkerEventPayload] = field(default_factory=list)
    completions: list[tuple[str, str | None]] = field(default_factory=list)
    attached: list[tuple[str, EventEvidence]] = field(default_factory=list)
    dispositions: list[tuple[str, str, str, str]] = field(default_factory=list)

    def stage(self, event: WorkerEventPayload) -> None:
        self.staged.append(event)

    def attach_snapshot(self, edge_event_id: str, snapshot: EventEvidence) -> None:
        self.attached.append((edge_event_id, snapshot))

    def record_snapshot_disposition(
        self, edge_event_id: str, snapshot_id: str, disposition: str, reason: str
    ) -> None:
        self.dispositions.append((edge_event_id, snapshot_id, disposition, reason))

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        self.completions.append((edge_event_id, clip_id))


@dataclass(slots=True)
class _RecordingRecorder:
    clip_id: str | None
    calls: list[tuple[str, BusinessEvent, bool, datetime]] = field(default_factory=list)

    def on_event(
        self,
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
        detected_at: datetime,
    ) -> str | None:
        self.calls.append((trigger_packet.camera_id, event, allow_new_clip, detected_at))
        return self.clip_id


def _event() -> BusinessEvent:
    return BusinessEvent(
        domain="fall",
        event_type="fall_detected",
        identity="event-123",
        camera_id="camera-1",
        facility_id="facility-1",
        time_sec=12.5,
        probability=0.91,
        person_id=7,
        audit={
            "clock_source": "edge_wall_clock",
            "model_version": "fall-model-v1",
            "detector_version": "detector-v1",
            "operating_threshold": 0.73,
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        },
    )


def _trigger_packet() -> FramePacket:
    return FramePacket(
        camera_id="camera-1",
        frame=Frame(4, 12.5, np.zeros((2, 2, 3), dtype=np.uint8)),
        pts=12.5,
        seq=4,
        width=2,
        height=2,
        decode_time_ms=0.1,
        worker_boot_id="boot-1",
        stream_epoch=3,
    )


def test_event_sink_stages_then_binds_the_admitted_business_event() -> None:
    from worker.pipeline.output.event_sink import EvidenceEventSink

    # Given: a durable stager and a recorder that reserves one clip.
    stager = _RecordingStager()
    recorder = _RecordingRecorder(clip_id="clip-123")
    sink = EvidenceEventSink(
        stager=stager,
        recorder=recorder,
        now=lambda: datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    # When: the decision pipeline emits an admitted immutable event.
    sink.emit_for_frame(_event(), _trigger_packet())

    # Then: its canonical relay payload is durable before its clip relation completes.
    assert stager.staged == [
        {
            "edge_event_id": "event-123",
            "event_type": "fall_detected",
            "probability": 0.91,
            "detected_at": "2026-07-31T12:00:00Z",
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "evidence": {
                "domain": "fall",
                "identity": "event-123",
                "time_sec": 12.5,
                "person_id": 7,
            },
            "audit": {
                "clock_source": "edge_wall_clock",
                "model_version": "fall-model-v1",
                "detector_version": "detector-v1",
                "operating_threshold": 0.73,
                "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            },
        }
    ]
    assert recorder.calls == [
        ("camera-1", _event(), True, datetime(2026, 7, 31, 12, 0, tzinfo=UTC))
    ]
    assert stager.completions == [("event-123", "clip-123")]
    assert isinstance(sink, EventSink)


def test_relay_detected_at_matches_the_published_manifest_byte_for_byte(tmp_path: Path) -> None:
    stager = _RecordingStager()
    recorder = _RecordingRecorder(clip_id="clip-123")
    detected_at = datetime(2026, 7, 31, 12, 0, 0, 123456, tzinfo=UTC)
    EvidenceEventSink(stager=stager, recorder=recorder, now=lambda: detected_at).emit_for_frame(
        _event(), _trigger_packet()
    )

    captured_at = recorder.calls[0][3]
    reservation = ClipIdAllocator(tmp_path, id_factory=lambda _camera: "clip-123").reserve(
        "camera-1"
    )
    published = ClipPublisher(tmp_path).publish_unavailable(
        reservation,
        ClipPublicationMetadata(
            camera_id="camera-1",
            event_refs=(EdgeEventId("00000000-0000-4000-8000-000000000123"),),
            event_type="fall_detected",
            clip_start_at=detected_at - timedelta(seconds=30),
            clip_end_at=detected_at + timedelta(seconds=30),
            finalized_at=detected_at + timedelta(seconds=31),
            started_at=detected_at - timedelta(seconds=30),
            detected_at=captured_at,
            duration_s=60.0,
            encoder="source-packet-remux",
        ),
        EvidenceReasonCode.NO_FRAMES,
    )

    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    assert manifest["detected_at"] == stager.staged[0]["detected_at"]


def test_event_sink_renders_backpressure_identity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = SnapshotStore(
        tmp_path,
        limits=SnapshotLimits(
            max_pending_global=2,
            max_pending_per_camera=1,
            max_files_global=10,
            max_files_per_camera=10,
            max_bytes_global=1024,
            max_bytes_per_camera=1024,
            max_age=timedelta(days=60),
            max_pending_age=timedelta(days=1),
        ),
    )
    first = store.stage(
        b"keep",
        snapshot_id="event-keep",
        captured_at="2026-07-31T12:00:00Z",
        camera_id="camera-1",
        edge_event_id="event-keep",
    )
    event = replace(_event(), snapshot_jpeg=b"jpeg")
    sink = EvidenceEventSink(
        stager=_RecordingStager(),
        recorder=_RecordingRecorder(clip_id=None),
        snapshot_store=store,
    )

    with caplog.at_level(logging.WARNING):
        sink.emit_for_frame(event, _trigger_packet())

    message = caplog.records[-1].getMessage()
    assert "camera_id=camera-1" in message
    assert "edge_event_id=event-123" in message
    assert "reason=per-camera pending files" in message
    assert store.staged_records() == (first,)
    assert store.stats.dropped_capacity == 1


def test_event_sink_renders_publication_identity_and_preserves_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = SnapshotStore(tmp_path)
    committed: list[StoredSnapshot] = []

    def fail_publish(self: SnapshotStore, snapshot: StoredSnapshot) -> None:
        del self, snapshot
        raise OSError("publish failed")

    def refuse_commit(self: SnapshotStore, snapshot: StoredSnapshot) -> None:
        del self
        committed.append(snapshot)
        raise AssertionError("commit must wait for publication")

    monkeypatch.setattr(SnapshotStore, "publish", fail_publish)
    monkeypatch.setattr(SnapshotStore, "commit", refuse_commit)
    sink = EvidenceEventSink(
        stager=_RecordingStager(),
        recorder=_RecordingRecorder(clip_id=None),
        snapshot_store=store,
    )

    with caplog.at_level(logging.ERROR):
        sink.emit_for_frame(replace(_event(), snapshot_jpeg=b"jpeg"), _trigger_packet())

    message = caplog.records[-1].getMessage()
    assert "camera_id=camera-1" in message
    assert "edge_event_id=event-123" in message
    assert caplog.records[-1].exc_info is not None
    assert committed == []
    assert store.stats.staged == 1
    assert store.stats.published == 0


def test_event_sink_rejects_invalid_runtime_manifest_before_any_side_effect() -> None:
    from worker.pipeline.output.event_sink import EvidenceEventSink

    stager = _RecordingStager()
    recorder = _RecordingRecorder(clip_id="clip-123")
    sink = EvidenceEventSink(stager=stager, recorder=recorder)
    invalid = replace(
        _event(),
        audit={"runtime_manifest_sha256": "B" * 64},
    )

    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        sink.emit_for_frame(invalid, _trigger_packet())

    assert stager.staged == []
    assert stager.completions == []
    assert recorder.calls == []


def test_event_sink_completes_without_clip_when_recording_is_unavailable() -> None:
    from worker.pipeline.output.event_sink import EvidenceEventSink

    # Given: a recorder that cannot reserve a new clip.
    stager = _RecordingStager()
    sink = EvidenceEventSink(
        stager=stager,
        recorder=_RecordingRecorder(clip_id=None),
        now=lambda: datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    # When: the event is emitted.
    sink.emit_for_frame(_event(), _trigger_packet())

    # Then: durable delivery remains ready rather than being dropped.
    assert stager.completions == [("event-123", None)]


def test_worker_relay_surface_delegates_http_to_the_shared_bounded_transport() -> None:
    # Given: the worker surfaces that emit facts and status to the backend.
    repo_root = Path(__file__).resolve().parents[1]
    relay_sources = (
        "worker/runtime/worker.py",
        "worker/runtime/telemetry/runtime_status_sender.py",
    )

    for relative in relay_sources:
        source = (repo_root / relative).read_text(encoding="utf-8")

        # When: that surface sends a request.
        # Then: it delegates to the one shared bounded transport, never opens HTTP itself.
        assert "bounded_request" in source, relative
        assert "urllib.request.urlopen" not in source, relative
