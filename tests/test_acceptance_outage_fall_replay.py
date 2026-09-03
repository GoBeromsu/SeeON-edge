"""Owner acceptance gate for fall replay across a backend outage."""

from __future__ import annotations

import base64
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.features.evidence.relay_projection import RelayEvidenceProjection
from backend.app.main import create_app, no_lifespan
from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from tests_support.alert_amplification_runtime import RELAY_TOKEN
from tests_support.compact_authority_db import prepare_compact_database
from worker.domains.fall import (
    FallPolicyDeciderV2,
    FallV2DomainDecider,
    FallV2Probabilities,
    FallWindowClassifierV2,
)
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import CoordinatedInference
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.types import BusinessEvent, FramePacket, ModuleResult

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000c1"
_DETECTED_AT = "2026-08-22T00:00:00.000Z"
_DECISION_ENVELOPE = {
    "config_version": 17,
    "detector_version": "fall-detector-2026.08",
    "model_version": "pose-model-4",
    "operating_threshold": 0.5,
    "runtime_manifest_sha256": "a" * 64,
}
_CAMERA_ID = "room-camera"
_FACILITY_ID = "facility-1"
_RUNTIME_MANIFEST = "a" * 64


class _FallModel:
    artifact_digest = "b" * 64

    def predict(self, window: tuple[tuple[float, ...], ...]) -> FallV2Probabilities:
        assert len(window) == 30
        assert all(len(row) == 56 for row in window)
        return FallV2Probabilities(background=0.01, fall_transition=0.97, fallen=0.01)


class _PoseRunner:
    def run(self, image: Image) -> RunnerResult:
        del image
        keypoints = tuple(
            coordinate for index in range(17) for coordinate in (40 + index, 30 + index, 0.9)
        )
        return pose_result((keypoints,), ((20, 10, 120, 115, 0.95),))


class _SingleFallResult:
    def __init__(self, packets: tuple[FramePacket, ...]) -> None:
        self._packets = iter(packets)

    def take(self, *, timeout_sec: float | None = None) -> CoordinatedInference | None:
        del timeout_sec
        try:
            packet = next(self._packets)
        except StopIteration:
            return None
        return CoordinatedInference(
            packet,
            ModuleResult("pose", _PoseRunner().run(packet.frame.image), 0.0, "pose"),
        )


class _NoClipRecorder:
    def on_event(
        self,
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
        detected_at: datetime,
    ) -> None:
        del trigger_packet, event, allow_new_clip, detected_at


def _packet(sequence: int = 7) -> FramePacket:
    pts = 12.5 + (sequence - 7) / 15
    return FramePacket(
        camera_id=_CAMERA_ID,
        frame=Frame(sequence, pts, np.zeros((120, 180, 3), dtype=np.uint8)),
        pts=pts,
        seq=sequence,
        width=180,
        height=120,
        decode_time_ms=0.25,
        worker_boot_id="acceptance-worker",
        stream_epoch=3,
    )


def _analytics() -> CompositeExtractor:
    runner = _PoseRunner()
    return CompositeExtractor(
        extractors=(
            NamedExtractor(
                module_name="pose",
                runner=runner,
                _call=runner.run,
                _clock=lambda: 1.0,
                output_adapter="pose",
            ),
        ),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(_CAMERA_ID),
    )


@dataclass
class _RelayTransport:
    """In-process adapter to the real relay routes; None models no reachable backend."""

    client: TestClient | None = None
    sent_events: list[dict[str, object]] = field(default_factory=list)

    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt | DeliveryFailure:
        if self.client is None:
            return DeliveryFailure(
                DeliveryDisposition.RETRY,
                "backend-down",
                transport_error="ConnectionError: backend unavailable",
            )
        self.sent_events.append(json.loads(payload_json))
        response = self.client.post(
            "/api/v1/relay/alerts",
            content=payload_json,
            headers={"Content-Type": "application/json", "X-Edge-Relay-Token": RELAY_TOKEN},
        )
        if response.status_code != 202:
            return DeliveryFailure(
                DeliveryDisposition.RETRY, "relay-unavailable", response.status_code
            )
        body = response.json()
        return EventReceipt(
            str(body["status"]),
            str(body.get("edge_event_id", edge_event_id)),
            str(body.get("event_id", f"relay:{edge_event_id}")),
        )

    def send_snapshot_attachment(self, payload: dict[str, object]) -> DeliveryFailure:
        del payload
        return DeliveryFailure(DeliveryDisposition.RETRY, "not-used")

    def send_snapshot_disposition(self, payload: dict[str, object]) -> DeliveryFailure | None:
        if self.client is None:
            return DeliveryFailure(DeliveryDisposition.RETRY, "backend-down")
        response = self.client.post(
            "/api/v1/relay/snapshot-dispositions",
            json=payload,
            headers={"X-Edge-Relay-Token": RELAY_TOKEN},
        )
        if response.status_code == 202:
            return None
        return DeliveryFailure(DeliveryDisposition.RETRY, "relay-unavailable", response.status_code)


def _stager(queue_directory: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        queue_directory,
        camera_id=_CAMERA_ID,
        facility_id=_FACILITY_ID,
        config_version=17,
    )


def _stage_scripted_fall(queue_directory: Path) -> str:
    """Drive a fall through pump, aggregator, attacher, and event-first sink."""
    stager = _stager(queue_directory)
    snapshot_root = queue_directory.parent / "broken-snapshot-store"
    snapshot_root.write_bytes(b"not-a-directory")
    detector = FallV2DomainDecider(
        classifier=FallWindowClassifierV2(_FallModel()),
        policy=FallPolicyDeciderV2(
            camera_id=_CAMERA_ID,
            facility_id=_FACILITY_ID,
            boot_id="acceptance-worker",
            stream_epoch="3",
            source_generation=0,
        ),
    )
    sink = EvidenceEventSink(
        stager=stager,
        recorder=_NoClipRecorder(),
        now=lambda: datetime.fromisoformat(_DETECTED_AT).astimezone(UTC),
        snapshot_store=SnapshotStore(snapshot_root),
    )
    pump = CameraPipelinePump(
        _CAMERA_ID,
        _SingleFallResult(tuple(_packet(sequence) for sequence in range(7, 52))),
        _analytics(),
        EventAggregator((detector,), IncidentManager(cooldown_sec=0.0), monotonic=lambda: 12.5),
        sink,
        max_frames=45,
        evidence_attacher=AlertEvidenceAttacher(
            domain_audit={"fall": _DECISION_ENVELOPE},
            overlay_renderer=_SnapshotRenderer(),
            runtime_manifest_sha256=_RUNTIME_MANIFEST,
        ),
    )
    pump.run()
    assert pump.failure_count == 0
    [event] = [
        entry for entry in DeliveryQueue(queue_directory).entries() if entry["kind"] == "EVENT"
    ]
    return str(event["edge_event_id"])


class _SnapshotRenderer:
    def encode_jpeg_bounded(self, *_: object) -> bytes:
        return b"actual-snapshot-capture"


def _relay_client(tmp_path: Path, database: Path) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = RELAY_TOKEN
    registry_path = tmp_path / "camera-registry.sqlite3"
    prepare_compact_database(registry_path)
    registry = CameraRegistryStore(registry_path)
    registry.create(
        camera_id="room-camera",
        label="room-camera",
        rtsp_url="rtsp://role-gateway:8554/room-camera",
        space_id=None,
        status="online",
        backend_camera_id="room-camera",
    )
    app.state.camera_registry = registry
    app.state.relay_evidence_projection = RelayEvidenceProjection(database)
    app.state.central_evidence_query = CentralEvidenceQuery(database)
    return TestClient(app)


def test_fall_survives_outage_and_replays_with_terminal_snapshot_absence(tmp_path: Path) -> None:
    """Event admission precedes optional media and remains publish-once through restart."""

    queue_directory = tmp_path / "delivery-queue"
    edge_event_id = _stage_scripted_fall(queue_directory)

    queue = DeliveryQueue(queue_directory)
    [event_entry] = [entry for entry in queue.entries() if entry["kind"] == "EVENT"]
    event_path = queue_directory / f"{event_entry['entry_id']}.json"
    assert event_path.is_file()
    assert json.loads(event_path.read_text(encoding="ascii")) == event_entry
    decision_trace = json.loads(base64.b64decode(str(event_entry["decision_trace_b64"])))
    assert decision_trace == _DECISION_ENVELOPE
    [snapshot_disposition] = [
        entry for entry in queue.entries() if entry["kind"] == "SNAPSHOT_DISPOSITION"
    ]
    assert snapshot_disposition["disposition"] == "UNAVAILABLE"
    assert snapshot_disposition["reason"] == "stage_failed"

    transport = _RelayTransport()
    sender_config = SenderConfig("http://relay.test", RELAY_TOKEN, "room-camera")
    assert EvidenceSender(queue_directory, sender_config, transport=transport).run_once() is (
        SenderStep.RETRY_SCHEDULED
    )
    # Reconstructing the sender simulates a sending-side restart. The committed
    # event file remains until an authenticated relay acknowledgement exists.
    assert EvidenceSender(queue_directory, sender_config, transport=transport).run_once() is (
        SenderStep.RETRY_SCHEDULED
    )
    assert event_path.is_file()

    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    with _relay_client(tmp_path, database) as relay:
        transport.client = relay
        sender = EvidenceSender(queue_directory, sender_config, transport=transport)
        assert sender.run_once() is SenderStep.EVENT_ACKED
        assert not event_path.exists()
        assert sender.run_once() is SenderStep.CLIP_ACKED
        assert (
            relay.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            ).status_code
            == 204
        )
        response = relay.get(f"/api/v1/incidents/{edge_event_id}")

    assert DeliveryQueue(queue_directory).accepted_count == 0
    assert response.status_code == 200
    summary = response.json()
    assert summary["edge_event_id"] == edge_event_id
    assert summary["event_type"] == "fall"
    assert summary["event_delivery_state"] == "ACKED"
    assert summary["snapshot_artifact_state"] == "UNAVAILABLE"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            """
            SELECT artifact.state, artifact.reason
            FROM incidents AS incident
            JOIN artifacts AS artifact ON artifact.incident_id = incident.incident_id
            WHERE incident.edge_event_id = ? AND artifact.kind = 'SNAPSHOT'
            """,
            (edge_event_id,),
        ).fetchone() == ("UNAVAILABLE", "UNAVAILABLE:stage_failed")


def test_replayed_fall_keeps_decision_envelope(
    tmp_path: Path,
) -> None:
    """The decision envelope must survive the outage, not just the event.

    This caught a real loss: the stager pops ``audit`` into ``decision_trace`` so
    it is admitted atomically with the event, but the sender transmitted only
    ``values``, so the decision basis never reached the backend and the relay
    projection recorded ``audit=None``. Fixed in ``evidence_sender._payload``.
    """

    queue_directory = tmp_path / "delivery-queue"
    edge_event_id = _stage_scripted_fall(queue_directory)
    [event_entry] = [
        entry for entry in DeliveryQueue(queue_directory).entries() if entry["kind"] == "EVENT"
    ]
    decision_trace = json.loads(base64.b64decode(str(event_entry["decision_trace_b64"])))
    assert decision_trace == _DECISION_ENVELOPE

    database = tmp_path / "edge.sqlite3"
    bootstrap_database(database)
    transport = _RelayTransport()
    sender_config = SenderConfig("http://relay.test", RELAY_TOKEN, "room-camera")
    with _relay_client(tmp_path, database) as relay:
        transport.client = relay
        assert (
            EvidenceSender(queue_directory, sender_config, transport=transport).run_once()
            is SenderStep.EVENT_ACKED
        )

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT edge_event_id FROM incidents WHERE edge_event_id=?",
            (edge_event_id,),
        ).fetchone() == (edge_event_id,)
    assert transport.sent_events[-1]["audit"] == _DECISION_ENVELOPE
