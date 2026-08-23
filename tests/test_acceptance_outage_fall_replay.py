"""Owner acceptance gate for fall replay across a backend outage."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.features.evidence.relay_projection import RelayEvidenceProjection
from backend.app.main import create_app, no_lifespan
from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from shared.detection_policies import FallPolicyV1, make_effective_policy
from shared.events.delivery_queue import DeliveryQueue
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from tests_support.alert_amplification_runtime import RELAY_TOKEN
from worker.domains.fall import FallEventLatch
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import CoordinatedInference
from worker.pipeline.output.annotated_derivative import AnnotatedDerivativeJob, DerivativeKind
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.evidence_sender import EvidenceSender, SenderConfig, SenderStep
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence.snapshot_store import SnapshotStore
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.pipeline.trace import (
    BoundedTraceWriter,
    TraceCapture,
    TraceIdentity,
    TraceRetentionPolicy,
)
from worker.pipeline.trace.models import DetailUnavailableReason
from worker.runtime.derivative_runtime import (
    DerivativeCommand,
    DerivativeCommandExecutor,
    DerivativeOutcome,
)
from worker.types import FallModelInput, FramePacket, ModuleResult
from worker.types.overlay_scene import (
    CoordinateTransform,
    ObservationSemantics,
    OverlayScene,
    SceneFrameIdentity,
    SceneValue,
)

_EDGE_EVENT_ID = "00000000-0000-4000-8000-0000000000c1"
_DETECTED_AT = "2026-08-22T00:00:00.000Z"
_DECISION_ENVELOPE = {
    "config_version": 17,
    "detector_version": "fall-detector-2026.08",
    "model_version": "pose-model-4",
    "operating_threshold": 0.91,
    "runtime_manifest_sha256": "a" * 64,
}
_CAMERA_ID = "room-camera"
_FACILITY_ID = "facility-1"
_RUNTIME_MANIFEST = "a" * 64


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 1
    stride: int = 1
    mode: str = "features"


class _FallModel:
    metadata = _FallMetadata()
    operating_threshold = 0.7
    artifact_digest = "b" * 64

    def predict(self, features: FallModelInput) -> float:
        del features
        return 0.97


class _PoseRunner:
    def run(self, image: Image) -> RunnerResult:
        del image
        keypoints = tuple(
            coordinate for index in range(17) for coordinate in (40 + index, 30 + index, 0.9)
        )
        return pose_result((keypoints,), ((20, 10, 120, 115, 0.95),))


class _SingleFallResult:
    def __init__(self, packet: FramePacket) -> None:
        self._packet: FramePacket | None = packet

    def take(self, *, timeout_sec: float | None = None) -> CoordinatedInference | None:
        del timeout_sec
        packet, self._packet = self._packet, None
        if packet is None:
            return None
        return CoordinatedInference(
            packet,
            ModuleResult("pose", _PoseRunner().run(packet.frame.image), 0.0, "pose"),
        )


class _NoClipRecorder:
    def on_event(self, trigger_packet: FramePacket, event: object, **_: object) -> None:
        del trigger_packet, event
        return None


def _packet() -> FramePacket:
    return FramePacket(
        camera_id=_CAMERA_ID,
        frame=Frame(7, 12.5, np.zeros((120, 180, 3), dtype=np.uint8)),
        pts=12.5,
        seq=7,
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

    def send_event(self, payload_json: str, edge_event_id: str) -> EventReceipt | DeliveryFailure:
        if self.client is None:
            return DeliveryFailure(
                DeliveryDisposition.RETRY,
                "backend-down",
                transport_error="ConnectionError: backend unavailable",
            )
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
    detector = FallEventLatch(
        _FallModel(),
        camera_id=_CAMERA_ID,
        facility_id=_FACILITY_ID,
        operating_threshold=0.7,
    )
    sink = EvidenceEventSink(
        stager=stager,
        recorder=_NoClipRecorder(),
        now=lambda: datetime.fromisoformat(_DETECTED_AT.replace("Z", "+00:00")).astimezone(UTC),
        snapshot_store=SnapshotStore(snapshot_root),
    )
    pump = CameraPipelinePump(
        _CAMERA_ID,
        _SingleFallResult(_packet()),
        _analytics(),
        EventAggregator((detector,), IncidentManager(cooldown_sec=0.0), monotonic=lambda: 12.5),
        sink,
        max_frames=1,
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


def _dropped_detail_reason(tmp_path: Path) -> DetailUnavailableReason:
    """Prune an actual persisted frame; do not invent a degradation reason."""
    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    capture = TraceCapture(
        (
            TraceIdentity(
                module_qualified_id="fall.v1",
                component_qualified_ids=(f"pose.sha256.{_RUNTIME_MANIFEST}",),
                policy_qualified_id="fall.policy.v1",
                effective_policy_id=policy.effective_policy_id,
                runtime_manifest_sha256=_RUNTIME_MANIFEST,
            ),
        )
    )
    writer = BoundedTraceWriter(
        tmp_path / "detail-cache",
        TraceRetentionPolicy(
            max_frames_per_camera=1,
            max_age_seconds=300.0,
            max_pending_frames=2,
            max_batch_size=1,
            max_numeric_values_per_decision=32,
            max_total_frames=1,
            max_total_rows=1_024,
            max_total_bytes=262_144,
        ),
    )
    writer.start()
    try:
        first = _packet()
        second = FramePacket(
            camera_id=_CAMERA_ID,
            frame=Frame(
                first.frame.index + 1,
                first.frame.time_sec + 1.0,
                np.zeros((120, 180, 3), dtype=np.uint8),
            ),
            pts=first.pts + 1.0 if first.pts is not None else None,
            seq=first.seq + 1,
            width=180,
            height=120,
            decode_time_ms=0.25,
            worker_boot_id=first.worker_boot_id,
            stream_epoch=first.stream_epoch,
        )
        for packet in (first, second):
            result = _analytics().process(
                packet,
                prefetched_results=(
                    ModuleResult("pose", _PoseRunner().run(packet.frame.image), 0.0, "pose"),
                ),
            )
            assert writer.submit(capture.build(packet, result, ()))
        writer.flush()
        recovered = writer.recover_camera(_CAMERA_ID)
    finally:
        writer.stop()
    assert recovered.truncation.pruned_frames == 1
    assert recovered.truncation.detail_unavailable_reason is DetailUnavailableReason.RETENTION_BOUND
    return recovered.truncation.detail_unavailable_reason


def _derivative_job(tmp_path: Path) -> AnnotatedDerivativeJob:
    source = tmp_path / "primary.mp4"
    source.write_bytes(b"primary")
    scene = OverlayScene(
        "scene",
        SceneFrameIdentity(
            "boot",
            _CAMERA_ID,
            1,
            1,
            SceneValue(0.0, ObservationSemantics.PRESENT),
            SceneValue(0.0, ObservationSemantics.PRESENT),
            "config",
        ),
        (1, 1),
        "source-pixels",
        CoordinateTransform(1, 1, 1, 1, 1.0, 1.0, 0.0, 0.0),
        (),
        (),
        (),
        (),
        (),
    )
    return AnnotatedDerivativeJob(
        "incident",
        "clip",
        source,
        hashlib.sha256(b"primary").hexdigest(),
        _RUNTIME_MANIFEST,
        "b" * 64,
        (scene,),
        len(b"primary"),
        derivative_kind=DerivativeKind.STILL,
    )


def _relay_client(tmp_path: Path, database: Path) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = RELAY_TOKEN
    registry = CameraRegistryStore(tmp_path / "camera-registry.sqlite3")
    registry.create(
        camera_id="room-camera",
        label="room-camera",
        rtsp_url="rtsp://role-gateway:8554/room-camera",
        space_id=None,
        status="online",
        backend_camera_id="room-camera",
    )
    app.state.camera_registry = registry
    app.state.catalog_store = CatalogStore.open(tmp_path / "relay-catalog.sqlite3")
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
    migrate_database(database)
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
            SELECT slot.state, slot.reason
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event USING (edge_event_id)
            JOIN evidence_artifact_slots AS slot ON slot.incident_id = incident.incident_id
            WHERE event.edge_event_id = ? AND slot.slot_name = 'SNAPSHOT'
            """,
            (edge_event_id,),
        ).fetchone() == ("UNAVAILABLE", "UNAVAILABLE:stage_failed")


def test_replayed_fall_keeps_decision_envelope_and_typed_derivative_degradation(
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
    migrate_database(database)
    transport = _RelayTransport()
    sender_config = SenderConfig("http://relay.test", RELAY_TOKEN, "room-camera")
    with _relay_client(tmp_path, database) as relay:
        transport.client = relay
        assert (
            EvidenceSender(queue_directory, sender_config, transport=transport).run_once()
            is SenderStep.EVENT_ACKED
        )

    detail_loss = _dropped_detail_reason(tmp_path)
    derivative = DerivativeCommandExecutor(tmp_path / "derivatives").execute(
        DerivativeCommand(_derivative_job(tmp_path), detail_loss)
    )
    assert derivative.outcome is DerivativeOutcome.UNAVAILABLE
    assert derivative.reason == detail_loss.value
    with sqlite3.connect(database) as connection:
        [payload_json] = connection.execute(
            """
            SELECT event.payload_json
            FROM evidence_incidents AS incident
            JOIN evidence_events AS event USING (edge_event_id)
            WHERE incident.edge_event_id = ?
            """,
            (edge_event_id,),
        ).fetchone()
    assert json.loads(payload_json)["audit"] == _DECISION_ENVELOPE
