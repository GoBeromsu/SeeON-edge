from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from shared.detection_policies import FallPolicyV1, make_effective_policy
from shared.edge_db.migrator import migrate_database
from worker.domains.fall import FallEventLatch
from worker.pipeline.analytics import CompositeExtractor, NamedExtractor
from worker.pipeline.bus import Scheduler
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.inference_coordinator import CoordinatedInference
from worker.pipeline.output.event_sink import EvidenceEventSink
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.output.evidence_attacher import AlertEvidenceAttacher
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.pipeline.trace import (
    BoundedTraceWriter,
    TraceCapture,
    TraceIdentity,
    TraceRetentionPolicy,
)
from worker.runtime.provenance import AppliedRuntimeManifestStore
from worker.runtime.provenance.models import AppliedRuntimeManifest
from worker.types import BusinessEvent, FallModelInput, FramePacket, ModuleResult

CAMERA_ID = "camera-fall-pipeline"
FACILITY_ID = "facility-fall-pipeline"
BOOT_ID = "boot-fall-pipeline"
POSE_SHA256 = "a" * 64
FALL_SHA256 = "b" * 64


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 1
    stride: int = 1
    mode: Literal["features"] = "features"


class _FallModel:
    metadata = _FallMetadata()
    operating_threshold = 0.7
    artifact_digest = FALL_SHA256

    def predict(self, features: FallModelInput) -> float:
        del features
        return 0.82


class _PoseRunner:
    def run(self, image: Image) -> RunnerResult:
        del image
        keypoints = tuple(
            coordinate for index in range(17) for coordinate in (40 + index, 30 + index, 0.9)
        )
        return pose_result((keypoints,), ((20, 10, 120, 115, 0.95),))


class _SinglePacketSubscription:
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

    def close(self) -> None:
        self._packet = None


class _RecordingClipRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[FramePacket, BusinessEvent]] = []

    def on_event(
        self,
        trigger_packet: FramePacket,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
    ) -> str | None:
        assert allow_new_clip is True
        self.calls.append((trigger_packet, event))
        return "clip-fall-pipeline"


def _packet() -> FramePacket:
    return FramePacket(
        camera_id=CAMERA_ID,
        frame=Frame(7, 12.5, np.zeros((120, 180, 3), dtype=np.uint8)),
        pts=12.5,
        seq=7,
        width=180,
        height=120,
        decode_time_ms=0.25,
        worker_boot_id=BOOT_ID,
        stream_epoch=3,
    )


def _analytics() -> CompositeExtractor:
    runner = _PoseRunner()
    extractor = NamedExtractor(
        module_name="pose",
        runner=runner,
        _call=runner.run,
        _clock=lambda: 1.0,
        output_adapter="pose",
    )
    return CompositeExtractor(
        extractors=(extractor,),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=SceneState(CAMERA_ID),
    )


def _seed_runtime_manifest(database: Path) -> AppliedRuntimeManifest:
    manifest = AppliedRuntimeManifest.from_content(
        {
            "manifest_schema_version": 1,
            "cameras": [{"camera_id": CAMERA_ID}],
        }
    )
    AppliedRuntimeManifestStore(database).persist(
        manifest,
        boot_instance_id=BOOT_ID,
        applied_at="2026-08-13T00:00:00Z",
    )
    return manifest


def test_fall_observation_persists_decision_trace_and_binds_event_to_clip(
    tmp_path: Path,
) -> None:
    """Exercise the real observation -> fall decision -> evidence transaction path."""
    database = tmp_path / "edge.sqlite3"
    migrate_database(database)
    manifest = _seed_runtime_manifest(database)
    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    detector = FallEventLatch(
        _FallModel(),
        camera_id=CAMERA_ID,
        facility_id=FACILITY_ID,
        operating_threshold=0.7,
    )
    capture = TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="fall.v1",
                component_qualified_ids=(
                    f"pose.sha256.{POSE_SHA256}",
                    f"fall-classifier.sha256.{FALL_SHA256}",
                ),
                policy_qualified_id="fall.policy.v1",
                effective_policy_id=policy.effective_policy_id,
                runtime_manifest_sha256=manifest.sha256,
                snapshot_provider=lambda: detector.last_trace_snapshots,
            ),
        )
    )
    writer = BoundedTraceWriter(database, TraceRetentionPolicy.testing())
    recorder = _RecordingClipRecorder()
    sink = EvidenceEventSink(
        stager=DurableEvidenceStager(
            database_path=database,
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            resident_id=None,
            config_version=1,
            clock=lambda: 12.5,
            runtime_manifest_sha256=manifest.sha256,
        ),
        recorder=recorder,
        now=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
    )
    pump = CameraPipelinePump(
        CAMERA_ID,
        _SinglePacketSubscription(_packet()),
        _analytics(),
        EventAggregator(
            (detector,),
            IncidentManager(cooldown_sec=0.0),
            monotonic=lambda: 12.5,
        ),
        sink,
        max_frames=1,
        evidence_attacher=AlertEvidenceAttacher(
            domain_audit={},
            runtime_manifest_sha256=manifest.sha256,
        ),
        trace_capture=capture,
        trace_writer=writer,
    )

    writer.start()
    try:
        pump.run()
    finally:
        writer.stop()

    assert pump.failure_count == 0
    assert pump.processed_count == 1
    assert len(recorder.calls) == 1
    trigger_packet, event = recorder.calls[0]
    assert trigger_packet.frame_key.seq == 7
    assert event.domain == "fall"
    assert event.probability == 0.82
    assert event.audit is not None
    trace_id = event.audit["decision_trace_id"]
    assert event.audit["runtime_manifest_sha256"] == manifest.sha256

    recovered = writer.recover_camera(CAMERA_ID)
    assert len(recovered.frames) == 1
    analysis = recovered.frames[0]
    assert analysis.frame_key == (BOOT_ID, CAMERA_ID, 3, 7)
    assert analysis.persons[0].track_id.value == 0
    assert analysis.persons[0].box == (20, 10, 120, 115)
    assert [(item.qualified_id, item.observation_state) for item in analysis.components] == [
        (f"pose.sha256.{POSE_SHA256}", "observed"),
        (f"fall-classifier.sha256.{FALL_SHA256}", "executed"),
    ]

    assert len(recovered.decisions) == 1
    decision = recovered.decisions[0]
    assert decision.trace_id == trace_id
    assert decision.analysis_trace_id == analysis.trace_id
    assert decision.module_qualified_id == "fall.v1"
    assert decision.policy_qualified_id == "fall.policy.v1"
    assert decision.effective_policy_id == policy.effective_policy_id
    assert decision.runtime_manifest_sha256 == manifest.sha256
    assert decision.snapshot.reason == "fall-onset"
    assert decision.snapshot.triggered is True
    assert decision.snapshot.track_id == 0
    assert decision.snapshot.values == {
        "fall_probability": 0.82,
        "operating_threshold": 0.7,
        "window_frames": 1,
    }

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT event.edge_event_id, event.state, link.clip_id, ref.decision_trace_id,
                   event.payload_json
            FROM evidence_events AS event
            JOIN evidence_event_trace_refs AS ref USING (edge_event_id)
            JOIN clip_events AS link USING (edge_event_id)
            """
        ).fetchone()
    assert row is not None
    edge_event_id, state, clip_id, persisted_trace_id, payload_json = row
    assert edge_event_id == str(event.identity)
    assert state == "READY"
    assert clip_id == "clip-fall-pipeline"
    assert persisted_trace_id == trace_id
    payload = json.loads(payload_json)
    assert payload["audit"]["decision_trace_id"] == trace_id
    assert payload["audit"]["runtime_manifest_sha256"] == manifest.sha256
