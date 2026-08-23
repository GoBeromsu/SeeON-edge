from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from shared.detection_policies import BedExitPolicyV1, FallPolicyV1, make_effective_policy
from worker.domains.fall import FallModelProtocol
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
from worker.pipeline.trace.models import TraceTruncation
from worker.pipeline.trace.store import TraceStore
from worker.replay.comparison import MismatchReason, compare_runs
from worker.replay.engine import (
    ReplayConfigurationError,
    assess_reproducibility,
    replay_camera,
)
from worker.runtime.provenance.models import AppliedRuntimeManifest
from worker.types import FallModelInput, FramePacket, ModuleResult

CAMERA_ID = "camera-replay"
FACILITY_ID = "facility-replay"
BOOT_ID = "boot-replay"
POSE_SHA256 = "a" * 64
FALL_SHA256 = "b" * 64


@dataclass(frozen=True, slots=True)
class _FallMetadata:
    window: int = 1
    stride: int = 1
    mode: Literal["features"] = "features"


class _FallModel:
    """Deterministic fixture model: probability rises with keypoint mean y."""

    metadata = _FallMetadata()
    operating_threshold = 0.7
    artifact_digest = FALL_SHA256

    def predict(self, features: FallModelInput) -> float:
        del features
        return 0.82


class _LowProbabilityFallModel(_FallModel):
    def predict(self, features: FallModelInput) -> float:
        del features
        return 0.10


class _PoseRunner:
    def __init__(self, box: tuple[int, int, int, int, float]) -> None:
        self._box = box

    def run(self, image: Image) -> RunnerResult:
        del image
        keypoints = tuple(
            coordinate for index in range(17) for coordinate in (40 + index, 30 + index, 0.9)
        )
        return pose_result((keypoints,), (self._box,))


def _packet(seq: int, pts: float) -> FramePacket:
    return FramePacket(
        camera_id=CAMERA_ID,
        frame=Frame(seq, pts, np.zeros((120, 180, 3), dtype=np.uint8)),
        pts=pts,
        seq=seq,
        width=180,
        height=120,
        decode_time_ms=0.25,
        worker_boot_id=BOOT_ID,
        stream_epoch=3,
    )


def _analytics(box: tuple[int, int, int, int, float]) -> CompositeExtractor:
    runner = _PoseRunner(box)
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


def _runtime_manifest(camera_id: str) -> AppliedRuntimeManifest:
    return AppliedRuntimeManifest.from_content(
        {"manifest_schema_version": 1, "cameras": [{"camera_id": camera_id}]}
    )


def _capture_real_frames(
    database: Path, *, fall_model: FallModelProtocol, boxes: list[tuple[int, int, int, int, float]]
) -> None:
    """Drive the real camera pipeline once per box to persist analysis traces."""
    from worker.domains.fall import FallEventLatch

    manifest = _runtime_manifest(CAMERA_ID)
    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    detector = FallEventLatch(
        fall_model,
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

    class _NullRecorder:
        def on_event(
            self, trigger_packet: object, event: object, *, allow_new_clip: bool = True
        ) -> str | None:
            del trigger_packet, event, allow_new_clip
            return "clip-replay"

    sink = EvidenceEventSink(
        stager=DurableEvidenceStager(
            queue_directory=database.parent / "delivery-queue",
            camera_id=CAMERA_ID,
            facility_id=FACILITY_ID,
            resident_id=None,
            config_version=1,
            clock=lambda: 1.0,
            runtime_manifest_sha256=manifest.sha256,
        ),
        recorder=_NullRecorder(),  # type: ignore[arg-type]
        now=lambda: __import__("datetime").datetime(2026, 8, 13, tzinfo=__import__("datetime").UTC),
    )
    writer.start()
    try:
        for seq, box in enumerate(boxes, start=1):
            pump = CameraPipelinePump(
                CAMERA_ID,
                _SinglePacketSubscription(
                    _packet(seq, float(seq)),
                    _PoseRunner(box).run(np.zeros((1, 1, 3), dtype=np.uint8)),
                ),
                _analytics(box),
                EventAggregator(
                    (detector,),
                    IncidentManager(cooldown_sec=0.0),
                    monotonic=lambda seq=seq: float(seq),
                ),
                sink,
                max_frames=1,
                evidence_attacher=AlertEvidenceAttacher(
                    domain_audit={}, runtime_manifest_sha256=manifest.sha256
                ),
                trace_capture=capture,
                trace_writer=writer,
            )
            pump.run()
    finally:
        writer.stop()


class _SinglePacketSubscription:
    def __init__(self, packet: FramePacket, result: RunnerResult) -> None:
        self._packet: FramePacket | None = packet
        self._result = result

    def take(self, *, timeout_sec: float | None = None) -> CoordinatedInference | None:
        del timeout_sec
        packet, self._packet = self._packet, None
        if packet is None:
            return None
        return CoordinatedInference(
            packet, ModuleResult("pose", self._result, 0.0, "pose")
        )

    def close(self) -> None:
        self._packet = None


def _boxes() -> list[tuple[int, int, int, int, float]]:
    return [(20, 10, 120, 115, 0.95) for _ in range(3)]


def test_replay_reproduces_identical_events_against_the_same_model(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _capture_real_frames(database, fall_model=_FallModel(), boxes=_boxes())

    recovered = TraceStore(database).recover_camera(CAMERA_ID)
    assert len(recovered.frames) == 3

    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )

    run_one = replay_camera(
        camera_id=CAMERA_ID,
        analyses=recovered.frames,
        module_id="fall",
        policy=policy,
        fall_model=_FallModel(),
    )
    run_two = replay_camera(
        camera_id=CAMERA_ID,
        analyses=recovered.frames,
        module_id="fall",
        policy=policy,
        fall_model=_FallModel(),
    )

    assert run_one == run_two
    assert run_one.event_count == 1
    comparison = compare_runs(run_one, run_two)
    assert comparison.identical is True
    assert comparison.mismatches == ()


def test_replay_threshold_change_produces_structured_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _capture_real_frames(database, fall_model=_FallModel(), boxes=_boxes())
    recovered = TraceStore(database).recover_camera(CAMERA_ID)

    baseline_policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    candidate_policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.9),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )

    baseline = replay_camera(
        camera_id=CAMERA_ID,
        analyses=recovered.frames,
        module_id="fall",
        policy=baseline_policy,
        fall_model=_FallModel(),
    )
    candidate = replay_camera(
        camera_id=CAMERA_ID,
        analyses=recovered.frames,
        module_id="fall",
        policy=candidate_policy,
        fall_model=_FallModel(),
    )

    assert baseline.event_count == 1
    assert candidate.event_count == 0

    comparison = compare_runs(baseline, candidate)
    assert comparison.identical is False
    reasons = {mismatch.reason for mismatch in comparison.mismatches}
    assert MismatchReason.EVENT_COUNT_DIFFERS in reasons
    assert MismatchReason.STATE_DIFFERS in reasons
    assert comparison.baseline_event_count == 1
    assert comparison.candidate_event_count == 0
    assert comparison.baseline_effective_policy_id != comparison.candidate_effective_policy_id


def test_replay_rejects_policy_schema_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _capture_real_frames(database, fall_model=_FallModel(), boxes=_boxes())
    recovered = TraceStore(database).recover_camera(CAMERA_ID)

    bed_exit_policy = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.5, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )

    import pytest

    with pytest.raises(ReplayConfigurationError, match="schema"):
        replay_camera(
            camera_id=CAMERA_ID,
            analyses=recovered.frames,
            module_id="fall",
            policy=bed_exit_policy,
            fall_model=_FallModel(),
        )


def test_replay_requires_fall_model_for_fall_module(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _capture_real_frames(database, fall_model=_FallModel(), boxes=_boxes())
    recovered = TraceStore(database).recover_camera(CAMERA_ID)
    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )

    import pytest

    with pytest.raises(ReplayConfigurationError, match="fall_model"):
        replay_camera(
            camera_id=CAMERA_ID,
            analyses=recovered.frames,
            module_id="fall",
            policy=policy,
            fall_model=None,
        )


BED_CAMERA_ID = "camera-replay-bed"
BED_BOOT_ID = "boot-replay-bed"
BED_SHA256 = "c" * 64


def _bed_analytics(
    person_box: tuple[int, int, int, int, float],
    bed_box: tuple[int, int, int, int, float],
) -> CompositeExtractor:
    runner = _PoseRunner(person_box)
    extractor = NamedExtractor(
        module_name="pose",
        runner=runner,
        _call=runner.run,
        _clock=lambda: 1.0,
        output_adapter="pose",
    )
    scene_state = SceneState(BED_CAMERA_ID)
    from contracts.observation import BoundingBox

    scene_state.persisted_bed_regions = (BoundingBox(*bed_box),)
    return CompositeExtractor(
        extractors=(extractor,),
        scheduler=Scheduler({"pose": 1}),
        tracker=GreedyIouTracker(),
        scene_state=scene_state,
    )


def _bed_packet(seq: int, pts: float) -> FramePacket:
    return FramePacket(
        camera_id=BED_CAMERA_ID,
        frame=Frame(seq, pts, np.zeros((120, 180, 3), dtype=np.uint8)),
        pts=pts,
        seq=seq,
        width=180,
        height=120,
        decode_time_ms=0.25,
        worker_boot_id=BED_BOOT_ID,
        stream_epoch=1,
    )


def _capture_bed_exit_frames(
    database: Path,
    *,
    person_boxes: list[tuple[int, int, int, int, float]],
    bed_box: tuple[int, int, int, int, float],
    policy_values: BedExitPolicyV1,
) -> None:
    """Drive the real camera pipeline to persist bed_exit analysis traces."""
    import datetime as _dt

    from worker.domains.bed_exit import BedExitConfig, BedExitMonitor

    manifest = _runtime_manifest(BED_CAMERA_ID)
    policy = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=policy_values,
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    detector = BedExitMonitor(
        config=BedExitConfig(
            camera_id=BED_CAMERA_ID,
            facility_id=FACILITY_ID,
            min_containment=policy_values.min_containment,
            hold_frames=policy_values.hold_frames,
            grace_frames=policy_values.grace_frames,
        ),
        clock=lambda: _dt.datetime(2026, 8, 13, tzinfo=_dt.UTC),
    )
    capture = TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="bed_exit.v1",
                component_qualified_ids=(
                    f"pose.sha256.{POSE_SHA256}",
                    f"bed.sha256.{BED_SHA256}",
                ),
                policy_qualified_id="bed_exit.policy.v1",
                effective_policy_id=policy.effective_policy_id,
                runtime_manifest_sha256=manifest.sha256,
                snapshot_provider=lambda: detector.last_trace_snapshots,
            ),
        )
    )
    writer = BoundedTraceWriter(database, TraceRetentionPolicy.testing())

    class _NullRecorder:
        def on_event(
            self, trigger_packet: object, event: object, *, allow_new_clip: bool = True
        ) -> str | None:
            del trigger_packet, event, allow_new_clip
            return "clip-replay-bed"

    sink = EvidenceEventSink(
        stager=DurableEvidenceStager(
            queue_directory=database.parent / "delivery-queue",
            camera_id=BED_CAMERA_ID,
            facility_id=FACILITY_ID,
            resident_id=None,
            config_version=1,
            clock=lambda: 1.0,
            runtime_manifest_sha256=manifest.sha256,
        ),
        recorder=_NullRecorder(),  # type: ignore[arg-type]
        now=lambda: _dt.datetime(2026, 8, 13, tzinfo=_dt.UTC),
    )
    writer.start()
    try:
        for seq, person_box in enumerate(person_boxes, start=1):
            pump = CameraPipelinePump(
                BED_CAMERA_ID,
                _SinglePacketSubscription(
                    _bed_packet(seq, float(seq)),
                    _PoseRunner(person_box).run(np.zeros((1, 1, 3), dtype=np.uint8)),
                ),
                _bed_analytics(person_box, bed_box),
                EventAggregator(
                    (detector,),
                    IncidentManager(cooldown_sec=0.0),
                    monotonic=lambda seq=seq: float(seq),
                ),
                sink,
                max_frames=1,
                evidence_attacher=AlertEvidenceAttacher(
                    domain_audit={}, runtime_manifest_sha256=manifest.sha256
                ),
                trace_capture=capture,
                trace_writer=writer,
            )
            pump.run()
    finally:
        writer.stop()


def test_replay_bed_exit_containment_change_produces_structured_mismatch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "edge.sqlite3"
    bed_box = (0, 0, 100, 100, 0.9)
    # Person overlaps the bed by half its area for the first two frames, then
    # moves fully outside the bed for the remaining frames (a genuine exit).
    person_boxes = [
        (50, 0, 150, 100, 0.9),
        (50, 0, 150, 100, 0.9),
        (200, 0, 300, 100, 0.9),
        (200, 0, 300, 100, 0.9),
    ]
    _capture_bed_exit_frames(
        database,
        person_boxes=person_boxes,
        bed_box=bed_box,
        policy_values=BedExitPolicyV1(min_containment=0.9, hold_frames=1, grace_frames=0),
    )
    recovered = TraceStore(database).recover_camera(BED_CAMERA_ID)
    assert len(recovered.frames) == 4

    # Baseline: min_containment=0.9 -> person (0.5 containment) never assigned to
    # the bed, so no exit can ever fire. Candidate: min_containment=0.4 -> person
    # is assigned while overlapping, then fires a real exit once they step away.
    baseline_policy = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.9, hold_frames=1, grace_frames=0),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    candidate_policy = make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=0.4, hold_frames=1, grace_frames=0),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )

    baseline = replay_camera(
        camera_id=BED_CAMERA_ID,
        analyses=recovered.frames,
        module_id="bed_exit",
        policy=baseline_policy,
    )
    candidate = replay_camera(
        camera_id=BED_CAMERA_ID,
        analyses=recovered.frames,
        module_id="bed_exit",
        policy=candidate_policy,
    )

    comparison = compare_runs(baseline, candidate)
    assert comparison.identical is False
    assert comparison.baseline_event_count != comparison.candidate_event_count

    # Same policy replayed twice is byte-for-byte identical (determinism).
    repeat = replay_camera(
        camera_id=BED_CAMERA_ID,
        analyses=recovered.frames,
        module_id="bed_exit",
        policy=baseline_policy,
    )
    assert baseline == repeat
    assert compare_runs(baseline, repeat).identical is True


def test_truncation_marks_run_non_reproducible(tmp_path: Path) -> None:
    database = tmp_path / "edge.sqlite3"
    _capture_real_frames(database, fall_model=_FallModel(), boxes=_boxes())
    recovered = TraceStore(database).recover_camera(CAMERA_ID)
    truncated = TraceTruncation(
        handoff_dropped_frames=2,
        pruned_frames=1,
        oldest_retained_seq=recovered.frames[0].frame_key[3],
        newest_retained_seq=recovered.frames[-1].frame_key[3],
        persistence_failed_frames=0,
        retention_blocked_frames=0,
        oldest_retained_key=recovered.frames[0].frame_key,
        newest_retained_key=recovered.frames[-1].frame_key,
    )
    policy = make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=0.7),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    run = replay_camera(
        camera_id=CAMERA_ID,
        analyses=recovered.frames,
        module_id="fall",
        policy=policy,
        fall_model=_FallModel(),
        truncation=truncated,
    )
    assert run.reproducible is False
    assert run.non_reproducible_reason is not None
    assert "handoff_dropped_frames=2" in run.non_reproducible_reason
    assert "pruned_frames=1" in run.non_reproducible_reason

    clean = assess_reproducibility(recovered.frames, recovered.truncation)
    # Fresh capture with no cursor activity remains reproducible.
    assert clean[0] is True



