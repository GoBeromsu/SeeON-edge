from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.runner import pose_result
from shared.detection_policies import FallPolicyV2, make_effective_policy
from shared.events.delivery_queue import DeliveryQueue, EventEntry
from worker.domains.bed_exit import BedExitConfig, BedExitMonitor
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallV2Probabilities
from worker.pipeline.analytics.composite import CompositeResult
from worker.pipeline.camera_pipeline import CameraPipelinePump
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.output.evidence.evidence_stager import DurableEvidenceStager
from worker.pipeline.trace import (
    BoundedTraceWriter,
    TraceCapture,
    TraceIdentity,
    TraceRetentionPolicy,
)
from worker.types import DecisionInput, FramePacket, ModuleResult

RUNTIME_MANIFEST_SHA256 = "a" * 64
COMPONENT_SHA256 = "b" * 64


class _ImmediateV2Classifier:
    def update(
        self, _rows: object, live_track_ids: tuple[int, ...]
    ) -> dict[int, FallV2Probabilities]:
        return {
            track_id: FallV2Probabilities(background=0.1, fall_transition=0.8, fallen=0.1)
            for track_id in live_track_ids
        }


def _traceable_fall_v2(*, camera_id: str, facility_id: str) -> FallV2DomainDecider:
    """The production V2 decider records compiled-vocabulary trace snapshots itself."""
    return FallV2DomainDecider(
        classifier=_ImmediateV2Classifier(),
        policy=FallPolicyDeciderV2(
            camera_id=camera_id,
            facility_id=facility_id,
            boot_id="boot-a",
            stream_epoch="1",
            source_generation=0,
            policy=FallPolicyV2(transition_votes=1),
        ),
    )


def _input(person: BoundingBox, *, frame_index: int, live: tuple[int, ...] = (9,)) -> DecisionInput:
    bed = BoundingBox(0, 0, 80, 100, 0.9)
    pose = tuple((index + 1, index + 2, 0.9) for index in range(17))
    observation = FrameObservation(
        detections=((person,), ()),
        poses=(pose,),
        regions=((bed,), ()),
        track_ids=(9,),
    )
    return DecisionInput(
        observation=observation,
        frame_width=180,
        frame_height=120,
        live_track_ids=live,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
    )


def test_fall_trace_records_v2_transition_confirmation() -> None:
    detector = _traceable_fall_v2(camera_id="camera-a", facility_id="facility-a")

    # transition_votes=1: the first qualifying frame confirms the transition.
    events = detector.update(_input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=1))

    assert len(events) == 1
    trace = detector.last_trace_snapshots[0]
    assert trace.reason == "transition-confirmed"
    assert trace.previous_state == "clear"
    assert trace.current_state == "transition-confirmed"
    assert trace.track_id == 9
    assert trace.values == {
        "fall_transition_probability": 0.8,
        "fallen_probability": 0.1,
        "transition_threshold": 0.5,
        "transition_votes": 1,
        "transition_window": 5,
    }


def test_bed_exit_trace_distinguishes_live_grace_from_stale_track_exit() -> None:
    detector = BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-bed",
            facility_id="facility-bed",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=2,
        ),
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        boot_id="test-boot",
        stream_epoch="test-epoch",
        source_generation=0,
    )
    inside = BoundingBox(10, 10, 70, 90, 0.9)
    outside = BoundingBox(100, 10, 160, 90, 0.9)
    assert detector.update(_input(inside, frame_index=0)) == ()
    assert detector.update(_input(outside, frame_index=1)) == ()
    live_trace = detector.last_trace_snapshots[0]

    events = detector.update(_input(outside, frame_index=2, live=()))

    assert live_trace.reason == "live-grace"
    assert live_trace.values["containment_ratio"] == 0.0
    assert live_trace.values["grace_frames_before"] == 0
    assert live_trace.values["grace_frames_after"] == 1
    assert len(events) == 1
    stale_trace = detector.last_trace_snapshots[0]
    assert stale_trace.reason == "stale-track-exit"
    assert stale_trace.previous_state == "live-grace"
    assert stale_trace.current_state == "triggered"
    assert stale_trace.values["grace_frames_before"] == 1
    assert stale_trace.values["grace_threshold"] == 2


def _packet() -> FramePacket:
    return FramePacket(
        camera_id="camera-a",
        frame=Frame(1, 1.0, np.zeros((4, 4, 3), dtype=np.uint8)),
        pts=1.0,
        seq=1,
        width=4,
        height=4,
        decode_time_ms=0.0,
        worker_boot_id="boot-a",
        stream_epoch=1,
    )


def _trace_capture(detector: FallV2DomainDecider) -> TraceCapture:
    policy = make_effective_policy(
        module_id="fall",
        module_version=2,
        values=FallPolicyV2(transition_votes=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )
    return TraceCapture(
        identities=(
            TraceIdentity(
                module_qualified_id="fall.v2",
                component_qualified_ids=(f"fall-classifier.sha256.{COMPONENT_SHA256}",),
                policy_qualified_id="fall.policy.v2",
                effective_policy_id=policy.effective_policy_id,
                runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
                snapshot_provider=lambda: detector.last_trace_snapshots,
            ),
        )
    )


def _stager(database: Path) -> DurableEvidenceStager:
    return DurableEvidenceStager(
        database_path=database,
        camera_id="camera-a",
        facility_id="facility-a",
        resident_id=None,
        config_version=1,
        clock=lambda: 1.0,
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
    )


def test_real_camera_pump_captures_before_emitting_the_admitted_event(
    tmp_path: Path,
) -> None:
    detector = _traceable_fall_v2(camera_id="camera-a", facility_id="facility-a")

    class _Analytics:
        def __init__(self) -> None:
            self.frame_index = 0

        def process(self, _packet: FramePacket, **_kwargs: object) -> CompositeResult:
            self.frame_index += 1
            decision_input = _input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=self.frame_index)
            return CompositeResult((), decision_input.observation, decision_input)

    class _Sink:
        def __init__(self) -> None:
            self.events: list[object] = []

        def emit_for_frame(self, event: object, _packet: FramePacket) -> None:
            self.events.append(event)

        def emit(self, event: object) -> None:
            self.events.append(event)

    class _Subscription:
        def take(self, timeout_sec: float | None = None) -> None:
            del timeout_sec

    writer = BoundedTraceWriter(tmp_path / "detail-cache", TraceRetentionPolicy.testing())
    writer.start()
    sink = _Sink()
    pump = CameraPipelinePump(
        "camera-a",
        _Subscription(),  # type: ignore[arg-type]
        _Analytics(),  # type: ignore[arg-type]
        EventAggregator((detector,), IncidentManager(cooldown_sec=0.0), monotonic=lambda: 1.0),
        sink,  # type: ignore[arg-type]
        trace_capture=_trace_capture(detector),
        trace_writer=writer,
    )
    try:
        for _ in range(3):
            pump._pump_one(  # noqa: SLF001
                _packet(), ModuleResult("pose", pose_result((), ()), 0.0, "pose")
            )
    finally:
        writer.stop()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert hasattr(event, "audit") and event.audit is not None
    trace_id = event.audit["decision_trace_id"]
    assert trace_id in {
        decision.trace_id for decision in writer.recover_camera("camera-a").decisions
    }


def test_admitted_event_decision_basis_is_atomic_in_delivery_queue(
    tmp_path: Path,
) -> None:
    detector = _traceable_fall_v2(camera_id="camera-a", facility_id="facility-a")
    # transition_votes=1: the very first qualifying frame opens the episode.
    decision_input = _input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=1)
    events = detector.update(decision_input)
    result = CompositeResult((), decision_input.observation, decision_input)
    writer = BoundedTraceWriter(tmp_path / "detail-cache", TraceRetentionPolicy.testing())
    writer.start()
    try:
        traced_events = _trace_capture(detector).capture(
            writer, _packet(), result, events, require_persisted=True
        )
    finally:
        writer.stop()
    assert len(traced_events) == 1
    audit = traced_events[0].audit
    assert audit is not None
    trace_id = audit["decision_trace_id"]
    assert isinstance(trace_id, str)

    decision = writer.recover_camera("camera-a").decisions[0]
    assert decision.trace_id == trace_id
    queue = DeliveryQueue(tmp_path / "delivery")
    entry = EventEntry(
        edge_event_id="event-valid",
        event_type="fall",
        detected_at="2026-08-13T00:00:01Z",
        camera_id="camera-a",
        facility_id="facility-a",
        decision_trace=decision.trace_id.encode(),
        values=b'{"fall_probability":0.8}',
    )
    assert queue.try_admit(entry).accepted
    queued = next(queue.entries())
    assert queued["edge_event_id"] == "event-valid"
    assert queued["decision_trace_b64"]
    assert queued["values_b64"]


def test_numeric_decision_trace_is_hardware_neutral_for_equal_inputs() -> None:
    cpu = _traceable_fall_v2(camera_id="camera-a", facility_id="f")
    nvidia = _traceable_fall_v2(camera_id="camera-a", facility_id="f")
    input_value = _input(BoundingBox(10, 10, 70, 90, 0.9), frame_index=1)

    assert cpu.update(input_value) == nvidia.update(input_value)
    assert cpu.last_trace_snapshots == nvidia.last_trace_snapshots
