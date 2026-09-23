"""Runtime composition for the optional execution-record export path."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from worker.domains.fall.policy import FallDomainDecider, FallPolicyDecider
from worker.interfaces.fall_model import FallProbabilities
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.perception import SceneState
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.execution_records import execution_records_settings_from_environment
from worker.runtime.execution_records import compose_execution_records
from worker.runtime.flow.execution_record_emit import emit_policy_consume
from worker.runtime.flow.metadata_slot import LatestMetadataSlot
from worker.runtime.flow.policy_pump import (
    DecisionIdentity,
    NativePolicyContext,
    NativePolicyPump,
)
from worker.types.metadata import (
    MetadataCounters,
    MetadataFrame,
    NativeObservationEvidence,
    SourceBinding,
)
from worker.types.perception_frame import (
    AssociationResult,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)


def test_settings_default_off_and_refuse_when_enabled_without_capacities() -> None:
    assert execution_records_settings_from_environment({}) is None
    with pytest.raises(WorkerConfigError, match="LANE_CAPACITY"):
        execution_records_settings_from_environment({"ML_WORKER_EXECUTION_RECORDS_ENABLED": "1"})
    settings = execution_records_settings_from_environment(
        {
            "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
            "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY": "16",
            "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX": "8",
            "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS": "50",
        }
    )
    assert settings is not None
    assert settings.lane_capacity == 16
    assert settings.batch_max == 8
    assert settings.flush_ms == 50


def test_compose_refuses_when_enabled_without_relay_token() -> None:
    config = SimpleNamespace(
        relay=SimpleNamespace(
            url="http://relay.test",
            token=SimpleNamespace(get_secret_value=lambda: ""),
        ),
        model_dump=lambda mode="json": {"version": 1},
        detection_policies=SimpleNamespace(
            defaults={"fall": SimpleNamespace(effective_policy_id="fall")}
        ),
    )
    env = {
        "ML_WORKER_EXECUTION_RECORDS_ENABLED": "1",
        "ML_WORKER_EXECUTION_RECORDS_LANE_CAPACITY": "4",
        "ML_WORKER_EXECUTION_RECORDS_BATCH_MAX": "2",
        "ML_WORKER_EXECUTION_RECORDS_FLUSH_MS": "20",
    }
    with pytest.raises(Exception, match="relay URL/token"):
        compose_execution_records(
            config,  # type: ignore[arg-type]
            env=env,
            worker_boot_id="boot",
            build_revision="abc123",
            image_digest="sha256:deadbeef",
            model_digest="model-1",
            calibration_digest="cal-1",
            preprocessing_identity="pose-bbox56/v1",
            policy_identity="fall.policy:2",
        )


def test_compose_off_returns_none_not_a_stub() -> None:
    config = SimpleNamespace(
        relay=SimpleNamespace(
            url="http://relay.test",
            token=SimpleNamespace(get_secret_value=lambda: "t"),
        )
    )
    settings, lanes, exporter = compose_execution_records(
        config,  # type: ignore[arg-type]
        env={},
        worker_boot_id="boot",
        build_revision="abc123",
        image_digest="digest",
        model_digest="model",
        calibration_digest="cal",
        preprocessing_identity="prep",
        policy_identity="policy",
    )
    assert settings is None
    assert lanes is None
    assert exporter is None


def _metadata(*, seq: int = 1, pts: int = 100, child: UUID | None = None) -> MetadataFrame:
    identity = PerceptionFrameIdentity("boot-1", "cam-1", 3, seq, pts)
    box = PersonBox(10, 10, 70, 90, 0.9)
    return MetadataFrame(
        frame=PerceptionFrameV1(
            identity=identity,
            person_box=PersonBoxChannel(ChannelState.INFERRED, (box,)),
            human_pose=HumanPoseChannel(
                ChannelState.INFERRED,
                (tuple(Keypoint(index + 1, index + 2, 0.9) for index in range(17)),),
            ),
            bed_region=BedRegionChannel(ChannelState.SKIPPED),
            association=AssociationResult(
                strategy="nvdcf",
                track_ids=(9,),
                selected_cue_indexes=(0,),
                identity=identity,
                live_track_ids=(9,),
            ),
        ),
        source_generation=1,
        child_instance_id=uuid4() if child is None else child,
        native_publish_sequence=seq,
        transform_id="transform-a",
        source_width=180,
        source_height=120,
        source_time_ns=pts,
        native_observation_evidence=NativeObservationEvidence(91, 17, True, 1, 1, 1),
    )


class _ImmediateClassifier:
    def __init__(self, fall_transition: float = 0.1) -> None:
        self._last: dict[int, FallProbabilities] = {}
        self._fall_transition = fall_transition

    def update(
        self, _rows: object, live_track_ids: tuple[int, ...]
    ) -> dict[int, FallProbabilities]:
        p = self._fall_transition
        scored = {track_id: FallProbabilities(1.0 - p, p, 0.0) for track_id in live_track_ids}
        self._last = scored
        return scored

    def probabilities_for(self, track_id: int) -> FallProbabilities | None:
        return self._last.get(track_id)

    def generation_for(self, track_id: int) -> int:
        del track_id
        return 0


def _pump(
    sink: ExecutionRecordLanes | None,
    *,
    identity: DecisionIdentity | None = None,
    emitted: list[object] | None = None,
    fall_transition: float = 0.1,
) -> NativePolicyPump:
    binding = SourceBinding("boot-1", str(uuid4()), "cam-1", 1, 3, "transform-a")
    slot = LatestMetadataSlot()
    slot.register_source(binding)
    slot.set_execution_record_sink(sink)
    classifier = _ImmediateClassifier(fall_transition)
    decider = FallDomainDecider(
        classifier=classifier,
        policy=FallPolicyDecider(
            camera_id="cam-1",
            facility_id="facility-a",
            boot_id="boot-1",
            stream_epoch="3",
            source_generation=1,
        ),
    )
    decision = EventAggregator(deciders=(decider,), incidents=IncidentManager())

    class _Control:
        def snapshot(self, camera_id: str) -> bytes:
            del camera_id
            raise RuntimeError("snapshot unused")

    class _Diagnostics:
        def update_measured_fps(self, *_args: object) -> None:
            return None

        def record_detection_completed(self, *_args: object) -> None:
            return None

        def record_native_detection_attempt(self, *_args: object) -> None:
            return None

        def record_track_id_switch(self, *_args: object) -> None:
            return None

        def record_track_id_switch_absorbed_total(self, *_args: object) -> None:
            return None

        def record_bed_polygon_source(self, *_args: object) -> None:
            return None

        def record_replay_trace_write_failure(self, *_args: object) -> None:
            return None

        def record_resample_gap_rows(self, *_args: object) -> None:
            return None

        def record_fall_inference_device(self, *_args: object) -> None:
            return None

    class _Sink:
        def emit_for_frame(self, event: object, trigger: object) -> None:
            del trigger
            if emitted is not None:
                emitted.append(event)

    class _Attacher:
        def attach_native(self, event: object, snapshot: object) -> object:
            del snapshot
            return event

    pump = NativePolicyPump(
        binding,
        NativePolicyContext(
            slot,
            _Control(),
            SceneState("cam-1"),
            decision,
            _Sink(),
            _Attacher(),
            _Diagnostics(),
            1,
            track_id_switch_absorbed_total=lambda _decision: 0,
            execution_records=sink,
            decision_identity=identity,
        ),
    )
    pump._slot = slot  # noqa: SLF001
    pump._child = UUID(binding.child_instance_id)  # noqa: SLF001
    return pump


def _kinds(lanes: ExecutionRecordLanes) -> set[str]:
    drained = lanes.drain_for("cam-1", "boot-1", limit=64)
    if drained is None:
        return set()
    return {record.record_kind for record in drained.records}


def test_policy_path_emits_sdk_model_and_policy_records_to_a_fake_sink() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=32)
    pump = _pump(lanes)
    slot = pump._slot  # noqa: SLF001
    metadata = _metadata(child=pump._child)  # noqa: SLF001
    assert slot.publish(metadata) is True
    pump._process(metadata)  # noqa: SLF001
    emit_policy_consume(
        lanes,
        metadata,
        before=MetadataCounters(),
        after=slot.counters(),
        processed_count=1,
    )
    kinds = _kinds(lanes)
    assert "sdk.frame" in kinds
    assert "model.score" in kinds
    assert "policy.decision" in kinds
    assert "policy.consume" in kinds


def test_policy_invariance_with_sink_on_and_off() -> None:
    off_pump = _pump(None)
    on_lanes = ExecutionRecordLanes(lane_capacity=32)
    on_pump = _pump(on_lanes)
    off_metadata = _metadata(child=off_pump._child)  # noqa: SLF001
    on_metadata = _metadata(child=on_pump._child)  # noqa: SLF001
    off_pump._process(off_metadata)  # noqa: SLF001
    on_pump._process(on_metadata)  # noqa: SLF001
    assert off_pump._decision.last_trace_snapshots == on_pump._decision.last_trace_snapshots  # noqa: SLF001
    assert on_lanes.queued() > 0


def test_alert_audit_and_policy_decision_record_share_one_decision_trace_id() -> None:
    from worker.domains.registry import FALL_MODULE_QUALIFIED_ID
    from worker.types.trace import decision_trace_id

    identity = DecisionIdentity(
        module_qualified_id=FALL_MODULE_QUALIFIED_ID,
        effective_policy_id="a" * 64,
    )
    lanes = ExecutionRecordLanes(lane_capacity=32)
    emitted: list[object] = []
    pump = _pump(lanes, identity=identity, emitted=emitted, fall_transition=0.9)
    # transition_votes=3 within transition_window=5: three scored frames trigger.
    for seq in range(3):
        # ~15 fps PTS spacing so the resampler sees three distinct rows.
        pump._process(  # noqa: SLF001
            _metadata(child=pump._child, seq=seq, pts=100 + seq * 66_666_667)  # noqa: SLF001
        )

    drained = lanes.drain_for("cam-1", "boot-1", limit=64)
    assert drained is not None
    decisions = [r for r in drained.records if r.record_kind == "policy.decision"]
    assert decisions
    triggered = [r for r in decisions if r.payload["triggered"] is True]
    assert triggered, "the immediate classifier must trigger a fall in this fixture"
    (record,) = triggered
    assert emitted, "a triggered decision must emit an alert"
    (event,) = emitted
    audit = event.audit  # type: ignore[attr-defined]
    assert audit is not None
    assert audit["decision_trace_id"] == record.payload["decision_trace_id"]
    # and both equal the single-source function over the triggering snapshot
    snapshot = next(
        s
        for s in pump._decision.last_trace_snapshots  # noqa: SLF001
        if s.triggered and s.track_id == event.person_id  # type: ignore[attr-defined]
    )
    assert audit["decision_trace_id"] == decision_trace_id(
        snapshot,
        module_qualified_id=FALL_MODULE_QUALIFIED_ID,
        effective_policy_id="a" * 64,
    )


def test_policy_decision_record_carries_no_trace_id_without_identity() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=32)
    pump = _pump(lanes, identity=None)
    pump._process(_metadata(child=pump._child))  # noqa: SLF001
    drained = lanes.drain_for("cam-1", "boot-1", limit=64)
    assert drained is not None
    for record in drained.records:
        if record.record_kind == "policy.decision":
            assert record.payload["decision_trace_id"] is None
