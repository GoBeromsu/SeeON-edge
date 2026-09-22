"""Runtime composition for the optional execution-record export path."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from worker.domains.fall.policy_v2 import FallPolicyDeciderV2, FallV2DomainDecider
from worker.interfaces.fall_model import FallV2Probabilities
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.perception import SceneState
from worker.runtime.config.errors import WorkerConfigError
from worker.runtime.config.execution_records import execution_records_settings_from_environment
from worker.runtime.execution_records import compose_execution_records
from worker.runtime.flow.execution_record_emit import emit_policy_consume
from worker.runtime.flow.metadata_slot import LatestMetadataSlot
from worker.runtime.flow.policy_pump import NativePolicyContext, NativePolicyPump
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
    def __init__(self) -> None:
        self._last: dict[int, FallV2Probabilities] = {}

    def update(
        self, _rows: object, live_track_ids: tuple[int, ...]
    ) -> dict[int, FallV2Probabilities]:
        scored = {track_id: FallV2Probabilities(0.8, 0.1, 0.1) for track_id in live_track_ids}
        self._last = scored
        return scored

    def probabilities_for(self, track_id: int) -> FallV2Probabilities | None:
        return self._last.get(track_id)

    def generation_for(self, track_id: int) -> int:
        del track_id
        return 0


def _pump(sink: ExecutionRecordLanes | None) -> NativePolicyPump:
    binding = SourceBinding("boot-1", str(uuid4()), "cam-1", 1, 3, "transform-a")
    slot = LatestMetadataSlot()
    slot.register_source(binding)
    slot.set_execution_record_sink(sink)
    classifier = _ImmediateClassifier()
    decider = FallV2DomainDecider(
        classifier=classifier,
        policy=FallPolicyDeciderV2(
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
            del event, trigger

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
