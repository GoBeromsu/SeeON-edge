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
        # Immediate classifier scores every live track on every call.
        self.current_call_missing_score_reasons: dict[int, str] = {}

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
    event_sink: object | None = None,
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
            if event_sink is not None:
                event_sink.emit_for_frame(event, trigger)
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


def test_missing_track_or_generation_never_aliases_onto_unit_zero() -> None:
    from worker.pipeline.diagnostics.record_builder import (
        NO_GENERATION,
        NO_TRACK,
        fall_causal_unit_id,
    )

    real_zero = fall_causal_unit_id("cam", "boot", 3, 0, 0)
    no_track = fall_causal_unit_id("cam", "boot", 3, None, 0)
    no_generation = fall_causal_unit_id("cam", "boot", 3, 0, None)
    assert real_zero == "cam:boot:3:0:0"
    assert no_track == f"cam:boot:3:{NO_TRACK}:0"
    assert no_generation == f"cam:boot:3:0:{NO_GENERATION}"
    assert len({real_zero, no_track, no_generation}) == 3


def test_window_gated_snapshot_without_track_gets_explicit_no_track_unit() -> None:
    from worker.pipeline.diagnostics.emit_policy import policy_decision_record
    from worker.pipeline.diagnostics.record_builder import NO_GENERATION, NO_TRACK
    from worker.types.trace import DecisionTraceSnapshot

    snapshot = DecisionTraceSnapshot(
        reason="outside-detection-window",
        previous_state="not-evaluated",
        current_state="not-evaluated",
        triggered=False,
        track_id=None,
        bed_id=None,
    )
    record = policy_decision_record(
        snapshot,
        camera_id="cam",
        worker_boot_id="boot",
        source_generation=1,
        stream_epoch=3,
        frame_seq=7,
        source_pts_ns=None,
        generation=None,
    )
    assert record is not None
    assert record.causal_unit_id == f"cam:boot:3:{NO_TRACK}:{NO_GENERATION}"
    assert record.payload["track_id"] is None


def test_model_score_is_not_emitted_for_tracks_the_classifier_skipped_this_call() -> None:
    """A stride-not-due frame must not re-emit the cached score as a new model call."""
    lanes = ExecutionRecordLanes(lane_capacity=64)
    pump = _pump(lanes, identity=None, fall_transition=0.9)

    class _StrideClassifier(_ImmediateClassifier):
        def __init__(self) -> None:
            super().__init__(0.9)
            self.current_call_missing_score_reasons: dict[int, str] = {}
            self._calls = 0

        def update(self, rows: object, live_track_ids: tuple[int, ...]) -> dict[int, object]:
            self._calls += 1
            if self._calls % 2 == 0:
                # even calls: stride not due, no score this call
                self.current_call_missing_score_reasons = dict.fromkeys(
                    live_track_ids, "classifier-stride-not-due"
                )
                return {}
            self.current_call_missing_score_reasons = {}
            return super().update(rows, live_track_ids)

    stride = _StrideClassifier()
    pump._decision.deciders[0].classifier = stride  # type: ignore[attr-defined]  # noqa: SLF001
    for seq in range(4):
        pump._process(  # noqa: SLF001
            _metadata(child=pump._child, seq=seq, pts=100 + seq * 66_666_667)  # noqa: SLF001
        )
    drained = lanes.drain_for("cam-1", "boot-1", limit=64)
    assert drained is not None
    scores = [r for r in drained.records if r.record_kind == "model.score"]
    decisions = [r for r in drained.records if r.record_kind == "policy.decision"]
    # 4 frames processed, classifier scored on 2 of them
    assert len(decisions) >= 4
    assert len(scores) == 2, [r.frame_seq for r in scores]


def test_try_emit_logs_sink_failure_with_camera_and_kind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from shared.events.execution_records import WireRecord
    from worker.pipeline.diagnostics.record_builder import try_emit

    class _Boom:
        def try_emit(self, record: object) -> bool:
            raise RuntimeError("sink exploded")

    record = WireRecord(
        record_kind="sdk.frame",
        camera_id="cam-log",
        worker_boot_id="boot-1",
        source_generation=0,
        stream_epoch=1,
        producer="sdk",
        producer_sequence=0,
        observed_at_ns=1_000,
        time_quality="monotonic",
        causal_unit_id="cam-log:boot-1:1:frame:0",
        outcome="accepted",
        payload={},
    )
    with caplog.at_level("WARNING", logger="worker.pipeline.diagnostics.record_builder"):
        assert try_emit(_Boom(), record) is False  # type: ignore[arg-type]
    message = caplog.records[-1].getMessage()
    assert "cam-log" in message
    assert "sdk.frame" in message


def test_make_record_logs_contract_error_and_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from worker.pipeline.diagnostics.record_builder import make_record

    with caplog.at_level("WARNING", logger="worker.pipeline.diagnostics.record_builder"):
        record = make_record(
            record_kind="not.a.kind",
            camera_id="cam-bad",
            worker_boot_id="boot-1",
            source_generation=0,
            stream_epoch=1,
            producer="sdk",
            observed_at_ns=1_000,
            time_quality="monotonic",
            causal_unit_id="cam-bad:boot-1:1:frame:0",
            outcome="accepted",
            payload={},
        )
    assert record is None
    message = caplog.records[-1].getMessage()
    assert "cam-bad" in message
    assert "not.a.kind" in message
