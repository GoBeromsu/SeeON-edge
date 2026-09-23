"""Generation-8 adversarial cases for the observability-16 attribution delta."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from observability_stack_fixtures import serve_backend, wait_until
from test_execution_record_wiring import _ImmediateClassifier, _pump
from test_flow_policy_pump_preview import _fall_input
from test_observability_end_to_end import (
    _BUDGET_BYTES,
    _QUERY_FROM_NS,
    _QUERY_TO_NS,
    _RELAY_TOKEN,
    _exporter,
)
from test_worker_domains_bed_exit import (
    BED_A,
    IN_BED_A,
)
from test_worker_domains_bed_exit import (
    _input as _bed_exit_input,
)
from test_worker_domains_bed_exit import (
    _monitor as _bed_exit_monitor,
)

from contracts.observation import BoundingBox
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall.policy import FallDomainDecider, FallPolicyDecider
from worker.domains.registry import FALL_MODULE_QUALIFIED_ID
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.diagnostics.record_builder import NO_MODULE, fall_causal_unit_id
from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.runtime.worker import _WindowGatedDecider
from worker.types import BusinessEvent, DecisionInput, DecisionTraceSnapshot
from worker.types.metadata import MetadataFrame, NativeObservationEvidence
from worker.types.perception_frame import (
    AssociationResult,
    BedRegion,
    BedRegionChannel,
    ChannelState,
    HumanPoseChannel,
    Keypoint,
    PerceptionFrameIdentity,
    PerceptionFrameV1,
    PersonBox,
    PersonBoxChannel,
)
from worker.types.trace import DecisionIdentity

_CAMERA = "cam-1"
_BED_MODULE = "bed_exit.v1"
_BED_ONLY_TRACK = 21
_FALL_TRACK = 9
_FALL_POLICY = "a" * 64
_BED_POLICY = "b" * 64
_PTS_STEP_NS = 66_666_667


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _bed_metadata(
    *,
    seq: int = 1,
    pts: int = 100,
    child: UUID | None = None,
    person: PersonBox | None = None,
    track_id: int = _FALL_TRACK,
) -> MetadataFrame:
    identity = PerceptionFrameIdentity("boot-1", "cam-1", 3, seq, pts)
    box = person if person is not None else PersonBox(10, 10, 70, 90, 0.9)
    region = BedRegion(0, 0, 80, 100, 0.99)
    pose = tuple(Keypoint(index + 1, index + 2, 0.9) for index in range(17))
    return MetadataFrame(
        frame=PerceptionFrameV1(
            identity=identity,
            person_box=PersonBoxChannel(ChannelState.INFERRED, (box,)),
            human_pose=HumanPoseChannel(ChannelState.INFERRED, (pose,)),
            bed_region=BedRegionChannel(ChannelState.INFERRED, (region,)),
            association=AssociationResult(
                strategy="nvdcf",
                track_ids=(track_id,),
                selected_cue_indexes=(0,),
                identity=identity,
                live_track_ids=(track_id,),
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


def _fall_identity() -> DecisionIdentity:
    return DecisionIdentity(FALL_MODULE_QUALIFIED_ID, _FALL_POLICY)


def _bed_identity() -> DecisionIdentity:
    return DecisionIdentity(_BED_MODULE, _BED_POLICY)


def _night_monitor(*, camera_id: str = "cam-1") -> object:
    return _bed_exit_monitor(camera_id=camera_id, hold_frames=1, grace_frames=2)


def _compose_bed_and_fall(
    pump: NativePolicyPump,
    monitor: object,
    *,
    identities: tuple[DecisionIdentity | None, ...],
) -> EventAggregator:
    original = pump._decision  # noqa: SLF001
    aggregator = EventAggregator(
        deciders=(*original.deciders, monitor),  # type: ignore[arg-type]
        incidents=original.incidents,
        identities=identities,
    )
    pump._decision = aggregator  # noqa: SLF001
    pump._scene.persisted_bed_regions = (BoundingBox(0, 0, 80, 100, 0.99),)  # noqa: SLF001
    return aggregator


def _comparable_event(event: BusinessEvent) -> tuple[object, ...]:
    return (
        event.domain,
        event.event_type,
        event.camera_id,
        event.facility_id,
        event.time_sec,
        event.probability,
        event.person_id,
        event.bed_id,
        None if event.audit is None else tuple(sorted(dict(event.audit).items())),
        event.snapshot_unavailable_reason,
    )


def _event_tuple_bytes(events: tuple[BusinessEvent, ...]) -> bytes:
    return repr(tuple(_comparable_event(event) for event in events)).encode()


class _ClampedShadowDecider:
    """Trace provider whose declared shadow count is not a valid slice length."""

    def __init__(self, *, count: int, snapshots: tuple[DecisionTraceSnapshot, ...]) -> None:
        self.last_shadow_trace_count = count
        self.last_trace_snapshots = snapshots

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        return ()

    def coast(self) -> tuple[BusinessEvent, ...]:
        return ()

    def release_onset(self, event: object) -> None:
        del event


def _two_snapshots() -> tuple[DecisionTraceSnapshot, ...]:
    return (
        DecisionTraceSnapshot(
            reason="contained",
            previous_state="in-bed",
            current_state="in-bed",
            triggered=False,
            track_id=1,
            bed_id=0,
        ),
        DecisionTraceSnapshot(
            reason="bed-observation-missing",
            previous_state="unknown",
            current_state="unknown",
            triggered=False,
            track_id=1,
            bed_id=None,
        ),
    )


def _domain_pair() -> tuple[FallDomainDecider, object]:
    fall = FallDomainDecider(
        classifier=_ImmediateClassifier(0.9),
        policy=FallPolicyDecider(
            camera_id="cam-1",
            facility_id="facility-a",
            boot_id="boot-1",
            stream_epoch="3",
            source_generation=1,
        ),
    )
    return fall, _night_monitor()


def test_g8_1_real_bed_exit_monitor_attributed_through_backend_query(tmp_path) -> None:
    """Real BedExitMonitor + fall decider: query names bed_exit.v1, never fall units."""
    lanes = ExecutionRecordLanes(lane_capacity=256)
    fall_identity = _fall_identity()
    bed_identity = _bed_identity()
    pump = _pump(lanes, identity=fall_identity, fall_transition=0.1)
    monitor = _night_monitor()
    _compose_bed_and_fall(pump, monitor, identities=(fall_identity, bed_identity))
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            child = pump._child  # noqa: SLF001
            assert isinstance(child, UUID)
            in_bed = PersonBox(10, 10, 70, 90, 0.95)
            only_bed = PersonBox(40, 120, 100, 190, 0.94)
            for seq in range(3):
                pump._process(  # noqa: SLF001
                    _bed_metadata(
                        child=child,
                        seq=seq,
                        pts=100 + seq * _PTS_STEP_NS,
                        person=in_bed,
                        track_id=_FALL_TRACK,
                    )
                )
            exclusive_seq = 3
            pump._process(  # noqa: SLF001
                _bed_metadata(
                    child=child,
                    seq=exclusive_seq,
                    pts=100 + exclusive_seq * _PTS_STEP_NS,
                    person=only_bed,
                    track_id=_BED_ONLY_TRACK,
                )
            )

            def _bed_rows_visible() -> bool:
                body = _query(backend)
                return any(
                    row["record_kind"] == "policy.decision"
                    and isinstance(row.get("payload"), dict)
                    and row["payload"].get("module_qualified_id") == _BED_MODULE
                    for row in body.get("records", ())
                )

            wait_until(
                _bed_rows_visible,
                timeout=5.0,
                what="bed_exit.v1 policy.decision rows on Backend",
            )
            body = _query(backend)
            decisions = [row for row in body["records"] if row["record_kind"] == "policy.decision"]
            bed_rows = [
                row for row in decisions if row["payload"].get("module_qualified_id") == _BED_MODULE
            ]
            fall_rows = [
                row
                for row in decisions
                if row["payload"].get("module_qualified_id") == FALL_MODULE_QUALIFIED_ID
            ]
            assert bed_rows, "Backend must surface bed_exit.v1 policy.decision rows"
            shadows = [row for row in bed_rows if row["payload"].get("authority_role") == "shadow"]
            for row in shadows:
                assert row["payload"]["triggered"] is False
                assert row["outcome"] != "triggered"
            bed_units = {row["causal_unit_id"] for row in bed_rows}
            fall_units = {row["causal_unit_id"] for row in fall_rows}
            assert bed_units.isdisjoint(fall_units)
            for row in bed_rows:
                assert f":{_BED_MODULE}:" in row["causal_unit_id"]
            fall_tracks: dict[object, set[object]] = defaultdict(set)
            bed_tracks: dict[object, set[object]] = defaultdict(set)
            score_tracks: dict[object, set[object]] = defaultdict(set)
            for row in body["records"]:
                seq = row.get("frame_seq")
                payload = row.get("payload")
                if not isinstance(payload, dict):
                    continue
                track_id = payload.get("track_id")
                if row["record_kind"] == "policy.decision" and track_id is not None:
                    module = payload.get("module_qualified_id")
                    if module == _BED_MODULE:
                        bed_tracks[seq].add(track_id)
                    elif module == FALL_MODULE_QUALIFIED_ID:
                        fall_tracks[seq].add(track_id)
                if row["record_kind"] == "model.score" and track_id is not None:
                    score_tracks[seq].add(track_id)
            for seq, tracks in bed_tracks.items():
                only_bed = tracks - fall_tracks[seq]
                assert score_tracks[seq].isdisjoint(only_bed), (
                    f"model.score minted for bed-exit-only tracks on frame_seq={seq}: "
                    f"{only_bed & score_tracks[seq]}"
                )
            exclusive_bed_only = bed_tracks.get(exclusive_seq, set()) - fall_tracks.get(
                exclusive_seq, set()
            )
            assert score_tracks.get(exclusive_seq, set()).isdisjoint(exclusive_bed_only)
        finally:
            if exporter is not None:
                exporter.stop()


def test_g8_2_identities_length_mismatch_raises_valueerror() -> None:
    fall, monitor = _domain_pair()
    with pytest.raises(ValueError, match="identities must be empty or parallel to deciders"):
        EventAggregator(
            deciders=(fall, monitor),  # type: ignore[arg-type]
            incidents=IncidentManager(),
            identities=(_fall_identity(),),
        )


def test_g8_3_empty_identities_leave_rows_unattributed_except_fall_units() -> None:
    lanes = ExecutionRecordLanes(lane_capacity=64)
    pump = _pump(lanes, identity=None, fall_transition=0.1)
    monitor = _night_monitor()
    aggregator = _compose_bed_and_fall(pump, monitor, identities=())
    child = pump._child  # noqa: SLF001
    assert isinstance(child, UUID)
    pump._process(  # noqa: SLF001
        _bed_metadata(child=child, seq=0, person=PersonBox(10, 10, 70, 90, 0.95))
    )
    drained = lanes.drain_for("cam-1", "boot-1", limit=64)
    assert drained is not None
    decisions = [record for record in drained.records if record.record_kind == "policy.decision"]
    assert decisions
    for record in decisions:
        assert record.payload["module_qualified_id"] is None
        assert record.payload["decision_trace_id"] is None
    attributed = aggregator.attributed_trace_snapshots()
    fall_index = aggregator.index_of(aggregator.deciders[0])
    assert fall_index is not None
    bed_index = aggregator.index_of(monitor)  # type: ignore[arg-type]
    assert bed_index is not None
    fall_unit = fall_causal_unit_id("cam-1", "boot-1", 3, _FALL_TRACK, 0)
    by_reason = {item.snapshot.reason: item for item in attributed}
    bed_rows = 0
    fall_rows = 0
    for record in decisions:
        item = by_reason[record.payload["reason"]]
        if item.producer_index == bed_index:
            bed_rows += 1
            assert f":{NO_MODULE}:" in record.causal_unit_id
            assert record.causal_unit_id != fall_unit
        elif item.producer_index == fall_index:
            fall_rows += 1
            # emit_policy selects the fall track unit only when
            # module_qualified_id == fall.v2. identities=() leaves every
            # snapshot unattributed, so the fall decider's rows also sit on
            # the NO_MODULE frame unit rather than cam-1:boot-1:3:9:0.
            assert f":{NO_MODULE}:" in record.causal_unit_id
            assert record.causal_unit_id != fall_unit
        else:
            raise AssertionError(f"unexpected producer_index {item.producer_index}")
    assert bed_rows > 0 and fall_rows > 0


def test_g8_4_window_gated_wrapper_is_attributed_to_inner_identity() -> None:
    inner = FallDomainDecider(
        classifier=_ImmediateClassifier(0.1),
        policy=FallPolicyDecider(
            camera_id="cam-1",
            facility_id="facility-a",
            boot_id="boot-1",
            stream_epoch="3",
            source_generation=1,
        ),
    )
    wrapped = _WindowGatedDecider(
        decider=inner,
        window=DetectionWindow(start="00:00", end="23:59", tz="UTC"),
        clock=lambda: datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )
    identity = _fall_identity()
    aggregator = EventAggregator(
        deciders=(wrapped,),
        incidents=IncidentManager(),
        identities=(identity,),
    )
    assert aggregator.index_of(wrapped) == 0
    assert aggregator.index_of(inner) == 0
    assert aggregator.identity_for(wrapped) is identity
    assert aggregator.identity_for(inner) is identity
    aggregator.update(_fall_input(time_sec=1.0, frame_index=1))
    attributed = aggregator.attributed_trace_snapshots()
    assert attributed
    assert all(item.identity is identity for item in attributed)
    assert all(item.producer_index == 0 for item in attributed)


def test_g8_5_shadow_count_is_clamped_without_raising() -> None:
    snapshots = _two_snapshots()
    over = EventAggregator(
        deciders=(_ClampedShadowDecider(count=99, snapshots=snapshots),),  # type: ignore[arg-type]
        incidents=IncidentManager(),
        identities=(_bed_identity(),),
    )
    under = EventAggregator(
        deciders=(_ClampedShadowDecider(count=-3, snapshots=snapshots),),  # type: ignore[arg-type]
        incidents=IncidentManager(),
        identities=(_bed_identity(),),
    )
    over_roles = [item.authority for item in over.attributed_trace_snapshots()]
    under_roles = [item.authority for item in under.attributed_trace_snapshots()]
    assert over_roles == ["shadow", "shadow"]
    assert under_roles == ["authoritative", "authoritative"]


def test_g8_6_identities_do_not_change_update_or_release_bytes(tmp_path) -> None:
    journal = tmp_path / "event-identities.jsonl"
    fall_blank, monitor_blank = _domain_pair()
    fall_named, monitor_named = _domain_pair()
    blank = EventAggregator(
        deciders=(fall_blank, monitor_blank),  # type: ignore[arg-type]
        incidents=IncidentManager(identity_path=journal),
        identities=(),
        monotonic=lambda: 0.0,
    )
    named = EventAggregator(
        deciders=(fall_named, monitor_named),  # type: ignore[arg-type]
        incidents=IncidentManager(identity_path=journal),
        identities=(_fall_identity(), _bed_identity()),
        monotonic=lambda: 0.0,
    )
    blank_events: tuple[BusinessEvent, ...] = ()
    named_events: tuple[BusinessEvent, ...] = ()
    for frame_index in (1, 2, 3):
        contained = _bed_exit_input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(7,),
            frame_index=frame_index,
        )
        blank_step = blank.update(contained)
        named_step = named.update(contained)
        assert _event_tuple_bytes(blank_step) == _event_tuple_bytes(named_step)
        blank_events = blank_events + blank_step
        named_events = named_events + named_step
    assert blank_events, "invariance must compare real admitted events, not two empty tuples"
    for left, right in zip(blank_events, named_events, strict=True):
        blank.release(left)
        named.release(right)
    contained = _bed_exit_input(
        person_boxes=(IN_BED_A,),
        bed_boxes=(BED_A,),
        track_ids=(7,),
        frame_index=4,
    )
    after_blank = blank.update(contained)
    after_named = named.update(contained)
    assert _event_tuple_bytes(after_blank) == _event_tuple_bytes(after_named)
