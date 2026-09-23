"""Generation-11 adversarial cases for the no-stale / no-silent decision-evidence delta."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from observability_stack_fixtures import serve_backend, wait_until
from test_execution_record_wiring import _ImmediateClassifier, _metadata, _pump
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
    OUTSIDE_BEDS,
)
from test_worker_domains_bed_exit import (
    _clock_at as _bed_clock_at,
)
from test_worker_domains_bed_exit import (
    _input as _bed_exit_input,
)
from test_worker_domains_bed_exit import (
    _monitor as _bed_exit_monitor,
)

from contracts.observation import BoundingBox
from shared.events.delivery_queue import AdmissionResult
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall.classifier import FallWindowClassifier
from worker.domains.fall.policy import FallDomainDecider, FallPolicyDecider
from worker.domains.registry import FALL_MODULE_QUALIFIED_ID
from worker.interfaces.fall_model import FallProbabilities
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.pipeline.diagnostics.lanes import ExecutionRecordLanes
from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublicationError
from worker.pipeline.output.evidence.flow_sealed_sidecar import FlowSealedSidecars
from worker.runtime.flow.evidence import FlowEvidenceBinding
from worker.runtime.flow.execution_record_emit import emit_model_and_decision
from worker.runtime.flow.policy_pump import NativePolicyPump
from worker.runtime.worker import _WindowGatedDecider
from worker.types import BusinessEvent
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
_FALL_POLICY = "a" * 64
_BED_POLICY = "b" * 64
_PTS_STEP_NS = 66_666_667
_TRACK = 9


def _query(backend: object) -> dict[str, Any]:
    return backend.query(_CAMERA, _QUERY_FROM_NS, _QUERY_TO_NS, limit=500)


def _fall_identity() -> DecisionIdentity:
    return DecisionIdentity(FALL_MODULE_QUALIFIED_ID, _FALL_POLICY)


def _bed_identity() -> DecisionIdentity:
    return DecisionIdentity(_BED_MODULE, _BED_POLICY)


def _night_monitor(*, camera_id: str = _CAMERA) -> object:
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


def _bed_metadata(
    *,
    seq: int,
    pts: int,
    child: UUID,
    person: PersonBox,
    track_id: int = _TRACK,
) -> MetadataFrame:
    identity = PerceptionFrameIdentity("boot-1", "cam-1", 3, seq, pts)
    region = BedRegion(0, 0, 80, 100, 0.99)
    pose = tuple(Keypoint(index + 1, index + 2, 0.9) for index in range(17))
    return MetadataFrame(
        frame=PerceptionFrameV1(
            identity=identity,
            person_box=PersonBoxChannel(ChannelState.INFERRED, (person,)),
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
        child_instance_id=child,
        native_publish_sequence=seq,
        transform_id="transform-a",
        source_width=180,
        source_height=120,
        source_time_ns=pts,
        native_observation_evidence=NativeObservationEvidence(91, 17, True, 1, 1, 1),
    )


def _person(box: BoundingBox) -> PersonBox:
    return PersonBox(box.x1, box.y1, box.x2, box.y2, box.confidence)


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


class _AdmitStager:
    def stage(self, event: dict[str, object]) -> AdmissionResult:
        del event
        return AdmissionResult(True)

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        del edge_event_id, clip_id


class _SilentActor:
    def admit(self, event_ref: str, detected_at: str) -> None:
        del event_ref, detected_at


class _NoClipPublisher:
    def publish(self, sealed: object, events: object) -> object:
        del sealed, events
        raise FlowClipPublicationError("clip publication unused in gen-11 red-team")


class _ConstFallModel:
    def predict(self, features: object) -> FallProbabilities:
        del features
        return FallProbabilities(0.9, 0.1, 0.0)


def _binding(tmp_path, lanes: ExecutionRecordLanes) -> FlowEvidenceBinding:
    return FlowEvidenceBinding(
        actor=_SilentActor(),  # type: ignore[arg-type]
        stager=_AdmitStager(),  # type: ignore[arg-type]
        publisher=_NoClipPublisher(),  # type: ignore[arg-type]
        sidecars=FlowSealedSidecars(tmp_path / "sidecars"),
        camera_id=_CAMERA,
        execution_records=lanes,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    assert isinstance(payload, dict)
    return payload


def _kind_rows(body: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [row for row in body.get("records", ()) if row["record_kind"] == kind]


def _fall_decisions(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _kind_rows(body, "policy.decision")
        if _payload(row).get("module_qualified_id") == FALL_MODULE_QUALIFIED_ID
    ]


def _bed_decisions(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _kind_rows(body, "policy.decision")
        if _payload(row).get("module_qualified_id") == _BED_MODULE
    ]


def _is_coast_row(row: dict[str, Any]) -> bool:
    payload = _payload(row)
    missing = payload.get("missing_values") or {}
    return (
        row.get("outcome") == "coasted"
        and payload.get("reason") == "score-missing"
        and missing.get("decision_state") == "resample-gap"
        and payload.get("track_id") is None
        and payload.get("decision_trace_id") is None
    )


def test_g11_1_duplicate_pts_coasts_through_backend_query(tmp_path) -> None:
    """Three frames; frame_seq 1 repeats PTS. Exactly one coasted fall decision."""
    lanes = ExecutionRecordLanes(lane_capacity=256)
    pump = _pump(lanes, identity=_fall_identity(), fall_transition=0.1)
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            child = pump._child  # noqa: SLF001
            assert isinstance(child, UUID)
            pump._process(_metadata(child=child, seq=0, pts=100))  # noqa: SLF001
            pump._process(_metadata(child=child, seq=1, pts=100))  # noqa: SLF001
            pump._process(_metadata(child=child, seq=2, pts=100 + _PTS_STEP_NS))  # noqa: SLF001

            def _coasted_visible() -> bool:
                body = _query(backend)
                return any(
                    row["frame_seq"] == 1 and _is_coast_row(row) for row in _fall_decisions(body)
                )

            wait_until(_coasted_visible, timeout=5.0, what="coasted policy.decision seq 1")
            body = _query(backend)
            fall_rows = _fall_decisions(body)
            seq0 = [row for row in fall_rows if row["frame_seq"] == 0]
            seq1 = [row for row in fall_rows if row["frame_seq"] == 1]
            seq2 = [row for row in fall_rows if row["frame_seq"] == 2]
            assert seq0 and all(row["outcome"] != "coasted" for row in seq0)
            assert seq2 and all(row["outcome"] != "coasted" for row in seq2)
            assert len(seq1) == 1, seq1
            assert _is_coast_row(seq1[0])
            assert all(_payload(row).get("track_id") is None for row in seq1)
            scores = [row for row in _kind_rows(body, "model.score") if row["frame_seq"] == 1]
            assert scores == []
        finally:
            if exporter is not None:
                exporter.stop()


def test_g11_2_nonmonotonic_pts_coasts_then_resumes_through_backend_query(tmp_path) -> None:
    """PTS rollback: ImmediateClassifier refuses; FallWindowClassifier resets.

    Duplicate PTS still coasts (G11-1). A strictly smaller PTS rebuilds the
    resampler rather than coasting the previous identity; the later monotonic
    frame is a fresh row.
    """
    refuse_lanes = ExecutionRecordLanes(lane_capacity=64)
    refuse_pump = _pump(refuse_lanes, identity=_fall_identity(), fall_transition=0.1)
    refuse_child = refuse_pump._child  # noqa: SLF001
    assert isinstance(refuse_child, UUID)
    refuse_pump._process(_metadata(child=refuse_child, seq=0, pts=200))  # noqa: SLF001
    try:
        refuse_pump._process(_metadata(child=refuse_child, seq=1, pts=100))  # noqa: SLF001
    except TypeError as error:
        assert "stream-epoch reset" in str(error)
    else:
        raise AssertionError("ImmediateClassifier must refuse a PTS rollback")

    lanes = ExecutionRecordLanes(lane_capacity=256)
    pump = _pump(lanes, identity=_fall_identity(), fall_transition=0.1)
    inner = pump._decision.deciders[0]
    inner.classifier = FallWindowClassifier(_ConstFallModel())  # type: ignore[attr-defined]
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            child = pump._child  # noqa: SLF001
            assert isinstance(child, UUID)
            first_pts = 200
            later_pts = first_pts + _PTS_STEP_NS
            pump._process(_metadata(child=child, seq=0, pts=first_pts))  # noqa: SLF001
            pump._process(_metadata(child=child, seq=1, pts=100))  # noqa: SLF001
            pump._process(_metadata(child=child, seq=2, pts=later_pts))  # noqa: SLF001

            def _seq2_visible() -> bool:
                return any(row["frame_seq"] == 2 for row in _fall_decisions(_query(backend)))

            wait_until(_seq2_visible, timeout=5.0, what="frame_seq 2 fall decision")
            body = _query(backend)
            fall_rows = _fall_decisions(body)
            seq1 = [row for row in fall_rows if row["frame_seq"] == 1]
            seq2 = [row for row in fall_rows if row["frame_seq"] == 2]
            assert seq1, "non-monotonic frame must be visible, not silent"
            assert all(not _is_coast_row(row) for row in seq1)
            assert all(row["outcome"] != "coasted" for row in seq1)
            assert seq2 and all(row["outcome"] != "coasted" for row in seq2)
        finally:
            if exporter is not None:
                exporter.stop()


def test_g11_3_bed_exit_episode_already_open_through_backend_query(tmp_path) -> None:
    """Onset is triggered+delivered; the next frame is episode-already-open with no delivery."""
    lanes = ExecutionRecordLanes(lane_capacity=256)
    pump = _pump(
        lanes,
        identity=_fall_identity(),
        fall_transition=0.1,
        event_sink=_binding(tmp_path, lanes),
    )
    monitor = _night_monitor()
    _compose_bed_and_fall(pump, monitor, identities=(_fall_identity(), _bed_identity()))
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            child = pump._child  # noqa: SLF001
            assert isinstance(child, UUID)
            in_bed = _person(IN_BED_A)
            outside = _person(OUTSIDE_BEDS)
            boxes = (in_bed, outside, outside, outside, outside)
            for seq, person in enumerate(boxes):
                pump._process(  # noqa: SLF001
                    _bed_metadata(
                        child=child,
                        seq=seq,
                        pts=100 + seq * _PTS_STEP_NS,
                        person=person,
                    )
                )

            def _onset_visible() -> bool:
                body = _query(backend)
                triggered = any(
                    _payload(row).get("triggered") is True for row in _bed_decisions(body)
                )
                delivered = any(
                    row["record_kind"] == "event.delivery" for row in body.get("records", ())
                )
                return triggered and delivered

            wait_until(_onset_visible, timeout=5.0, what="bed-exit onset + delivery")
            body = _query(backend)
            bed_rows = _bed_decisions(body)
            triggered = [row for row in bed_rows if _payload(row).get("triggered") is True]
            assert triggered, "onset frame must surface a triggered bed_exit.v1 row"
            onset_seq = triggered[0]["frame_seq"]
            deliveries = _kind_rows(body, "event.delivery")
            assert deliveries, "onset must admit an event.delivery"
            assert any(row["frame_seq"] == onset_seq for row in deliveries)
            follow_seq = onset_seq + 1
            follow = [
                row
                for row in bed_rows
                if row["frame_seq"] == follow_seq
                and _payload(row).get("reason") == "episode-already-open"
            ]
            assert follow, f"frame_seq {follow_seq} must explain the open episode"
            for row in follow:
                assert _payload(row).get("triggered") is False
                assert _payload(row).get("track_id") == _TRACK
            assert not [row for row in deliveries if row["frame_seq"] == follow_seq]
        finally:
            if exporter is not None:
                exporter.stop()


def test_g11_4_bed_exit_outside_window_through_backend_query(tmp_path) -> None:
    """Clock outside the night window: explicit non-event, zero deliveries."""
    lanes = ExecutionRecordLanes(lane_capacity=256)
    pump = _pump(
        lanes,
        identity=_fall_identity(),
        fall_transition=0.1,
        event_sink=_binding(tmp_path, lanes),
    )
    monitor = _night_monitor()
    monitor._clock = _bed_clock_at(hour=12)  # noqa: SLF001
    _compose_bed_and_fall(pump, monitor, identities=(_fall_identity(), _bed_identity()))
    exporter = None
    with serve_backend(tmp_path, budget_bytes=_BUDGET_BYTES, relay_token=_RELAY_TOKEN) as backend:
        try:
            exporter = _exporter(lanes, backend.base_url, backend.relay_token)
            exporter.start()
            child = pump._child  # noqa: SLF001
            assert isinstance(child, UUID)
            in_bed = _person(IN_BED_A)
            outside = _person(OUTSIDE_BEDS)
            boxes = (in_bed, outside, outside, outside)
            for seq, person in enumerate(boxes):
                pump._process(  # noqa: SLF001
                    _bed_metadata(
                        child=child,
                        seq=seq,
                        pts=100 + seq * _PTS_STEP_NS,
                        person=person,
                    )
                )

            def _bed_visible() -> bool:
                return bool(_bed_decisions(_query(backend)))

            wait_until(_bed_visible, timeout=5.0, what="bed_exit.v1 policy.decision")
            body = _query(backend)
            bed_rows = _bed_decisions(body)
            assert bed_rows
            assert all(_payload(row).get("triggered") is False for row in bed_rows)
            assert any(
                _payload(row).get("reason") == "outside-detection-window" for row in bed_rows
            )
            assert _kind_rows(body, "event.delivery") == []
        finally:
            if exporter is not None:
                exporter.stop()


def _update_fall(
    decider: FallPolicyDecider, transition: float, frame: int, *, fallen: float = 0.0
) -> tuple[BusinessEvent, ...]:
    probability = FallProbabilities(1.0 - transition, transition, fallen)
    return decider.update(
        {_TRACK: probability}, (_TRACK,), frame_index=frame, time_sec=float(frame)
    )


def test_g11_5_fall_recovery_is_not_labelled_suppression() -> None:
    """After onset, 0.9 is episode-already-open; five clears re-arm a new onset."""
    decider = FallPolicyDecider(
        camera_id=_CAMERA,
        facility_id="facility-a",
        boot_id="boot-1",
        source_generation=0,
        stream_epoch="epoch",
    )
    assert _update_fall(decider, 0.7, 0) == ()
    assert _update_fall(decider, 0.7, 1) == ()
    (onset,) = _update_fall(decider, 0.7, 2)
    assert onset.event_type == "fall"
    (fired,) = decider.last_trace_snapshots
    assert fired.triggered is True and fired.reason == "transition-confirmed"

    assert _update_fall(decider, 0.9, 3) == ()
    (suppressed,) = decider.last_trace_snapshots
    assert suppressed.triggered is False
    assert suppressed.reason == "episode-already-open"

    clear_reasons: list[str] = []
    for frame in range(4, 9):
        assert _update_fall(decider, 0.1, frame) == ()
        (row,) = decider.last_trace_snapshots
        clear_reasons.append(row.reason)
        assert row.reason != "episode-already-open"
        assert row.triggered is False
    assert "episode-already-open" not in clear_reasons

    assert _update_fall(decider, 0.7, 9) == ()
    assert _update_fall(decider, 0.7, 10) == ()
    (second,) = _update_fall(decider, 0.7, 11)
    assert second.event_type == "fall"
    assert second.identity != onset.identity
    (rearmed,) = decider.last_trace_snapshots
    assert rearmed.triggered is True
    assert rearmed.reason == "transition-confirmed"
    assert rearmed.reason != "episode-already-open"


def _bed_sequence(monitor: object) -> tuple[BusinessEvent, ...]:
    events: tuple[BusinessEvent, ...] = ()
    events += monitor.update(  # type: ignore[union-attr]
        _bed_exit_input(
            person_boxes=(IN_BED_A,),
            bed_boxes=(BED_A,),
            track_ids=(_TRACK,),
            frame_index=0,
        )
    )
    for frame_index in (1, 2, 3, 4):
        events += monitor.update(  # type: ignore[union-attr]
            _bed_exit_input(
                person_boxes=(OUTSIDE_BEDS,),
                bed_boxes=(BED_A,),
                track_ids=(_TRACK,),
                frame_index=frame_index,
            )
        )
    return events


def _fall_sequence(decider: FallPolicyDecider) -> tuple[BusinessEvent, ...]:
    events: tuple[BusinessEvent, ...] = ()
    for frame, transition in (
        (0, 0.7),
        (1, 0.7),
        (2, 0.7),
        (3, 0.9),
        (4, 0.1),
        (5, 0.1),
        (6, 0.1),
        (7, 0.1),
        (8, 0.1),
        (9, 0.7),
        (10, 0.7),
        (11, 0.7),
    ):
        events += _update_fall(decider, transition, frame)
    return events


def test_g11_6_domain_decider_events_are_byte_identical() -> None:
    """Identical inputs: fall and bed-exit domain events are byte-identical (not records)."""
    fall_a = FallPolicyDecider(
        camera_id=_CAMERA,
        facility_id="facility-a",
        boot_id="boot-1",
        source_generation=0,
        stream_epoch="epoch",
    )
    fall_b = FallPolicyDecider(
        camera_id=_CAMERA,
        facility_id="facility-a",
        boot_id="boot-1",
        source_generation=0,
        stream_epoch="epoch",
    )
    fall_events_a = _fall_sequence(fall_a)
    fall_events_b = _fall_sequence(fall_b)
    assert fall_events_a, "invariance must compare real fall onsets, not two empty tuples"
    assert _event_tuple_bytes(fall_events_a) == _event_tuple_bytes(fall_events_b)

    bed_a = _night_monitor()
    bed_b = _night_monitor()
    bed_events_a = _bed_sequence(bed_a)
    bed_events_b = _bed_sequence(bed_b)
    assert bed_events_a, "invariance must compare a real bed-exit onset"
    assert _event_tuple_bytes(bed_events_a) == _event_tuple_bytes(bed_events_b)

    fall_direct = FallDomainDecider(
        classifier=_ImmediateClassifier(0.1),
        policy=FallPolicyDecider(
            camera_id=_CAMERA,
            facility_id="facility-a",
            boot_id="boot-1",
            stream_epoch="3",
            source_generation=1,
        ),
    )
    fall_wrapped = FallDomainDecider(
        classifier=_ImmediateClassifier(0.1),
        policy=FallPolicyDecider(
            camera_id=_CAMERA,
            facility_id="facility-a",
            boot_id="boot-1",
            stream_epoch="3",
            source_generation=1,
        ),
    )
    aggregator = EventAggregator(
        deciders=(fall_wrapped,),
        incidents=IncidentManager(cooldown_sec=0.0),
        identities=(_fall_identity(),),
        monotonic=lambda: 0.0,
    )
    direct_events: tuple[BusinessEvent, ...] = ()
    wrapped_events: tuple[BusinessEvent, ...] = ()
    for frame_index in (0, 1, 2):
        decision_input = _fall_input(time_sec=float(frame_index), frame_index=frame_index)
        direct_events += fall_direct.update(decision_input)
        wrapped_events += aggregator.update(decision_input)
    assert _event_tuple_bytes(direct_events) == _event_tuple_bytes(wrapped_events)


def test_g11_7_window_gate_does_not_leak_inner_coast_outside_window() -> None:
    """In-window rows are inner/fresh; outside is one authoritative gate row, no stale leak."""
    inner = FallDomainDecider(
        classifier=_ImmediateClassifier(0.1),
        policy=FallPolicyDecider(
            camera_id=_CAMERA,
            facility_id="facility-a",
            boot_id="boot-1",
            stream_epoch="3",
            source_generation=1,
        ),
    )
    clock = [datetime(2026, 7, 31, 23, 0, tzinfo=UTC)]
    gated = _WindowGatedDecider(
        decider=inner,
        window=DetectionWindow(start="21:00", end="06:00", tz="UTC"),
        clock=lambda: clock[0],
    )
    aggregator = EventAggregator(
        deciders=(gated,),
        incidents=IncidentManager(),
        identities=(_fall_identity(),),
    )
    lanes = ExecutionRecordLanes(lane_capacity=32)

    aggregator.update(_fall_input(time_sec=1.0, frame_index=0))
    inside = aggregator.attributed_trace_snapshots()
    assert inside
    assert all(item.snapshot.reason != "outside-detection-window" for item in inside)
    assert all(item.fresh for item in inside)
    emit_model_and_decision(lanes, _metadata(seq=0, pts=1_000_000_000), aggregator)
    first = lanes.drain_for("cam-1", "boot-1", limit=32)
    assert first is not None
    inside_decisions = [row for row in first.records if row.record_kind == "policy.decision"]
    assert inside_decisions
    assert all(row.payload["reason"] != "outside-detection-window" for row in inside_decisions)
    assert all(row.outcome != "coasted" for row in inside_decisions)

    aggregator.update(_fall_input(time_sec=1.0, frame_index=1))
    coasted = aggregator.attributed_trace_snapshots()
    assert coasted
    assert all(item.fresh is False for item in coasted)
    assert inner.last_update_evaluated is False

    clock[0] = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    aggregator.update(_fall_input(time_sec=2.0, frame_index=2))
    outside = aggregator.attributed_trace_snapshots()
    assert len(outside) == 1
    (row,) = outside
    assert row.fresh is True
    assert row.authority == "authoritative"
    assert row.snapshot.reason == "outside-detection-window"
    assert row.snapshot.triggered is False
    assert row.snapshot.track_id is None
    assert row.snapshot.previous_state == "not-evaluated"
    emit_model_and_decision(lanes, _metadata(seq=2, pts=2_000_000_000), aggregator)
    second = lanes.drain_for("cam-1", "boot-1", limit=32)
    assert second is not None
    outside_decisions = [item for item in second.records if item.record_kind == "policy.decision"]
    assert len(outside_decisions) == 1
    record = outside_decisions[0]
    assert record.outcome != "coasted"
    assert record.payload["reason"] == "outside-detection-window"
    assert record.payload["triggered"] is False
    assert record.payload["track_id"] is None
    assert record.payload["authority_role"] == "authoritative"
    assert not [item for item in second.records if item.record_kind == "model.score"]
