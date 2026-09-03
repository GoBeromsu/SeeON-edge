from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from contracts.observation import BedRegionCacheState, BedRegionDebugSnapshot, FrameObservation
from worker.pipeline.decision import EventAggregator, IncidentManager
from worker.types import BusinessEvent, DecisionInput


def _event(
    *,
    domain: str = "fall",
    event_type: str = "fall",
    identity: str | int = 1,
    camera_id: str = "camera-1",
    facility_id: str = "facility-1",
    time_sec: float = 12.5,
    probability: float = 0.91,
    person_id: int | None = None,
    bed_id: int | None = None,
) -> BusinessEvent:
    return BusinessEvent(
        domain=domain,
        event_type=event_type,
        identity=identity,
        camera_id=camera_id,
        facility_id=facility_id,
        time_sec=time_sec,
        probability=probability,
        person_id=person_id,
        bed_id=bed_id,
    )


def _input() -> DecisionInput:
    return DecisionInput(
        observation=FrameObservation(),
        frame_width=640,
        frame_height=360,
        live_track_ids=(),
        time_sec=12.5,
        frame_index=1,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


@dataclass(slots=True)  # policy: MUTABLE_OK - test changes scripted events between frames
class _StaticDecider:
    events: tuple[BusinessEvent, ...]

    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        return self.events


@dataclass(slots=True)  # policy: MUTABLE_OK - output probe records observable calls
class _OutputProbe:
    clips: list[str] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    relays: list[str] = field(default_factory=list)

    def emit(self, event: BusinessEvent) -> None:
        event_id = str(event.identity)
        self.clips.append(event_id)
        self.snapshots.append(event_id)
        self.relays.append(event_id)


def _dispatch(probe: _OutputProbe, events: tuple[BusinessEvent, ...]) -> None:
    for event in events:
        probe.emit(event)


def test_fall_admission_preserves_enrichment_and_assigns_a_uuid4(tmp_path: Path) -> None:
    # Given: an immutable domain event carrying its fall counter and probability.
    source = _event(identity=3, time_sec=8.0, probability=0.87)
    manager = IncidentManager(
        cooldown_sec=30.0,
        identity_path=tmp_path / "identities.jsonl",
    )

    # When: the incident is admitted.
    admitted = manager.admit(source, now_sec=100.0)

    # Then: its domain identity remains untouched and the emitted copy is enriched.
    assert admitted is not None
    parsed = UUID(str(admitted.identity))
    assert parsed.version == 4
    assert source.identity == 3
    assert admitted.camera_id == "camera-1"
    assert admitted.facility_id == "facility-1"
    assert admitted.time_sec == 8.0
    assert admitted.probability == 0.87
    assert manager.last_audit_snapshot is not None
    assert manager.last_audit_snapshot.source_identity == 3
    assert manager.last_audit_snapshot.edge_event_id == str(parsed)


def test_fall_identity_repeat_is_suppressed_until_exact_cooldown_boundary(
    tmp_path: Path,
) -> None:
    # Given: one admitted fall episode, a *distinct* second episode, and a
    # repeat of the first episode's own identity.
    manager = IncidentManager(cooldown_sec=30.0, identity_path=tmp_path / "ids.jsonl")
    first = _event(identity=1, time_sec=1.0)
    second_episode = _event(identity=2, time_sec=2.0)
    repeat = _event(identity=1, time_sec=3.0)

    # When: all three are offered inside the cooldown, and the repeat again at
    # the boundary.
    admitted_first = manager.admit(first, now_sec=100.0)
    admitted_second = manager.admit(second_episode, now_sec=101.0)
    suppressed = manager.admit(repeat, now_sec=102.0)
    readmitted = manager.admit(repeat, now_sec=130.0)

    # Then: a distinct episode is never suppressed -- the episode authority is
    # the lifecycle owner and a second episode is a second resident or a
    # confirmed-recovery re-arm. Only a producer re-emitting an identity it
    # already emitted is overload, and that suppression is counted.
    assert admitted_first is not None
    assert admitted_second is not None
    assert suppressed is None
    assert manager.cooldown_suppressed_total == 1
    assert readmitted is not None
    assert readmitted.identity != admitted_first.identity


def test_distinct_bed_episodes_are_admitted_and_only_a_repeat_is_suppressed(
    tmp_path: Path,
) -> None:
    # Given: two distinct bed episodes and a repeat of the first identity.
    manager = IncidentManager(cooldown_sec=30.0, identity_path=tmp_path / "ids.jsonl")
    first = _event(
        domain="bed_exit",
        event_type="bed-exit",
        identity="boot:1:7:0:0:1",
        person_id=7,
        bed_id=1,
    )
    other_bed = _event(
        domain="bed_exit",
        event_type="bed-exit",
        identity="boot:1:9:0:0:2",
        person_id=9,
        bed_id=2,
    )
    repeat = _event(
        domain="bed_exit",
        event_type="bed-exit",
        identity="boot:1:7:0:0:1",
        person_id=7,
        bed_id=1,
    )

    # When: all three candidates are offered within one cooldown.
    admitted = manager.admit(first, now_sec=100.0)
    independent = manager.admit(other_bed, now_sec=101.0)
    suppressed = manager.admit(repeat, now_sec=101.0)

    # Then: a second bed episode is admitted -- id churn inside one episode is
    # the episode authority's re-association job, not the cooldown's -- and only
    # a producer re-emitting an already-emitted identity is suppressed.
    assert admitted is not None
    assert admitted.person_id == 7
    assert admitted.bed_id == 1
    assert independent is not None
    assert independent.bed_id == 2
    assert suppressed is None
    assert manager.cooldown_suppressed_total == 1


def test_duplicate_within_cooldown_has_zero_output_side_effects(tmp_path: Path) -> None:
    # Given: an aggregator and output probe separated by the admission boundary.
    first = _event(identity=1, time_sec=1.0)
    # The same episode identity re-emitted: overload, not a second episode.
    repeat = _event(identity=1, time_sec=2.0)
    now_values = iter((100.0, 101.0))
    decider = _StaticDecider((first,))
    aggregator = EventAggregator(
        deciders=(decider,),
        incidents=IncidentManager(
            cooldown_sec=30.0,
            identity_path=tmp_path / "identities.jsonl",
        ),
        monotonic=lambda: next(now_values),
    )
    output = _OutputProbe()

    # When: output handles the first admission and then a duplicate rising edge.
    _dispatch(output, aggregator.update(_input()))
    decider.events = (repeat,)
    duplicate = aggregator.update(_input())
    _dispatch(output, duplicate)

    # Then: no clip, snapshot, or relay call crosses the boundary for the duplicate.
    assert duplicate == ()
    assert len(output.clips) == 1
    assert len(output.snapshots) == 1
    assert len(output.relays) == 1
