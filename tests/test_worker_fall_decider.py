from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    FrameObservation,
)
from worker.domains.fall import FallEventLatch
from worker.interfaces.decision import Decider
from worker.types import BusinessEvent, DecisionInput, FallModelInput


@dataclass(frozen=True, slots=True)
class _Metadata:
    window: int = 1
    stride: int = 1
    mode: Literal["sequence"] = "sequence"


@dataclass(slots=True)  # policy: MUTABLE_OK - records predictions for assertions
class _Model:
    probabilities: tuple[float, ...]
    metadata: _Metadata = field(default_factory=_Metadata)
    operating_threshold: float = 0.5
    inputs: list[FallModelInput] = field(default_factory=list)

    def predict(self, features: FallModelInput) -> float:
        self.inputs.append(features)
        index = min(len(self.inputs) - 1, len(self.probabilities) - 1)
        return self.probabilities[index]


@dataclass(frozen=True, slots=True)
class _CoordinateModel:
    metadata: _Metadata = field(default_factory=_Metadata)
    operating_threshold: float = 0.5

    def predict(self, features: FallModelInput) -> float:
        first_row = features[0]
        first_value = first_row[0] if isinstance(first_row, tuple) else first_row
        return 0.9 if first_value >= 0.5 else 0.1


def _input(
    *,
    frame_index: int,
    track_id: int | None = 1,
    x: int = 90,
    live_track_ids: tuple[int, ...] | None = None,
    has_timestamp: bool = True,
) -> DecisionInput:
    pose = tuple((x, 50, 0.9) for _ in range(17))
    poses = () if track_id is None else (pose,)
    track_ids = () if track_id is None else (track_id,)
    live_ids = track_ids if live_track_ids is None else live_track_ids
    return DecisionInput(
        observation=FrameObservation(poses=poses, track_ids=track_ids),
        frame_width=100,
        frame_height=100,
        live_track_ids=live_ids,
        time_sec=float(frame_index) if has_timestamp else None,
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_repeated_positive_frames_emit_one_typed_rising_edge() -> None:
    # Given
    model = _Model((0.87,))
    detector = FallEventLatch(model, camera_id="camera-1", facility_id="facility-1")

    # When
    emitted = tuple(
        detector.update(_input(frame_index=index))
        for index in range(3)
    )

    # Then
    assert isinstance(detector, Decider)
    assert emitted == (
        (
            BusinessEvent(
                domain="fall",
                event_type="fall",
                identity=1,
                camera_id="camera-1",
                facility_id="facility-1",
                time_sec=0.0,
                probability=0.87,
            ),
        ),
        (),
        (),
    )
    assert detector.event_count == 1
    assert detector.first_event_sec == 0.0


def test_negative_transition_allows_second_identity() -> None:
    # Given
    model = _Model((0.9, 0.1, 0.95))
    detector = FallEventLatch(model, camera_id="camera", facility_id="facility")

    # When
    first = detector.update(_input(frame_index=1))
    negative = detector.update(_input(frame_index=2))
    second = detector.update(_input(frame_index=3))

    # Then
    assert first[0].identity == 1
    assert first[0].probability == 0.9
    assert negative == ()
    assert second[0].identity == 2
    assert second[0].probability == 0.95
    assert detector.event_count == 2
    assert detector.first_event_sec == 1.0


def test_two_cameras_share_model_but_own_temporal_state() -> None:
    # Given
    model = _Model((0.76,))
    first = FallEventLatch(model, camera_id="camera-a", facility_id="facility")
    second = FallEventLatch(model, camera_id="camera-b", facility_id="facility")

    # When
    first_event = first.update(_input(frame_index=4))
    second_event = second.update(_input(frame_index=8))

    # Then
    assert first.classifier is not second.classifier
    assert first.classifier.model is model
    assert second.classifier.model is model
    assert first_event[0].camera_id == "camera-a"
    assert second_event[0].camera_id == "camera-b"
    assert first_event[0].identity == second_event[0].identity == 1
    assert first.first_event_sec == 4.0
    assert second.first_event_sec == 8.0


def test_eviction_and_camera_boundary_do_not_leak_probability() -> None:
    # Given
    model = _CoordinateModel()
    first = FallEventLatch(model, camera_id="camera-a", facility_id="facility")
    second = FallEventLatch(model, camera_id="camera-b", facility_id="facility")

    # When
    positive = first.update(_input(frame_index=0, track_id=1, x=90))
    evicted = first.update(
        _input(frame_index=1, track_id=None, live_track_ids=())
    )
    replacement = first.update(_input(frame_index=2, track_id=2, x=10))
    other_camera = second.update(_input(frame_index=0, track_id=1, x=10))

    # Then
    assert positive[0].probability == 0.9
    assert evicted == ()
    assert replacement == ()
    assert other_camera == ()
    assert first.event_count == 1
    assert second.event_count == 0


def test_missing_timestamp_maps_to_zero() -> None:
    # Given
    detector = FallEventLatch(
        _Model((0.6,)), camera_id="camera", facility_id="facility"
    )

    # When
    event = detector.update(_input(frame_index=5, has_timestamp=False))

    # Then
    assert event[0].time_sec == 0.0


def test_an_unstaged_onset_can_be_reported_again_by_a_later_frame() -> None:
    """Releasing must re-open the real rising edge, not just the cooldown.

    `update_signal` consumes the edge, so after an onset the next positive frame
    is merely "still falling" and emits nothing. If that onset's envelope never
    reached durable storage, leaving the edge consumed destroys the fall
    outright -- it cannot be re-derived from any later frame.

    An earlier version of this repair undid only the incident-manager cooldown,
    which looks correct against a decider that emits every frame and is useless
    against the real latch.
    """
    from worker.pipeline.decision.event_aggregator import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager

    model = _Model((0.9, 0.9, 0.9))
    detector = FallEventLatch(model, camera_id="camera-1", facility_id="facility-1")
    aggregator = EventAggregator(
        deciders=(detector,), incidents=IncidentManager(cooldown_sec=300.0)
    )

    first = aggregator.update(_input(frame_index=0))
    assert len(first) == 1, "the onset was never reported"

    # The envelope failed to reach durable storage.
    aggregator.release(first[0])

    second = aggregator.update(_input(frame_index=1))

    assert len(second) == 1, (
        "the later frame reported nothing; the rising edge stayed consumed and "
        "the fall was destroyed by a transient staging failure"
    )


def test_release_reaches_a_wrapped_decider() -> None:
    """Production wraps deciders, and a plain getattr on the wrapper is None.

    `_WindowGatedDecider` sits around the real latch whenever a detection window
    is configured. A release that only inspects the outermost object therefore
    never arrives, and the fall stays destroyed exactly as before the repair.
    """
    from datetime import UTC, datetime

    from worker.pipeline.decision.event_aggregator import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager
    from worker.runtime.worker import _WindowGatedDecider

    class _AlwaysOpen:
        def contains(self, *_args: object, **_kwargs: object) -> bool:
            return True

        def __getattr__(self, _name: str) -> object:
            return lambda *_a, **_k: True

    latch = FallEventLatch(
        _Model((0.9, 0.9, 0.9)), camera_id="camera-1", facility_id="facility-1"
    )
    aggregator = EventAggregator(
        deciders=(
            _WindowGatedDecider(latch, _AlwaysOpen(), clock=lambda: datetime.now(UTC)),
        ),
        incidents=IncidentManager(cooldown_sec=300.0),
    )

    first = aggregator.update(_input(frame_index=0))
    assert len(first) == 1
    aggregator.release(first[0])

    assert len(aggregator.update(_input(frame_index=1))) == 1, (
        "the release never reached the latch through the production wrapper"
    )


def test_an_event_never_attempted_is_also_released() -> None:
    """Events behind a failure were admitted and must not be silently consumed.

    The aggregator admits the whole tuple before the emission loop begins. When
    an earlier event raises, the ones behind it are never attempted at all, yet
    their cooldowns are already spent -- a fall sitting after a bed exit that
    raised is lost without ever being tried.
    """
    from worker.pipeline.decision.incident_manager import IncidentManager
    from worker.types import BusinessEvent

    incidents = IncidentManager(cooldown_sec=300.0)

    def _fall(time_sec: float) -> BusinessEvent:
        return BusinessEvent(
            "fall", "fall.detected", "src", "camera-1", "facility-1", time_sec, 0.99
        )

    admitted = incidents.admit(_fall(100.0), now_sec=100.0)
    assert admitted is not None
    assert incidents.admit(_fall(101.0), now_sec=101.0) is None, "cooldown not in effect"

    incidents.release(admitted)

    assert incidents.admit(_fall(102.0), now_sec=102.0) is not None, (
        "an admitted-but-never-attempted fall stayed consumed and was lost"
    )


def test_releasing_another_domains_event_leaves_the_fall_latch_alone() -> None:
    """Release must reach the producing decider and no other.

    An earlier version matched on a `domain` attribute the real latches do not
    declare, so `getattr(decider, "domain", event.domain)` always compared equal
    and the check silently passed for every decider. A bed-exit failure then
    re-opened a fall onset that had been delivered perfectly well, and the next
    positive frame reported a second fall for the same uninterrupted condition.
    """
    from worker.pipeline.decision.event_aggregator import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager
    from worker.types import BusinessEvent

    latch = FallEventLatch(
        _Model((0.9, 0.9, 0.9)), camera_id="camera-1", facility_id="facility-1"
    )
    aggregator = EventAggregator(
        deciders=(latch,), incidents=IncidentManager(cooldown_sec=300.0)
    )

    emitted = aggregator.update(_input(frame_index=0))
    assert len(emitted) == 1
    def _decision_state() -> tuple[object, ...]:
        # observation_age_sec ticks with wall time and says nothing about
        # whether the decision was consumed.
        snapshot = latch.status_snapshot
        return (latch.event_count, latch.first_event_sec, snapshot.is_fall, snapshot.stale)

    before = _decision_state()

    # An event this aggregator never produced.
    aggregator.release(
        BusinessEvent(
            "bed_exit", "bed.exit", "other", "camera-1", "facility-1", 1.0, 0.9, bed_id=7
        )
    )

    assert _decision_state() == before, (
        "releasing another domain's event mutated the fall latch, which would "
        "report a second fall for the same uninterrupted condition"
    )

    # Its own event still re-opens it.
    aggregator.release(emitted[0])
    assert latch.event_count == before[0] - 1


def test_the_latch_refuses_an_onset_release_it_does_not_own() -> None:
    """Defence in depth at the owner, independent of how release is routed.

    The aggregator routes a release to the producing decider, but an earlier
    version of that routing was a permissive default that matched every decider.
    Checking ownership here as well means a future caller cannot repeat that
    mistake silently: an event this latch did not produce is refused, and so is
    one whose ownership cannot be established at all.
    """
    from worker.types import BusinessEvent

    latch = FallEventLatch(
        _Model((0.9, 0.9)), camera_id="camera-1", facility_id="facility-1"
    )
    assert len(latch.update(_input(frame_index=0))) == 1
    consumed = (latch.event_count, latch._previous_fall)  # noqa: SLF001

    def _event(domain: str, camera_id: str) -> BusinessEvent:
        return BusinessEvent(
            domain, "x", "src", camera_id, "facility-1", 1.0, 0.9, bed_id=7
        )

    latch.release_onset(_event("bed_exit", "camera-1"))
    latch.release_onset(_event("fall", "camera-OTHER"))
    latch.release_onset(None)

    assert (latch.event_count, latch._previous_fall) == consumed, (  # noqa: SLF001
        "the latch acted on a release for an event it never produced"
    )

    latch.release_onset(_event("fall", "camera-1"))
    assert latch.event_count == consumed[0] - 1, "its own onset was not re-opened"


def test_a_release_arriving_a_frame_late_still_reaches_the_latch() -> None:
    """The producer record must outlive the frame that created it.

    The pump releases inside the same iteration that produced the events, so
    clearing the record each frame would be safe today. It would also be
    fragile: a release arriving one frame later would find no producer and
    silently do nothing, leaving the fall destroyed with no error anywhere.
    """
    from worker.pipeline.decision.event_aggregator import EventAggregator
    from worker.pipeline.decision.incident_manager import IncidentManager

    latch = FallEventLatch(
        _Model((0.9,) * 6), camera_id="camera-1", facility_id="facility-1"
    )
    aggregator = EventAggregator(
        deciders=(latch,), incidents=IncidentManager(cooldown_sec=300.0)
    )

    first = aggregator.update(_input(frame_index=0))
    assert len(first) == 1
    aggregator.update(_input(frame_index=1))  # a frame passes

    aggregator.release(first[0])

    assert len(aggregator.update(_input(frame_index=2))) == 1, (
        "a release one frame late found no producer and did nothing; the fall "
        "stayed destroyed"
    )
