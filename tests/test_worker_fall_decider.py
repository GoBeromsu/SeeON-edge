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


@dataclass(slots=True)  # noqa: MUTABLE_OK - records predictions for assertions
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
