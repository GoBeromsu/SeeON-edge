from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    FrameObservation,
)
from worker.domains.fall import FallWindowClassifier
from worker.types import DecisionInput, FallModelInput


@dataclass(frozen=True, slots=True)
class _Metadata:
    window: int
    stride: int
    mode: Literal["features", "sequence"]


@dataclass(slots=True)  # policy: MUTABLE_OK - records model inputs for assertions
class _Model:
    metadata: _Metadata
    probabilities: tuple[float, ...]
    operating_threshold: float
    inputs: list[FallModelInput] = field(default_factory=list)

    def predict(self, features: FallModelInput) -> float:
        self.inputs.append(features)
        index = min(len(self.inputs) - 1, len(self.probabilities) - 1)
        return self.probabilities[index]


def _pose(
    *, x: int = 25, y: int = 100, confidence: float = 0.9
) -> tuple[tuple[int, int, float], ...]:
    return tuple((x + index, y + index, confidence) for index in range(17))


def _input(
    *,
    frame_index: int,
    track_id: int | None,
    pose: tuple[tuple[int, int, float], ...] | None,
    live_track_ids: tuple[int, ...],
) -> DecisionInput:
    poses = () if pose is None else (pose,)
    track_ids = () if track_id is None else (track_id,)
    return DecisionInput(
        observation=FrameObservation(poses=poses, track_ids=track_ids),
        frame_width=100,
        frame_height=200,
        live_track_ids=live_track_ids,
        time_sec=float(frame_index),
        frame_index=frame_index,
        bed_region=BedRegionDebugSnapshot(BedRegionCacheState.EMPTY),
    )


def test_sequence_mode_normalizes_window_and_preserves_probability() -> None:
    # Given
    model = _Model(_Metadata(window=2, stride=1, mode="sequence"), (0.812345,), 0.812345)
    classifier = FallWindowClassifier(model)

    # When
    first = classifier.classify(
        _input(frame_index=0, track_id=7, pose=_pose(), live_track_ids=(7,))
    )
    second = classifier.classify(
        _input(frame_index=1, track_id=7, pose=_pose(x=35), live_track_ids=(7,))
    )

    # Then
    assert first.labels[0].confidence == 0.0
    assert second.labels[0].confidence == 0.812345
    assert second.labels[0].is_fall
    assert len(model.inputs) == 1
    captured = model.inputs[0]
    assert isinstance(captured, tuple)
    assert len(captured) == 2  # window rows
    rows = [row for row in captured if isinstance(row, tuple)]
    assert len(rows) == 2
    assert all(len(row) == 51 for row in rows)
    assert rows[0][:3] == (0.25, 0.5, 0.9)
    assert rows[1][:3] == (0.35, 0.5, 0.9)


def test_engineered_feature_mode_preserves_probability() -> None:
    # Given
    model = _Model(_Metadata(window=2, stride=1, mode="features"), (0.623456,), 0.6)
    classifier = FallWindowClassifier(model)
    missing_pose = _pose(confidence=0.0)

    # When
    _ = classifier.classify(
        _input(frame_index=0, track_id=8, pose=missing_pose, live_track_ids=(8,))
    )
    result = classifier.classify(
        _input(frame_index=1, track_id=8, pose=missing_pose, live_track_ids=(8,))
    )

    # Then
    assert result.labels[0].confidence == 0.623456
    assert len(model.inputs) == 1
    captured = model.inputs[0]
    assert isinstance(captured, tuple)
    assert len(captured) == 45  # flat engineered-feature vector
    assert all(value == 0.0 for value in captured)


def test_live_track_without_pose_appends_zero_pose() -> None:
    # Given
    model = _Model(_Metadata(window=3, stride=1, mode="sequence"), (0.7,), 0.5)
    classifier = FallWindowClassifier(model)

    # When
    _ = classifier.classify(
        _input(frame_index=0, track_id=3, pose=_pose(), live_track_ids=(3,))
    )
    _ = classifier.classify(
        _input(frame_index=1, track_id=None, pose=None, live_track_ids=(3,))
    )
    _ = classifier.classify(
        _input(frame_index=2, track_id=3, pose=_pose(), live_track_ids=(3,))
    )

    # Then
    assert len(model.inputs) == 1
    middle_row = model.inputs[0][1]
    assert middle_row == (0.0,) * 51


def test_evicted_track_cannot_reuse_probability_or_window() -> None:
    # Given
    model = _Model(_Metadata(window=2, stride=1, mode="sequence"), (0.9, 0.1), 0.5)
    classifier = FallWindowClassifier(model)
    first_track = _input(frame_index=0, track_id=1, pose=_pose(x=80), live_track_ids=(1,))

    # When
    _ = classifier.classify(first_track)
    positive = classifier.classify(first_track)
    _ = classifier.classify(
        _input(frame_index=1, track_id=None, pose=None, live_track_ids=())
    )
    new_track_first = classifier.classify(
        _input(frame_index=2, track_id=2, pose=_pose(x=10), live_track_ids=(2,))
    )
    new_track_second = classifier.classify(
        _input(frame_index=3, track_id=2, pose=_pose(x=10), live_track_ids=(2,))
    )

    # Then
    assert positive.labels[0].is_fall
    assert new_track_first.labels[0].confidence == 0.0
    assert not new_track_first.labels[0].is_fall
    assert new_track_second.labels[0].confidence == 0.1
    assert len(model.inputs) == 2


def test_stride_reuses_probability_until_next_due_frame() -> None:
    # Given
    model = _Model(_Metadata(window=2, stride=2, mode="sequence"), (0.8, 0.2), 0.5)
    classifier = FallWindowClassifier(model)

    # When
    results = tuple(
        classifier.classify(
            _input(frame_index=index, track_id=4, pose=_pose(), live_track_ids=(4,))
        )
        for index in range(4)
    )

    # Then
    assert tuple(result.labels[0].confidence for result in results) == (0.0, 0.8, 0.8, 0.2)
    assert len(model.inputs) == 2
