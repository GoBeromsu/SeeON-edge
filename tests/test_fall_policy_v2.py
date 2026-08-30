"""Structural contracts for the inactive V2 fall temporal policy."""

from __future__ import annotations

import pytest

from worker.domains.fall import (
    FallPolicyDeciderV2,
    FallV2Probabilities,
    FallWindowClassifierV2,
)


def _probability(transition: float, fallen: float = 0.0) -> FallV2Probabilities:
    return FallV2Probabilities(0.0, transition, fallen)


def _update(
    decider: FallPolicyDeciderV2,
    probability: FallV2Probabilities,
    frame: int,
    track: int = 7,
) -> tuple:
    return decider.update({track: probability}, (track,), frame_index=frame, time_sec=float(frame))


def test_transition_confirmation_emits_once_with_deterministic_camera_winner() -> None:
    decider = FallPolicyDeciderV2(camera_id="camera", facility_id="facility")

    assert _update(decider, _probability(0.7), 0) == ()
    assert _update(decider, _probability(0.7), 1) == ()
    event = _update(decider, _probability(0.7), 2)[0]

    assert event.event_type == "fall"
    assert event.person_id == 7
    assert event.identity == "7:0"
    assert event.probability == 0.7
    assert _update(decider, _probability(0.9), 3) == ()


def test_fallen_is_internal_and_starting_fallen_does_not_alert() -> None:
    decider = FallPolicyDeciderV2(camera_id="camera", facility_id="facility")

    assert _update(decider, _probability(0.0, 0.8), 0) == ()
    assert decider.is_fallen(7)
    for frame in range(1, 4):
        assert _update(decider, _probability(0.0, 0.8), frame) == ()


def test_recovery_requires_five_joint_clear_scores() -> None:
    decider = FallPolicyDeciderV2(camera_id="camera", facility_id="facility")
    for frame in range(3):
        _update(decider, _probability(0.0, 0.8), frame)
    assert decider.is_fallen(7)

    for frame in range(3, 7):
        _update(decider, _probability(0.39, 0.49), frame)
        assert decider.is_fallen(7)
    _update(decider, _probability(0.39, 0.49), 7)
    assert not decider.is_fallen(7)


def test_eviction_reconnects_with_a_new_generation() -> None:
    decider = FallPolicyDeciderV2(camera_id="camera", facility_id="facility")
    _update(decider, _probability(0.0), 0)
    decider.update({}, (), frame_index=44, time_sec=44.0)
    assert decider.generation_for(7) == 0
    decider.update({}, (), frame_index=45, time_sec=45.0)
    assert decider.generation_for(7) is None

    _update(decider, _probability(0.0), 46)
    assert decider.generation_for(7) == 1


def test_nonlive_track_cannot_confirm_an_alert_and_reconnect_before_ttl_keeps_generation() -> None:
    decider = FallPolicyDeciderV2(camera_id="camera", facility_id="facility")
    for frame in range(2):
        assert _update(decider, _probability(0.7), frame) == ()

    # A score associated with an absent identity is not a camera-OR candidate.
    assert decider.update({7: _probability(1.0)}, (), frame_index=2, time_sec=2.0) == ()
    assert decider.generation_for(7) == 0

    event = _update(decider, _probability(0.7), 3)[0]
    assert event.identity == "7:0"


class _RecordingModel:
    def __init__(self) -> None:
        self.windows: list[tuple[tuple[float, ...], ...]] = []

    def predict(self, features: object) -> FallV2Probabilities:
        assert isinstance(features, tuple)
        self.windows.append(features)
        return _probability(0.0)


def test_missing_live_coasts_until_exact_ttl_then_reconnect_zero_fills_fresh_window() -> None:
    model = _RecordingModel()
    classifier = FallWindowClassifierV2(model)
    row = (0.25,) * 56

    for _ in range(30):
        classifier.update({7: row}, (7,))
    assert model.windows[-1] == (row,) * 30

    # The first 44 absent ticks preserve the same resident and coast its row.
    for _ in range(44):
        classifier.update({}, ())
    assert classifier.probabilities_for(7) is not None
    classifier.update({7: None}, (7,))
    assert model.windows[-1] == (row,) * 30

    # Tick 45 after that reconnect is the exact expiration boundary.
    for _ in range(45):
        classifier.update({}, ())
    assert classifier.probabilities_for(7) is None

    # The reused number has no row history. Its new window is zero-filled and
    # therefore cannot carry a stale transition onset from the prior resident.
    for _ in range(30):
        classifier.update({7: None}, (7,))
    assert model.windows[-1] == ((0.0,) * 56,) * 30


def test_release_reopens_only_the_exact_failed_onset() -> None:
    decider = FallPolicyDeciderV2(camera_id="camera", facility_id="facility")
    for frame in range(2):
        _update(decider, _probability(0.7), frame)
    event = _update(decider, _probability(0.7), 2)[0]

    decider.release_onset(event)
    assert _update(decider, _probability(0.7), 3)[0].identity == event.identity
    decider.release_onset(event)
    assert _update(decider, _probability(0.7), 4) == ()


def test_rejects_nonfinite_or_wrong_arity_outputs() -> None:
    with pytest.raises(ValueError):
        FallV2Probabilities(0.0, float("nan"), 0.0)
