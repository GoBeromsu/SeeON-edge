"""Structural contracts for the inactive V2 fall temporal policy."""

from __future__ import annotations

import pytest

from worker.domains.fall import FallPolicyDeciderV2, FallV2Probabilities


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
    decider.update({}, (), frame_index=45, time_sec=45.0)

    _update(decider, _probability(0.0), 46)
    assert decider.generation_for(7) == 1


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
