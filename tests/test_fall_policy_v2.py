"""Structural contracts for the inactive V2 fall temporal policy."""

from __future__ import annotations

import json
from pathlib import Path

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
    decider = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot",
        source_generation=0,
        stream_epoch="epoch",
    )

    assert _update(decider, _probability(0.7), 0) == ()
    assert _update(decider, _probability(0.7), 1) == ()
    event = _update(decider, _probability(0.7), 2)[0]

    assert event.event_type == "fall"
    assert event.person_id == 7
    assert event.identity == "boot:epoch:7:0:0:1"
    assert event.probability == 0.7
    assert _update(decider, _probability(0.9), 3) == ()


def test_fallen_is_internal_and_starting_fallen_does_not_alert() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot",
        source_generation=0,
        stream_epoch="epoch",
    )

    assert _update(decider, _probability(0.0, 0.8), 0) == ()
    assert decider.is_fallen(7)
    for frame in range(1, 4):
        assert _update(decider, _probability(0.0, 0.8), frame) == ()


def test_recovery_requires_five_joint_clear_scores() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot",
        source_generation=0,
        stream_epoch="epoch",
    )
    for frame in range(3):
        _update(decider, _probability(0.0, 0.8), frame)
    assert decider.is_fallen(7)

    for frame in range(3, 7):
        _update(decider, _probability(0.39, 0.49), frame)
        assert decider.is_fallen(7)
    _update(decider, _probability(0.39, 0.49), 7)
    assert not decider.is_fallen(7)


def test_eviction_reconnects_with_a_new_generation() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot",
        source_generation=0,
        stream_epoch="epoch",
    )
    _update(decider, _probability(0.0), 0)
    decider.update({}, (), frame_index=44, time_sec=44.0)
    assert decider.generation_for(7) == 0
    decider.update({}, (), frame_index=45, time_sec=45.0)
    assert decider.generation_for(7) is None

    _update(decider, _probability(0.0), 46)
    assert decider.generation_for(7) == 1


def test_nonlive_track_cannot_confirm_an_alert_and_reconnect_before_ttl_keeps_generation() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot",
        source_generation=0,
        stream_epoch="epoch",
    )
    for frame in range(2):
        assert _update(decider, _probability(0.7), frame) == ()

    # A score associated with an absent identity is not a camera-OR candidate.
    assert decider.update({7: _probability(1.0)}, (), frame_index=2, time_sec=2.0) == ()
    assert decider.generation_for(7) == 0

    event = _update(decider, _probability(0.7), 3)[0]
    assert event.identity == "boot:epoch:7:0:0:1"


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


def test_committed_reconnect_after_eviction_case_preloads_a_fresh_generation_window() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures_fall_pose_bbox56_v1.json").read_text())
    reconnect_case = next(
        case for case in fixture["raw_cases"] if case["case_id"] == "reconnect-after-eviction"
    )
    representative_rows = tuple(tuple(row) for row in reconnect_case["expected_windows"][0]["rows"])
    reconnect_row = representative_rows[-1]
    model = _RecordingModel()
    classifier = FallWindowClassifierV2(model)

    # Align the initial live observation so the exact reconnect tick is stride
    # due: 3 idle ticks + first live tick + 45 absent ticks + reconnect tick.
    for _ in range(3):
        classifier.update({}, ())
    classifier.update({7: reconnect_row}, (7,))
    assert classifier.generation_for(7) == 0
    for _ in range(45):
        classifier.update({}, ())
    assert classifier.generation_for(7) is None

    due = classifier.update({7: reconnect_row}, (7,))

    assert classifier.generation_for(7) == 1
    assert due[7] == _probability(0.0)
    assert len(model.windows[-1]) == 30
    assert model.windows[-1][:29] == ((0.0,) * 56,) * 29
    assert model.windows[-1][-1] == reconnect_row


def test_release_does_not_reopen_an_emitted_episode() -> None:
    decider = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot",
        source_generation=0,
        stream_epoch="epoch",
    )
    for frame in range(2):
        _update(decider, _probability(0.7), frame)
    event = _update(decider, _probability(0.7), 2)[0]

    decider.release_onset(event)
    assert _update(decider, _probability(0.7), 3) == ()
    decider.release_onset(event)
    assert _update(decider, _probability(0.7), 4) == ()


def test_rejects_nonfinite_or_wrong_arity_outputs() -> None:
    with pytest.raises(ValueError):
        FallV2Probabilities(0.0, float("nan"), 0.0)


def test_policy_requires_immutable_boot_and_epoch_and_binds_onset_identity() -> None:
    with pytest.raises(TypeError):
        FallPolicyDeciderV2(camera_id="camera", facility_id="facility")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="boot and source epoch"):
        FallPolicyDeciderV2(
            camera_id="camera",
            facility_id="facility",
            boot_id="",
            source_generation=0,
            stream_epoch="epoch",
        )

    first = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot-a",
        source_generation=0,
        stream_epoch="epoch-a",
    )
    second = FallPolicyDeciderV2(
        camera_id="camera",
        facility_id="facility",
        boot_id="boot-b",
        source_generation=0,
        stream_epoch="epoch-b",
    )
    for frame in range(2):
        _update(first, _probability(0.7), frame)
        _update(second, _probability(0.7), frame)

    assert _update(first, _probability(0.7), 2)[0].identity == "boot-a:epoch-a:7:0:0:1"
    assert _update(second, _probability(0.7), 2)[0].identity == "boot-b:epoch-b:7:0:0:1"
    first.update({}, (), frame_index=47, time_sec=47.0)
    for frame in range(48, 53):
        _update(first, _probability(0.1, fallen=0.1), frame)
    for frame in range(53, 55):
        _update(first, _probability(0.7), frame)
    assert _update(first, _probability(0.7), 55)[0].identity == "boot-a:epoch-a:7:0:1:2"
