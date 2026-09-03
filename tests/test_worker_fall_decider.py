from __future__ import annotations

from shared.detection_policies import FallPolicyV2
from worker.domains.fall.policy_v2 import FallPolicyDeciderV2
from worker.domains.fall.pose_bbox56 import pose_bbox56_row
from worker.interfaces.fall_model import FallV2Probabilities

_CAMERA_ID = "camera-fall"
_FACILITY_ID = "facility-fall"
_BOOT_ID = "boot-fall"


def _policy(**overrides: object) -> FallPolicyV2:
    values = {
        "transition_threshold": 0.5,
        "transition_votes": 2,
        "transition_window": 2,
        "fallen_threshold": 0.8,
        "fallen_consecutive": 2,
        "recovery_transition_max": 0.4,
        "recovery_fallen_max": 0.5,
        "recovery_consecutive": 2,
        "track_ttl_frames": 45,
        "cooldown_frames": 1,
    }
    values.update(overrides)
    return FallPolicyV2(**values)


def _decider(policy: FallPolicyV2 | None = None) -> FallPolicyDeciderV2:
    return FallPolicyDeciderV2(
        camera_id=_CAMERA_ID,
        facility_id=_FACILITY_ID,
        boot_id=_BOOT_ID,
        stream_epoch="epoch-1",
        source_generation=0,
        policy=policy or _policy(),
    )


def test_transition_votes_emit_one_rising_edge_at_default_threshold() -> None:
    decider = _decider()
    transition = FallV2Probabilities(0.5, 0.5, 0.0)

    # The first qualifying frame is the first vote (plan G7: CANDIDATE on the
    # first qualifying proposal, OPEN after transition_votes), so a 2-of-2
    # policy promotes on the second frame -- there is no silent warm-up frame.
    assert decider.update({7: transition}, (7,), frame_index=1, time_sec=1.0) == ()
    events = decider.update({7: transition}, (7,), frame_index=2, time_sec=2.0)

    assert len(events) == 1
    assert events[0].domain == "fall"
    assert events[0].event_type == "fall"
    assert events[0].probability == 0.5
    assert decider.update({7: transition}, (7,), frame_index=3, time_sec=3.0) == ()


def test_fallen_state_recovers_only_after_configured_safe_streak() -> None:
    decider = _decider()
    fallen = FallV2Probabilities(0.05, 0.9, 0.9)
    safe = FallV2Probabilities(0.9, 0.1, 0.1)

    decider.update({4: fallen}, (4,), frame_index=1, time_sec=1.0)
    decider.update({4: fallen}, (4,), frame_index=2, time_sec=2.0)
    assert decider.is_fallen(4)
    decider.update({4: safe}, (4,), frame_index=3, time_sec=3.0)
    assert decider.is_fallen(4)
    decider.update({4: safe}, (4,), frame_index=4, time_sec=4.0)
    assert not decider.is_fallen(4)


def test_confirmed_recovery_rearms_the_shared_episode_authority() -> None:
    decider = _decider(_policy(transition_votes=1, transition_window=1))
    transition = FallV2Probabilities(0.5, 0.5, 0.0)
    safe = FallV2Probabilities(0.9, 0.1, 0.1)

    # A 1-of-1 policy opens on the first qualifying frame; the episode then
    # stays silent until a confirmed recovery (2 consecutive clear scores)
    # re-arms it, and only then may a second episode open.
    event = decider.update({9: transition}, (9,), frame_index=1, time_sec=1.0)[0]
    assert decider.update({9: transition}, (9,), frame_index=2, time_sec=2.0) == ()
    assert decider.update({9: safe}, (9,), frame_index=3, time_sec=3.0) == ()
    assert decider.update({9: safe}, (9,), frame_index=4, time_sec=4.0) == ()
    retry = decider.update({9: transition}, (9,), frame_index=5, time_sec=5.0)

    assert retry[0].identity != event.identity
    assert retry[0].time_sec == 5.0


def test_pose_bbox56_encoder_produces_exactly_30_by_56_compatible_rows() -> None:
    keypoints = tuple((20.0 + index, 30.0 + index, 0.9) for index in range(17))
    row = pose_bbox56_row(keypoints, (10.0, 15.0, 110.0, 115.0), 200, 150)

    assert len(row) == 56
    assert row[55] == 1.0
    assert len(tuple(row for _ in range(30))) == 30
