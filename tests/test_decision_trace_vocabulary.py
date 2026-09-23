"""Closed decision-trace vocabulary: baseline membership plus the additive bed-exit,
fall (classifier dispositions, resample-gap) and episode-suppression tokens."""

from __future__ import annotations

from enum import StrEnum

import pytest

from worker.types import DecisionTraceSnapshot
from worker.types.trace import (
    DecisionTraceMissingReason,
    DecisionTraceReason,
    DecisionTraceState,
    DecisionTraceValueName,
    canonical_trace_number,
)

# Frozen membership of the four closed vocabularies at the commit this test
# was introduced against. Live snapshots reference these tokens; they must
# remain a subset of whatever the enums grow into.
BASELINE_REASONS: frozenset[str] = frozenset(
    {
        "trace-unavailable",
        "outside-detection-window",
        "score-missing",
        "fall-onset",
        "fall-active",
        "below-threshold",
        "bed-region-unavailable",
        "bed-observation-missing",
        "stale-track-exit",
        "stale-track-clear",
        "assigned",
        "assignment-hold",
        "below-containment",
        "contained",
        "contained-in-other-bed",
        "live-grace-exit",
        "live-grace",
        "person-observation-missing",
    }
)
BASELINE_STATES: frozenset[str] = frozenset(
    {
        "unknown",
        "not-evaluated",
        "no-decision",
        "clear",
        "fall",
        "live-grace",
        "contained",
        "triggered",
        "retired",
        "unassigned",
        "other-bed",
    }
)
BASELINE_VALUE_NAMES: frozenset[str] = frozenset(
    {
        "operating_threshold",
        "window_frames",
        "fall_probability",
        "containment_ratio",
        "max_other_containment_ratio",
        "min_containment",
        "candidate_frames",
        "hold_frames_threshold",
        "grace_frames_before",
        "grace_frames_after",
        "grace_threshold",
        "bed_id",
        "decision_state",
    }
)
BASELINE_MISSING_REASONS: frozenset[str] = frozenset(
    {
        "adapter-not-provided",
        "adapter-returned-no-data",
        "outside-detection-window",
        "no-live-classified-track",
        "bed-region-unavailable",
        "bed-observation-missing",
        "track-no-longer-live",
        "no-observed-person",
    }
)

BED_EXIT_STATES: frozenset[str] = frozenset(
    {
        "in-bed",
        "sitting-up",
        "edge-sitting",
        "out-of-bed",
        "uncertain",
        "absent",
    }
)
BED_EXIT_VALUE_NAMES: frozenset[str] = frozenset(
    {
        "torso_in_frac",
        "lower_in_frac",
        "keypoint_in_frac",
        "hip_depth",
        "torso_angle",
        "centroid_displacement",
        "hip_x_rel",
        "hip_y_rel",
        "observability",
        "dwell_frames",
        "dwell_threshold",
    }
)
BED_EXIT_MISSING_REASONS: frozenset[str] = frozenset(
    {
        "pose-unavailable",
        "bed-polygon-invalid",
    }
)
FALL_V2_REASONS: frozenset[str] = frozenset(
    {
        "transition-candidate",
        "transition-confirmed",
        "fall-recovered",
    }
)
FALL_V2_STATES: frozenset[str] = frozenset(
    {
        "transition-candidate",
        "transition-confirmed",
        "fallen",
    }
)
FALL_VALUE_NAMES: frozenset[str] = frozenset(
    {
        "fall_transition_probability",
        "fallen_probability",
        "transition_threshold",
        "transition_votes",
        "transition_window",
    }
)
FALL_MISSING_REASONS: frozenset[str] = frozenset(
    {
        "classifier-warmup",
        "classifier-stride-not-due",
        # A frame the resampler yielded no row for: the module coasted.
        "resample-gap",
    }
)
EPISODE_REASONS: frozenset[str] = frozenset(
    {
        # Onset computed but suppressed by the episode authority; the row
        # keeps triggered=False so a non-event is explained, not silent.
        "episode-already-open",
        "episode-reassociated",
        "episode-resolved-hold",
        "episode-candidate",
    }
)
BED_EXIT_REASONS: frozenset[str] = frozenset(
    {
        "in-bed-hold",
        "sitting-up-hold",
        "edge-sitting-hold",
        "out-of-bed-hold",
        "uncertain-hold",
        "absent-hold",
        "entered-in-bed",
        "entered-sitting-up",
        "entered-edge-sitting",
        "entered-out-of-bed",
        "entered-uncertain",
        "entered-absent",
        "pose-unavailable",
        "bed-polygon-invalid",
    }
)

_HOLD_REASON_BY_STATE: dict[str, str] = {
    "in-bed": "in-bed-hold",
    "sitting-up": "sitting-up-hold",
    "edge-sitting": "edge-sitting-hold",
    "out-of-bed": "out-of-bed-hold",
    "uncertain": "uncertain-hold",
    "absent": "absent-hold",
}
_ENTRY_REASON_BY_STATE: dict[str, str] = {
    "in-bed": "entered-in-bed",
    "sitting-up": "entered-sitting-up",
    "edge-sitting": "entered-edge-sitting",
    "out-of-bed": "entered-out-of-bed",
    "uncertain": "entered-uncertain",
    "absent": "entered-absent",
}


def _values(enum_type: type[StrEnum]) -> frozenset[str]:
    return frozenset(member.value for member in enum_type)


def _new_token_snapshots() -> tuple[DecisionTraceSnapshot, ...]:
    snapshots: list[DecisionTraceSnapshot] = []
    for state, reason in _HOLD_REASON_BY_STATE.items():
        snapshots.append(
            DecisionTraceSnapshot(
                reason=reason,
                previous_state=state,
                current_state=state,
                triggered=False,
                track_id=5,
                bed_id=0,
                values={
                    "torso_in_frac": 0.81,
                    "lower_in_frac": 0.74,
                    "keypoint_in_frac": 0.66,
                    "hip_depth": 0.05,
                    "torso_angle": 0.41,
                    "centroid_displacement": 0.01,
                    "hip_x_rel": 0.48,
                    "hip_y_rel": 0.52,
                    "observability": 0.92,
                    "dwell_frames": 3,
                    "dwell_threshold": 5,
                },
            )
        )
    for state, reason in _ENTRY_REASON_BY_STATE.items():
        snapshots.append(
            DecisionTraceSnapshot(
                reason=reason,
                previous_state="absent" if state != "absent" else "uncertain",
                current_state=state,
                triggered=state == "out-of-bed",
                track_id=5,
                bed_id=0,
                values={"dwell_frames": 1, "dwell_threshold": 3},
            )
        )
    snapshots.append(
        DecisionTraceSnapshot(
            reason="pose-unavailable",
            previous_state="unknown",
            current_state="uncertain",
            triggered=False,
            track_id=5,
            bed_id=0,
            missing_values={
                "torso_in_frac": "pose-unavailable",
                "hip_depth": "pose-unavailable",
            },
        )
    )
    snapshots.append(
        DecisionTraceSnapshot(
            reason="bed-polygon-invalid",
            previous_state="unknown",
            current_state="no-decision",
            triggered=False,
            track_id=5,
            bed_id=None,
            missing_values={
                "torso_in_frac": "bed-polygon-invalid",
                "hip_depth": "bed-polygon-invalid",
            },
        )
    )
    return tuple(snapshots)


def test_baseline_closed_vocabularies_remain_intact() -> None:
    """Pin the pre-extension membership so persisted tokens cannot silently vanish."""
    assert _values(DecisionTraceReason) >= BASELINE_REASONS
    assert _values(DecisionTraceState) >= BASELINE_STATES
    assert _values(DecisionTraceValueName) >= BASELINE_VALUE_NAMES
    assert _values(DecisionTraceMissingReason) >= BASELINE_MISSING_REASONS


def test_baseline_vocabularies_are_exactly_the_pre_extension_sets() -> None:
    """Fail if a later change silently drops or renames a persisted token.

    Before the bed-exit extension this is an exact-set pin. After the
    extension the same names stay a subset (asserted above) and the
    additive tokens are asserted separately.
    """
    current_reasons = _values(DecisionTraceReason)
    current_states = _values(DecisionTraceState)
    current_value_names = _values(DecisionTraceValueName)
    current_missing = _values(DecisionTraceMissingReason)

    extra_reasons = current_reasons - BASELINE_REASONS
    extra_states = current_states - BASELINE_STATES
    extra_value_names = current_value_names - BASELINE_VALUE_NAMES
    extra_missing = current_missing - BASELINE_MISSING_REASONS

    if extra_reasons | extra_states | extra_value_names | extra_missing:
        assert extra_reasons == BED_EXIT_REASONS | FALL_V2_REASONS | EPISODE_REASONS
        assert extra_states == BED_EXIT_STATES | FALL_V2_STATES
        assert extra_value_names == BED_EXIT_VALUE_NAMES | FALL_VALUE_NAMES
        assert extra_missing == BED_EXIT_MISSING_REASONS | FALL_MISSING_REASONS
    else:
        assert current_reasons == BASELINE_REASONS
        assert current_states == BASELINE_STATES
        assert current_value_names == BASELINE_VALUE_NAMES
        assert current_missing == BASELINE_MISSING_REASONS


def test_bed_exit_and_fall_v2_tokens_are_additive_and_closed() -> None:
    assert (
        _values(DecisionTraceReason)
        == BASELINE_REASONS | BED_EXIT_REASONS | FALL_V2_REASONS | EPISODE_REASONS
    )
    assert _values(DecisionTraceState) == BASELINE_STATES | BED_EXIT_STATES | FALL_V2_STATES
    assert (
        _values(DecisionTraceValueName)
        == BASELINE_VALUE_NAMES | BED_EXIT_VALUE_NAMES | FALL_VALUE_NAMES
    )
    assert (
        _values(DecisionTraceMissingReason)
        == BASELINE_MISSING_REASONS | BED_EXIT_MISSING_REASONS | FALL_MISSING_REASONS
    )


def test_unknown_reason_still_raises() -> None:
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="not-a-real-token",
            previous_state="unknown",
            current_state="unknown",
            triggered=False,
            track_id=None,
            bed_id=None,
        )


def test_misspelled_new_token_fails_closed() -> None:
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="entered-inbed",
            previous_state="in-bed",
            current_state="in-bed",
            triggered=False,
            track_id=5,
            bed_id=0,
        )
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="in-bed-hold",
            previous_state="inbed",
            current_state="in-bed",
            triggered=False,
            track_id=5,
            bed_id=0,
        )
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="in-bed-hold",
            previous_state="in-bed",
            current_state="in-bed",
            triggered=False,
            track_id=5,
            bed_id=0,
            values={"torso_in_frac_": 0.8},
        )
    with pytest.raises(ValueError, match="compiled vocabulary"):
        DecisionTraceSnapshot(
            reason="pose-unavailable",
            previous_state="unknown",
            current_state="uncertain",
            triggered=False,
            track_id=5,
            bed_id=0,
            missing_values={"torso_in_frac": "pose_unavailable"},
        )


def test_canonical_trace_number_is_unchanged() -> None:
    assert canonical_trace_number(1) == 1
    assert type(canonical_trace_number(1)) is int
    assert canonical_trace_number(1.23456789) == 1.234568
    assert canonical_trace_number(-0.0) == 0.0
    assert str(canonical_trace_number(-0.0)) == "0.0"
    with pytest.raises(ValueError, match="finite"):
        canonical_trace_number(float("nan"))


def test_new_tokens_round_trip_through_trace_adapter_and_writer() -> None:
    snapshots = _new_token_snapshots()
    seen_reasons = {snapshot.reason for snapshot in snapshots}
    seen_states = {snapshot.current_state for snapshot in snapshots} | {
        snapshot.previous_state for snapshot in snapshots
    }
    seen_value_names: set[str] = set()
    seen_missing_reasons: set[str] = set()
    for snapshot in snapshots:
        seen_value_names.update(str(name) for name in snapshot.values)
        seen_value_names.update(str(name) for name in snapshot.missing_values)
        seen_missing_reasons.update(str(reason) for reason in snapshot.missing_values.values())

    assert seen_reasons >= BED_EXIT_REASONS
    assert seen_states >= BED_EXIT_STATES
    assert seen_value_names >= BED_EXIT_VALUE_NAMES
    assert seen_missing_reasons >= BED_EXIT_MISSING_REASONS

    recovered_by_key = {
        (
            str(snapshot.reason),
            str(snapshot.previous_state),
            str(snapshot.current_state),
            snapshot.triggered,
        ): snapshot
        for snapshot in snapshots
    }
    for original in snapshots:
        recovered_snapshot = recovered_by_key[
            (
                str(original.reason),
                str(original.previous_state),
                str(original.current_state),
                original.triggered,
            )
        ]
        for name, value in original.values.items():
            assert recovered_snapshot.values[name] == pytest.approx(float(value))
        for name, reason in original.missing_values.items():
            assert str(recovered_snapshot.missing_values[name]) == str(reason)
