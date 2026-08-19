"""Pose-feature bed-exit state machine (shadow, non-emitting).

States: ABSENT | IN_BED | SITTING_UP | EDGE_SITTING | OUT_OF_BED | UNCERTAIN.

This module replaces the area-containment rule that scores an edge-sitting
resident at 0.50 and therefore never accumulates grace. It is wired in
shadow mode only: it records ``DecisionTraceSnapshot``s and never emits a
``BusinessEvent``. The legacy containment path remains the only emitter.

Why the bands ignore ``torso_angle`` as a gate
----------------------------------------------
The plan proposed ``IN_BED: torso_angle <= 0.61``. A recumbent pose lying
along the image y-axis produces ``torso_angle = atan2(|dy|, |dx|) = π/2``,
which fails that band. Independently measured on a realistic camera view,
ALL FOUR postures returned ``torso_angle = 1.571``. The angle is therefore
camera-dependent and does not discriminate. Bands are built on
``lower_in_frac`` + ``hip_depth`` + ``torso_in_frac``. ``torso_angle`` is
traced as optional corroboration only.

Guard order (non-negotiable)
----------------------------
1. ``bed_polygon_valid is False`` -> no state decision. Must run BEFORE any
   numeric band: an invalid polygon zeros every bed-relative field, so
   ``hip_depth == 0.0`` would otherwise satisfy the EDGE band
   ``|hip_depth| <= 0.08``.
2. ``observability < 0.35`` -> UNCERTAIN. This is the only correct answer
   to a blanket-occluded / missing-pose resident on a *valid* polygon
   (all-zero fractions with ``bed_polygon_valid=True``). Without it,
   occlusion is silently coerced into a confident state.
3. Only then the posture bands.

Measured feature values this machine is tuned against
-----------------------------------------------------
IN_BED          torso=1.000  lower=1.000  hip_depth=+0.257
SITTING_UP      torso=1.000  lower=1.000  hip_depth=+0.236
EDGE_SITTING    torso=1.000  lower=0.000  hip_depth=+0.043   (rim-perched)
OUT_OF_BED      torso=0.000  lower=0.000  hip_depth=-0.289

Todo 6's EDGE fixture (hips mid-mattress, hip_depth=+0.233) is NOT an
edge-sitting case and must not reach EDGE_SITTING. That is intentional.

Dwell seconds (converted through TemporalProfile, never hardcoded frames)
------------------------------------------------------------------------
->IN_BED                  2.0 s
IN_BED -> SITTING_UP      0.6 s
SITTING_UP -> EDGE        0.6 s
EDGE -> OUT_OF_BED        0.4 s
->UNCERTAIN               1.0 s
UNCERTAIN ->              1.0 s
other confirmed hops      0.4 s  (smallest defensible choice: same as
                                  EDGE->OUT so a clean OUT hop is not
                                  slower than the documented exit)

Hysteresis is the asymmetric 0.25 / 0.80 ``torso_in_frac`` pair plus the
``lower_in_frac`` 0.35 / 0.50 split. ``ABSENT`` is driven by
``live_track_ids``, never by dwell -- the tracker already coasts
``max_misses=30``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from worker.types import (
    CURRENT_TEMPORAL_PROFILE,
    BedPoseFeatures,
    DecisionTraceSnapshot,
    TemporalProfile,
)

# Observability below this is UNCERTAIN (plan proposal; catches missing pose).
_MIN_OBSERVABILITY: Final = 0.35

# Posture bands. Built on lower_in_frac + hip_depth + torso_in_frac.
# torso_angle is NEVER a gate -- see module docstring.
# Deep-inside split between measured IN_BED (+0.257) and SITTING_UP (+0.236).
# Without torso_angle the plan's IN_BED/SITTING_UP bands overlap; this is the
# only axis that still separates the two measured postures.
_IN_BED_HIP_DEPTH_MIN: Final = 0.24
_IN_BED_TORSO_MIN: Final = 0.80
_IN_BED_LOWER_MIN: Final = 0.50

# E2 sitting-up shares EDGE's torso=0.55 / hip_depth=0.0 and differs only by
# lower_in_frac=0.80, so the sitting-up floor is the EDGE torso floor, not 0.70.
_SITTING_UP_TORSO_MIN: Final = 0.50
_SITTING_UP_LOWER_MIN: Final = 0.50

_EDGE_TORSO_MIN: Final = 0.50
_EDGE_HIP_DEPTH_ABS_MAX: Final = 0.08
_EDGE_LOWER_MAX: Final = 0.35

_OUT_TORSO_MAX: Final = 0.25
_OUT_HIP_DEPTH_MAX: Final = -0.05
_OUT_KEYPOINT_MAX: Final = 0.30

# Dwell durations in seconds. Converted via TemporalProfile.frames_for_seconds.
_DWELL_ENTER_IN_BED_SEC: Final = 2.0
_DWELL_IN_BED_TO_SITTING_UP_SEC: Final = 0.6
_DWELL_SITTING_UP_TO_EDGE_SEC: Final = 0.6
_DWELL_EDGE_TO_OUT_SEC: Final = 0.4
_DWELL_ENTER_UNCERTAIN_SEC: Final = 1.0
_DWELL_LEAVE_UNCERTAIN_SEC: Final = 1.0
_DWELL_DEFAULT_SEC: Final = 0.4


class BedExitState(StrEnum):
    """Live posture / occupancy state of one track relative to one bed."""

    ABSENT = "absent"
    IN_BED = "in-bed"
    SITTING_UP = "sitting-up"
    EDGE_SITTING = "edge-sitting"
    OUT_OF_BED = "out-of-bed"
    UNCERTAIN = "uncertain"


# Instantaneous classification when the polygon is valid. Distinct from the
# committed BedExitState so a dwell can hold the previous committed state.
class _Instant(StrEnum):
    UNCERTAIN = "uncertain"
    IN_BED = "in-bed"
    SITTING_UP = "sitting-up"
    EDGE_SITTING = "edge-sitting"
    OUT_OF_BED = "out-of-bed"
    UNRESOLVED = "unresolved"


_STATE_FROM_INSTANT: Final[dict[_Instant, BedExitState]] = {
    _Instant.UNCERTAIN: BedExitState.UNCERTAIN,
    _Instant.IN_BED: BedExitState.IN_BED,
    _Instant.SITTING_UP: BedExitState.SITTING_UP,
    _Instant.EDGE_SITTING: BedExitState.EDGE_SITTING,
    _Instant.OUT_OF_BED: BedExitState.OUT_OF_BED,
}

_HOLD_REASON: Final[dict[BedExitState, str]] = {
    BedExitState.ABSENT: "absent-hold",
    BedExitState.IN_BED: "in-bed-hold",
    BedExitState.SITTING_UP: "sitting-up-hold",
    BedExitState.EDGE_SITTING: "edge-sitting-hold",
    BedExitState.OUT_OF_BED: "out-of-bed-hold",
    BedExitState.UNCERTAIN: "uncertain-hold",
}

_ENTER_REASON: Final[dict[BedExitState, str]] = {
    BedExitState.ABSENT: "entered-absent",
    BedExitState.IN_BED: "entered-in-bed",
    BedExitState.SITTING_UP: "entered-sitting-up",
    BedExitState.EDGE_SITTING: "entered-edge-sitting",
    BedExitState.OUT_OF_BED: "entered-out-of-bed",
    BedExitState.UNCERTAIN: "entered-uncertain",
}

_EXIT_STATES: Final[frozenset[BedExitState]] = frozenset(
    {BedExitState.EDGE_SITTING, BedExitState.OUT_OF_BED}
)


@dataclass(frozen=True, slots=True)
class BedExitStateDecision:
    """One frame of committed state for one track. Never an event."""

    track_id: int
    bed_id: int | None
    previous_state: BedExitState
    current_state: BedExitState
    reason: str
    dwell_frames: int
    dwell_threshold: int
    would_trigger: bool
    snapshot: DecisionTraceSnapshot


@dataclass
class _TrackState:
    state: BedExitState = BedExitState.ABSENT
    bed_id: int | None = None
    candidate: _Instant | None = None
    candidate_frames: int = 0
    last_live_state: BedExitState | None = None
    last_live_bed_id: int | None = None
    fired: bool = False


class BedExitStateMachine:
    """Per-camera posture machine over ``BedPoseFeatures``.

    Shadow contract: ``observe`` / ``mark_absent`` / ``coast`` never emit
    events. ``would_trigger`` on a decision is a recorded predicate only.
    """

    def __init__(self, *, temporal_profile: TemporalProfile | None = None) -> None:
        self._profile: TemporalProfile = (
            CURRENT_TEMPORAL_PROFILE if temporal_profile is None else temporal_profile
        )
        self._tracks: dict[int, _TrackState] = {}

    @property
    def temporal_profile(self) -> TemporalProfile:
        return self._profile

    def dwell_frames(self, seconds: float) -> int:
        return self._profile.frames_for_seconds(seconds)

    def coast(self) -> tuple[()]:
        """Hold every track's committed state across a missing inference."""
        return ()

    def track_state(self, track_id: int) -> BedExitState:
        track = self._tracks.get(track_id)
        return BedExitState.ABSENT if track is None else track.state

    def known_track_ids(self) -> tuple[int, ...]:
        return tuple(self._tracks)

    def observe(self, features: BedPoseFeatures) -> BedExitStateDecision | None:
        """Advance one live track. ``None`` means no state decision."""
        if not features.bed_polygon_valid:
            # All-zeros trap (i): do not invent EDGE_SITTING from hip_depth=0,
            # and do not register the track. No state decision.
            return None
        track = self._tracks.setdefault(features.track_id, _TrackState())
        previous = track.state

        instant = _classify(features)
        threshold = self._dwell_threshold(previous, instant)
        committed, dwell_frames = self._apply_dwell(track, instant, threshold)
        entered = committed != previous
        if committed is not BedExitState.ABSENT:
            track.last_live_state = committed
            track.last_live_bed_id = features.bed_id
        if committed is BedExitState.IN_BED:
            # A confirmed return to bed arms a fresh exit; rising-edge only.
            track.fired = False
        would_trigger = (
            entered
            and committed is BedExitState.OUT_OF_BED
            and previous in (BedExitState.EDGE_SITTING, BedExitState.SITTING_UP, BedExitState.IN_BED)
            and not track.fired
        )
        if would_trigger:
            track.fired = True
        reason = _ENTER_REASON[committed] if entered else _HOLD_REASON[committed]
        snapshot = _snapshot(
            features=features,
            previous=previous,
            current=committed,
            reason=reason,
            dwell_frames=dwell_frames,
            dwell_threshold=threshold,
            triggered=False,
        )
        return BedExitStateDecision(
            track_id=features.track_id,
            bed_id=features.bed_id,
            previous_state=previous,
            current_state=committed,
            reason=reason,
            dwell_frames=dwell_frames,
            dwell_threshold=threshold,
            would_trigger=would_trigger,
            snapshot=snapshot,
        )

    def mark_absent(self, track_id: int) -> BedExitStateDecision | None:
        """Drive ABSENT from live_track_ids. Track loss may satisfy trigger."""
        track = self._tracks.get(track_id)
        if track is None:
            return None
        previous = track.state
        last_live = track.last_live_state
        last_bed = track.last_live_bed_id
        would_trigger = last_live in _EXIT_STATES and not track.fired
        if would_trigger:
            track.fired = True
        del self._tracks[track_id]
        reason = "entered-absent"
        snapshot = DecisionTraceSnapshot(
            reason=reason,
            previous_state=previous.value,
            current_state=BedExitState.ABSENT.value,
            triggered=False,
            track_id=track_id,
            bed_id=last_bed,
            values={
                "dwell_frames": 0,
                "dwell_threshold": 0,
            },
            missing_values={
                "torso_in_frac": "track-no-longer-live",
                "lower_in_frac": "track-no-longer-live",
                "hip_depth": "track-no-longer-live",
                "observability": "track-no-longer-live",
            },
        )
        return BedExitStateDecision(
            track_id=track_id,
            bed_id=last_bed,
            previous_state=previous,
            current_state=BedExitState.ABSENT,
            reason=reason,
            dwell_frames=0,
            dwell_threshold=0,
            would_trigger=would_trigger,
            snapshot=snapshot,
        )

    def _dwell_threshold(self, previous: BedExitState, instant: _Instant) -> int:
        target = _STATE_FROM_INSTANT.get(instant)
        if target is None:
            return self.dwell_frames(_DWELL_DEFAULT_SEC)
        if previous is target:
            return 1
        if target is BedExitState.IN_BED:
            return self.dwell_frames(_DWELL_ENTER_IN_BED_SEC)
        if target is BedExitState.UNCERTAIN:
            return self.dwell_frames(_DWELL_ENTER_UNCERTAIN_SEC)
        if previous is BedExitState.UNCERTAIN:
            return self.dwell_frames(_DWELL_LEAVE_UNCERTAIN_SEC)
        if previous is BedExitState.IN_BED and target is BedExitState.SITTING_UP:
            return self.dwell_frames(_DWELL_IN_BED_TO_SITTING_UP_SEC)
        if previous is BedExitState.SITTING_UP and target is BedExitState.EDGE_SITTING:
            return self.dwell_frames(_DWELL_SITTING_UP_TO_EDGE_SEC)
        if previous is BedExitState.EDGE_SITTING and target is BedExitState.OUT_OF_BED:
            return self.dwell_frames(_DWELL_EDGE_TO_OUT_SEC)
        return self.dwell_frames(_DWELL_DEFAULT_SEC)

    def _apply_dwell(
        self,
        track: _TrackState,
        instant: _Instant,
        threshold: int,
    ) -> tuple[BedExitState, int]:
        target = _STATE_FROM_INSTANT.get(instant)
        if target is None:
            track.candidate = None
            track.candidate_frames = 0
            return track.state, 0
        if track.state is target:
            track.candidate = instant
            track.candidate_frames = 0
            return track.state, 0
        if track.candidate is instant:
            track.candidate_frames += 1
        else:
            track.candidate = instant
            track.candidate_frames = 1
        if track.candidate_frames >= threshold:
            track.state = target
            track.candidate = None
            track.candidate_frames = 0
            return track.state, threshold
        return track.state, track.candidate_frames


def classify_posture(features: BedPoseFeatures) -> BedExitState | None:
    """Instantaneous classification with the same guard order as the machine.

    Returns ``None`` when the polygon is invalid (no state decision).
    """
    if not features.bed_polygon_valid:
        return None
    instant = _classify(features)
    if instant is _Instant.UNRESOLVED:
        return None
    return _STATE_FROM_INSTANT[instant]


def _classify(features: BedPoseFeatures) -> _Instant:
    # Guard (2): low observability, including valid-polygon missing pose.
    if features.observability < _MIN_OBSERVABILITY:
        return _Instant.UNCERTAIN

    torso = features.torso_in_frac
    lower = features.lower_in_frac
    hip = features.hip_depth
    keypoints = features.keypoint_in_frac

    # EDGE first: rim-perched hips (hip_depth ~ +0.043) with legs off
    # (lower ~ 0). Todo 6's mid-mattress "EDGE" fixture (hip=+0.233)
    # fails |hip| <= 0.08 and must NOT land here.
    if (
        torso >= _EDGE_TORSO_MIN
        and abs(hip) <= _EDGE_HIP_DEPTH_ABS_MAX
        and lower <= _EDGE_LOWER_MAX
    ):
        return _Instant.EDGE_SITTING
    if torso <= _OUT_TORSO_MAX and hip <= _OUT_HIP_DEPTH_MAX and keypoints <= _OUT_KEYPOINT_MAX:
        return _Instant.OUT_OF_BED
    if torso >= _IN_BED_TORSO_MIN and hip >= _IN_BED_HIP_DEPTH_MIN and lower >= _IN_BED_LOWER_MIN:
        return _Instant.IN_BED
    if torso >= _SITTING_UP_TORSO_MIN and lower >= _SITTING_UP_LOWER_MIN:
        return _Instant.SITTING_UP
    # Hysteresis leftovers (e.g. mid-transition, or mid-mattress legs-off)
    # stay unresolved so the committed state holds.
    return _Instant.UNRESOLVED


def _snapshot(
    *,
    features: BedPoseFeatures,
    previous: BedExitState,
    current: BedExitState,
    reason: str,
    dwell_frames: int,
    dwell_threshold: int,
    triggered: bool,
) -> DecisionTraceSnapshot:
    values: dict[str, float | int] = {
        "torso_in_frac": features.torso_in_frac,
        "lower_in_frac": features.lower_in_frac,
        "keypoint_in_frac": features.keypoint_in_frac,
        "hip_depth": features.hip_depth,
        "torso_angle": features.torso_angle,
        "centroid_displacement": features.centroid_displacement,
        "hip_x_rel": features.hip_x_rel,
        "hip_y_rel": features.hip_y_rel,
        "observability": features.observability,
        "dwell_frames": dwell_frames,
        "dwell_threshold": dwell_threshold,
    }
    if features.bed_id is not None:
        values["bed_id"] = features.bed_id
    return DecisionTraceSnapshot(
        reason=reason,
        previous_state=previous.value,
        current_state=current.value,
        triggered=triggered,
        track_id=features.track_id,
        bed_id=features.bed_id,
        values=values,
    )


__all__ = [
    "BedExitState",
    "BedExitStateDecision",
    "BedExitStateMachine",
    "classify_posture",
]
