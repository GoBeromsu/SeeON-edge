from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final, TypeAlias, TypeVar, overload

NumericTraceValue: TypeAlias = int | float
TRACE_FLOAT_DECIMAL_PLACES: Final = 6


class DecisionTraceReason(StrEnum):
    """Finite reason vocabulary emitted by the compiled detection modules."""

    TRACE_UNAVAILABLE = "trace-unavailable"
    OUTSIDE_DETECTION_WINDOW = "outside-detection-window"
    SCORE_MISSING = "score-missing"
    FALL_ONSET = "fall-onset"
    FALL_ACTIVE = "fall-active"
    TRANSITION_CONFIRMED = "transition-confirmed"
    TRANSITION_CANDIDATE = "transition-candidate"
    FALL_RECOVERED = "fall-recovered"
    BELOW_THRESHOLD = "below-threshold"
    BED_REGION_UNAVAILABLE = "bed-region-unavailable"
    BED_OBSERVATION_MISSING = "bed-observation-missing"
    STALE_TRACK_EXIT = "stale-track-exit"
    STALE_TRACK_CLEAR = "stale-track-clear"
    ASSIGNED = "assigned"
    ASSIGNMENT_HOLD = "assignment-hold"
    BELOW_CONTAINMENT = "below-containment"
    CONTAINED = "contained"
    CONTAINED_IN_OTHER_BED = "contained-in-other-bed"
    LIVE_GRACE_EXIT = "live-grace-exit"
    LIVE_GRACE = "live-grace"
    PERSON_OBSERVATION_MISSING = "person-observation-missing"
    IN_BED_HOLD = "in-bed-hold"
    SITTING_UP_HOLD = "sitting-up-hold"
    EDGE_SITTING_HOLD = "edge-sitting-hold"
    OUT_OF_BED_HOLD = "out-of-bed-hold"
    UNCERTAIN_HOLD = "uncertain-hold"
    ABSENT_HOLD = "absent-hold"
    ENTERED_IN_BED = "entered-in-bed"
    ENTERED_SITTING_UP = "entered-sitting-up"
    ENTERED_EDGE_SITTING = "entered-edge-sitting"
    ENTERED_OUT_OF_BED = "entered-out-of-bed"
    ENTERED_UNCERTAIN = "entered-uncertain"
    ENTERED_ABSENT = "entered-absent"
    POSE_UNAVAILABLE = "pose-unavailable"
    BED_POLYGON_INVALID = "bed-polygon-invalid"


class DecisionTraceState(StrEnum):
    """Finite state vocabulary; values are persisted as stable qualified tokens."""

    UNKNOWN = "unknown"
    NOT_EVALUATED = "not-evaluated"
    NO_DECISION = "no-decision"
    CLEAR = "clear"
    FALL = "fall"
    TRANSITION_CANDIDATE = "transition-candidate"
    TRANSITION_CONFIRMED = "transition-confirmed"
    FALLEN = "fallen"
    LIVE_GRACE = "live-grace"
    CONTAINED = "contained"
    TRIGGERED = "triggered"
    RETIRED = "retired"
    UNASSIGNED = "unassigned"
    OTHER_BED = "other-bed"
    IN_BED = "in-bed"
    SITTING_UP = "sitting-up"
    EDGE_SITTING = "edge-sitting"
    OUT_OF_BED = "out-of-bed"
    UNCERTAIN = "uncertain"
    ABSENT = "absent"


class DecisionTraceValueName(StrEnum):
    """Finite names for numeric values persisted by compiled trace adapters."""

    OPERATING_THRESHOLD = "operating_threshold"
    WINDOW_FRAMES = "window_frames"
    FALL_PROBABILITY = "fall_probability"
    FALL_TRANSITION_PROBABILITY = "fall_transition_probability"
    FALLEN_PROBABILITY = "fallen_probability"
    TRANSITION_THRESHOLD = "transition_threshold"
    TRANSITION_VOTES = "transition_votes"
    TRANSITION_WINDOW = "transition_window"
    CONTAINMENT_RATIO = "containment_ratio"
    MAX_OTHER_CONTAINMENT_RATIO = "max_other_containment_ratio"
    MIN_CONTAINMENT = "min_containment"
    CANDIDATE_FRAMES = "candidate_frames"
    HOLD_FRAMES_THRESHOLD = "hold_frames_threshold"
    GRACE_FRAMES_BEFORE = "grace_frames_before"
    GRACE_FRAMES_AFTER = "grace_frames_after"
    GRACE_THRESHOLD = "grace_threshold"
    BED_ID = "bed_id"
    DECISION_STATE = "decision_state"
    TORSO_IN_FRAC = "torso_in_frac"
    LOWER_IN_FRAC = "lower_in_frac"
    KEYPOINT_IN_FRAC = "keypoint_in_frac"
    HIP_DEPTH = "hip_depth"
    TORSO_ANGLE = "torso_angle"
    CENTROID_DISPLACEMENT = "centroid_displacement"
    HIP_X_REL = "hip_x_rel"
    HIP_Y_REL = "hip_y_rel"
    OBSERVABILITY = "observability"
    DWELL_FRAMES = "dwell_frames"
    DWELL_THRESHOLD = "dwell_threshold"


class DecisionTraceMissingReason(StrEnum):
    """Finite missing-value vocabulary; arbitrary free text is never persisted."""

    ADAPTER_NOT_PROVIDED = "adapter-not-provided"
    ADAPTER_RETURNED_NO_DATA = "adapter-returned-no-data"
    OUTSIDE_DETECTION_WINDOW = "outside-detection-window"
    NO_LIVE_CLASSIFIED_TRACK = "no-live-classified-track"
    BED_REGION_UNAVAILABLE = "bed-region-unavailable"
    BED_OBSERVATION_MISSING = "bed-observation-missing"
    TRACK_NO_LONGER_LIVE = "track-no-longer-live"
    NO_OBSERVED_PERSON = "no-observed-person"
    POSE_UNAVAILABLE = "pose-unavailable"
    BED_POLYGON_INVALID = "bed-polygon-invalid"


@overload
def canonical_trace_number(value: int) -> int: ...


@overload
def canonical_trace_number(value: float) -> float: ...


def canonical_trace_number(value: NumericTraceValue) -> NumericTraceValue:
    """Return the canonical hardware-neutral representation of a trace scalar.

    Integers remain exact integers. Floats must be finite and are rounded to six
    decimal places, enough for the compiled confidence/geometry decisions while
    removing backend-specific sub-micro-unit noise. Negative floating-point zero
    is normalized to positive ``0.0``. Keeping int and float types distinct makes
    their JSON/content-id semantics explicit and deterministic.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("decision trace values must be numeric scalars")
    if isinstance(value, int):
        return value
    if not math.isfinite(value):
        raise ValueError("decision trace values must be finite")
    normalized = round(value, TRACE_FLOAT_DECIMAL_PLACES)
    return 0.0 if normalized == 0.0 else normalized


TokenT = TypeVar("TokenT", bound=StrEnum)


def _compiled_token(
    enum_type: type[TokenT],
    value: str,
    field_name: str,
) -> TokenT:
    if not isinstance(value, str):
        raise TypeError(f"decision trace {field_name} must use compiled vocabulary")
    try:
        return enum_type(value)
    except ValueError:
        # Do not echo rejected input: it may itself contain a credential or path.
        raise ValueError(f"decision trace {field_name} must use compiled vocabulary") from None


@dataclass(frozen=True, slots=True)
class DecisionTraceSnapshot:
    """Privacy-safe, hardware-neutral state emitted by one camera-local decider."""

    reason: str
    previous_state: str
    current_state: str
    triggered: bool
    track_id: int | None
    bed_id: int | None
    values: Mapping[str, NumericTraceValue] = field(default_factory=dict)
    missing_values: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reason = _compiled_token(DecisionTraceReason, self.reason, "reason")
        previous_state = _compiled_token(DecisionTraceState, self.previous_state, "previous_state")
        current_state = _compiled_token(DecisionTraceState, self.current_state, "current_state")
        values: dict[DecisionTraceValueName, NumericTraceValue] = {}
        for raw_name, raw_value in self.values.items():
            name = _compiled_token(DecisionTraceValueName, raw_name, "value name")
            values[name] = canonical_trace_number(raw_value)
        missing: dict[DecisionTraceValueName, DecisionTraceMissingReason] = {}
        for raw_name, raw_reason in self.missing_values.items():
            name = _compiled_token(DecisionTraceValueName, raw_name, "missing value name")
            missing[name] = _compiled_token(
                DecisionTraceMissingReason,
                raw_reason,
                "missing reason",
            )
        if set(values) & set(missing):
            raise ValueError("decision trace values cannot be both known and missing")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "previous_state", previous_state)
        object.__setattr__(self, "current_state", current_state)
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "missing_values", MappingProxyType(missing))


__all__ = [
    "TRACE_FLOAT_DECIMAL_PLACES",
    "DecisionTraceMissingReason",
    "DecisionTraceReason",
    "DecisionTraceSnapshot",
    "DecisionTraceState",
    "DecisionTraceValueName",
    "NumericTraceValue",
    "canonical_trace_number",
]
