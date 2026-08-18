"""Pure bounded-state derivation for per-camera detection health."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

EVALUATION_WINDOW_SEC = 10.0
TIMEOUT_SEC = 120.0

DetectionState: TypeAlias = Literal["starting", "healthy", "blind", "unknown", "disabled"]
DetectionReason: TypeAlias = Literal[
    "pose_not_completing",
    "decision_not_completing",
    "no_completed_cycles",
    "telemetry_stale",
    "telemetry_missing",
    "counter_reset",
]


@dataclass(frozen=True, slots=True)
class DetectionCounters:
    expected: bool
    inference_admitted: int
    inference_succeeded: int
    inference_overwritten: int
    decision_completed: int


@dataclass(frozen=True, slots=True)
class DetectionHealth:
    previous: DetectionCounters | None = None
    previous_at: float | None = None
    first_seen_at: float | None = None
    last_completed_at: float | None = None
    pose_failure_streak: int = 0
    pose_failure_started_at: float | None = None
    decision_failure_streak: int = 0
    decision_failure_started_at: float | None = None
    state: DetectionState = "starting"
    reason: DetectionReason | None = None
    recent_success_rate: float | None = None


def parse_detection_counters(value: object) -> DetectionCounters | None:
    """Parse already-validated relay telemetry without accepting bool counters."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast(Mapping[object, object], value)
    expected = mapping.get("expected")
    admitted = mapping.get("inference_admitted")
    succeeded = mapping.get("inference_succeeded")
    overwritten = mapping.get("inference_overwritten")
    completed = mapping.get("decision_completed")
    if (
        not isinstance(expected, bool)
        or isinstance(admitted, bool)
        or not isinstance(admitted, int)
        or isinstance(succeeded, bool)
        or not isinstance(succeeded, int)
        or isinstance(overwritten, bool)
        or not isinstance(overwritten, int)
        or isinstance(completed, bool)
        or not isinstance(completed, int)
    ):
        return None
    return DetectionCounters(expected, admitted, succeeded, overwritten, completed)


def accept_detection_sample(
    previous_health: DetectionHealth | None,
    counters: DetectionCounters,
    *,
    accepted_at: float,
) -> DetectionHealth:
    """Return new bounded health state for one accepted cumulative sample."""
    if not counters.expected:
        return DetectionHealth(state="disabled")
    if previous_health is None or previous_health.previous is None:
        return DetectionHealth(
            previous=counters,
            previous_at=accepted_at,
            first_seen_at=accepted_at,
        )

    prior = previous_health.previous
    if _decreased(counters, prior):
        return DetectionHealth(
            previous=counters,
            previous_at=accepted_at,
            first_seen_at=accepted_at,
            reason="counter_reset",
        )

    admitted_delta = counters.inference_admitted - prior.inference_admitted
    succeeded_delta = counters.inference_succeeded - prior.inference_succeeded
    completed_delta = counters.decision_completed - prior.decision_completed
    success_rate = completed_delta / admitted_delta if admitted_delta > 0 else None

    if completed_delta > 0:
        return DetectionHealth(
            previous=counters,
            previous_at=accepted_at,
            first_seen_at=previous_health.first_seen_at,
            last_completed_at=accepted_at,
            state="healthy",
            recent_success_rate=success_rate,
        )

    pose_bad = admitted_delta > 0 and succeeded_delta == 0
    decision_bad = succeeded_delta > 0 and completed_delta == 0
    pose_streak, pose_started = _next_streak(
        previous_health.pose_failure_streak,
        previous_health.pose_failure_started_at,
        previous_health.previous_at,
        pose_bad,
    )
    decision_streak, decision_started = _next_streak(
        previous_health.decision_failure_streak,
        previous_health.decision_failure_started_at,
        previous_health.previous_at,
        decision_bad,
    )
    state: DetectionState = "starting"
    reason: DetectionReason | None = None
    if _window_complete(pose_streak, pose_started, accepted_at):
        state, reason = "blind", "pose_not_completing"
    elif _window_complete(decision_streak, decision_started, accepted_at):
        state, reason = "blind", "decision_not_completing"

    return DetectionHealth(
        previous=counters,
        previous_at=accepted_at,
        first_seen_at=previous_health.first_seen_at,
        last_completed_at=previous_health.last_completed_at,
        pose_failure_streak=pose_streak,
        pose_failure_started_at=pose_started,
        decision_failure_streak=decision_streak,
        decision_failure_started_at=decision_started,
        state=state,
        reason=reason,
        recent_success_rate=success_rate,
    )


def detection_health_fields(
    health: DetectionHealth | None,
    *,
    now: float,
    stale: bool,
    missing: bool,
) -> dict[str, object]:
    """Project state with snapshot-time stale and no-cycle precedence."""
    if stale:
        state: DetectionState = "unknown"
        reason: DetectionReason | None = "telemetry_stale"
    elif missing:
        state, reason = "unknown", "telemetry_missing"
    elif health is None:
        state, reason = "starting", None
    elif health.state == "disabled":
        state, reason = "disabled", None
    elif health.reason == "counter_reset":
        state, reason = "starting", "counter_reset"
    elif health.state == "blind":
        state, reason = health.state, health.reason
    elif _timed_out(health, now):
        state, reason = "blind", "no_completed_cycles"
    else:
        state, reason = health.state, health.reason
    return {
        "state": state,
        "reason": reason,
        "recent_success_rate": None if health is None else health.recent_success_rate,
        "last_completed_at_sec": None if health is None else health.last_completed_at,
        "evaluation_window_sec": EVALUATION_WINDOW_SEC,
        "timeout_sec": TIMEOUT_SEC,
    }


def _decreased(current: DetectionCounters, previous: DetectionCounters) -> bool:
    return any(
        current_value < previous_value
        for current_value, previous_value in zip(
            (
                current.inference_admitted,
                current.inference_succeeded,
                current.inference_overwritten,
                current.decision_completed,
            ),
            (
                previous.inference_admitted,
                previous.inference_succeeded,
                previous.inference_overwritten,
                previous.decision_completed,
            ),
            strict=True,
        )
    )


def _next_streak(
    streak: int,
    started_at: float | None,
    previous_at: float | None,
    failed: bool,
) -> tuple[int, float | None]:
    if not failed:
        return 0, None
    if streak == 0:
        return 1, previous_at
    return streak + 1, started_at


def _window_complete(streak: int, started_at: float | None, accepted_at: float) -> bool:
    return (
        streak >= 2 and started_at is not None and accepted_at - started_at >= EVALUATION_WINDOW_SEC
    )


def _timed_out(health: DetectionHealth, now: float) -> bool:
    baseline = health.last_completed_at
    if baseline is None:
        baseline = health.first_seen_at
    return baseline is not None and now - baseline >= TIMEOUT_SEC


__all__ = [
    "DetectionHealth",
    "accept_detection_sample",
    "detection_health_fields",
    "parse_detection_counters",
]
