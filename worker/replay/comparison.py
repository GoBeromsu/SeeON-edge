"""Structured, machine-consumed comparison between two deterministic replays.

Compares event count/onset/probability, containment/state transitions, and
per-frame decision snapshots between an original captured run (persisted
``DecisionTrace`` rows) and a replayed run, or between two replayed runs (A/B
across policy/module/profile revisions). Every difference carries an explicit,
finite reason -- never a free-text diff.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from worker.replay.engine import ReplayFrameResult, ReplayRun
from worker.types import BusinessEvent, DecisionTraceSnapshot


class MismatchReason(StrEnum):
    EVENT_COUNT_DIFFERS = "event-count-differs"
    EVENT_DOMAIN_DIFFERS = "event-domain-differs"
    EVENT_TYPE_DIFFERS = "event-type-differs"
    EVENT_IDENTITY_DIFFERS = "event-identity-differs"
    EVENT_CAMERA_DIFFERS = "event-camera-differs"
    EVENT_FACILITY_DIFFERS = "event-facility-differs"
    EVENT_ONSET_TIME_DIFFERS = "event-onset-time-differs"
    EVENT_PROBABILITY_DIFFERS = "event-probability-differs"
    EVENT_TRACK_DIFFERS = "event-track-differs"
    EVENT_BED_DIFFERS = "event-bed-differs"
    EVENT_AUDIT_DIFFERS = "event-audit-differs"
    SNAPSHOT_COUNT_DIFFERS = "snapshot-count-differs"
    PREVIOUS_STATE_DIFFERS = "previous-state-differs"
    STATE_DIFFERS = "state-differs"
    TRIGGERED_DIFFERS = "triggered-differs"
    REASON_DIFFERS = "reason-differs"
    SNAPSHOT_TRACK_DIFFERS = "snapshot-track-differs"
    SNAPSHOT_BED_DIFFERS = "snapshot-bed-differs"
    VALUE_DIFFERS = "value-differs"
    MISSING_VALUE_DIFFERS = "missing-value-differs"
    FRAME_MISSING_IN_OTHER = "frame-missing-in-other"
    REPRODUCIBILITY_DIFFERS = "reproducibility-differs"


@dataclass(frozen=True, slots=True)
class FrameMismatch:
    frame_key: tuple[str, str, int, int]
    reason: MismatchReason
    detail: str


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    """Deterministic, order-independent comparison summary between two runs."""

    baseline_effective_policy_id: str
    candidate_effective_policy_id: str
    baseline_event_count: int
    candidate_event_count: int
    identical: bool
    mismatches: tuple[FrameMismatch, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_effective_policy_id": self.baseline_effective_policy_id,
            "candidate_effective_policy_id": self.candidate_effective_policy_id,
            "baseline_event_count": self.baseline_event_count,
            "candidate_event_count": self.candidate_event_count,
            "identical": self.identical,
            "mismatches": [
                {
                    "frame_key": list(mismatch.frame_key),
                    "reason": mismatch.reason.value,
                    "detail": mismatch.detail,
                }
                for mismatch in self.mismatches
            ],
        }


_RUN_LEVEL_FRAME_KEY: tuple[str, str, int, int] = ("", "", -1, -1)


def compare_runs(baseline: ReplayRun, candidate: ReplayRun) -> ReplayComparison:
    """Compare two replay runs of the same camera trace frame-by-frame.

    Both runs must have replayed the same ordered frame-key sequence (the
    normal case: two policy/module revisions over one recovered camera
    trace). A frame key present in one run but not the other is itself a
    structured mismatch, not a silent skip. Snapshot and event extras are
    never dropped via non-strict zip -- cardinality is an explicit reason.
    """
    mismatches: list[FrameMismatch] = []
    if baseline.reproducible != candidate.reproducible or (
        baseline.non_reproducible_reason != candidate.non_reproducible_reason
    ):
        mismatches.append(
            FrameMismatch(
                _RUN_LEVEL_FRAME_KEY,
                MismatchReason.REPRODUCIBILITY_DIFFERS,
                (
                    f"baseline=reproducible={baseline.reproducible}/"
                    f"{baseline.non_reproducible_reason!r} "
                    f"candidate=reproducible={candidate.reproducible}/"
                    f"{candidate.non_reproducible_reason!r}"
                ),
            )
        )
    baseline_by_key = {frame.frame_key: frame for frame in baseline.frames}
    candidate_by_key = {frame.frame_key: frame for frame in candidate.frames}
    all_keys = sorted(set(baseline_by_key) | set(candidate_by_key))
    for key in all_keys:
        base_frame = baseline_by_key.get(key)
        cand_frame = candidate_by_key.get(key)
        if base_frame is None or cand_frame is None:
            side = "candidate" if base_frame is None else "baseline"
            mismatches.append(
                FrameMismatch(key, MismatchReason.FRAME_MISSING_IN_OTHER, f"missing in {side}")
            )
            continue
        mismatches.extend(_compare_frame(key, base_frame, cand_frame))
    return ReplayComparison(
        baseline_effective_policy_id=baseline.effective_policy_id,
        candidate_effective_policy_id=candidate.effective_policy_id,
        baseline_event_count=baseline.event_count,
        candidate_event_count=candidate.event_count,
        identical=not mismatches,
        mismatches=tuple(mismatches),
    )


def _compare_frame(
    frame_key: tuple[str, str, int, int],
    base_frame: ReplayFrameResult,
    cand_frame: ReplayFrameResult,
) -> list[FrameMismatch]:
    mismatches: list[FrameMismatch] = []
    if len(base_frame.events) != len(cand_frame.events):
        mismatches.append(
            FrameMismatch(
                frame_key,
                MismatchReason.EVENT_COUNT_DIFFERS,
                f"baseline={len(base_frame.events)} candidate={len(cand_frame.events)}",
            )
        )
    else:
        for base_event, cand_event in zip(base_frame.events, cand_frame.events, strict=True):
            mismatches.extend(_compare_event(frame_key, base_event, cand_event))
    mismatches.extend(_compare_snapshots(frame_key, base_frame, cand_frame))
    return mismatches


def _compare_event(
    frame_key: tuple[str, str, int, int],
    base_event: BusinessEvent,
    cand_event: BusinessEvent,
) -> list[FrameMismatch]:
    mismatches: list[FrameMismatch] = []
    field_checks: tuple[tuple[MismatchReason, object, object, str], ...] = (
        (MismatchReason.EVENT_DOMAIN_DIFFERS, base_event.domain, cand_event.domain, "domain"),
        (
            MismatchReason.EVENT_TYPE_DIFFERS,
            base_event.event_type,
            cand_event.event_type,
            "event_type",
        ),
        (
            MismatchReason.EVENT_IDENTITY_DIFFERS,
            base_event.identity,
            cand_event.identity,
            "identity",
        ),
        (
            MismatchReason.EVENT_CAMERA_DIFFERS,
            base_event.camera_id,
            cand_event.camera_id,
            "camera_id",
        ),
        (
            MismatchReason.EVENT_FACILITY_DIFFERS,
            base_event.facility_id,
            cand_event.facility_id,
            "facility_id",
        ),
        (
            MismatchReason.EVENT_ONSET_TIME_DIFFERS,
            base_event.time_sec,
            cand_event.time_sec,
            "time_sec",
        ),
        (
            MismatchReason.EVENT_PROBABILITY_DIFFERS,
            base_event.probability,
            cand_event.probability,
            "probability",
        ),
        (
            MismatchReason.EVENT_TRACK_DIFFERS,
            base_event.person_id,
            cand_event.person_id,
            "person_id",
        ),
        (MismatchReason.EVENT_BED_DIFFERS, base_event.bed_id, cand_event.bed_id, "bed_id"),
    )
    for reason, base_value, cand_value, label in field_checks:
        if base_value != cand_value:
            mismatches.append(
                FrameMismatch(
                    frame_key,
                    reason,
                    f"{label} baseline={base_value!r} candidate={cand_value!r}",
                )
            )
    base_audit = _canonical_audit(base_event.audit)
    cand_audit = _canonical_audit(cand_event.audit)
    if base_audit != cand_audit:
        mismatches.append(
            FrameMismatch(
                frame_key,
                MismatchReason.EVENT_AUDIT_DIFFERS,
                f"audit baseline={base_audit} candidate={cand_audit}",
            )
        )
    return mismatches


def _compare_snapshots(
    frame_key: tuple[str, str, int, int],
    base_frame: ReplayFrameResult,
    cand_frame: ReplayFrameResult,
) -> list[FrameMismatch]:
    mismatches: list[FrameMismatch] = []
    if len(base_frame.snapshots) != len(cand_frame.snapshots):
        mismatches.append(
            FrameMismatch(
                frame_key,
                MismatchReason.SNAPSHOT_COUNT_DIFFERS,
                f"baseline={len(base_frame.snapshots)} candidate={len(cand_frame.snapshots)}",
            )
        )
        return mismatches
    for index, (base_snap, cand_snap) in enumerate(
        zip(base_frame.snapshots, cand_frame.snapshots, strict=True)
    ):
        mismatches.extend(_compare_snapshot(frame_key, index, base_snap, cand_snap))
    return mismatches


def _compare_snapshot(
    frame_key: tuple[str, str, int, int],
    index: int,
    base_snap: DecisionTraceSnapshot,
    cand_snap: DecisionTraceSnapshot,
) -> list[FrameMismatch]:
    mismatches: list[FrameMismatch] = []
    field_checks: tuple[tuple[MismatchReason, object, object, str], ...] = (
        (
            MismatchReason.PREVIOUS_STATE_DIFFERS,
            base_snap.previous_state,
            cand_snap.previous_state,
            "previous_state",
        ),
        (
            MismatchReason.STATE_DIFFERS,
            base_snap.current_state,
            cand_snap.current_state,
            "current_state",
        ),
        (MismatchReason.REASON_DIFFERS, base_snap.reason, cand_snap.reason, "reason"),
        (
            MismatchReason.TRIGGERED_DIFFERS,
            base_snap.triggered,
            cand_snap.triggered,
            "triggered",
        ),
        (
            MismatchReason.SNAPSHOT_TRACK_DIFFERS,
            base_snap.track_id,
            cand_snap.track_id,
            "track_id",
        ),
        (MismatchReason.SNAPSHOT_BED_DIFFERS, base_snap.bed_id, cand_snap.bed_id, "bed_id"),
    )
    for reason, base_value, cand_value, label in field_checks:
        if base_value != cand_value:
            mismatches.append(
                FrameMismatch(
                    frame_key,
                    reason,
                    f"snapshot[{index}] {label} baseline={base_value!r} candidate={cand_value!r}",
                )
            )
    value_keys = set(base_snap.values) | set(cand_snap.values)
    for name in sorted(value_keys, key=str):
        if base_snap.values.get(name) != cand_snap.values.get(name):
            detail = (
                f"snapshot[{index}] {name} baseline={base_snap.values.get(name)!r} "
                + f"candidate={cand_snap.values.get(name)!r}"
            )
            mismatches.append(FrameMismatch(frame_key, MismatchReason.VALUE_DIFFERS, detail))
    missing_keys = set(base_snap.missing_values) | set(cand_snap.missing_values)
    for name in sorted(missing_keys, key=str):
        if base_snap.missing_values.get(name) != cand_snap.missing_values.get(name):
            detail = (
                f"snapshot[{index}] {name} "
                + f"baseline={base_snap.missing_values.get(name)!r} "
                + f"candidate={cand_snap.missing_values.get(name)!r}"
            )
            mismatches.append(
                FrameMismatch(frame_key, MismatchReason.MISSING_VALUE_DIFFERS, detail)
            )
    return mismatches


def _canonical_audit(audit: Mapping[str, object] | None) -> str:
    if audit is None:
        return "null"
    return json.dumps(
        _jsonable(dict(audit)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {str(key): _jsonable(item) for key, item in items}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return repr(value)


__all__ = ["FrameMismatch", "MismatchReason", "ReplayComparison", "compare_runs"]
