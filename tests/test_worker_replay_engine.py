from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from contracts.replay_trace import decode_document
from shared.detection_policies import BedExitPolicyV1, FallPolicyV2, make_effective_policy
from worker.domains.fall import FallV2Probabilities
from worker.pipeline.trace.models import TraceTruncation
from worker.replay.comparison import MismatchReason, compare_runs
from worker.replay.engine import ReplayConfigurationError, replay, replay_camera


def _rows(name: str):
    return decode_document(Path(f"tests/fixtures/replay/{name}.json").read_text())[1]


def _fall_policy(threshold: float = 0.5):
    return make_effective_policy(
        module_id="fall",
        module_version=2,
        values=FallPolicyV2(transition_threshold=threshold),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


def _bed_policy(containment: float = 0.5):
    return make_effective_policy(
        module_id="bed_exit",
        module_version=1,
        values=BedExitPolicyV1(min_containment=containment, hold_frames=1, grace_frames=1),
        source="image-default",
        facility_revision_id=None,
        camera_revision_id=None,
    )


class _FallModel:
    def predict(self, _: tuple[tuple[float, ...], ...]) -> FallV2Probabilities:
        return FallV2Probabilities(0.1, 0.8, 0.1)


def _fall_rows():
    template = _rows("gap-control-v2")[0]
    return tuple(replace(template, pts_ns=index * 66_666_667, seq=index) for index in range(45))


def test_replay_reproduces_identical_events_against_the_same_v2_model() -> None:
    first = replay(
        camera_id="fixture",
        rows=_fall_rows(),
        module_id="fall",
        policy=_fall_policy(),
        fall_model=_FallModel(),
    )
    second = replay(
        camera_id="fixture",
        rows=_fall_rows(),
        module_id="fall",
        policy=_fall_policy(),
        fall_model=_FallModel(),
    )
    assert first == second
    assert compare_runs(first, second).identical is True


def test_replay_threshold_change_produces_structured_mismatch() -> None:
    baseline = replay(
        camera_id="fixture",
        rows=_fall_rows(),
        module_id="fall",
        policy=_fall_policy(0.5),
        fall_model=_FallModel(),
    )
    candidate = replay(
        camera_id="fixture",
        rows=_fall_rows(),
        module_id="fall",
        policy=_fall_policy(0.9),
        fall_model=_FallModel(),
    )
    comparison = compare_runs(baseline, candidate)
    assert baseline.event_count == 1
    assert candidate.event_count == 0
    assert MismatchReason.EVENT_COUNT_DIFFERS in {item.reason for item in comparison.mismatches}
    assert baseline.effective_policy_id != candidate.effective_policy_id


def test_replay_rejects_policy_schema_mismatch() -> None:
    with pytest.raises(ReplayConfigurationError, match="schema"):
        replay(
            camera_id="fixture",
            rows=_fall_rows(),
            module_id="fall",
            policy=_bed_policy(),
            fall_model=_FallModel(),
        )


def test_replay_requires_fall_model_for_fall_module() -> None:
    with pytest.raises(ReplayConfigurationError, match="fall_model"):
        replay(
            camera_id="fixture",
            rows=_fall_rows(),
            module_id="fall",
            policy=_fall_policy(),
        )


def test_replay_bed_exit_containment_change_produces_structured_mismatch() -> None:
    source = _rows("reconnect-control-v2")
    rows = tuple(
        replace(
            row,
            tracks=(replace(row.tracks[0], bbox=(0.25, 0.0, 0.75, 0.5, 0.9)),)
            if row.seq < 2
            else row.tracks,
        )
        for row in source
    )
    baseline = replay(camera_id="fixture", rows=rows, module_id="bed_exit", policy=_bed_policy(0.9))
    candidate = replay(
        camera_id="fixture", rows=rows, module_id="bed_exit", policy=_bed_policy(0.4)
    )
    comparison = compare_runs(baseline, candidate)
    assert comparison.identical is False
    assert comparison.baseline_event_count != comparison.candidate_event_count


def test_truncation_marks_run_non_reproducible() -> None:
    truncation = TraceTruncation(2, 1, None, None)
    run = replay_camera(
        camera_id="fixture",
        analyses=(),
        module_id="fall",
        policy=_fall_policy(),
        fall_model=_FallModel(),
        truncation=truncation,
    )
    assert run.reproducible is False
    assert run.non_reproducible_reason is not None
    assert "handoff_dropped_frames=2" in run.non_reproducible_reason
    assert "pruned_frames=1" in run.non_reproducible_reason
