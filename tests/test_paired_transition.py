"""Adversarial coverage for the frozen five-pair non-regression reducer."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from tests_support import paired_transition as _module
from tests_support.paired_transition import (
    ONE_SIDED_95_COEFFICIENT,
    PAIR_COUNT,
    PAIRED_DEGREES_OF_FREEDOM,
    CalibrationLeg,
    CapacityBudget,
    DiagnosedCorrection,
    Direction,
    ExpectedCall,
    InvalidateAttempt,
    LegRole,
    MetricObservation,
    MetricSpec,
    ModelCall,
    PairSummary,
    ResourceObservation,
    ResultStatus,
    SealProtocol,
    SubmitAttempt,
    Transform,
    detection_resolution,
    initial_state,
    measurement_pair_quantization,
    reduce,
    required_pair_order,
    sample_variance,
    statistical_resolution,
)

GPU = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
PINS = {
    "input_id": "input-1",
    "model_id": "model-1",
    "config_id": "config-1",
}
SOURCE_PINS = {
    "baseline_source_id": "source-baseline",
    "candidate_source_id": "source-candidate",
}
EXPECTED_CALLS = (
    ExpectedCall("live", "pose", 4),
    ExpectedCall("live", "fall", 4),
    ExpectedCall("on_demand", "bed", 2),
    ExpectedCall("stored_clip", "pose", 3),
    ExpectedCall("stored_clip", "bed", 2),
)
LEVELS = (10.0, 10.0, 10.0, 10.0, 10.0)
EQUAL_CANDIDATES = (9.5, 10.5, 9.5, 10.5, 10.0)
CALIBRATION = (10.0, 12.0, 11.0, 9.0, 13.0)
CORRECTION = DiagnosedCorrection(
    diagnosis="paired clock skewed after warm-up",
    correction="replace the instrument and rerun the frozen budget",
)


def _codes(reasons: tuple[object, ...]) -> set[str]:
    return {reason.code for reason in reasons}


def _spec(**overrides: object) -> MetricSpec:
    payload: dict[str, object] = {
        "name": "latency_p95",
        "estimator": "p95",
        "direction": Direction.HIGHER_ADVERSE,
        "transform": Transform.ABSOLUTE,
        "instrument_resolution": 0.001,
    }
    payload.update(overrides)
    return MetricSpec(**payload)  # type: ignore[arg-type]


def _resources(vram: float = 100.0, wal: float = 50.0) -> tuple[ResourceObservation, ...]:
    return (ResourceObservation("vram", vram), ResourceObservation("wal", wal))


def _baseline_calls() -> tuple[ModelCall, ...]:
    return (
        ModelCall("live", "pose", GPU, 4, 0),
        ModelCall("live", "fall", CPU, 4, 4),
        ModelCall("on_demand", "bed", CPU, 2, 2),
        ModelCall("stored_clip", "pose", CPU, 3, 3),
        ModelCall("stored_clip", "bed", CPU, 2, 2),
    )


def _candidate_calls() -> tuple[ModelCall, ...]:
    return tuple(
        ModelCall(call.execution_mode, call.role, GPU, call.call_count, 0)
        for call in EXPECTED_CALLS
    )


def _replace_call(
    calls: tuple[ModelCall, ...], mode: str, target_role: str, **changes: object
) -> tuple[ModelCall, ...]:
    return tuple(
        _replace(call, **changes)
        if (call.execution_mode, call.role) == (mode, target_role)
        else call
        for call in calls
    )


def _calibration(values: tuple[float, ...], spec: MetricSpec) -> tuple[CalibrationLeg, ...]:
    legs: list[CalibrationLeg] = []
    for index, value in enumerate(values, start=1):
        legs.append(
            CalibrationLeg(
                leg_index=index,
                evidence_id=f"cal-{index}",
                sequence=index,
                leg=LegRole.BASELINE,
                source_id=SOURCE_PINS["baseline_source_id"],
                metrics=(MetricObservation(spec.name, spec.estimator, value),),
                **PINS,
            )
        )
    return tuple(legs)


def _action(
    spec: MetricSpec,
    calibration: tuple[CalibrationLeg, ...],
    **overrides: object,
) -> SealProtocol:
    payload: dict[str, object] = {
        "gpu_provider": GPU,
        "cpu_provider": CPU,
        "metrics": (spec,),
        "budgets": (CapacityBudget("vram", 1000.0), CapacityBudget("wal", 500.0)),
        "expected_calls": EXPECTED_CALLS,
        "calibration": calibration,
        "freeze_sequence": 10,
        **PINS,
        **SOURCE_PINS,
    }
    payload.update(overrides)
    return SealProtocol(**payload)  # type: ignore[arg-type]


def _sealed(
    calibration_values: tuple[float, ...] = CALIBRATION,
    spec_overrides: dict[str, object] | None = None,
    **protocol_overrides: object,
) -> tuple[object, MetricSpec]:
    spec = _spec(**(spec_overrides or {}))
    state = reduce(
        initial_state(), _action(spec, _calibration(calibration_values, spec), **protocol_overrides)
    )
    return state, spec


def _leg(
    role: LegRole,
    evidence_id: str,
    sequence: int,
    value: float,
    spec: MetricSpec,
    **overrides: object,
):
    payload: dict[str, object] = {
        "evidence_id": evidence_id,
        "sequence": sequence,
        "leg": role,
        "source_id": SOURCE_PINS[f"{role.value}_source_id"],
        "metrics": (MetricObservation(spec.name, spec.estimator, value),),
        "outcomes": ("fall", "bed") if role is LegRole.BASELINE else ("bed", "fall"),
        "state_facts": ("epoch=1", "track=1")
        if role is LegRole.BASELINE
        else ("track=1", "epoch=1"),
        "delivery_loss": 0,
        "delivery_duplicate": 0,
        "unknown_gap": 0,
        "model_calls": _baseline_calls() if role is LegRole.BASELINE else _candidate_calls(),
        "resources": _resources(),
        **PINS,
    }
    payload.update(overrides)
    from tests_support.paired_transition import LegSummary

    return LegSummary(**payload)  # type: ignore[arg-type]


def _pairs(
    spec: MetricSpec,
    baselines: tuple[float, ...],
    candidates: tuple[float, ...],
    *,
    prefix: str = "run",
    sequence_offset: int = 0,
) -> tuple[PairSummary, ...]:
    pairs: list[PairSummary] = []
    for index, (baseline_value, candidate_value) in enumerate(
        zip(baselines, candidates, strict=True),
        start=1,
    ):
        order = required_pair_order(index)
        values = {LegRole.BASELINE: baseline_value, LegRole.CANDIDATE: candidate_value}
        first_sequence = 10 + sequence_offset + (index - 1) * 2 + 1
        legs = [
            _leg(
                role, f"{prefix}-p{index}-{role.value}", first_sequence + offset, values[role], spec
            )
            for offset, role in enumerate(order)
        ]
        pairs.append(
            PairSummary(
                pair_index=index,
                evidence_id=f"{prefix}-p{index}",
                first=legs[0],
                second=legs[1],
            )
        )
    return tuple(pairs)


def _replace_role(pairs: tuple[PairSummary, ...], index: int, role: LegRole, **changes: object):
    updated: list[PairSummary] = []
    for pair in pairs:
        if pair.pair_index != index or pair.first is None or pair.second is None:
            updated.append(pair)
            continue
        first = pair.first
        second = pair.second
        if first.leg is role:
            first = _replace(first, **changes)
        else:
            second = _replace(second, **changes)
        updated.append(_replace(pair, first=first, second=second))
    return tuple(updated)


def _replace_all(
    pairs: tuple[PairSummary, ...], role: LegRole, **changes: object
) -> tuple[PairSummary, ...]:
    updated = pairs
    for index in range(1, PAIR_COUNT + 1):
        updated = _replace_role(updated, index, role, **changes)
    return updated


def _replace(item: object, **changes: object) -> object:
    from dataclasses import replace

    return replace(item, **changes)  # type: ignore[arg-type]


def _submit(pairs: tuple[PairSummary, ...], **seal_kwargs: object):
    state, spec = _sealed(**seal_kwargs)
    assert state.protocol is not None
    return (
        reduce(state, SubmitAttempt(pairs or _pairs(spec, LEVELS, EQUAL_CANDIDATES))),
        state,
        spec,
    )


def _bounds(differences: tuple[float, ...]) -> tuple[float, float, float]:
    point = math.fsum(differences) / PAIR_COUNT
    variance = math.fsum((value - point) ** 2 for value in differences) / PAIRED_DEGREES_OF_FREEDOM
    standard_error = math.sqrt(variance / PAIR_COUNT)
    half = ONE_SIDED_95_COEFFICIENT * standard_error
    return point, point - half, point + half


def _invalidate(state, correction: DiagnosedCorrection = CORRECTION):
    return reduce(state, InvalidateAttempt(state.attempts[0].evidence_ids, correction))


def test_pair_order_alternates_by_index() -> None:
    assert required_pair_order(1) == (LegRole.BASELINE, LegRole.CANDIDATE)
    assert required_pair_order(2) == (LegRole.CANDIDATE, LegRole.BASELINE)
    assert required_pair_order(3) == (LegRole.BASELINE, LegRole.CANDIDATE)
    assert required_pair_order(4) == (LegRole.CANDIDATE, LegRole.BASELINE)
    assert required_pair_order(5) == (LegRole.BASELINE, LegRole.CANDIDATE)


def test_equal_nonzero_variance_passes_when_upper_bound_is_below_rm() -> None:
    state, spec = _sealed()
    assert state.decision.status is ResultStatus.OPEN
    assert state.decision.reasons[0].code == "protocol_frozen"
    pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    assert pairs[0].first is not None and pairs[0].second is not None
    assert pairs[0].first.outcomes != pairs[0].second.outcomes
    result = reduce(state, SubmitAttempt(pairs))
    metric = result.attempts[0].metrics[0]
    point, lower, upper = _bounds(metric.differences)
    statistical = ONE_SIDED_95_COEFFICIENT * math.sqrt(2.0 * metric.baseline_variance / PAIR_COUNT)
    assert metric.baseline_variance == sample_variance(CALIBRATION) == 2.5
    assert metric.difference_variance > 0
    assert metric.point == point == 0.0
    assert metric.lower_bound == lower
    assert metric.upper_bound == upper
    assert metric.pair_quantization == measurement_pair_quantization(spec.instrument_resolution)
    assert metric.statistical_resolution == statistical
    assert metric.rm == statistical + metric.pair_quantization
    assert metric.rm == detection_resolution(metric.baseline_variance, spec.instrument_resolution)
    assert metric.degrees_of_freedom == 4
    assert metric.coefficient == ONE_SIDED_95_COEFFICIENT
    assert metric.upper_bound < metric.rm
    assert metric.status is ResultStatus.PASS
    assert result.decision.status is ResultStatus.PASS
    assert result.decision.terminal is True
    assert _codes(result.decision.reasons) == {"paired_non_adverse_below_resolution"}
    assert all(gate.status is ResultStatus.PASS for gate in result.attempts[0].gates)
    assert result.protocol is state.protocol


def test_zero_baseline_variance_still_passes_on_pair_quantization() -> None:
    state, spec = _sealed(calibration_values=LEVELS)
    result = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, LEVELS)))
    metric = result.attempts[0].metrics[0]
    assert metric.baseline_variance == 0.0
    assert metric.statistical_resolution == 0.0
    assert metric.pair_quantization == 0.001
    assert metric.rm == 0.001
    assert metric.point == 0.0
    assert metric.upper_bound == 0.0
    assert metric.upper_bound < metric.rm
    assert result.decision.status is ResultStatus.PASS


def test_tiny_confirmed_adverse_fails_even_below_rm() -> None:
    _state, spec = _sealed()
    candidates = tuple(10.0 + 1e-6 for _ in range(PAIR_COUNT))
    result, _sealed_state, _spec = _submit(_pairs(spec, LEVELS, candidates))
    metric = result.attempts[0].metrics[0]
    assert metric.point > 0.0
    assert metric.point < metric.rm
    assert metric.upper_bound < metric.rm
    assert metric.lower_bound > 0.0
    assert metric.difference_variance == 0.0
    assert metric.status is ResultStatus.FAIL
    assert result.decision.status is ResultStatus.FAIL
    assert result.decision.terminal is True
    assert _codes(result.decision.reasons) == {"confirmed_adverse"}


def test_unresolved_adverse_point_is_inconclusive_even_when_upper_is_below_rm() -> None:
    _state, spec = _sealed()
    result, sealed, _spec = _submit(_pairs(spec, LEVELS, (11.0, 11.0, 11.0, 11.0, 7.0)))
    metric = result.attempts[0].metrics[0]
    assert metric.differences == (1.0, 1.0, 1.0, 1.0, -3.0)
    assert metric.point > 0.0
    assert metric.lower_bound < 0.0
    assert metric.upper_bound < metric.rm
    assert metric.status is ResultStatus.INCONCLUSIVE
    assert result.decision.status is ResultStatus.INCONCLUSIVE
    assert result.decision.terminal is False
    assert result.decision.status is not ResultStatus.NO_GO
    assert _codes(result.decision.reasons) == {"adverse_unresolved"}
    assert all(gate.status is ResultStatus.PASS for gate in result.attempts[0].gates)
    assert sealed.protocol.metrics[0].rm == metric.rm


def test_non_adverse_point_whose_upper_bound_reaches_rm_is_inconclusive() -> None:
    state, spec = _sealed()
    result = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, (0.0, 20.0, 0.0, 20.0, 10.0))))
    metric = result.attempts[0].metrics[0]
    assert metric.point == 0.0
    assert metric.lower_bound < 0.0
    assert metric.difference_variance > metric.baseline_variance
    assert metric.upper_bound > metric.rm
    assert metric.rm == detection_resolution(
        sample_variance(CALIBRATION), spec.instrument_resolution
    )
    assert result.decision.status is ResultStatus.INCONCLUSIVE
    assert "resolution_unexcluded" in _codes(result.decision.reasons)
    assert "confirmed_adverse" not in _codes(result.decision.reasons)


def test_upper_bound_must_be_strictly_below_frozen_resolution() -> None:
    candidates = (9.0, 11.0, 9.0, 11.0, 10.0)
    _point, _lower, upper = _bounds((-1.0, 1.0, -1.0, 1.0, 0.0))
    for resolution, status in (
        (upper, ResultStatus.INCONCLUSIVE),
        (math.nextafter(upper, math.inf), ResultStatus.PASS),
    ):
        state, spec = _sealed(
            calibration_values=LEVELS,
            spec_overrides={"instrument_resolution": resolution},
        )
        result = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, candidates)))
        metric = result.attempts[0].metrics[0]
        assert metric.point == 0.0
        assert metric.upper_bound == upper
        assert metric.rm == resolution
        assert result.decision.status is status


def test_positive_transformed_difference_is_adverse_and_log_ratio_is_natural_log() -> None:
    higher, higher_spec = _sealed()
    improved = reduce(higher, SubmitAttempt(_pairs(higher_spec, LEVELS, (9.0, 9.0, 9.0, 9.0, 9.0))))
    assert improved.attempts[0].metrics[0].differences == (-1.0, -1.0, -1.0, -1.0, -1.0)
    assert improved.decision.status is ResultStatus.PASS

    lower, lower_spec = _sealed(spec_overrides={"direction": Direction.LOWER_ADVERSE})
    worsened = reduce(lower, SubmitAttempt(_pairs(lower_spec, LEVELS, (9.0, 9.0, 9.0, 9.0, 9.0))))
    assert worsened.attempts[0].metrics[0].differences == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert worsened.attempts[0].metrics[0].lower_bound > 0.0
    assert worsened.attempts[0].metrics[0].point < worsened.attempts[0].metrics[0].rm
    assert worsened.decision.status is ResultStatus.FAIL

    logged, logged_spec = _sealed(
        calibration_values=LEVELS,
        spec_overrides={"transform": Transform.LOG_RATIO},
    )
    ratio = reduce(logged, SubmitAttempt(_pairs(logged_spec, LEVELS, (5.0, 5.0, 5.0, 5.0, 5.0))))
    expected = math.log(5.0) - math.log(10.0)
    assert ratio.attempts[0].metrics[0].differences == (
        expected,
        expected,
        expected,
        expected,
        expected,
    )
    assert ratio.decision.status is ResultStatus.PASS
    rejected = reduce(logged, SubmitAttempt(_pairs(logged_spec, LEVELS, (0.0, 0.0, 0.0, 0.0, 0.0))))
    assert rejected.decision.status is ResultStatus.REJECTED
    assert "log_ratio_not_strictly_positive" in _codes(rejected.decision.reasons)
    negative = reduce(
        logged, SubmitAttempt(_pairs(logged_spec, LEVELS, (-1.0, -1.0, -1.0, -1.0, -1.0)))
    )
    assert "log_ratio_not_strictly_positive" in _codes(negative.decision.reasons)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 10**400])
def test_nonfinite_metric_evidence_is_rejected(bad: float | int) -> None:
    state, spec = _sealed()
    pairs = _replace_role(
        _pairs(spec, LEVELS, EQUAL_CANDIDATES),
        1,
        LegRole.CANDIDATE,
        metrics=(MetricObservation(spec.name, spec.estimator, bad),),
    )
    result = reduce(state, SubmitAttempt(pairs))
    assert result.decision.status is ResultStatus.REJECTED
    assert result.decision.terminal is False
    assert "nonfinite" in _codes(result.decision.reasons)
    assert result.attempts[0].metrics == ()


def test_missing_leg_wrong_order_duplicate_and_pins_fail_closed() -> None:
    state, spec = _sealed()
    pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    missing = reduce(state, SubmitAttempt((_replace(pairs[0], first=None), *pairs[1:])))
    assert missing.decision.status is ResultStatus.REJECTED
    assert "missing_leg" in _codes(missing.decision.reasons)

    for index in range(1, PAIR_COUNT + 1):
        swapped = tuple(
            _replace(pair, first=pair.second, second=pair.first)
            if pair.pair_index == index
            else pair
            for pair in pairs
        )
        ordered = reduce(state, SubmitAttempt(swapped))
        assert ordered.decision.status is ResultStatus.REJECTED
        assert "wrong_order" in _codes(ordered.decision.reasons)

    duplicated = reduce(
        state,
        SubmitAttempt(_replace_role(pairs, 2, LegRole.BASELINE, evidence_id=pairs[0].evidence_id)),
    )
    assert duplicated.decision.status is ResultStatus.REJECTED
    assert "duplicate_evidence" in _codes(duplicated.decision.reasons)


@pytest.mark.parametrize("field", ["input_id", "model_id", "config_id"])
def test_wrong_identity_pin_is_rejected(field: str) -> None:
    state, spec = _sealed()
    pairs = _replace_role(
        _pairs(spec, LEVELS, EQUAL_CANDIDATES), 4, LegRole.CANDIDATE, **{field: "other"}
    )
    result = reduce(state, SubmitAttempt(pairs))
    assert result.decision.status is ResultStatus.REJECTED
    assert "wrong_pin" in _codes(result.decision.reasons)
    assert any(field in reason.detail for reason in result.decision.reasons)


def test_declared_sources_are_role_specific_and_cannot_be_swapped() -> None:
    state, spec = _sealed()
    pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    accepted = reduce(state, SubmitAttempt(pairs))
    assert accepted.decision.status is ResultStatus.PASS

    swapped = _replace_all(pairs, LegRole.BASELINE, source_id=SOURCE_PINS["candidate_source_id"])
    swapped = _replace_all(swapped, LegRole.CANDIDATE, source_id=SOURCE_PINS["baseline_source_id"])
    rejected = reduce(state, SubmitAttempt(swapped))
    assert rejected.decision.status is ResultStatus.REJECTED
    assert {reason.detail.rsplit(":", 1)[-1] for reason in rejected.decision.reasons} == {
        "baseline_source_id",
        "candidate_source_id",
    }
    for role in (LegRole.BASELINE, LegRole.CANDIDATE):
        unexpected = _replace_role(pairs, 3, role, source_id="undeclared-source")
        result = reduce(state, SubmitAttempt(unexpected))
        assert result.decision.status is ResultStatus.REJECTED
        assert _codes(result.decision.reasons) == {"wrong_pin"}
        assert result.decision.reasons[0].detail.endswith(f"{role.value}_source_id")


@pytest.mark.parametrize("role", [LegRole.BASELINE, LegRole.CANDIDATE])
def test_each_source_is_frozen_in_the_digest_and_reseal_is_refused(role: LegRole) -> None:
    state, spec = _sealed()
    field = f"{role.value}_source_id"
    source = f"other-{role.value}-source"
    calibration = _calibration(CALIBRATION, spec)
    if role is LegRole.BASELINE:
        calibration = tuple(_replace(leg, source_id=source) for leg in calibration)
    action = _action(spec, calibration, **{field: source})
    changed = reduce(initial_state(), action)
    assert changed.protocol is not None
    assert changed.protocol.protocol_id != state.protocol.protocol_id
    pairs = _replace_all(_pairs(spec, LEVELS, LEVELS), role, source_id=source)
    assert reduce(changed, SubmitAttempt(pairs)).decision.status is ResultStatus.PASS
    assert reduce(state, SubmitAttempt(pairs)).decision.status is ResultStatus.REJECTED

    resealed = reduce(state, action)
    assert resealed.protocol is state.protocol
    assert resealed.rejected_actions[-1].code == "protocol_already_frozen"

    missing = reduce(initial_state(), _action(spec, calibration, **{field: ""}))
    assert missing.protocol is None
    assert ("invalid_pin", field) in {
        (reason.code, reason.detail) for reason in missing.decision.reasons
    }


def test_calibration_requires_the_baseline_source_not_the_candidate_source() -> None:
    spec = _spec()
    calibration = _calibration(CALIBRATION, spec)
    wrong_source = _replace(calibration[0], source_id=SOURCE_PINS["candidate_source_id"])
    result = reduce(initial_state(), _action(spec, (wrong_source, *calibration[1:])))
    assert result.protocol is None
    assert _codes(result.decision.reasons) == {"wrong_pin"}
    assert result.decision.reasons[0].detail == "calibration:1:baseline_source_id"


def test_estimator_mismatch_and_candidate_before_freeze_are_rejected() -> None:
    state, spec = _sealed()
    pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    misestimated = _replace_role(
        pairs,
        1,
        LegRole.CANDIDATE,
        metrics=(MetricObservation(spec.name, "p50", 10.0),),
    )
    estimated = reduce(state, SubmitAttempt(misestimated))
    assert "wrong_pin" in _codes(estimated.decision.reasons)
    assert any("estimator" in reason.detail for reason in estimated.decision.reasons)

    pair = pairs[1]
    assert pair.first is not None and pair.first.leg is LegRole.CANDIDATE
    early = (pairs[0], _replace(pair, first=_replace(pair.first, sequence=10)), *pairs[2:])
    frozen = reduce(state, SubmitAttempt(early))
    assert frozen.decision.status is ResultStatus.REJECTED
    assert "candidate_before_freeze" in _codes(frozen.decision.reasons)
    assert "wrong_order" not in _codes(frozen.decision.reasons)


def test_calibration_must_be_baseline_and_strictly_before_freeze() -> None:
    spec = _spec()
    legs = _calibration(CALIBRATION, spec)
    late = reduce(initial_state(), _action(spec, (_replace(legs[0], sequence=10), *legs[1:])))
    assert late.protocol is None
    assert "calibration_not_before_freeze" in _codes(late.decision.reasons)

    candidate = reduce(
        initial_state(),
        _action(spec, (_replace(legs[0], leg=LegRole.CANDIDATE), *legs[1:])),
    )
    assert candidate.protocol is None
    assert "calibration_not_baseline" in _codes(candidate.decision.reasons)

    nonpositive = _sealed(
        calibration_values=(10.0, 10.0, 10.0, 10.0, 0.0),
        spec_overrides={"transform": Transform.LOG_RATIO},
    )
    assert nonpositive[0].protocol is None
    assert "log_ratio_not_strictly_positive" in _codes(nonpositive[0].decision.reasons)


def test_exact_multiset_and_state_mismatch_fail_independently_of_latency() -> None:
    state, spec = _sealed()
    base = _pairs(spec, LEVELS, LEVELS)
    extra = _replace_role(base, 5, LegRole.CANDIDATE, outcomes=("fall", "bed", "fall"))
    mismatched = reduce(state, SubmitAttempt(extra))
    assert mismatched.decision.status is ResultStatus.FAIL
    assert _codes(mismatched.decision.reasons) == {"correctness_multiset_mismatch"}

    shifted = _replace_role(base, 2, LegRole.BASELINE, state_facts=("epoch=1", "track=2"))
    state_result = reduce(state, SubmitAttempt(shifted))
    assert state_result.decision.status is ResultStatus.FAIL
    assert _codes(state_result.decision.reasons) == {"state_mismatch"}

    both = _replace_role(extra, 2, LegRole.BASELINE, state_facts=("epoch=9",))
    combined = reduce(state, SubmitAttempt(both))
    assert _codes(combined.decision.reasons) == {"correctness_multiset_mismatch", "state_mismatch"}


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("delivery_loss", "delivery_loss"),
        ("delivery_duplicate", "delivery_duplicate"),
        ("unknown_gap", "unknown_gap"),
    ],
)
@pytest.mark.parametrize("role", [LegRole.BASELINE, LegRole.CANDIDATE])
def test_delivery_loss_duplicate_and_unknown_gap_are_zero_defect(
    field: str,
    code: str,
    role: LegRole,
) -> None:
    state, spec = _sealed()
    pairs = _replace_role(_pairs(spec, LEVELS, LEVELS), 3, role, **{field: 1})
    result = reduce(state, SubmitAttempt(pairs))
    assert result.decision.status is ResultStatus.FAIL
    assert _codes(result.decision.reasons) == {code}


def test_resource_budget_is_absolute_and_independent_of_the_latency_pass() -> None:
    state, spec = _sealed()
    base = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    increased = _replace_role(
        base, 1, LegRole.CANDIDATE, resources=_resources(vram=900.0, wal=50.0)
    )
    allowed = reduce(state, SubmitAttempt(increased))
    assert allowed.decision.status is ResultStatus.PASS
    assert increased[0].second is not None and increased[0].first is not None
    assert increased[0].second.resources[0].value > increased[0].first.resources[0].value

    at_limit = _replace_all(base, LegRole.CANDIDATE, resources=_resources(vram=1000.0, wal=500.0))
    equal = reduce(state, SubmitAttempt(at_limit))
    assert equal.decision.status is ResultStatus.PASS

    overflow = _replace_role(base, 4, LegRole.BASELINE, resources=_resources(vram=1001.0, wal=50.0))
    failed = reduce(state, SubmitAttempt(overflow))
    assert failed.decision.status is ResultStatus.FAIL
    assert failed.attempts[0].metrics[0].status is ResultStatus.PASS
    assert _codes(failed.decision.reasons) == {"resource_overflow"}
    assert any(reason.detail == "vram" for reason in failed.decision.reasons)

    missing = _replace_role(
        base, 1, LegRole.CANDIDATE, resources=(ResourceObservation("vram", 1.0),)
    )
    rejected = reduce(state, SubmitAttempt(missing))
    assert rejected.decision.status is ResultStatus.REJECTED
    assert "resource_missing" in _codes(rejected.decision.reasons)


def test_reported_gpu_provider_and_call_count_gates() -> None:
    state, spec = _sealed()
    base = _pairs(spec, LEVELS, LEVELS)
    passed = reduce(state, SubmitAttempt(base))
    gpu_gate = next(gate for gate in passed.attempts[0].gates if gate.gate == "gpu_provider")
    assert gpu_gate.status is ResultStatus.PASS
    assert passed.decision.status is ResultStatus.PASS

    gpu_baseline = _replace_all(
        base,
        LegRole.BASELINE,
        model_calls=_candidate_calls(),
    )
    assert reduce(state, SubmitAttempt(gpu_baseline)).decision.status is ResultStatus.PASS

    fallback_cases = (
        {"provider": CPU, "cpu_count": 4},
        {"provider": GPU, "cpu_count": 1},
        {"provider": CPU, "cpu_count": 0},
    )
    for changes in fallback_cases:
        calls = _replace_call(_candidate_calls(), "live", "fall", **changes)
        failed = reduce(
            state, SubmitAttempt(_replace_role(base, 3, LegRole.CANDIDATE, model_calls=calls))
        )
        assert failed.decision.status is ResultStatus.FAIL
        assert "gpu_fallback" in _codes(failed.decision.reasons)

    absent = reduce(state, SubmitAttempt(_replace_role(base, 2, LegRole.CANDIDATE, model_calls=())))
    assert absent.decision.status is ResultStatus.REJECTED
    assert "model_evidence_absent" in _codes(absent.decision.reasons)

    counted = _replace_call(_candidate_calls(), "live", "fall", call_count=3)
    mismatch = reduce(
        state, SubmitAttempt(_replace_role(base, 1, LegRole.CANDIDATE, model_calls=counted))
    )
    assert mismatch.decision.status is ResultStatus.REJECTED
    assert "model_count_mismatch" in _codes(mismatch.decision.reasons)

    posed = _replace_call(_candidate_calls(), "live", "fall", execution_mode="on_demand")
    wrong_role = reduce(
        state, SubmitAttempt(_replace_role(base, 5, LegRole.CANDIDATE, model_calls=posed))
    )
    assert wrong_role.decision.status is ResultStatus.REJECTED
    assert "model_role_mismatch" in _codes(wrong_role.decision.reasons)
    assert "model_evidence_absent" in _codes(wrong_role.decision.reasons)

    toy = _replace_call(_candidate_calls(), "live", "fall", provider="gpu")
    invalid = reduce(
        state, SubmitAttempt(_replace_role(base, 1, LegRole.CANDIDATE, model_calls=toy))
    )
    assert invalid.decision.status is ResultStatus.REJECTED
    assert "model_provider_invalid" in _codes(invalid.decision.reasons)


@pytest.mark.parametrize("mode", ["live", "stored_clip"])
def test_baseline_cpu_pose_exception_is_mode_specific(mode: str) -> None:
    state, spec = _sealed()
    pairs = _pairs(spec, LEVELS, LEVELS)
    calls = _baseline_calls()
    pose = next(call for call in calls if (call.execution_mode, call.role) == (mode, "pose"))
    pairs = _replace_all(
        pairs,
        LegRole.BASELINE,
        model_calls=_replace_call(calls, mode, "pose", provider=CPU, cpu_count=pose.call_count),
    )
    result = reduce(state, SubmitAttempt(pairs))
    if mode == "live":
        assert result.decision.status is ResultStatus.REJECTED
        assert "baseline_cpu_role_rejected" in _codes(result.decision.reasons)
    else:
        assert result.decision.status is ResultStatus.PASS


@pytest.mark.parametrize(
    ("mode", "role"),
    [
        ("live", "pose"),
        ("live", "fall"),
        ("on_demand", "bed"),
        ("stored_clip", "pose"),
        ("stored_clip", "bed"),
    ],
)
def test_every_lane_is_required_at_seal_and_in_each_leg(mode: str, role: str) -> None:
    expected = tuple(
        call for call in EXPECTED_CALLS if (call.execution_mode, call.role) != (mode, role)
    )
    rejected_seal, _spec = _sealed(expected_calls=expected)
    assert rejected_seal.protocol is None
    assert "missing_model_inventory" in _codes(rejected_seal.decision.reasons)
    assert any(reason.detail == f"{mode}:{role}" for reason in rejected_seal.decision.reasons)

    state, spec = _sealed()
    for leg_role, calls in (
        (LegRole.BASELINE, _baseline_calls()),
        (LegRole.CANDIDATE, _candidate_calls()),
    ):
        missing = tuple(call for call in calls if (call.execution_mode, call.role) != (mode, role))
        pairs = _replace_role(_pairs(spec, LEVELS, LEVELS), 2, leg_role, model_calls=missing)
        rejected = reduce(state, SubmitAttempt(pairs))
        assert rejected.decision.status is ResultStatus.REJECTED
        assert "model_evidence_absent" in _codes(rejected.decision.reasons)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("fallback", ["cpu_provider", "cpu_count"])
def test_every_candidate_lane_rejects_reported_cpu_execution(index: int, fallback: str) -> None:
    state, spec = _sealed()
    call = _candidate_calls()[index]
    changes = {"provider": CPU} if fallback == "cpu_provider" else {"cpu_count": 1}
    calls = _replace_call(_candidate_calls(), call.execution_mode, call.role, **changes)
    pairs = _replace_role(_pairs(spec, LEVELS, LEVELS), 3, LegRole.CANDIDATE, model_calls=calls)
    result = reduce(state, SubmitAttempt(pairs))
    assert result.decision.status is ResultStatus.FAIL
    assert _codes(result.decision.reasons) == {"gpu_fallback"}
    assert result.attempts[0].metrics[0].status is ResultStatus.PASS
    gate = next(gate for gate in result.attempts[0].gates if gate.gate == "gpu_provider")
    assert f"{call.execution_mode}:{call.role}" in gate.detail


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("execution_mode", "", "invalid_evidence"),
        ("execution_mode", None, "invalid_evidence"),
        ("execution_mode", "on_demand", "model_role_mismatch"),
        ("role", "", "invalid_evidence"),
        ("role", None, "invalid_evidence"),
        ("role", "other", "model_role_mismatch"),
        ("call_count", 0, "model_count_mismatch"),
        ("call_count", -1, "model_count_mismatch"),
        ("call_count", True, "model_count_mismatch"),
    ],
)
def test_inventory_requires_valid_modes_roles_and_positive_counts(
    field: str, value: object, code: str
) -> None:
    expected = (_replace(EXPECTED_CALLS[0], **{field: value}), *EXPECTED_CALLS[1:])
    sealed, _spec = _sealed(expected_calls=expected)
    assert sealed.protocol is None
    assert code in _codes(sealed.decision.reasons)

    state, spec = _sealed()
    calls = _replace_call(_candidate_calls(), "live", "pose", **{field: value})
    pairs = _replace_role(_pairs(spec, LEVELS, LEVELS), 1, LegRole.CANDIDATE, model_calls=calls)
    submitted = reduce(state, SubmitAttempt(pairs))
    assert submitted.decision.status is ResultStatus.REJECTED
    call_code = "invalid_count" if value is True else code
    assert call_code in _codes(submitted.decision.reasons)


def test_duplicate_lane_and_extra_lane_are_not_inventory_substitutes() -> None:
    duplicate, _spec = _sealed(expected_calls=(*EXPECTED_CALLS, EXPECTED_CALLS[0]))
    assert duplicate.protocol is None
    assert "duplicate_evidence" in _codes(duplicate.decision.reasons)
    extra, _spec = _sealed(expected_calls=(*EXPECTED_CALLS, ExpectedCall("on_demand", "fall", 4)))
    assert extra.protocol is None
    assert "model_role_mismatch" in _codes(extra.decision.reasons)

    state, spec = _sealed()
    calls = _candidate_calls()
    pairs = _replace_role(
        _pairs(spec, LEVELS, LEVELS), 1, LegRole.CANDIDATE, model_calls=(*calls, calls[0])
    )
    result = reduce(state, SubmitAttempt(pairs))
    assert result.decision.status is ResultStatus.REJECTED
    assert "duplicate_evidence" in _codes(result.decision.reasons)


def test_bad_retry_selective_replay_and_terminal_inconclusive() -> None:
    state, spec = _sealed()
    passing_pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    passed = reduce(state, SubmitAttempt(passing_pairs))
    assert passed.decision.status is ResultStatus.PASS
    ignored = reduce(passed, SubmitAttempt(_pairs(spec, LEVELS, EQUAL_CANDIDATES, prefix="again")))
    assert ignored.decision == passed.decision
    assert ignored.attempts == passed.attempts
    assert ignored.rejected_actions[-1].code == "bad_retry"

    unresolved = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, (11.0, 11.0, 11.0, 11.0, 7.0))))
    assert unresolved.decision.status is ResultStatus.INCONCLUSIVE
    premature = reduce(unresolved, SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="early")))
    assert premature.decision == unresolved.decision
    assert premature.rejected_actions[-1].code == "bad_retry"
    partial = reduce(
        unresolved,
        InvalidateAttempt(unresolved.attempts[0].evidence_ids[:-1], CORRECTION),
    )
    assert partial.attempts[0].invalidated is False
    assert partial.rejected_actions[-1].code == "partial_invalidation"
    blank = reduce(
        unresolved,
        InvalidateAttempt(unresolved.attempts[0].evidence_ids, DiagnosedCorrection("  ", "x")),
    )
    assert blank.rejected_actions[-1].code == "missing_correction"

    opened = _invalidate(unresolved)
    assert opened.attempts[0].invalidated is True
    assert opened.decision.authoritative_attempt is None
    assert opened.decision.terminal is False
    assert opened.correction == CORRECTION
    assert opened.protocol is state.protocol

    reused = _pairs(spec, LEVELS, LEVELS, prefix="retry", sequence_offset=10)
    stolen = reused[0].evidence_id
    old_id = unresolved.attempts[0].evidence_ids[0]
    selective_pairs = (_replace(reused[0], evidence_id=old_id), *reused[1:])
    assert stolen != old_id
    selective = reduce(opened, SubmitAttempt(selective_pairs))
    assert selective.decision.terminal is True
    assert selective.decision.status is ResultStatus.REJECTED
    assert "selective_replay" in _codes(selective.attempts[1].reasons)
    assert "attempt_budget_exhausted" in _codes(selective.decision.reasons)
    blocked = reduce(
        selective, SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="third", sequence_offset=20))
    )
    assert len(blocked.attempts) == ATTEMPT_COUNT
    assert blocked.decision == selective.decision
    assert blocked.rejected_actions[-1].code == "attempt_limit"

    short = reduce(
        opened,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="short", sequence_offset=10)[:4]),
    )
    assert short.decision.terminal is True
    assert short.decision.status is not ResultStatus.PASS
    assert "pair_count" in _codes(short.attempts[1].reasons)

    retried = reduce(
        opened,
        SubmitAttempt(_pairs(spec, LEVELS, EQUAL_CANDIDATES, prefix="full", sequence_offset=10)),
    )
    assert retried.decision.status is ResultStatus.PASS
    assert retried.decision.authoritative_attempt == 2
    assert retried.attempts[0].invalidated is True
    assert retried.attempts[0].metrics[0].point > 0.0
    assert retried.attempts[1].metrics[0].point == 0.0
    assert retried.protocol is state.protocol
    assert retried.attempts[1].metrics[0].rm == state.protocol.metrics[0].rm
    assert "adverse_unresolved" not in _codes(retried.decision.reasons)
    third = reduce(
        retried, SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="nope", sequence_offset=20))
    )
    assert len(third.attempts) == 2
    assert third.decision == retried.decision
    assert third.rejected_actions[-1].code == "attempt_limit"

    again = reduce(
        opened,
        SubmitAttempt(
            _pairs(spec, LEVELS, (11.0, 11.0, 11.0, 11.0, 7.0), prefix="second", sequence_offset=10)
        ),
    )
    assert again.attempts[1].status is ResultStatus.INCONCLUSIVE
    assert again.decision.status is ResultStatus.NO_GO
    assert again.decision.terminal is True
    assert "terminal_inconclusive" in _codes(again.decision.reasons)
    assert "adverse_unresolved" in _codes(again.decision.reasons)
    no_third = reduce(
        again, SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="later", sequence_offset=20))
    )
    assert no_third.decision == again.decision
    assert no_third.rejected_actions[-1].code == "attempt_limit"


# The retry test compares attempt length with this local name.
ATTEMPT_COUNT = 2


def test_conclusive_failure_cannot_be_invalidated_or_resealed() -> None:
    state, spec = _sealed()
    failed = reduce(
        state, SubmitAttempt(_pairs(spec, LEVELS, tuple(10.0 + 1e-6 for _ in range(5))))
    )
    assert failed.decision.status is ResultStatus.FAIL
    kept = reduce(failed, InvalidateAttempt(failed.attempts[0].evidence_ids, CORRECTION))
    assert kept.attempts[0].invalidated is False
    assert kept.decision == failed.decision
    assert kept.rejected_actions[-1].code == "conclusive_attempt_not_invalidated"
    resealed = reduce(failed, _action(spec, _calibration(CALIBRATION, spec)))
    assert resealed.protocol is failed.protocol
    assert resealed.protocol.protocol_id == failed.protocol.protocol_id
    assert resealed.rejected_actions[-1].code == "protocol_already_frozen"


def test_rejected_attempt_can_be_fully_invalidated_then_rerun() -> None:
    state, spec = _sealed()
    partial = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="bad")[:4]))
    assert partial.decision.status is ResultStatus.REJECTED
    assert partial.decision.terminal is False
    opened = _invalidate(partial)
    repaired = reduce(
        opened,
        SubmitAttempt(
            _pairs(spec, LEVELS, EQUAL_CANDIDATES, prefix="repaired", sequence_offset=10)
        ),
    )
    assert repaired.decision.status is ResultStatus.PASS
    assert repaired.decision.authoritative_attempt == 2
    assert repaired.protocol.budgets == state.protocol.budgets


@pytest.mark.parametrize(
    ("first_offset", "retry_offset"),
    [(0, 0), (0, 9), (30, 10)],
)
def test_retry_cannot_rename_old_or_partially_later_sequences(
    first_offset: int, retry_offset: int
) -> None:
    state, spec = _sealed()
    first_pairs = _pairs(
        spec,
        LEVELS,
        (11.0, 11.0, 11.0, 11.0, 7.0),
        sequence_offset=first_offset,
    )
    first = reduce(state, SubmitAttempt(first_pairs))
    assert first.decision.status is ResultStatus.INCONCLUSIVE
    opened = _invalidate(first)
    renamed = _pairs(spec, LEVELS, LEVELS, prefix="renamed", sequence_offset=retry_offset)
    rejected = reduce(opened, SubmitAttempt(renamed))
    assert rejected.decision.status is ResultStatus.REJECTED
    assert rejected.decision.terminal is True
    assert len(rejected.attempts) == 2
    assert "sequence_replay" in _codes(rejected.decision.reasons)
    assert "selective_replay" not in _codes(rejected.decision.reasons)
    assert "attempt_budget_exhausted" in _codes(rejected.decision.reasons)
    third = reduce(
        rejected,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="third", sequence_offset=100)),
    )
    assert third.decision == rejected.decision
    assert len(third.attempts) == 2
    assert third.rejected_actions[-1].code == "attempt_limit"


@pytest.mark.parametrize(
    "malformation",
    ["bad_index", "duplicate_index", "extra_index", "bad_role", "missing_leg", "bad_item"],
)
def test_malformed_submissions_burn_all_valid_leg_sequences(malformation: str) -> None:
    state, spec = _sealed()
    pairs = _pairs(spec, LEVELS, LEVELS)
    first = _replace(pairs[0].first, sequence=80)
    second = _replace(pairs[0].second, sequence=81)
    malformed = _replace(pairs[0], first=first, second=second)
    expected_sequences = {80, 81}
    if malformation == "bad_index":
        malformed = _replace(malformed, pair_index="not-an-index")
    elif malformation == "duplicate_index":
        malformed = _replace(malformed, pair_index=2)
    elif malformation == "extra_index":
        malformed = _replace(malformed, pair_index=6)
    elif malformation == "bad_role":
        malformed = _replace(malformed, first=_replace(first, leg="not-a-role"))
    elif malformation == "missing_leg":
        malformed = _replace(malformed, second=None)
        expected_sequences = {80}
    submitted = (malformed, *pairs[1:])
    if malformation == "bad_item":
        submitted = (*submitted, object())
    rejected = reduce(state, SubmitAttempt(submitted))
    assert rejected.decision.status is ResultStatus.REJECTED
    assert rejected.decision.terminal is False
    assert len(rejected.attempts) == 1
    assert expected_sequences <= set(rejected.consumed_sequences)
    assert state.consumed_sequences == ()
    opened = _invalidate(rejected)
    assert opened.consumed_sequences == rejected.consumed_sequences

    not_later = _pairs(spec, LEVELS, LEVELS, prefix="too-early", sequence_offset=10)
    replay = reduce(opened, SubmitAttempt(not_later))
    assert replay.decision.status is ResultStatus.REJECTED
    assert "sequence_replay" in _codes(replay.decision.reasons)

    later = _pairs(spec, LEVELS, EQUAL_CANDIDATES, prefix="later", sequence_offset=71)
    retried = reduce(opened, SubmitAttempt(later))
    assert retried.decision.status is ResultStatus.PASS
    assert retried.protocol is state.protocol
    assert expected_sequences <= set(retried.consumed_sequences)
    assert set(range(82, 92)) <= set(retried.consumed_sequences)


def test_blocked_submissions_also_advance_the_sequence_frontier() -> None:
    state, spec = _sealed()
    first = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, (11.0, 11.0, 11.0, 11.0, 7.0))))
    premature = reduce(
        first,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="premature", sequence_offset=20)),
    )
    assert premature.rejected_actions[-1].code == "bad_retry"
    assert premature.attempts == first.attempts
    assert premature.decision == first.decision
    assert max(premature.consumed_sequences) == 40
    assert max(first.consumed_sequences) == 20
    opened = _invalidate(premature)
    too_early = reduce(
        opened,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="too-early", sequence_offset=10)),
    )
    assert too_early.decision.status is ResultStatus.REJECTED
    assert "sequence_replay" in _codes(too_early.decision.reasons)
    later = reduce(
        opened,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="later", sequence_offset=30)),
    )
    assert later.decision.status is ResultStatus.PASS
    third = reduce(
        later,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="third", sequence_offset=40)),
    )
    assert third.rejected_actions[-1].code == "attempt_limit"
    assert third.attempts == later.attempts
    assert max(third.consumed_sequences) == 60


@pytest.mark.parametrize(
    "sequences",
    [
        ((11, 14), (12, 13), (15, 16), (17, 18), (19, 20)),
        ((13, 14), (11, 12), (15, 16), (17, 18), (19, 20)),
    ],
    ids=["interleaved", "reordered-blocks"],
)
def test_pair_indexes_must_follow_execution_chronology(
    sequences: tuple[tuple[int, int], ...],
) -> None:
    state, spec = _sealed()
    pairs = tuple(
        _replace(
            pair,
            first=_replace(pair.first, sequence=first),
            second=_replace(pair.second, sequence=second),
        )
        for pair, (first, second) in zip(
            _pairs(spec, LEVELS, EQUAL_CANDIDATES), sequences, strict=True
        )
    )
    for submitted in (pairs, tuple(reversed(pairs))):
        rejected = reduce(state, SubmitAttempt(submitted))
        assert rejected.decision.status is ResultStatus.REJECTED
        assert _codes(rejected.decision.reasons) == {"pair_chronology"}


def test_pair_submission_order_does_not_change_indexed_differences() -> None:
    state, spec = _sealed()
    pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    forward = reduce(state, SubmitAttempt(pairs))
    backward = reduce(state, SubmitAttempt(tuple(reversed(pairs))))
    assert forward.attempts[0].metrics[0].differences == backward.attempts[0].metrics[0].differences
    assert forward.decision.status is backward.decision.status is ResultStatus.PASS


def test_reducer_does_not_mutate_prior_state_or_admit_before_seal() -> None:
    state, spec = _sealed()
    snapshot = state.decision
    result = reduce(state, SubmitAttempt(_pairs(spec, LEVELS, EQUAL_CANDIDATES)))
    assert state.decision == snapshot
    assert state.attempts == ()
    assert result.decision.status is ResultStatus.PASS
    unsealed = reduce(initial_state(), SubmitAttempt(()))
    assert unsealed.protocol is None
    assert unsealed.rejected_actions[-1].code == "protocol_not_frozen"
    with pytest.raises(TypeError):
        reduce(initial_state(), object())  # type: ignore[arg-type]


def test_same_protocol_inputs_have_one_id_and_resolution_changes_it() -> None:
    left, _left_spec = _sealed()
    right, _right_spec = _sealed()
    assert left.protocol.protocol_id == right.protocol.protocol_id
    coarser, _spec = _sealed(spec_overrides={"instrument_resolution": 0.5})
    assert coarser.protocol.protocol_id != left.protocol.protocol_id
    assert coarser.protocol.metrics[0].rm > left.protocol.metrics[0].rm
    assert (
        coarser.protocol.metrics[0].baseline_variance == left.protocol.metrics[0].baseline_variance
    )


def test_seal_snapshots_mutable_calibration_legs_and_nested_metrics() -> None:
    spec = _spec()
    original = _calibration(CALIBRATION, spec)
    observations = [list(leg.metrics) for leg in original]
    calibration = [
        _replace(leg, metrics=metrics) for leg, metrics in zip(original, observations, strict=True)
    ]
    sealed = reduce(initial_state(), _action(spec, calibration))
    protocol = sealed.protocol
    assert protocol is not None
    digest = protocol.protocol_id
    frozen_metrics = protocol.metrics
    pairs = _pairs(spec, LEVELS, EQUAL_CANDIDATES)
    before = reduce(sealed, SubmitAttempt(pairs))
    assert before.decision.status is ResultStatus.PASS

    observations[0][0] = MetricObservation(spec.name, spec.estimator, 1e100)
    observations[1].clear()
    observations[2].append(MetricObservation("extra", "p50", 10.0))
    calibration.clear()
    assert protocol.calibration == original
    assert isinstance(protocol.calibration, tuple)
    assert all(isinstance(leg.metrics, tuple) for leg in protocol.calibration)
    assert protocol.metrics == frozen_metrics
    assert protocol.protocol_id == digest

    rebuilt = reduce(initial_state(), _action(spec, protocol.calibration))
    assert rebuilt.protocol == protocol
    after = reduce(sealed, SubmitAttempt(pairs))
    assert after.decision == before.decision
    assert after.attempts[0].metrics == before.attempts[0].metrics


@pytest.mark.parametrize(
    ("calibration", "code"),
    [
        ((10**400, 10.0, 10.0, 10.0, 10.0), "nonfinite"),
        ((1e308, 1e308, 1e308, 1e308, 1e308), "nonfinite_derived"),
        ((1e200, 0.0, 0.0, 0.0, 0.0), "nonfinite_derived"),
    ],
    ids=["integer-conversion", "calibration-sum", "calibration-variance"],
)
def test_calibration_arithmetic_overflow_is_an_explicit_seal_rejection(
    calibration: tuple[float, ...], code: str
) -> None:
    state, _spec = _sealed(calibration_values=calibration)
    assert state.protocol is None
    assert state.decision.status is ResultStatus.REJECTED
    assert code in _codes(state.decision.reasons)
    assert state.attempts == ()


@pytest.mark.parametrize(
    ("baselines", "candidates"),
    [
        ((-1e308,) * 5, (1e308,) * 5),
        (LEVELS, (1e308,) * 5),
        ((0.0,) * 5, (1e200, -1e200, 1e200, -1e200, 0.0)),
    ],
    ids=["difference", "point-sum", "variance"],
)
def test_derived_attempt_overflow_rejects_and_consumes_the_retry_budget(
    baselines: tuple[float, ...], candidates: tuple[float, ...]
) -> None:
    state, spec = _sealed()
    first = reduce(state, SubmitAttempt(_pairs(spec, baselines, candidates)))
    assert first.decision.status is ResultStatus.REJECTED
    assert _codes(first.decision.reasons) == {"nonfinite_derived"}
    assert first.decision.terminal is False
    assert len(first.attempts) == 1
    assert first.attempts[0].metrics == ()
    assert all(gate.status is ResultStatus.PASS for gate in first.attempts[0].gates)
    assert first.consumed_sequences == tuple(range(11, 21))
    opened = _invalidate(first)

    repaired = reduce(
        opened,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="repaired", sequence_offset=10)),
    )
    assert repaired.decision.status is ResultStatus.PASS
    assert repaired.protocol is state.protocol
    second_bad = reduce(
        opened,
        SubmitAttempt(_pairs(spec, baselines, candidates, prefix="bad", sequence_offset=10)),
    )
    assert second_bad.decision.status is ResultStatus.REJECTED
    assert second_bad.decision.terminal is True
    assert "attempt_budget_exhausted" in _codes(second_bad.decision.reasons)
    assert len(second_bad.attempts) == 2
    third = reduce(
        second_bad,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="third", sequence_offset=20)),
    )
    assert third.decision == second_bad.decision
    assert third.rejected_actions[-1].code == "attempt_limit"


def test_huge_integer_evidence_cannot_escape_or_reset_the_attempt_budget() -> None:
    state, spec = _sealed()
    pairs = _replace_role(
        _pairs(spec, LEVELS, LEVELS),
        2,
        LegRole.CANDIDATE,
        metrics=(MetricObservation(spec.name, spec.estimator, 10**400),),
    )
    first = reduce(state, SubmitAttempt(pairs))
    assert first.decision.status is ResultStatus.REJECTED
    assert "nonfinite" in _codes(first.decision.reasons)
    assert len(first.attempts) == 1
    opened = _invalidate(first)
    renamed = _pairs(spec, LEVELS, LEVELS, prefix="renamed")
    second = reduce(opened, SubmitAttempt(renamed))
    assert second.decision.status is ResultStatus.REJECTED
    assert "sequence_replay" in _codes(second.decision.reasons)
    assert second.decision.terminal is True
    third = reduce(
        second,
        SubmitAttempt(_pairs(spec, LEVELS, LEVELS, prefix="third", sequence_offset=10)),
    )
    assert third.rejected_actions[-1].code == "attempt_limit"
    assert len(third.attempts) == 2


def test_resolution_helpers_reject_non_finite_inputs() -> None:
    with pytest.raises(ValueError):
        measurement_pair_quantization(0.0)
    with pytest.raises(ValueError):
        measurement_pair_quantization(float("nan"))
    with pytest.raises(ValueError):
        sample_variance((1.0, 1.0, 1.0, 1.0, float("inf")))


@pytest.mark.parametrize("value", [True, "1.0", None])
def test_statistical_resolution_rejects_invalid_types(value: object) -> None:
    with pytest.raises(TypeError):
        statistical_resolution(value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), 10**400, 1e308])
def test_statistical_resolution_rejects_nonfinite_inputs_and_results(value: float) -> None:
    with pytest.raises(ValueError):
        statistical_resolution(value)
    with pytest.raises(ValueError):
        detection_resolution(value, 0.001)


def test_module_is_stdlib_only_and_does_not_invoke_models() -> None:
    source_path = Path(_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "typing",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module is not None
            imported.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"system", "popen", "run", "Popen", "check_call"}
    assert imported <= allowed
    assert "worker" not in imported
    assert "backend" not in imported
