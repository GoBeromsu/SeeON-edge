"""Pure reducer for the frozen five-pair non-regression protocol.

The reducer is typed, immutable, and limited to the standard library. It does not
read product configuration, start a process, invoke a model, or synthesize
measurements. Callers supply calibration and paired summaries; this module only
judges them.

Before any candidate evidence, the seal pins shared input, model, and config
identities and separate baseline and candidate source identities; each metric's
estimator, adverse direction, transform, and instrument
resolution; absolute capacity budgets; the GPU and CPU provider identities; and
exactly five baseline calibration legs. Detection resolution ``R_m`` is frozen
from that baseline variance and from measurement-pair quantization. ``R_m`` is
the procedure's identification limit, not an allowed regression.

Analysis scale: absolute metrics use the reported estimator value. Log-ratio
metrics use the natural log and accept only strictly positive values. The paired
transform is signed so a positive difference is adverse. Five pairs use a paired
Student t with 4 degrees of freedom. The one-sided 95% coefficient is
``2.131846786326649``.

Approved admission rule for each metric, after the hard gates accept the evidence:

- PASS requires paired point <= 0 and upper one-sided bound < ``R_m``.
- Confirmed adverse evidence (lower one-sided bound > 0) is FAIL at any size.
- Every other finite result is INCONCLUSIVE.

Correctness multisets, state, delivery loss, duplicate, and unknown-gap, reported
provider and call counts, and absolute resource budgets are independent hard
gates. The inventory covers live pose/fall, on-demand bed, and stored-clip
pose/bed calls, not arbitrary benchmark configuration. The unchanged baseline
uses CPU for live fall, on-demand bed, and stored-clip pose/bed; only live pose
is GPU-owned. These exceptions are mode-specific, not role-only. Every
candidate call must report the pinned GPU provider and a CPU count of 0. These
self-reported summaries do not prove ORT runtime placement or qualify a product;
producer provenance must be verified separately. Absent lanes and mismatched
counts are rejected. Attempt 2 requires a diagnosed correction and complete
invalidation of attempt 1, on the same frozen protocol and budget, with all ten
legs later than every previously submitted leg. Attempt 3 is refused. A second
inconclusive result is a terminal no-go.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final

PAIR_COUNT: Final = 5
PAIRED_DEGREES_OF_FREEDOM: Final = 4
ONE_SIDED_95_COEFFICIENT: Final = 2.131846786326649
ATTEMPT_LIMIT: Final = 2
BASELINE_CPU_LANES: Final = frozenset(
    (("live", "fall"), ("on_demand", "bed"), ("stored_clip", "pose"), ("stored_clip", "bed"))
)
REQUIRED_MODEL_LANES: Final = frozenset(
    (
        ("live", "pose"),
        ("live", "fall"),
        ("on_demand", "bed"),
        ("stored_clip", "pose"),
        ("stored_clip", "bed"),
    )
)
_PIN_FIELDS: Final = ("input_id", "model_id", "config_id")


class ResultStatus(Enum):
    """Admission status. OPEN means the protocol is not yet judged."""

    OPEN = "open"
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"
    NO_GO = "no_go"


class Direction(Enum):
    """Sign convention before the adverse transform. Positive means adverse."""

    HIGHER_ADVERSE = "higher_adverse"
    LOWER_ADVERSE = "lower_adverse"


class Transform(Enum):
    """ABSOLUTE subtracts estimator values. LOG_RATIO subtracts natural logs."""

    ABSOLUTE = "absolute"
    LOG_RATIO = "log_ratio"


class LegRole(Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class Reason:
    code: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    estimator: str
    direction: Direction
    transform: Transform
    instrument_resolution: float


@dataclass(frozen=True, slots=True)
class MetricObservation:
    name: str
    estimator: str
    value: float


@dataclass(frozen=True, slots=True)
class CapacityBudget:
    name: str
    absolute_limit: float


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class ExpectedCall:
    execution_mode: str
    role: str
    call_count: int


@dataclass(frozen=True, slots=True)
class ModelCall:
    execution_mode: str
    role: str
    provider: str
    call_count: int
    cpu_count: int


@dataclass(frozen=True, slots=True)
class CalibrationLeg:
    """One pre-candidate baseline leg. Candidate rows are not calibration."""

    leg_index: int
    evidence_id: str
    sequence: int
    leg: LegRole
    input_id: str
    model_id: str
    config_id: str
    source_id: str
    metrics: tuple[MetricObservation, ...]


@dataclass(frozen=True, slots=True)
class LegSummary:
    evidence_id: str
    sequence: int
    leg: LegRole
    input_id: str
    model_id: str
    config_id: str
    source_id: str
    metrics: tuple[MetricObservation, ...]
    outcomes: tuple[str, ...]
    state_facts: tuple[str, ...]
    delivery_loss: int
    delivery_duplicate: int
    unknown_gap: int
    model_calls: tuple[ModelCall, ...]
    resources: tuple[ResourceObservation, ...]


@dataclass(frozen=True, slots=True)
class PairSummary:
    pair_index: int
    evidence_id: str
    first: LegSummary | None
    second: LegSummary | None


@dataclass(frozen=True, slots=True)
class SealProtocol:
    input_id: str
    model_id: str
    config_id: str
    baseline_source_id: str
    candidate_source_id: str
    gpu_provider: str
    cpu_provider: str
    metrics: tuple[MetricSpec, ...]
    budgets: tuple[CapacityBudget, ...]
    expected_calls: tuple[ExpectedCall, ...]
    calibration: tuple[CalibrationLeg, ...]
    freeze_sequence: int


@dataclass(frozen=True, slots=True)
class DiagnosedCorrection:
    diagnosis: str
    correction: str


@dataclass(frozen=True, slots=True)
class SubmitAttempt:
    pairs: tuple[PairSummary, ...]


@dataclass(frozen=True, slots=True)
class InvalidateAttempt:
    evidence_ids: tuple[str, ...]
    correction: DiagnosedCorrection


@dataclass(frozen=True, slots=True)
class FrozenMetric:
    spec: MetricSpec
    baseline_variance: float
    pair_quantization: float
    statistical_resolution: float
    rm: float


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    input_id: str
    model_id: str
    config_id: str
    baseline_source_id: str
    candidate_source_id: str
    gpu_provider: str
    cpu_provider: str
    metrics: tuple[FrozenMetric, ...]
    budgets: tuple[CapacityBudget, ...]
    expected_calls: tuple[ExpectedCall, ...]
    calibration: tuple[CalibrationLeg, ...]
    freeze_sequence: int
    protocol_id: str
    reserved_evidence_ids: tuple[str, ...]
    reserved_sequences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MetricAssessment:
    name: str
    estimator: str
    direction: Direction
    transform: Transform
    differences: tuple[float, ...]
    point: float
    difference_variance: float
    standard_error: float
    degrees_of_freedom: int
    coefficient: float
    lower_bound: float
    upper_bound: float
    baseline_variance: float
    pair_quantization: float
    statistical_resolution: float
    rm: float
    status: ResultStatus
    reason: Reason


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    status: ResultStatus
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AttemptResult:
    index: int
    status: ResultStatus
    reasons: tuple[Reason, ...]
    evidence_ids: tuple[str, ...]
    metrics: tuple[MetricAssessment, ...]
    gates: tuple[GateResult, ...]
    invalidated: bool


@dataclass(frozen=True, slots=True)
class Decision:
    status: ResultStatus
    reasons: tuple[Reason, ...]
    terminal: bool
    authoritative_attempt: int | None


@dataclass(frozen=True, slots=True)
class TransitionState:
    protocol: FrozenProtocol | None
    attempts: tuple[AttemptResult, ...]
    burned_evidence_ids: tuple[str, ...]
    consumed_sequences: tuple[int, ...]
    correction: DiagnosedCorrection | None
    rejected_actions: tuple[Reason, ...]
    decision: Decision


def measurement_pair_quantization(instrument_resolution: float) -> float:
    """Quantization of a paired difference of two measurements on a grid of q.

    Each leg is resolved in steps of ``q``, so the difference is resolved in
    steps of ``q``. The value is not an allowed degradation.
    """

    resolution = _positive_resolution(instrument_resolution)
    if resolution is None:
        raise ValueError("instrument resolution must be a positive finite number")
    return resolution


def statistical_resolution(baseline_variance: float) -> float:
    """One-sided critical mean difference under the calibration variance model.

    Legs are treated as independent with equal variance, so a paired difference
    has variance ``2 * s_b^2``. The standard error of the five-pair mean is
    ``sqrt(2 * s_b^2 / 5)``. Multiplied by the df=4 one-sided 95% coefficient,
    that is the statistical term of ``R_m``. Candidate samples are not used.
    """

    if isinstance(baseline_variance, bool) or not isinstance(baseline_variance, (int, float)):
        raise TypeError("baseline variance must be a number")
    variance = _finite_float(baseline_variance)
    if variance is None or variance < 0.0:
        raise ValueError("baseline variance must be a non-negative finite number")
    return _finite_result(ONE_SIDED_95_COEFFICIENT * math.sqrt(2.0 * variance / PAIR_COUNT))


def detection_resolution(baseline_variance: float, instrument_resolution: float) -> float:
    """Frozen ``R_m``: statistical resolution plus measurement-pair quantization."""

    statistical = statistical_resolution(baseline_variance)
    return _finite_result(statistical + measurement_pair_quantization(instrument_resolution))


def sample_variance(values: Sequence[float]) -> float:
    """Unbiased sample variance of exactly five finite numbers."""

    numbers = _five_floats(values)
    mean = _finite_result(math.fsum(numbers) / PAIR_COUNT)
    squared = _finite_result(math.fsum((number - mean) ** 2 for number in numbers))
    return _finite_result(squared / PAIRED_DEGREES_OF_FREEDOM)


def required_pair_order(pair_index: int) -> tuple[LegRole, LegRole]:
    """Odd pairs run baseline then candidate. Even pairs run candidate then baseline."""

    if not _is_int(pair_index) or not 1 <= pair_index <= PAIR_COUNT:
        raise ValueError("pair index must be an integer from 1 through 5")
    if pair_index % 2 == 1:
        return (LegRole.BASELINE, LegRole.CANDIDATE)
    return (LegRole.CANDIDATE, LegRole.BASELINE)


def initial_state() -> TransitionState:
    return TransitionState(
        protocol=None,
        attempts=(),
        burned_evidence_ids=(),
        consumed_sequences=(),
        correction=None,
        rejected_actions=(),
        decision=Decision(
            status=ResultStatus.OPEN,
            reasons=(Reason("protocol_not_frozen"),),
            terminal=False,
            authoritative_attempt=None,
        ),
    )


def reduce(
    state: TransitionState,
    action: SealProtocol | InvalidateAttempt | SubmitAttempt,
) -> TransitionState:
    """Return the next immutable state. The input state is not modified."""

    if not isinstance(state, TransitionState):
        raise TypeError("state must be a TransitionState")
    if isinstance(action, SealProtocol):
        return _seal(state, action)
    if isinstance(action, InvalidateAttempt):
        return _invalidate(state, action)
    if isinstance(action, SubmitAttempt):
        return _submit(state, action)
    raise TypeError(f"unknown action: {type(action).__name__}")


def _seal(state: TransitionState, action: SealProtocol) -> TransitionState:
    if state.protocol is not None:
        return _reject_action(state, Reason("protocol_already_frozen"))
    reasons, protocol = _freeze(action)
    if protocol is None:
        return replace(
            state,
            rejected_actions=(*state.rejected_actions, *reasons),
            decision=Decision(
                status=ResultStatus.REJECTED,
                reasons=tuple(reasons),
                terminal=False,
                authoritative_attempt=None,
            ),
        )
    return replace(
        state,
        protocol=protocol,
        decision=Decision(
            status=ResultStatus.OPEN,
            reasons=(Reason("protocol_frozen"),),
            terminal=False,
            authoritative_attempt=None,
        ),
    )


def _invalidate(state: TransitionState, action: InvalidateAttempt) -> TransitionState:
    if state.protocol is None:
        return _reject_action(state, Reason("protocol_not_frozen"))
    if len(state.attempts) != 1:
        code = "bad_retry" if not state.attempts else "attempt_limit"
        return _reject_action(state, Reason(code))
    attempt = state.attempts[0]
    if attempt.invalidated:
        return _reject_action(state, Reason("already_invalidated"))
    open_statuses = (ResultStatus.INCONCLUSIVE, ResultStatus.REJECTED)
    if state.decision.terminal or attempt.status not in open_statuses:
        return _reject_action(state, Reason("conclusive_attempt_not_invalidated"))
    if not _valid_correction(action.correction):
        return _reject_action(state, Reason("missing_correction"))
    identifiers = _identifier_tuple(action.evidence_ids)
    if identifiers is None or identifiers != attempt.evidence_ids:
        return _reject_action(state, Reason("partial_invalidation"))
    burned = tuple(sorted(set(state.burned_evidence_ids).union(identifiers)))
    return replace(
        state,
        attempts=(replace(attempt, invalidated=True),),
        burned_evidence_ids=burned,
        correction=action.correction,
        decision=Decision(
            status=attempt.status,
            reasons=(Reason("attempt_invalidated"), Reason("awaiting_retry")),
            terminal=False,
            authoritative_attempt=None,
        ),
    )


def _submit(state: TransitionState, action: SubmitAttempt) -> TransitionState:
    prior_sequences = state.consumed_sequences
    consumed = set(prior_sequences).union(_submitted_sequences(action.pairs))
    state = replace(
        state,
        consumed_sequences=tuple(sorted(consumed)),
    )
    if state.protocol is None:
        return _reject_action(state, Reason("protocol_not_frozen"))
    if len(state.attempts) >= ATTEMPT_LIMIT or state.decision.status is ResultStatus.NO_GO:
        return _reject_action(state, Reason("attempt_limit"))
    retry_slot = len(state.attempts) == 1
    if retry_slot:
        current = state.attempts[0]
        blocked = not current.invalidated or state.correction is None or state.decision.terminal
        if blocked:
            return _reject_action(state, Reason("bad_retry"))
    prior = state.attempts[0].evidence_ids if retry_slot else ()
    evaluated = _evaluate(state.protocol, action.pairs, prior, prior_sequences)
    result = replace(evaluated, index=len(state.attempts) + 1)
    return replace(
        state,
        attempts=(*state.attempts, result),
        decision=_admission(result, retry_slot=retry_slot),
    )


def _admission(result: AttemptResult, *, retry_slot: bool) -> Decision:
    if retry_slot and result.status is ResultStatus.INCONCLUSIVE:
        return Decision(
            status=ResultStatus.NO_GO,
            reasons=(Reason("terminal_inconclusive"), *result.reasons),
            terminal=True,
            authoritative_attempt=result.index,
        )
    if retry_slot and result.status is ResultStatus.REJECTED:
        return Decision(
            status=ResultStatus.REJECTED,
            reasons=(*result.reasons, Reason("attempt_budget_exhausted")),
            terminal=True,
            authoritative_attempt=result.index,
        )
    terminal = result.status in (ResultStatus.PASS, ResultStatus.FAIL)
    return Decision(
        status=result.status,
        reasons=result.reasons,
        terminal=terminal,
        authoritative_attempt=result.index,
    )


def _reject_action(state: TransitionState, reason: Reason) -> TransitionState:
    return replace(state, rejected_actions=(*state.rejected_actions, reason))


def _submitted_sequences(pairs: object) -> tuple[int, ...]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        return ()
    return tuple(
        leg.sequence
        for pair in pairs
        if isinstance(pair, PairSummary)
        for leg in (pair.first, pair.second)
        if isinstance(leg, LegSummary) and _is_int(leg.sequence)
    )


def _evaluate(
    protocol: FrozenProtocol,
    pairs: object,
    prior_evidence_ids: tuple[str, ...],
    prior_sequences: tuple[int, ...],
) -> AttemptResult:
    reasons: list[Reason] = []
    found: dict[int, PairSummary] = {}
    collected: list[PairSummary] = []
    identifiers: list[str] = []
    _collect_pairs(pairs, found, collected, identifiers, reasons)
    _structural(protocol, collected, prior_evidence_ids, identifiers, reasons)
    if prior_sequences:
        frontier = max(prior_sequences)
        for sequence in _submitted_sequences(pairs):
            if sequence <= frontier:
                reasons.append(Reason("sequence_replay", f"{sequence}<={frontier}"))
    for index in range(1, PAIR_COUNT + 1):
        if index not in found:
            reasons.append(Reason("missing_pair", str(index)))
    _check_pair_chronology(found, reasons)
    evidence_ids = tuple(sorted(set(identifiers)))
    if reasons:
        return _rejected_attempt(evidence_ids, reasons)
    ordered = tuple(found[index] for index in range(1, PAIR_COUNT + 1))
    gates = _hard_gates(protocol, ordered)
    try:
        metrics = _metric_assessments(protocol, ordered)
    except (OverflowError, ValueError) as error:
        return replace(
            _rejected_attempt(evidence_ids, [Reason("nonfinite_derived", str(error))]),
            gates=tuple(gates),
        )
    status, status_reasons = _combine(gates, metrics)
    return AttemptResult(
        index=0,
        status=status,
        reasons=status_reasons,
        evidence_ids=evidence_ids,
        metrics=metrics,
        gates=tuple(gates),
        invalidated=False,
    )


def _rejected_attempt(evidence_ids: tuple[str, ...], reasons: list[Reason]) -> AttemptResult:
    return AttemptResult(
        index=0,
        status=ResultStatus.REJECTED,
        reasons=tuple(reasons),
        evidence_ids=evidence_ids,
        metrics=(),
        gates=(),
        invalidated=False,
    )


def _collect_pairs(
    pairs: object,
    found: dict[int, PairSummary],
    collected: list[PairSummary],
    identifiers: list[str],
    reasons: list[Reason],
) -> None:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        reasons.append(Reason("invalid_evidence", "pairs"))
        return
    if len(pairs) != PAIR_COUNT:
        reasons.append(Reason("pair_count", str(len(pairs))))
    for item in pairs:
        if not isinstance(item, PairSummary):
            reasons.append(Reason("invalid_evidence", "pair"))
            continue
        collected.append(item)
        _remember_id(item.evidence_id, identifiers, reasons)
        if not _is_int(item.pair_index):
            reasons.append(Reason("invalid_evidence", "pair_index"))
            continue
        if not 1 <= item.pair_index <= PAIR_COUNT:
            reasons.append(Reason("missing_pair", str(item.pair_index)))
            continue
        if item.pair_index in found:
            reasons.append(Reason("duplicate_evidence", f"pair_index:{item.pair_index}"))
            continue
        found[item.pair_index] = item


def _check_pair_chronology(found: dict[int, PairSummary], reasons: list[Reason]) -> None:
    for index in range(1, PAIR_COUNT):
        current = found.get(index)
        following = found.get(index + 1)
        if current is None or following is None:
            continue
        second = current.second
        first = following.first
        if (
            isinstance(second, LegSummary)
            and isinstance(first, LegSummary)
            and _is_int(second.sequence)
            and _is_int(first.sequence)
            and second.sequence >= first.sequence
        ):
            reasons.append(Reason("pair_chronology", f"{index}:{index + 1}"))


def _structural(
    protocol: FrozenProtocol,
    collected: list[PairSummary],
    prior_evidence_ids: tuple[str, ...],
    identifiers: list[str],
    reasons: list[Reason],
) -> None:
    prior = set(prior_evidence_ids)
    reserved_ids = set(protocol.reserved_evidence_ids)
    reserved_sequences = set(protocol.reserved_sequences)
    seen_sequences: set[int] = set()
    for pair in collected:
        _structural_pair(
            protocol,
            pair,
            identifiers,
            reasons,
            reserved_sequences,
            seen_sequences,
        )
    _mark_collisions(identifiers, prior, reserved_ids, reasons)


def _structural_pair(
    protocol: FrozenProtocol,
    pair: PairSummary,
    identifiers: list[str],
    reasons: list[Reason],
    reserved_sequences: set[int],
    seen_sequences: set[int],
) -> None:
    valid: list[LegSummary] = []
    if pair.first is None or pair.second is None:
        reasons.append(Reason("missing_leg", str(pair.pair_index)))
    for leg in (pair.first, pair.second):
        if leg is None:
            continue
        if not isinstance(leg, LegSummary):
            reasons.append(Reason("invalid_evidence", f"leg:{pair.pair_index}"))
            continue
        valid.append(leg)
        _remember_id(leg.evidence_id, identifiers, reasons)
        _structural_leg(protocol, pair.pair_index, leg, reasons, reserved_sequences, seen_sequences)
    if not _is_int(pair.pair_index) or not 1 <= pair.pair_index <= PAIR_COUNT or len(valid) != 2:
        return
    expected = required_pair_order(pair.pair_index)
    roles_match = valid[0].leg is expected[0] and valid[1].leg is expected[1]
    if not roles_match:
        reasons.append(Reason("wrong_order", str(pair.pair_index)))
        return
    if (
        _is_int(valid[0].sequence)
        and _is_int(valid[1].sequence)
        and valid[0].sequence >= valid[1].sequence
    ):
        reasons.append(Reason("wrong_order", str(pair.pair_index)))


def _structural_leg(
    protocol: FrozenProtocol,
    pair_index: int,
    leg: LegSummary,
    reasons: list[Reason],
    reserved_sequences: set[int],
    seen_sequences: set[int],
) -> None:
    if not isinstance(leg.leg, LegRole):
        reasons.append(Reason("invalid_evidence", str(pair_index)))
        return
    where = f"{pair_index}:{leg.leg.value}"
    if (
        leg.leg is LegRole.CANDIDATE
        and _is_int(leg.sequence)
        and leg.sequence <= protocol.freeze_sequence
    ):
        reasons.append(Reason("candidate_before_freeze", where))
    if (
        leg.leg is LegRole.BASELINE
        and _is_int(leg.sequence)
        and leg.sequence <= protocol.freeze_sequence
    ):
        reasons.append(Reason("paired_before_freeze", where))
    if not _is_int(leg.sequence):
        reasons.append(Reason("invalid_evidence", f"sequence:{where}"))
    elif leg.sequence in reserved_sequences or leg.sequence in seen_sequences:
        reasons.append(Reason("duplicate_evidence", f"sequence:{leg.sequence}"))
    else:
        seen_sequences.add(leg.sequence)
    _check_pins(protocol, leg, reasons, where)
    _check_metrics(protocol, leg.metrics, reasons, where)
    _check_resources(protocol, leg.resources, reasons, where)
    _check_texts(leg.outcomes, "invalid_outcome", where, reasons)
    _check_texts(leg.state_facts, "invalid_state", where, reasons)
    _check_delivery(leg, reasons, where)
    _check_model_shape(protocol, leg, reasons, where)


def _mark_collisions(
    identifiers: list[str],
    prior: set[str],
    reserved_ids: set[str],
    reasons: list[Reason],
) -> None:
    for identifier, count in sorted(Counter(identifiers).items()):
        if count > 1 or identifier in reserved_ids:
            reasons.append(Reason("duplicate_evidence", identifier))
        elif identifier in prior:
            reasons.append(Reason("selective_replay", identifier))


def _hard_gates(protocol: FrozenProtocol, pairs: tuple[PairSummary, ...]) -> list[GateResult]:
    reference_outcomes, reference_state = _reference(pairs[0])
    outcome_mismatch = False
    state_mismatch = False
    for pair in pairs:
        baseline, candidate = _ordered_legs(pair)
        if (
            Counter(baseline.outcomes) != reference_outcomes
            or Counter(candidate.outcomes) != reference_outcomes
        ):
            outcome_mismatch = True
        states = (Counter(baseline.state_facts), Counter(candidate.state_facts))
        if any(observed != reference_state for observed in states):
            state_mismatch = True
    gates = [
        _pass_fail_gate("correctness_multiset", "correctness_multiset_mismatch", outcome_mismatch),
        _pass_fail_gate("state", "state_mismatch", state_mismatch),
    ]
    gates.extend(_delivery_gates(pairs))
    gates.append(_gpu_gate(protocol, pairs))
    gates.extend(_resource_gates(protocol, pairs))
    return gates


def _delivery_gates(pairs: tuple[PairSummary, ...]) -> list[GateResult]:
    totals = {"delivery_loss": 0, "delivery_duplicate": 0, "unknown_gap": 0}
    for pair in pairs:
        for leg in _ordered_legs(pair):
            totals["delivery_loss"] += leg.delivery_loss
            totals["delivery_duplicate"] += leg.delivery_duplicate
            totals["unknown_gap"] += leg.unknown_gap
    return [
        _pass_fail_gate(code, code, total > 0, detail=str(total) if total > 0 else "")
        for code, total in totals.items()
    ]


def _gpu_gate(protocol: FrozenProtocol, pairs: tuple[PairSummary, ...]) -> GateResult:
    failures: list[str] = []
    for pair in pairs:
        for leg in _ordered_legs(pair):
            failures.extend(_provider_failures(protocol, leg))
    if not failures:
        return GateResult(gate="gpu_provider", status=ResultStatus.PASS, reason="gpu_provider")
    return GateResult(
        gate="gpu_provider",
        status=ResultStatus.FAIL,
        reason="gpu_fallback",
        detail=",".join(failures),
    )


def _provider_failures(protocol: FrozenProtocol, leg: LegSummary) -> list[str]:
    failures: list[str] = []
    for call in leg.model_calls:
        label = f"{leg.leg.value}:{call.execution_mode}:{call.role}"
        if call.provider == protocol.gpu_provider and call.cpu_count != 0:
            failures.append(f"{label}:cpu_count")
        elif call.provider != protocol.gpu_provider and leg.leg is LegRole.CANDIDATE:
            failures.append(f"{label}:{call.provider}")
    return failures


def _resource_gates(protocol: FrozenProtocol, pairs: tuple[PairSummary, ...]) -> list[GateResult]:
    gates: list[GateResult] = []
    for budget in protocol.budgets:
        overflow = False
        for pair in pairs:
            for leg in _ordered_legs(pair):
                observed = _resource_value(leg, budget.name)
                if observed is not None and observed > budget.absolute_limit:
                    overflow = True
        gates.append(
            _pass_fail_gate("resource_budget", "resource_overflow", overflow, detail=budget.name)
        )
    return gates


def _metric_assessments(
    protocol: FrozenProtocol,
    pairs: tuple[PairSummary, ...],
) -> tuple[MetricAssessment, ...]:
    return tuple(
        _assess_metric(frozen, _differences(frozen.spec, pairs)) for frozen in protocol.metrics
    )


def _differences(spec: MetricSpec, pairs: tuple[PairSummary, ...]) -> tuple[float, ...]:
    return tuple(_pair_difference(spec, pair) for pair in pairs)


def _assess_metric(frozen: FrozenMetric, differences: tuple[float, ...]) -> MetricAssessment:
    point = _finite_result(math.fsum(differences) / PAIR_COUNT)
    variance = sample_variance(differences)
    standard_error = _finite_result(math.sqrt(variance / PAIR_COUNT))
    half_width = _finite_result(ONE_SIDED_95_COEFFICIENT * standard_error)
    lower = _finite_result(point - half_width)
    upper = _finite_result(point + half_width)
    status, code = _metric_rule(point, lower, upper, frozen.rm)
    return MetricAssessment(
        name=frozen.spec.name,
        estimator=frozen.spec.estimator,
        direction=frozen.spec.direction,
        transform=frozen.spec.transform,
        differences=differences,
        point=point,
        difference_variance=variance,
        standard_error=standard_error,
        degrees_of_freedom=PAIRED_DEGREES_OF_FREEDOM,
        coefficient=ONE_SIDED_95_COEFFICIENT,
        lower_bound=lower,
        upper_bound=upper,
        baseline_variance=frozen.baseline_variance,
        pair_quantization=frozen.pair_quantization,
        statistical_resolution=frozen.statistical_resolution,
        rm=frozen.rm,
        status=status,
        reason=Reason(code, frozen.spec.name),
    )


def _metric_rule(point: float, lower: float, upper: float, rm: float) -> tuple[ResultStatus, str]:
    # Confirmed adverse evidence fails even when the shift is smaller than R_m.
    if lower > 0.0:
        return ResultStatus.FAIL, "confirmed_adverse"
    if point <= 0.0 and upper < rm:
        return ResultStatus.PASS, "paired_non_adverse_below_resolution"
    if point > 0.0:
        return ResultStatus.INCONCLUSIVE, "adverse_unresolved"
    return ResultStatus.INCONCLUSIVE, "resolution_unexcluded"


def _combine(
    gates: list[GateResult],
    metrics: tuple[MetricAssessment, ...],
) -> tuple[ResultStatus, tuple[Reason, ...]]:
    failures = [
        Reason(gate.reason, gate.detail) for gate in gates if gate.status is ResultStatus.FAIL
    ]
    failures.extend(metric.reason for metric in metrics if metric.status is ResultStatus.FAIL)
    if failures:
        return ResultStatus.FAIL, tuple(failures)
    unresolved = tuple(
        metric.reason for metric in metrics if metric.status is ResultStatus.INCONCLUSIVE
    )
    if unresolved:
        return ResultStatus.INCONCLUSIVE, unresolved
    if metrics and all(metric.status is ResultStatus.PASS for metric in metrics):
        return ResultStatus.PASS, tuple(metric.reason for metric in metrics)
    return ResultStatus.REJECTED, (Reason("missing_metric"),)


def _freeze(action: SealProtocol) -> tuple[list[Reason], FrozenProtocol | None]:
    reasons: list[Reason] = []
    for field in (*_PIN_FIELDS, "baseline_source_id", "candidate_source_id"):
        _require_pin(getattr(action, field), field, reasons)
    _require_pin(action.gpu_provider, "gpu_provider", reasons)
    _require_pin(action.cpu_provider, "cpu_provider", reasons)
    same_provider = (
        isinstance(action.gpu_provider, str)
        and isinstance(action.cpu_provider, str)
        and action.gpu_provider != ""
        and action.gpu_provider == action.cpu_provider
    )
    if same_provider:
        reasons.append(Reason("provider_identity", "gpu_provider"))
    if not _is_int(action.freeze_sequence):
        reasons.append(Reason("invalid_freeze"))
    metrics = _freeze_metric_specs(action.metrics, reasons)
    budgets = _freeze_budgets(action.budgets, reasons)
    expected = _freeze_expected(action.expected_calls, reasons)
    calibration = _freeze_calibration(action, metrics, reasons)
    if reasons or metrics is None or budgets is None or expected is None or calibration is None:
        return reasons or [Reason("invalid_evidence", "protocol")], None
    try:
        frozen_metrics = _frozen_metrics(metrics, calibration)
    except (OverflowError, ValueError) as error:
        return [Reason("nonfinite_derived", str(error))], None
    protocol = FrozenProtocol(
        input_id=action.input_id,
        model_id=action.model_id,
        config_id=action.config_id,
        baseline_source_id=action.baseline_source_id,
        candidate_source_id=action.candidate_source_id,
        gpu_provider=action.gpu_provider,
        cpu_provider=action.cpu_provider,
        metrics=frozen_metrics,
        budgets=budgets,
        expected_calls=expected,
        calibration=calibration,
        freeze_sequence=action.freeze_sequence,
        protocol_id="",
        reserved_evidence_ids=tuple(sorted(leg.evidence_id for leg in calibration)),
        reserved_sequences=tuple(sorted(leg.sequence for leg in calibration)),
    )
    return [], replace(protocol, protocol_id=_protocol_id(protocol))


def _frozen_metrics(
    metrics: tuple[MetricSpec, ...],
    calibration: tuple[CalibrationLeg, ...],
) -> tuple[FrozenMetric, ...]:
    frozen: list[FrozenMetric] = []
    for spec in metrics:
        samples = tuple(
            _analysis_value(spec, _metric_value(leg.metrics, spec.name)) for leg in calibration
        )
        variance = sample_variance(samples)
        quantization = measurement_pair_quantization(spec.instrument_resolution)
        statistical = statistical_resolution(variance)
        frozen.append(
            FrozenMetric(
                spec=spec,
                baseline_variance=variance,
                pair_quantization=quantization,
                statistical_resolution=statistical,
                rm=_finite_result(statistical + quantization),
            )
        )
    return tuple(frozen)


def _freeze_metric_specs(metrics: object, reasons: list[Reason]) -> tuple[MetricSpec, ...] | None:
    items = _named_tuple(metrics, "metrics", reasons)
    if items is None:
        return None
    specs: list[MetricSpec] = []
    names: set[str] = set()
    for item in items:
        if not isinstance(item, MetricSpec):
            reasons.append(Reason("invalid_evidence", "metric_spec"))
            continue
        if not _nonempty_str(item.name) or not _nonempty_str(item.estimator):
            reasons.append(Reason("invalid_pin", "metric"))
            continue
        if item.name in names:
            reasons.append(Reason("duplicate_evidence", f"metric:{item.name}"))
        names.add(item.name)
        if not isinstance(item.direction, Direction) or not isinstance(item.transform, Transform):
            reasons.append(Reason("invalid_evidence", item.name))
        if _positive_resolution(item.instrument_resolution) is None:
            reasons.append(Reason("invalid_resolution", item.name))
        specs.append(item)
    if not specs:
        reasons.append(Reason("missing_metric", "protocol"))
    if reasons:
        return None
    return tuple(specs)


def _freeze_budgets(budgets: object, reasons: list[Reason]) -> tuple[CapacityBudget, ...] | None:
    items = _named_tuple(budgets, "budgets", reasons)
    if items is None:
        return None
    if not items:
        reasons.append(Reason("missing_budget"))
        return None
    names: set[str] = set()
    parsed: list[CapacityBudget] = []
    for item in items:
        if not isinstance(item, CapacityBudget) or not _nonempty_str(item.name):
            reasons.append(Reason("invalid_evidence", "budget"))
            continue
        if item.name in names:
            reasons.append(Reason("duplicate_evidence", f"budget:{item.name}"))
        names.add(item.name)
        limit = _finite_float(item.absolute_limit)
        if limit is None or limit < 0.0:
            reasons.append(Reason("invalid_budget", item.name))
        parsed.append(item)
    if reasons:
        return None
    return tuple(parsed)


def _freeze_expected(expected: object, reasons: list[Reason]) -> tuple[ExpectedCall, ...] | None:
    items = _named_tuple(expected, "expected_calls", reasons)
    if items is None:
        return None
    if not items:
        reasons.append(Reason("missing_model_inventory"))
        return None
    lanes: set[tuple[str, str]] = set()
    parsed: list[ExpectedCall] = []
    for item in items:
        if (
            not isinstance(item, ExpectedCall)
            or not _nonempty_str(item.execution_mode)
            or not _nonempty_str(item.role)
        ):
            reasons.append(Reason("invalid_evidence", "expected_call"))
            continue
        lane = (item.execution_mode, item.role)
        label = f"{item.execution_mode}:{item.role}"
        if lane in lanes:
            reasons.append(Reason("duplicate_evidence", f"lane:{label}"))
        lanes.add(lane)
        if not _is_int(item.call_count) or item.call_count < 1:
            reasons.append(Reason("model_count_mismatch", label))
        parsed.append(item)
    for mode, role in sorted(REQUIRED_MODEL_LANES - lanes):
        reasons.append(Reason("missing_model_inventory", f"{mode}:{role}"))
    for mode, role in sorted(lanes - REQUIRED_MODEL_LANES):
        reasons.append(Reason("model_role_mismatch", f"{mode}:{role}"))
    if reasons:
        return None
    return tuple(parsed)


def _freeze_calibration(
    action: SealProtocol,
    metrics: tuple[MetricSpec, ...] | None,
    reasons: list[Reason],
) -> tuple[CalibrationLeg, ...] | None:
    items = _named_tuple(action.calibration, "calibration", reasons)
    if items is None or metrics is None:
        return None
    if len(items) != PAIR_COUNT:
        reasons.append(Reason("pair_count", "calibration"))
    found: dict[int, CalibrationLeg] = {}
    identifiers: list[str] = []
    sequences: set[int] = set()
    for item in items:
        if not isinstance(item, CalibrationLeg) or not _is_int(item.leg_index):
            reasons.append(Reason("invalid_evidence", "calibration"))
            continue
        if item.leg_index in found:
            reasons.append(Reason("duplicate_evidence", f"calibration:{item.leg_index}"))
            continue
        found[item.leg_index] = item
        _remember_id(item.evidence_id, identifiers, reasons)
        if item.leg is not LegRole.BASELINE:
            reasons.append(Reason("calibration_not_baseline", str(item.leg_index)))
        _check_calibration_sequence(action, item, sequences, reasons)
        where = f"calibration:{item.leg_index}"
        _check_pins(action, item, reasons, where)
        _check_metrics_against(metrics, item.metrics, reasons, where)
    for index in range(1, PAIR_COUNT + 1):
        if index not in found:
            reasons.append(Reason("missing_leg", f"calibration:{index}"))
    for identifier, count in sorted(Counter(identifiers).items()):
        if count > 1:
            reasons.append(Reason("duplicate_evidence", identifier))
    if reasons or len(found) != PAIR_COUNT:
        return None
    return tuple(
        replace(found[index], metrics=tuple(found[index].metrics))
        for index in range(1, PAIR_COUNT + 1)
    )


def _check_calibration_sequence(
    action: SealProtocol,
    item: CalibrationLeg,
    sequences: set[int],
    reasons: list[Reason],
) -> None:
    if not _is_int(item.sequence):
        reasons.append(Reason("invalid_evidence", f"calibration_sequence:{item.leg_index}"))
        return
    if not _is_int(action.freeze_sequence) or item.sequence >= action.freeze_sequence:
        reasons.append(Reason("calibration_not_before_freeze", str(item.leg_index)))
        return
    if item.sequence in sequences:
        reasons.append(Reason("duplicate_evidence", f"sequence:{item.sequence}"))
        return
    sequences.add(item.sequence)


def _check_pins(
    protocol: FrozenProtocol | SealProtocol,
    leg: CalibrationLeg | LegSummary,
    reasons: list[Reason],
    where: str,
) -> None:
    for field in _PIN_FIELDS:
        observed = getattr(leg, field)
        if not isinstance(observed, str) or observed != getattr(protocol, field):
            reasons.append(Reason("wrong_pin", f"{where}:{field}"))
    source_field = (
        "baseline_source_id"
        if isinstance(leg, CalibrationLeg) or leg.leg is LegRole.BASELINE
        else "candidate_source_id"
    )
    if not isinstance(leg.source_id, str) or leg.source_id != getattr(protocol, source_field):
        reasons.append(Reason("wrong_pin", f"{where}:{source_field}"))


def _check_metrics(
    protocol: FrozenProtocol,
    observations: object,
    reasons: list[Reason],
    where: str,
) -> None:
    _check_metrics_against(
        tuple(item.spec for item in protocol.metrics), observations, reasons, where
    )


def _check_metrics_against(
    specs: tuple[MetricSpec, ...],
    observations: object,
    reasons: list[Reason],
    where: str,
) -> None:
    items = _named_tuple(observations, f"metrics:{where}", reasons)
    if items is None:
        return
    by_name = {spec.name: spec for spec in specs}
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, MetricObservation) or not isinstance(item.name, str):
            reasons.append(Reason("invalid_evidence", f"metric:{where}"))
            continue
        if item.name in seen:
            reasons.append(Reason("duplicate_evidence", f"metric:{item.name}"))
        seen.add(item.name)
        spec = by_name.get(item.name)
        if spec is None:
            reasons.append(Reason("unexpected_metric", item.name))
            continue
        if item.estimator != spec.estimator:
            reasons.append(Reason("wrong_pin", f"{item.name}:estimator"))
        number = _finite_float(item.value)
        if number is None:
            reasons.append(Reason("nonfinite", item.name))
        elif spec.transform is Transform.LOG_RATIO and number <= 0.0:
            reasons.append(Reason("log_ratio_not_strictly_positive", item.name))
    for spec in specs:
        if spec.name not in seen:
            reasons.append(Reason("missing_metric", spec.name))


def _check_resources(
    protocol: FrozenProtocol,
    observations: object,
    reasons: list[Reason],
    where: str,
) -> None:
    items = _named_tuple(observations, f"resources:{where}", reasons)
    if items is None:
        return
    expected = {budget.name for budget in protocol.budgets}
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, ResourceObservation) or not isinstance(item.name, str):
            reasons.append(Reason("invalid_evidence", f"resource:{where}"))
            continue
        if item.name in seen:
            reasons.append(Reason("duplicate_evidence", f"resource:{item.name}"))
        seen.add(item.name)
        if item.name not in expected:
            reasons.append(Reason("unexpected_resource", item.name))
            continue
        number = _finite_float(item.value)
        if number is None:
            reasons.append(Reason("nonfinite", item.name))
        elif number < 0.0:
            reasons.append(Reason("invalid_resource", item.name))
    for name in expected:
        if name not in seen:
            reasons.append(Reason("resource_missing", name))


def _check_texts(values: object, code: str, where: str, reasons: list[Reason]) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        reasons.append(Reason(code, where))
        return
    if any(not isinstance(value, str) for value in values):
        reasons.append(Reason(code, where))


def _check_delivery(leg: LegSummary, reasons: list[Reason], where: str) -> None:
    for field in ("delivery_loss", "delivery_duplicate", "unknown_gap"):
        value = getattr(leg, field)
        if not _is_int(value) or value < 0:
            reasons.append(Reason("invalid_count", f"{where}:{field}"))


def _check_model_shape(
    protocol: FrozenProtocol,
    leg: LegSummary,
    reasons: list[Reason],
    where: str,
) -> None:
    items = _named_tuple(leg.model_calls, f"model_calls:{where}", reasons)
    if items is None:
        return
    expected = {
        (item.execution_mode, item.role): item.call_count for item in protocol.expected_calls
    }
    seen: dict[tuple[str, str], ModelCall] = {}
    for item in items:
        if (
            not isinstance(item, ModelCall)
            or not _nonempty_str(item.execution_mode)
            or not _nonempty_str(item.role)
        ):
            reasons.append(Reason("invalid_evidence", f"model_call:{where}"))
            continue
        lane = (item.execution_mode, item.role)
        if lane in seen:
            reasons.append(
                Reason("duplicate_evidence", f"model_lane:{item.execution_mode}:{item.role}")
            )
            continue
        seen[lane] = item
    missing = [f"{mode}:{role}" for mode, role in expected if (mode, role) not in seen]
    extra = [f"{mode}:{role}" for mode, role in seen if (mode, role) not in expected]
    if not items or missing:
        detail = ",".join(missing) if missing else where
        reasons.append(Reason("model_evidence_absent", detail))
    if extra:
        reasons.append(Reason("model_role_mismatch", ",".join(extra)))
    for lane, call_count in expected.items():
        call = seen.get(lane)
        if call is not None:
            _check_one_call(protocol, leg, call, call_count, reasons, where)


def _check_one_call(
    protocol: FrozenProtocol,
    leg: LegSummary,
    call: ModelCall,
    expected_count: int,
    reasons: list[Reason],
    where: str,
) -> None:
    label = f"{where}:{call.execution_mode}:{call.role}"
    if not _is_int(call.call_count) or not _is_int(call.cpu_count):
        reasons.append(Reason("invalid_count", label))
        return
    if call.call_count != expected_count or call.cpu_count < 0 or call.cpu_count > call.call_count:
        reasons.append(Reason("model_count_mismatch", label))
        return
    if not _nonempty_str(call.provider):
        reasons.append(Reason("model_provider_invalid", label))
        return
    if call.provider == protocol.gpu_provider:
        return
    if call.provider == protocol.cpu_provider and leg.leg is LegRole.CANDIDATE:
        return
    if call.provider == protocol.cpu_provider and leg.leg is LegRole.BASELINE:
        if (call.execution_mode, call.role) not in BASELINE_CPU_LANES:
            reasons.append(Reason("baseline_cpu_role_rejected", label))
        elif call.cpu_count != call.call_count:
            reasons.append(Reason("model_count_mismatch", label))
        return
    reasons.append(Reason("model_provider_invalid", f"{label}:{call.provider}"))


def _pair_difference(spec: MetricSpec, pair: PairSummary) -> float:
    baseline, candidate = _ordered_legs(pair)
    baseline_value = _analysis_value(spec, _metric_value(baseline.metrics, spec.name))
    candidate_value = _analysis_value(spec, _metric_value(candidate.metrics, spec.name))
    if spec.direction is Direction.HIGHER_ADVERSE:
        return _finite_result(candidate_value - baseline_value)
    return _finite_result(baseline_value - candidate_value)


def _analysis_value(spec: MetricSpec, value: float) -> float:
    if spec.transform is Transform.LOG_RATIO:
        return math.log(value)
    return value


def _metric_value(observations: tuple[MetricObservation, ...], name: str) -> float:
    for observation in observations:
        if observation.name == name:
            return float(observation.value)
    raise ValueError(f"missing metric {name}")


def _resource_value(leg: LegSummary, name: str) -> float | None:
    for observation in leg.resources:
        if observation.name == name:
            return float(observation.value)
    return None


def _ordered_legs(pair: PairSummary) -> tuple[LegSummary, LegSummary]:
    first = pair.first
    second = pair.second
    if not isinstance(first, LegSummary) or not isinstance(second, LegSummary):
        raise TypeError("pair requires two LegSummary instances")
    if first.leg is LegRole.BASELINE:
        return first, second
    return second, first


def _reference(pair: PairSummary) -> tuple[Counter[str], Counter[str]]:
    baseline, _candidate = _ordered_legs(pair)
    return Counter(baseline.outcomes), Counter(baseline.state_facts)


def _pass_fail_gate(gate: str, reason: str, failed: bool, detail: str = "") -> GateResult:
    if failed:
        return GateResult(gate=gate, status=ResultStatus.FAIL, reason=reason, detail=detail)
    return GateResult(gate=gate, status=ResultStatus.PASS, reason=gate, detail=detail)


def _protocol_id(protocol: FrozenProtocol) -> str:
    payload = {
        "budgets": [(item.name, item.absolute_limit) for item in protocol.budgets],
        "calibration": [
            {
                "id": leg.evidence_id,
                "index": leg.leg_index,
                "metrics": [(item.name, item.estimator, item.value) for item in leg.metrics],
                "pins": [leg.input_id, leg.model_id, leg.config_id, leg.source_id],
                "sequence": leg.sequence,
            }
            for leg in protocol.calibration
        ],
        "calls": [
            (item.execution_mode, item.role, item.call_count) for item in protocol.expected_calls
        ],
        "freeze_sequence": protocol.freeze_sequence,
        "metrics": [
            {
                "direction": item.spec.direction.value,
                "estimator": item.spec.estimator,
                "name": item.spec.name,
                "resolution": item.spec.instrument_resolution,
                "rm": item.rm,
                "transform": item.spec.transform.value,
                "variance": item.baseline_variance,
            }
            for item in protocol.metrics
        ],
        "pins": [protocol.input_id, protocol.model_id, protocol.config_id],
        "sources": [protocol.baseline_source_id, protocol.candidate_source_id],
        "providers": [protocol.gpu_provider, protocol.cpu_provider],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_correction(correction: object) -> bool:
    if not isinstance(correction, DiagnosedCorrection):
        return False
    diagnosis = correction.diagnosis
    fix = correction.correction
    if not isinstance(diagnosis, str) or not isinstance(fix, str):
        return False
    return bool(diagnosis.strip()) and bool(fix.strip())


def _identifier_tuple(value: object) -> tuple[str, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    identifiers = tuple(value)
    if any(not isinstance(identifier, str) or identifier == "" for identifier in identifiers):
        return None
    if len(identifiers) != len(set(identifiers)):
        return None
    return tuple(sorted(identifiers))


def _remember_id(identifier: object, identifiers: list[str], reasons: list[Reason]) -> None:
    if not isinstance(identifier, str) or identifier == "":
        reasons.append(Reason("invalid_evidence_id"))
        return
    identifiers.append(identifier)


def _named_tuple(value: object, label: str, reasons: list[Reason]) -> tuple[object, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        reasons.append(Reason("invalid_evidence", label))
        return None
    return tuple(value)


def _require_pin(value: object, field: str, reasons: list[Reason]) -> None:
    if not _nonempty_str(value):
        reasons.append(Reason("invalid_pin", field))


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _positive_resolution(value: object) -> float | None:
    number = _finite_float(value)
    if number is None or number <= 0.0:
        return None
    return number


def _five_floats(values: Sequence[float]) -> tuple[float, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != PAIR_COUNT
    ):
        raise ValueError("variance requires exactly 5 finite samples")
    numbers: list[float] = []
    for item in values:
        number = _finite_float(item)
        if number is None:
            raise ValueError("variance samples must be finite")
        numbers.append(number)
    return tuple(numbers)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_result(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("derived statistic must be finite")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
