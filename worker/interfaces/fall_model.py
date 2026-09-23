"""Fall-model ports shared by domain classifiers and concrete adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from worker.types import FallModelInput


@dataclass(frozen=True, slots=True)
class BinaryFallScoreEvidence:
    """Observed binary score and calibration used to produce one result.

    ``class_origins`` follows the probability field order below: background is
    the complement of the calibrated score, fall_transition is the
    temperature-scaled sigmoid, and fallen is a synthetic constant zero.
    """

    raw_logit: float
    applied_temperature: float
    class_origins: tuple[
        Literal["derived_complement"],
        Literal["temperature_sigmoid"],
        Literal["constant_zero"],
    ] = field(
        default=("derived_complement", "temperature_sigmoid", "constant_zero"),
        init=False,
    )


@dataclass(frozen=True, slots=True)
class FallV2Probabilities:
    """Policy probabilities with optional evidence from a binary source.

    ``fall_transition`` is already calibrated. ``model_evidence=None`` means
    that no source score was observed; it does not imply a native three-class
    result, a unit temperature, or a reconstructed logit.
    """

    background: float
    fall_transition: float
    fallen: float
    model_evidence: BinaryFallScoreEvidence | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        for value in (self.background, self.fall_transition, self.fallen):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("fall v2 probabilities must be finite values in [0, 1]")


@runtime_checkable
class FallV2ModelProtocol(Protocol):
    """V2 models score one ``(30, 56)`` pose+bbox56 window on the CPU."""

    def predict(self, features: FallModelInput) -> FallV2Probabilities: ...


__all__ = [
    "BinaryFallScoreEvidence",
    "FallV2ModelProtocol",
    "FallV2Probabilities",
]
