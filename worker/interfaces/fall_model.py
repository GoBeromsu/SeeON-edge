"""Fall-model ports shared by domain classifiers and concrete adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from worker.types import FallModelInput


@dataclass(frozen=True, slots=True)
class FallV2Probabilities:
    background: float
    fall_transition: float
    fallen: float

    def __post_init__(self) -> None:
        for value in (self.background, self.fall_transition, self.fallen):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("fall v2 probabilities must be finite values in [0, 1]")


@runtime_checkable
class FallV2ModelProtocol(Protocol):
    """V2 models score one ``(30, 56)`` pose+bbox56 window on the CPU."""

    def predict(self, features: FallModelInput) -> FallV2Probabilities: ...


__all__ = [
    "FallV2ModelProtocol",
    "FallV2Probabilities",
]
