"""Fall domain: temporal state, window classifier and rising-edge decisions."""

from __future__ import annotations

from worker.domains.fall.classifier import (
    FallModelMetadataProtocol,
    FallModelProtocol,
    FallWindowClassifier,
)
from worker.domains.fall.detector import FallEventLatch, FallLatchStatus
from worker.domains.fall.schema import FallEvent

__all__ = [
    "FallEvent",
    "FallEventLatch",
    "FallLatchStatus",
    "FallModelMetadataProtocol",
    "FallModelProtocol",
    "FallWindowClassifier",
]
