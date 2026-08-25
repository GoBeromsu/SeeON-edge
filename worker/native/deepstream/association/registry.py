"""Compiled association-strategy registry: exactly one active identity.

Mirrors the compiled-registry pattern in `worker/domains/registry.py`: strategy
availability is declared data, never an `if profile` branch inside a pipeline
stage. `ACTIVE_ASSOCIATION_STRATEGY_ID` is the single source of truth cutover
gates and tests read to prove stock `nvtracker` and the pose-aware strategy are
both impossible to activate without a separate, explicit change here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from worker.native.deepstream.association.legacy_greedy_iou import (
    LEGACY_GREEDY_BBOX_IOU_V1,
    LegacyGreedyBboxIouStrategy,
)
from worker.native.deepstream.association.pose_aware import (
    POSE_AWARE_BBOX_IOU_V1,
    PoseAwareAssociationStrategy,
)
from worker.native.deepstream.association.strategy import AssociationStrategy


@dataclass(frozen=True, slots=True)
class AssociationStrategyRegistration:
    identity: str
    enabled: bool
    factory: Callable[[], AssociationStrategy]


_REGISTRATIONS: Final = (
    AssociationStrategyRegistration(
        identity=LEGACY_GREEDY_BBOX_IOU_V1,
        enabled=True,
        factory=LegacyGreedyBboxIouStrategy,
    ),
    AssociationStrategyRegistration(
        identity=POSE_AWARE_BBOX_IOU_V1,
        enabled=False,
        factory=PoseAwareAssociationStrategy,
    ),
)

ASSOCIATION_STRATEGY_REGISTRY: Mapping[str, AssociationStrategyRegistration] = MappingProxyType(
    {registration.identity: registration for registration in _REGISTRATIONS}
)

#: The one strategy identity the native child may activate at cutover.
ACTIVE_ASSOCIATION_STRATEGY_ID: Final = LEGACY_GREEDY_BBOX_IOU_V1


class AssociationStrategyDisabledError(RuntimeError):
    def __init__(self, identity: str) -> None:
        super().__init__(f"association strategy {identity!r} is registered but disabled")


class UnknownAssociationStrategyError(RuntimeError):
    def __init__(self, identity: str) -> None:
        known = ", ".join(sorted(ASSOCIATION_STRATEGY_REGISTRY))
        super().__init__(f"unknown association strategy {identity!r}; known: {known}")


def build_active_association_strategy() -> AssociationStrategy:
    """Construct the one strategy the compiled registry marks active."""
    return build_association_strategy(ACTIVE_ASSOCIATION_STRATEGY_ID)


def build_association_strategy(identity: str) -> AssociationStrategy:
    """Construct a registered strategy by identity, refusing a disabled one."""
    registration = ASSOCIATION_STRATEGY_REGISTRY.get(identity)
    if registration is None:
        raise UnknownAssociationStrategyError(identity)
    if not registration.enabled:
        raise AssociationStrategyDisabledError(identity)
    return registration.factory()


__all__ = [
    "ACTIVE_ASSOCIATION_STRATEGY_ID",
    "ASSOCIATION_STRATEGY_REGISTRY",
    "AssociationStrategyDisabledError",
    "AssociationStrategyRegistration",
    "UnknownAssociationStrategyError",
    "build_active_association_strategy",
    "build_association_strategy",
]
