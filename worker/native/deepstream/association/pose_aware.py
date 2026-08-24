"""``pose-aware-bbox-iou.v1``: disabled/dark association strategy.

Task 4 allows a pose-aware strategy to exist only as explicitly disabled dark
code with its own version identity. It must be structurally present (so the
registry and cutover-gate tests can prove it is registered but inert) and must
never run: activating it changes event timelines, which is an explicit
behavior-change decision Task 4 does not make.

This stub intentionally raises rather than approximating pose-aware matching --
a partially-implemented strategy that "mostly works" is more dangerous than one
that refuses outright, because a future accidental activation would silently
change fall/bed-exit timelines instead of failing loudly.
"""

from __future__ import annotations

from typing import Final, final

from worker.types.perception_frame import (
    AssociationResult,
    PerceptionFrameIdentity,
    PersonBoxChannel,
)

POSE_AWARE_BBOX_IOU_V1: Final = "pose-aware-bbox-iou.v1"


class PoseAwareStrategyDisabledError(RuntimeError):
    """Raised on any call: this strategy is dark code, never activated."""

    def __init__(self, identity: str) -> None:
        super().__init__(
            f"association strategy {identity!r} is disabled dark code; "
            "activating it is a separate behavior-change decision outside Task 4"
        )


@final
class PoseAwareAssociationStrategy:
    """Present but inert. Every call raises `PoseAwareStrategyDisabledError`."""

    identity: str = POSE_AWARE_BBOX_IOU_V1
    enabled: Final[bool] = False

    @property
    def live_ids(self) -> frozenset[int]:
        raise PoseAwareStrategyDisabledError(self.identity)

    def observe(
        self,
        identity: PerceptionFrameIdentity,
        person_box: PersonBoxChannel,
    ) -> AssociationResult:
        del identity, person_box
        raise PoseAwareStrategyDisabledError(self.identity)

    def coast(self) -> None:
        raise PoseAwareStrategyDisabledError(self.identity)

    def reset(self) -> None:
        raise PoseAwareStrategyDisabledError(self.identity)


__all__ = [
    "POSE_AWARE_BBOX_IOU_V1",
    "PoseAwareAssociationStrategy",
    "PoseAwareStrategyDisabledError",
]
