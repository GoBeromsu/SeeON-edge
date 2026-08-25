"""Native modular person-association strategies.

This package is the executable specification the custom `GstBaseTransform`
association stage implements, mirroring the C3 `parity/` package convention:
pure Python, no GPU/DeepStream/TensorRT dependency, so the parity suite runs
on an ordinary CI runner while pinning the exact behavior the native code
owns.

- `strategy`: the `AssociationStrategy` port every implementation satisfies.
  `observe()` takes a `PerceptionFrameIdentity` plus a typed `PersonBoxChannel`
  and returns the real C1 `AssociationResult` -- a `BedRegionChannel` has no
  parameter shape to satisfy here, so bed regions are structurally
  unrepresentable at this boundary.
- `legacy_greedy_iou`: the active `legacy-greedy-bbox-iou.v1` strategy, an
  INDEPENDENT reimplementation of `worker/pipeline/perception/tracker.py`'s
  observable contract -- it does not import the oracle's `greedy_match`, so a
  tie/order regression cannot move both sides of the differential comparator
  together.
- `pose_aware`: `pose-aware-bbox-iou.v1`, registered but disabled dark code.
- `registry`: the compiled registry proving exactly one strategy is active.
- `comparator`: binary differential comparator against the Python oracle.

The Python `GreedyIouTracker` stays the oracle for differential shadow
comparison until cutover acceptance (Task 4 guardrail). This package never
imports `worker.pipeline`; the oracle import lives only in test code, keeping
the native/dark-strategy specification importable as loose Python.
"""

from __future__ import annotations

from worker.native.deepstream.association.comparator import (
    AssociationFrameTrace,
    AssociationParityMismatch,
    compare_traces,
)
from worker.native.deepstream.association.legacy_greedy_iou import (
    DEFAULT_MAX_MISSES,
    DEFAULT_MIN_IOU,
    LEGACY_GREEDY_BBOX_IOU_V1,
    LegacyGreedyBboxIouStrategy,
)
from worker.native.deepstream.association.pose_aware import (
    POSE_AWARE_BBOX_IOU_V1,
    PoseAwareAssociationStrategy,
    PoseAwareStrategyDisabledError,
)
from worker.native.deepstream.association.registry import (
    ACTIVE_ASSOCIATION_STRATEGY_ID,
    ASSOCIATION_STRATEGY_REGISTRY,
    AssociationStrategyDisabledError,
    AssociationStrategyRegistration,
    UnknownAssociationStrategyError,
    build_active_association_strategy,
    build_association_strategy,
)
from worker.native.deepstream.association.strategy import AssociationStrategy

__all__ = [
    "ACTIVE_ASSOCIATION_STRATEGY_ID",
    "ASSOCIATION_STRATEGY_REGISTRY",
    "DEFAULT_MAX_MISSES",
    "DEFAULT_MIN_IOU",
    "LEGACY_GREEDY_BBOX_IOU_V1",
    "POSE_AWARE_BBOX_IOU_V1",
    "AssociationFrameTrace",
    "AssociationParityMismatch",
    "AssociationStrategy",
    "AssociationStrategyDisabledError",
    "AssociationStrategyRegistration",
    "LegacyGreedyBboxIouStrategy",
    "PoseAwareAssociationStrategy",
    "PoseAwareStrategyDisabledError",
    "UnknownAssociationStrategyError",
    "build_active_association_strategy",
    "build_association_strategy",
    "compare_traces",
]
