"""Registry / cutover-gate contract for `worker.native.deepstream.association`.

Proves exactly one strategy identity (`legacy-greedy-bbox-iou.v1`) is active
at cutover, the dark `pose-aware-bbox-iou.v1` strategy is registered but
structurally refuses to run, the active strategy satisfies the
`AssociationStrategy` protocol, and the active strategy's greedy matcher is
independently implemented (never imports the oracle's own `greedy_match`).
"""

from __future__ import annotations

import ast
import inspect
from types import ModuleType

import pytest

from worker.native.deepstream.association import (
    ACTIVE_ASSOCIATION_STRATEGY_ID,
    ASSOCIATION_STRATEGY_REGISTRY,
    LEGACY_GREEDY_BBOX_IOU_V1,
    POSE_AWARE_BBOX_IOU_V1,
    AssociationStrategy,
    AssociationStrategyDisabledError,
    PoseAwareAssociationStrategy,
    PoseAwareStrategyDisabledError,
    build_active_association_strategy,
    build_association_strategy,
)
from worker.native.deepstream.association.legacy_greedy_iou import LegacyGreedyBboxIouStrategy
from worker.types.perception_frame import ChannelState, PerceptionFrameIdentity, PersonBoxChannel

_OWN_GREEDY_MODULE_MARKER = "worker.pipeline.perception.features.geometry"
_IDENTITY = PerceptionFrameIdentity(
    worker_boot_id="boot-1", camera_id="camera-1", stream_epoch=0, seq=0
)
_EMPTY_PERSON_BOX = PersonBoxChannel(state=ChannelState.INFERRED_EMPTY, boxes=())


def _imported_module_names(module: ModuleType) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_legacy_strategy_module_does_not_import_the_oracles_greedy_match() -> None:
    module = __import__(
        "worker.native.deepstream.association.legacy_greedy_iou",
        fromlist=["LegacyGreedyBboxIouStrategy"],
    )
    imported_modules = _imported_module_names(module)
    assert _OWN_GREEDY_MODULE_MARKER not in imported_modules, (
        "candidate strategy must not import the oracle tracker's own "
        "greedy_match/geometry module -- it must own independent greedy "
        "iteration so an order/tie regression cannot move both sides together"
    )
    source = inspect.getsource(LegacyGreedyBboxIouStrategy.observe)
    assert "for" in source or "sort" in source, "observe() must run its own matching loop"


def test_only_legacy_greedy_strategy_is_active_at_cutover() -> None:
    assert ACTIVE_ASSOCIATION_STRATEGY_ID == LEGACY_GREEDY_BBOX_IOU_V1
    active = tuple(
        identity for identity, reg in ASSOCIATION_STRATEGY_REGISTRY.items() if reg.enabled
    )
    assert active == (LEGACY_GREEDY_BBOX_IOU_V1,)
    disabled = tuple(
        identity for identity, reg in ASSOCIATION_STRATEGY_REGISTRY.items() if not reg.enabled
    )
    assert POSE_AWARE_BBOX_IOU_V1 in disabled


def test_pose_aware_strategy_is_registered_but_refuses_to_activate() -> None:
    with pytest.raises(AssociationStrategyDisabledError):
        build_association_strategy(POSE_AWARE_BBOX_IOU_V1)


def test_pose_aware_strategy_instance_refuses_every_call() -> None:
    strategy = PoseAwareAssociationStrategy()
    with pytest.raises(PoseAwareStrategyDisabledError):
        strategy.observe(_IDENTITY, _EMPTY_PERSON_BOX)
    with pytest.raises(PoseAwareStrategyDisabledError):
        strategy.coast()
    with pytest.raises(PoseAwareStrategyDisabledError):
        strategy.reset()


def test_legacy_strategy_satisfies_the_association_strategy_protocol() -> None:
    strategy = build_active_association_strategy()
    assert isinstance(strategy, AssociationStrategy)
    assert strategy.identity == LEGACY_GREEDY_BBOX_IOU_V1
