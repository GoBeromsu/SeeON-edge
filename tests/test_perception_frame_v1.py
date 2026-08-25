"""RED/GREEN contract for worker-internal PerceptionFrameV1.

Imports stay inside helpers so a missing module fails as an assertion, not as
a collection-time ImportError from a typo.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest

from contracts.runner import bed_result, person_result, pose_result

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_TREES = ("contracts", "backend", "front")
_REQUIRED_TYPE_SYMBOLS = (
    "AssociationResult",
    "BedRegionChannel",
    "ChannelState",
    "HumanPoseChannel",
    "Keypoint",
    "PerceptionFrameFailure",
    "PerceptionFrameIdentity",
    "PerceptionFrameV1",
    "PersonBoxChannel",
)
_REQUIRED_INTERFACE_SYMBOLS = (
    "PerceptionFrameAdapter",
)


def _load_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _require_module(module_name: str) -> Any:
    module = _load_module(module_name)
    assert module is not None, f"missing module {module_name}"
    return module


def _require_symbol(module: Any, name: str) -> Any:
    assert hasattr(module, name), f"{module.__name__} is missing {name}"
    return getattr(module, name)


def _perception_types() -> Any:
    types_pkg = _require_module("worker.types")
    for name in _REQUIRED_TYPE_SYMBOLS:
        _require_symbol(types_pkg, name)
    return types_pkg


def _perception_interfaces() -> Any:
    interfaces_pkg = _require_module("worker.interfaces")
    for name in _REQUIRED_INTERFACE_SYMBOLS:
        _require_symbol(interfaces_pkg, name)
    return interfaces_pkg


def _adapter() -> Any:
    module = _require_module("worker.adapters.perception")
    adapter_cls = _require_symbol(module, "PythonInferencePerceptionAdapter")
    return adapter_cls()


def _coco17(*, origin: int = 0, confidence: float = 0.8) -> tuple[tuple[int, int, float], ...]:
    return tuple((origin + index, origin + index + 1, confidence) for index in range(17))


def _identity(types: Any, *, epoch: int = 3, seq: int = 12, source_pts: int | None = 90000) -> Any:
    return types.PerceptionFrameIdentity(
        worker_boot_id="boot-7",
        camera_id="camera-a",
        stream_epoch=epoch,
        seq=seq,
        source_pts=source_pts,
    )


def test_perception_frame_symbols_live_on_worker_types_and_interfaces() -> None:
    types_pkg = _perception_types()
    interfaces_pkg = _perception_interfaces()
    assert types_pkg.ChannelState.INFERRED == "inferred"
    assert types_pkg.ChannelState.INFERRED_EMPTY == "inferred_empty"
    assert types_pkg.ChannelState.SKIPPED == "skipped"
    assert inspect.isclass(types_pkg.PersonBoxChannel)
    assert inspect.isclass(types_pkg.HumanPoseChannel)
    assert inspect.isclass(types_pkg.BedRegionChannel)
    assert inspect.isclass(types_pkg.AssociationResult)
    assert inspect.isclass(types_pkg.Keypoint)
    assert inspect.isclass(interfaces_pkg.PerceptionFrameAdapter)
    assert not hasattr(interfaces_pkg, "PersonBoxChannel")
    assert not hasattr(interfaces_pkg, "HumanPoseChannel")
    assert not hasattr(interfaces_pkg, "BedRegionChannel")
    assert not hasattr(interfaces_pkg, "AssociationResult")


def test_perception_frame_v1_identity_and_channel_states_are_exact() -> None:
    types = _perception_types()
    identity = _identity(types)
    pose = pose_result(poses=(_coco17(),), boxes=((10, 20, 110, 220, 0.91),))
    adapter = _adapter()
    outcome = adapter.adapt(
        identity=identity,
        pose=pose,
        bed=None,
        track_ids=(4,),
        selected_cue_indexes=(0,),
    )

    assert type(outcome).__name__ == "PerceptionFrameV1"
    assert outcome.identity == identity
    assert outcome.identity.worker_boot_id == "boot-7"
    assert outcome.identity.camera_id == "camera-a"
    assert outcome.identity.stream_epoch == 3
    assert outcome.identity.seq == 12
    assert outcome.identity.source_pts == 90000
    assert outcome.identity.durable_key == ("boot-7", "camera-a", 3, 90000)
    assert outcome.person_box.state == types.ChannelState.INFERRED
    assert outcome.human_pose.state == types.ChannelState.INFERRED
    assert outcome.bed_region.state == types.ChannelState.SKIPPED
    assert outcome.bed_region.regions == ()
    assert outcome.association is not None
    assert outcome.association.strategy == "legacy-greedy-bbox-iou.v1"
    assert outcome.association.track_ids == (4,)
    assert outcome.association.selected_cue_indexes == (0,)
    assert outcome.association.cue_source == "person_box"
    assert outcome.association.identity == identity
    assert tuple(item.name for item in fields(type(outcome))) == (
        "identity",
        "person_box",
        "human_pose",
        "bed_region",
        "association",
    )
    with pytest.raises(FrozenInstanceError):
        outcome.identity = identity  # type: ignore[misc]


def test_adapter_maps_empty_python_outputs_to_inferred_empty() -> None:
    types = _perception_types()
    adapter = _adapter()
    outcome = adapter.adapt(
        identity=_identity(types),
        pose=pose_result(poses=(), boxes=()),
        person=person_result(boxes=()),
        bed=bed_result(()),
    )

    assert type(outcome).__name__ == "PerceptionFrameV1"
    assert outcome.person_box.state == types.ChannelState.INFERRED_EMPTY
    assert outcome.human_pose.state == types.ChannelState.INFERRED_EMPTY
    assert outcome.bed_region.state == types.ChannelState.INFERRED_EMPTY
    assert outcome.association is None


def test_epoch_mismatched_association_returns_typed_failure() -> None:
    types = _perception_types()
    adapter = _adapter()
    identity = _identity(types, epoch=3)
    outcome = adapter.adapt(
        identity=identity,
        pose=pose_result(poses=(_coco17(),), boxes=((1, 2, 3, 4, 0.9),)),
        track_ids=(0,),
        selected_cue_indexes=(0,),
        association_identity=_identity(types, epoch=4),
    )

    assert type(outcome).__name__ == "PerceptionFrameFailure"
    assert outcome.code == "epoch_mismatch"
    assert "epoch" in outcome.message.lower()


def test_bed_to_person_identity_cue_returns_typed_failure() -> None:
    types = _perception_types()
    adapter = _adapter()
    identity = _identity(types)
    outcome = adapter.adapt(
        identity=identity,
        pose=pose_result(poses=(_coco17(),), boxes=((1, 2, 3, 4, 0.9),)),
        bed=bed_result(((10, 10, 40, 40, 0.8),)),
        track_ids=(0,),
        selected_cue_indexes=(0,),
        association_cue_source="bed_region",
    )

    assert type(outcome).__name__ == "PerceptionFrameFailure"
    assert outcome.code == "bed_identity_cue"
    parsed = adapter.parse(
        {
            "identity": {
                "worker_boot_id": "boot-7",
                "camera_id": "camera-a",
                "stream_epoch": 3,
                "seq": 12,
                "source_pts": 90000,
            },
            "person_box": {"state": "inferred", "boxes": []},
            "human_pose": {"state": "skipped", "poses": []},
            "bed_region": {
                "state": "inferred",
                "regions": [{"x1": 10, "y1": 10, "x2": 40, "y2": 40, "confidence": 0.8}],
            },
            "association": {
                "strategy": "legacy-greedy-bbox-iou.v1",
                "track_ids": [0],
                "selected_cue_indexes": [0],
                "cue_source": "bed_region",
                "identity": {
                    "worker_boot_id": "boot-7",
                    "camera_id": "camera-a",
                    "stream_epoch": 3,
                    "seq": 12,
                    "source_pts": 90000,
                },
            },
        }
    )
    assert type(parsed).__name__ == "PerceptionFrameFailure"
    assert parsed.code == "bed_identity_cue"


def test_malformed_identity_and_stale_epoch_return_typed_failures() -> None:
    types = _perception_types()
    adapter = _adapter()
    malformed = adapter.adapt(
        identity=types.PerceptionFrameIdentity(
            worker_boot_id="",
            camera_id="camera-a",
            stream_epoch=0,
            seq=0,
            source_pts=None,
        ),
        pose=pose_result(poses=(), boxes=()),
    )
    assert type(malformed).__name__ == "PerceptionFrameFailure"
    assert malformed.code == "malformed_identity"

    stale = adapter.adapt(
        identity=_identity(types, epoch=5),
        pose=pose_result(poses=(_coco17(),), boxes=((1, 2, 3, 4, 0.9),)),
        track_ids=(0,),
        selected_cue_indexes=(0,),
        association_identity=_identity(types, epoch=4),
    )
    assert type(stale).__name__ == "PerceptionFrameFailure"
    assert stale.code == "stale_epoch"


def test_person_result_overwrites_pose_boxes_and_preserves_pose_row_order() -> None:
    types = _perception_types()
    adapter = _adapter()
    outcome = adapter.adapt(
        identity=_identity(types),
        pose=pose_result(
            poses=(_coco17(origin=1), _coco17(origin=20)),
            boxes=((1, 2, 11, 22, 0.4), (3, 4, 13, 24, 0.5)),
        ),
        person=person_result(boxes=((100, 110, 200, 220, 0.95), (8, 9, 18, 19, 0.7))),
        bed=None,
        track_ids=(2, 3),
        selected_cue_indexes=(0, 1),
    )

    assert type(outcome).__name__ == "PerceptionFrameV1"
    assert outcome.person_box.state == types.ChannelState.INFERRED
    assert outcome.person_box.boxes[0].x1 == 100
    assert outcome.person_box.boxes[1].x1 == 8
    assert outcome.human_pose.poses[0][0] == types.Keypoint(x=1, y=2, score=0.8)
    assert outcome.human_pose.poses[1][0] == types.Keypoint(x=20, y=21, score=0.8)
    assert outcome.association.track_ids == (2, 3)


def test_public_adapter_matches_perception_frame_adapter_protocol() -> None:
    interfaces = _perception_interfaces()
    adapter = _adapter()
    assert isinstance(adapter, interfaces.PerceptionFrameAdapter)
    assert isinstance(adapter.adapt, object)
    assert isinstance(adapter.parse, object)
    assert isinstance(adapter.diagnostic, object)


def test_adapter_protocol_rejects_varargs_adapt_signature() -> None:
    interfaces = _perception_interfaces()
    protocol_params = inspect.signature(interfaces.PerceptionFrameAdapter.adapt).parameters

    class VarargsOnly:
        def adapt(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("varargs adapt should not be called")

        def parse(self, payload: object) -> object:
            raise AssertionError("varargs parse should not be called")

        def diagnostic(self, frame: object) -> object:
            raise AssertionError("varargs diagnostic should not be called")

    required = tuple(name for name in protocol_params if name != "self")
    assert required == (
        "identity",
        "pose",
        "person",
        "bed",
        "track_ids",
        "selected_cue_indexes",
        "association_identity",
        "association_strategy",
        "association_cue_source",
    )
    assert all(
        param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for param in protocol_params.values()
    )
    varargs_names = tuple(
        name for name in inspect.signature(VarargsOnly.adapt).parameters if name != "self"
    )
    assert varargs_names != required


def test_channel_and_association_names_are_unambiguous() -> None:
    types_pkg = _require_module("worker.types")
    interface_modules = (
        _require_module("worker.interfaces"),
        _require_module("worker.interfaces.perception"),
    )
    for name in (
        "PersonBoxChannel",
        "HumanPoseChannel",
        "BedRegionChannel",
        "AssociationResult",
    ):
        type_obj = _require_symbol(types_pkg, name)
        for module in interface_modules:
            if hasattr(module, name):
                assert getattr(module, name) is type_obj, (
                    f"{name} on {module.__name__} is a different object "
                    f"than worker.types.{name}"
                )


def test_keypoints_expose_named_x_y_score_fields() -> None:
    types = _require_module("worker.types")
    keypoint_cls = _require_symbol(types, "Keypoint")
    assert tuple(item.name for item in fields(keypoint_cls)) == ("x", "y", "score")
    adapter = _adapter()
    outcome = adapter.adapt(
        identity=_identity(_perception_types()),
        pose=pose_result(
            poses=(((10.9, 20.1, 0.83), (4.2, 5.8, 0.4)),),
            boxes=((1, 2, 3, 4, 0.9),),
        ),
    )
    assert type(outcome).__name__ == "PerceptionFrameV1"
    first, second = outcome.human_pose.poses[0]
    assert type(first) is keypoint_cls
    assert first.x == 10
    assert first.y == 20
    assert first.score == 0.83
    assert second.x == 4
    assert second.y == 5
    assert second.score == 0.4
    with pytest.raises(FrozenInstanceError):
        first.x = 99  # type: ignore[misc]


def test_diagnostic_round_trip_preserves_channel_states() -> None:
    types = _perception_types()
    adapter = _adapter()
    identity = _identity(types)
    frame = adapter.adapt(
        identity=identity,
        pose=pose_result(poses=(_coco17(),), boxes=((10, 20, 30, 40, 0.9),)),
        bed=None,
        track_ids=(9,),
        selected_cue_indexes=(0,),
    )
    assert type(frame).__name__ == "PerceptionFrameV1"
    payload = adapter.diagnostic(frame)
    assert payload["version"] == "PerceptionFrameV1"
    assert payload["person_box"]["state"] == "inferred"
    assert payload["human_pose"]["state"] == "inferred"
    assert payload["bed_region"]["state"] == "skipped"
    assert payload["association"]["strategy"] == "legacy-greedy-bbox-iou.v1"
    restored = adapter.parse(payload)
    assert restored == frame


def test_perception_frame_v1_symbol_is_absent_from_public_trees() -> None:
    hits: list[str] = []
    for tree in _FORBIDDEN_TREES:
        root = _REPO_ROOT / tree
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "PerceptionFrameV1" in text:
                hits.append(str(path.relative_to(_REPO_ROOT)))
    assert hits == []


def test_adapter_does_not_mutate_decision_input_or_runner_contracts() -> None:
    from worker.types.decision_input import DecisionInput

    names = tuple(item.name for item in fields(DecisionInput))
    assert names[:7] == (
        "observation",
        "frame_width",
        "frame_height",
        "live_track_ids",
        "time_sec",
        "frame_index",
        "bed_region",
    )
    runner = _require_module("contracts.runner")
    assert not hasattr(runner, "PerceptionFrameV1")
    assert callable(pose_result)
    assert isinstance({"version": "PerceptionFrameV1"}, Mapping)
