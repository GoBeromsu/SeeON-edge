from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import final

import numpy as np
import pytest
import yaml

from contracts.frame import Frame
from contracts.observation import (
    BedRegionCacheState,
    BedRegionDebugSnapshot,
    BoundingBox,
    FrameObservation,
)
from contracts.runner import Image, RunnerResult, pose_result
from shared.detection_policies import default_policy_bundle
from worker.domains.bed_exit.detector import BedExitMonitor
from worker.domains.bed_exit.schema import BedExitConfig
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallV2Probabilities
from worker.domains.fall.pose_bbox56 import POSE_BBOX56_PREPROCESSING_IDENTITY
from worker.domains.module_compiler import compile_detection_module_registry
from worker.domains.module_definition import (
    CameraModuleContext,
    DetectionModuleActivationError,
    DetectionModuleCompilationError,
    SharedComponentIdentity,
)
from worker.domains.registry import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_DEFINITIONS,
    DETECTION_MODULE_REGISTRY,
)
from worker.pipeline.analytics import CompositeExtractor, ExtractorSpec, provision_extractors
from worker.pipeline.analytics.merge import result_merger_names
from worker.pipeline.bus import Scheduler
from worker.pipeline.perception import GreedyIouTracker, SceneState
from worker.runtime.model_composition import SharedComponentPool
from worker.runtime.profile.registry import PROFILE_REGISTRY, runtime_descriptor_for
from worker.runtime.worker import WorkerRuntime
from worker.types import CURRENT_TEMPORAL_PROFILE, DecisionInput, FramePacket

_PACKAGED_FALL_METADATA = (
    Path(__file__).resolve().parents[1] / "models" / "fall" / "pose-bbox56-gru" / "metadata.yaml"
)


@final
class _FallMetadata:
    window = 2
    stride = 1
    mode = "sequence"


@final
class _FallModel:
    metadata = _FallMetadata()
    operating_threshold = 0.5

    def predict(self, _features: object) -> FallV2Probabilities:
        return FallV2Probabilities(1.0, 0.0, 0.0)


def _context(camera_id: str, model: _FallModel | None = None) -> CameraModuleContext:
    return CameraModuleContext(
        camera_id=camera_id,
        facility_id="facility-1",
        shared_components={"fall-classifier": model or _FallModel()},
        camera_components={
            "person-tracker": object(),
            "fall-v2-identity": ("boot-1", "stream-1", 0),
        },
        detection_window=None,
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        diagnostics=None,
        policy=default_policy_bundle().resolve(camera_id, "fall", 2),
    )


def test_default_registry_compiles_versioned_multi_component_modules() -> None:
    fall = DETECTION_MODULE_REGISTRY.get("fall")
    bed_exit = DETECTION_MODULE_REGISTRY.get("bed_exit")

    assert fall.qualified_id == "fall.v2"
    assert bed_exit.qualified_id == "bed_exit.v1"
    assert fall.policy_schema.qualified_id == "fall.policy.v2"
    assert bed_exit.policy_schema.qualified_id == "bed_exit.policy.v1"
    assert fall.event_types == frozenset({"fall"})
    assert bed_exit.event_types == frozenset({"bed-exit"})

    fall_components = {binding.component_id for binding in fall.component_bindings}
    bed_exit_components = {binding.component_id for binding in bed_exit.component_bindings}
    assert fall_components == {
        "pose",
        "person-tracker",
        "fall-classifier",
        "fall-v2",
    }
    assert bed_exit_components == {
        "pose",
        "person",
        "bed",
        "person-tracker",
        "containment",
        "bed-assignment",
        "bed-exit-state",
        "bed-exit-latch",
    }
    assert len([binding for binding in bed_exit.component_bindings if binding.model_family]) == 3


def test_compiled_fall_binding_matches_the_packaged_model_identity() -> None:
    metadata = yaml.safe_load(_PACKAGED_FALL_METADATA.read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)
    binding = next(
        binding
        for binding in DETECTION_MODULE_REGISTRY.get("fall", 2).shared_bindings
        if binding.component_id == "fall-classifier"
    )

    assert binding.artifact_digest is not None and len(binding.artifact_digest) == 64
    assert binding.preprocessing_identity == POSE_BBOX56_PREPROCESSING_IDENTITY
    assert metadata["model_family"] == "gru_source_proxy_v0"


def test_camera_module_factories_isolate_temporal_state_while_sharing_model() -> None:
    definition = DETECTION_MODULE_REGISTRY.get("fall")
    model = _FallModel()
    first = definition.create_camera_module(_context("camera-a", model))
    second = definition.create_camera_module(_context("camera-b", model))

    assert first.state is not second.state
    assert first.decider is not second.decider
    assert first.camera_component_ids == second.camera_component_ids
    assert isinstance(first.decider, FallV2DomainDecider)
    assert isinstance(second.decider, FallV2DomainDecider)
    assert isinstance(first.decider.policy, FallPolicyDeciderV2)
    assert isinstance(second.decider.policy, FallPolicyDeciderV2)
    assert first.decider.classifier.model is second.decider.classifier.model


def _module_context(camera_id: str, module_id: str, model: _FallModel) -> CameraModuleContext:
    """A context with NO pre-injected camera components.

    Unlike ``_context`` above (which hands in a ready-made ``person-tracker``),
    this forces ``create_camera_module`` to run every declared
    ``camera_factory`` itself -- which is what production does, and the only
    way the per-camera-object assertions below actually test the factories.
    """
    return CameraModuleContext(
        camera_id=camera_id,
        facility_id="facility-1",
        shared_components={"fall-classifier": model},
        camera_components={"fall-v2-identity": ("boot-1", "stream-1", 0)},
        detection_window=None,
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        diagnostics=None,
        policy=default_policy_bundle().resolve(
            camera_id, module_id, DETECTION_MODULE_REGISTRY.get(module_id).version
        ),
    )


def test_no_camera_local_component_object_is_shared_between_cameras_in_either_module() -> None:
    """Per-camera temporal state is never shared -- across 13 cameras, both modules.

    Cross-camera batched inference (nvidia-multistream-serving) shares model
    runners aggressively; the invariant that keeps that safe is that NOTHING
    holding per-camera history is shared. A single leaked object here (a
    tracker, a fall window, a bed assignment map, a latch) would silently
    braid two residents' timelines together once one coordinator thread
    drives all 13 cameras.

    ``containment`` is deliberately exempt: it is bound with
    ``component_kind == "rule"`` and resolves to the pure function
    ``containment_ratio``, which holds no state -- sharing it is correct.
    """
    model = _FallModel()
    camera_ids = tuple(f"camera-{index}" for index in range(1, 14))
    modules = {
        module_id: [
            DETECTION_MODULE_REGISTRY.get(module_id).create_camera_module(
                _module_context(camera_id, module_id, model)
            )
            for camera_id in camera_ids
        ]
        for module_id in ("fall", "bed_exit")
    }

    stateful_ids = {
        module_id: frozenset(
            binding.component_id
            for binding in DETECTION_MODULE_REGISTRY.get(module_id).component_bindings
            if not binding.shared and binding.component_kind != "rule"
        )
        for module_id in modules
    }
    assert stateful_ids["fall"] == {"person-tracker", "fall-v2"}
    assert stateful_ids["bed_exit"] == {
        "person-tracker",
        "bed-assignment",
        "bed-exit-state",
        "bed-exit-latch",
    }

    for module_id, camera_modules in modules.items():
        assert len({id(item.state) for item in camera_modules}) == len(camera_ids)
        assert len({id(item.decider) for item in camera_modules}) == len(camera_ids)
        for component_id in stateful_ids[module_id]:
            component_ids = {id(item.camera_components[component_id]) for item in camera_modules}
            assert len(component_ids) == len(camera_ids), (
                f"{module_id}.{component_id} is shared across cameras"
            )

    # Isolation must not be achieved by accidentally duplicating shared
    # things: the stateless containment rule stays one object for everyone.
    bed_exit_modules = modules["bed_exit"]
    assert len({id(item.camera_components["containment"]) for item in bed_exit_modules}) == 1

    # No camera-local object leaks ACROSS modules either (fall's tracker is
    # not bed_exit's tracker, even for the same camera).
    for fall_module, bed_exit_module in zip(modules["fall"], modules["bed_exit"], strict=True):
        assert (
            fall_module.camera_components["person-tracker"]
            is not bed_exit_module.camera_components["person-tracker"]
        )


def test_shared_component_pool_keys_every_execution_and_artifact_identity_field() -> None:
    baseline = SharedComponentIdentity(
        component_id="pose",
        artifact_digest="a" * 64,
        runtime="onnxruntime",
        device="cpu",
        preprocessing_identity="rgb24-v1",
    )
    calls: list[str] = []
    pool = SharedComponentPool()

    first = pool.get_or_create(baseline, lambda: calls.append("created") or object())
    second = pool.get_or_create(baseline, lambda: calls.append("duplicate") or object())

    assert first is second
    assert calls == ["created"]
    assert (
        len(
            {
                baseline,
                replace(baseline, artifact_digest="b" * 64),
                replace(baseline, runtime="tensorrt"),
                replace(baseline, device="cuda:0"),
                replace(baseline, preprocessing_identity="rgb24-v2"),
            }
        )
        == 5
    )


def test_runtime_decider_composition_has_no_domain_name_branch() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(WorkerRuntime._build_decider)))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "fall" not in literals
    assert "bed_exit" not in literals


def test_compiled_modules_are_profile_independent() -> None:
    identities = DETECTION_MODULE_REGISTRY.qualified_ids

    for profile_name in ("cpu-host", "nvidia"):
        spec = PROFILE_REGISTRY[profile_name]
        descriptor = runtime_descriptor_for(spec, requested_profile=profile_name)
        activation = DETECTION_MODULE_REGISTRY.activation(
            module_ids=("fall", "bed_exit"),
            available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
            available_component_ids=DETECTION_MODULE_REGISTRY.shared_component_ids(
                ("fall", "bed_exit"), flags={"person-box-source": True}
            ),
            warmed_component_ids=DETECTION_MODULE_REGISTRY.shared_component_ids(
                ("fall", "bed_exit"), flags={"person-box-source": True}
            ),
            output_adapter_ids=result_merger_names(),
            camera_frame_stride=5,
            flags={"person-box-source": True, "persisted-bed-region": False},
            temporal_profile=CURRENT_TEMPORAL_PROFILE,
        )

        assert activation.qualified_ids == identities
        assert activation.schedule == {"pose": 5, "person": 5, "bed": 90}
        assert descriptor.canonical_profile == profile_name


def test_activation_fails_closed_for_unavailable_runtime_dependencies() -> None:
    module_ids = ("fall", "bed_exit")
    flags = {"person-box-source": True, "persisted-bed-region": False}
    components = DETECTION_MODULE_REGISTRY.shared_component_ids(module_ids, flags=flags)

    def activate(
        *,
        channels: frozenset[str] = AVAILABLE_OBSERVATION_CHANNELS,
        available: frozenset[str] = components,
        warmed: frozenset[str] = components,
        outputs: frozenset[str] = result_merger_names(),
    ) -> None:
        _ = DETECTION_MODULE_REGISTRY.activation(
            module_ids=module_ids,
            available_observation_channels=channels,
            available_component_ids=available,
            warmed_component_ids=warmed,
            output_adapter_ids=outputs,
            camera_frame_stride=5,
            flags=flags,
            temporal_profile=CURRENT_TEMPORAL_PROFILE,
        )

    with pytest.raises(DetectionModuleActivationError, match="observation channel"):
        activate(channels=frozenset())
    with pytest.raises(DetectionModuleActivationError, match="component binding"):
        activate(available=components - {"fall-classifier"})
    with pytest.raises(DetectionModuleActivationError, match="component warmup"):
        activate(warmed=components - {"fall-classifier"})
    with pytest.raises(DetectionModuleActivationError, match="output adapter"):
        activate(outputs=frozenset())

    fall, bed_exit = DETECTION_MODULE_DEFINITIONS
    conflicting_bed_exit = replace(
        bed_exit,
        schedule_rules=tuple(
            replace(rule, interval_source="fixed", interval=2)
            if rule.component_id == "pose"
            else rule
            for rule in bed_exit.schedule_rules
        ),
    )
    conflicting_registry = compile_detection_module_registry(
        (fall, conflicting_bed_exit),
        available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
        output_adapter_ids=result_merger_names(),
        temporal_profile=CURRENT_TEMPORAL_PROFILE,
    )
    with pytest.raises(DetectionModuleActivationError, match="schedule conflict"):
        _ = conflicting_registry.activation(
            module_ids=module_ids,
            available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
            available_component_ids=components,
            warmed_component_ids=components,
            output_adapter_ids=result_merger_names(),
            camera_frame_stride=5,
            flags=flags,
            temporal_profile=CURRENT_TEMPORAL_PROFILE,
        )


def test_compiler_fails_closed_for_incomplete_or_conflicting_definitions() -> None:
    fall, bed_exit = DETECTION_MODULE_DEFINITIONS
    pose = next(binding for binding in fall.component_bindings if binding.component_id == "pose")

    malformed = (
        (replace(fall, required_observation_channels=frozenset({"missing"})), bed_exit, "channel"),
        (
            replace(
                fall,
                schedule_rules=tuple(
                    rule for rule in fall.schedule_rules if rule.component_id != "pose"
                ),
            ),
            bed_exit,
            "schedule",
        ),
        (
            replace(
                fall,
                component_bindings=tuple(
                    replace(binding, output_adapter=None)
                    if binding.component_id == "pose"
                    else binding
                    for binding in fall.component_bindings
                ),
            ),
            bed_exit,
            "output adapter",
        ),
        (
            replace(
                fall,
                component_bindings=tuple(
                    replace(binding, warmup_required=False)
                    if binding.component_id == "fall-classifier"
                    else binding
                    for binding in fall.component_bindings
                ),
            ),
            bed_exit,
            "warmup",
        ),
        (
            replace(
                fall,
                component_bindings=tuple(
                    replace(binding, artifact_digest="")
                    if binding.component_id == "fall-classifier"
                    else binding
                    for binding in fall.component_bindings
                ),
            ),
            bed_exit,
            "provenance",
        ),
        (replace(fall, event_types=bed_exit.event_types), bed_exit, "event type"),
        (
            replace(fall, policy_schema=replace(fall.policy_schema, schema_id="")),
            bed_exit,
            "policy",
        ),
        (replace(fall, trace_adapter=None), bed_exit, "trace adapter"),
    )

    for broken_fall, valid_bed_exit, expected in malformed:
        with pytest.raises(DetectionModuleCompilationError, match=expected):
            _ = compile_detection_module_registry(
                (broken_fall, valid_bed_exit),
                available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
                output_adapter_ids=result_merger_names(),
                temporal_profile=CURRENT_TEMPORAL_PROFILE,
            )

    with pytest.raises(DetectionModuleCompilationError, match="binding"):
        _ = compile_detection_module_registry(
            (
                replace(
                    fall,
                    component_bindings=tuple(
                        binding for binding in fall.component_bindings if binding != pose
                    ),
                ),
                bed_exit,
            ),
            available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
            output_adapter_ids=result_merger_names(),
            temporal_profile=CURRENT_TEMPORAL_PROFILE,
        )


# --- sampling / staleness semantics (nvidia-multistream-serving todo 2d) ----
#
# Under cross-camera batching a camera's frames are decoded faster than they
# are inferred: most frames carry NO pose result at all. Today the pipeline
# cannot tell "nobody was there" from "nobody looked", and treats the second
# as the first. The first two tests below CHARACTERIZE that current behavior
# (they pass today, deliberately); the strict-xfail specs after them state
# the behavior todo 9 must deliver, at which point the characterization
# tests are the ones that get rewritten.


def _tracked_box() -> BoundingBox:
    return BoundingBox(x1=10, y1=10, x2=90, y2=90, confidence=0.9)


@final
class _PoseRunner:
    """Always sees the same person; only the SCHEDULER decides if it runs."""

    def __init__(self, box: BoundingBox) -> None:
        self._box = box
        self.calls = 0

    def run(self, _image: Image) -> RunnerResult:
        self.calls += 1
        keypoints = tuple(float(value) for index in range(17) for value in (index, index, 0.9))
        return pose_result(
            (keypoints,),
            ((self._box.x1, self._box.y1, self._box.x2, self._box.y2, self._box.confidence),),
        )


@final
class _PoseServing:
    def __init__(self, runner: _PoseRunner) -> None:
        self._runner = runner

    def create(self, task: str, **_options: object) -> _PoseRunner:
        assert task == "pose"
        return self._runner


def _frame_packet(index: int) -> FramePacket:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return FramePacket(
        camera_id="camera-sampling",
        frame=Frame(index=index, time_sec=float(index), image=image),
        pts=float(index),
        seq=index,
        width=8,
        height=8,
        decode_time_ms=0.0,
    )


def _pose_composite(runner: _PoseRunner) -> CompositeExtractor:
    return CompositeExtractor(
        extractors=provision_extractors(_PoseServing(runner), (ExtractorSpec("pose", "pose"),)),
        scheduler=Scheduler({"pose": 5}),
        tracker=GreedyIouTracker(max_misses=3),
        scene_state=SceneState("camera-sampling"),
    )


def test_characterization_empty_tracker_update_ages_and_expires_live_tracks() -> None:
    """POST-FIX: update(()) coasts; observe(()) is an actual negative."""
    tracker = GreedyIouTracker(max_misses=3)
    box = _tracked_box()

    assert tracker.update((box,)) == (0,)
    for _ in range(10):
        assert tracker.update(()) == ()
    assert tracker.live_ids == frozenset({0})
    assert tracker.update((box,)) == (0,)

    for _ in range(4):
        assert tracker.observe(()) == ()
    assert tracker.live_ids == frozenset()
    assert tracker.observe((box,)) == (1,)


def test_characterization_non_inferred_frame_advances_scene_state_as_empty() -> None:
    """POST-FIX: SceneState.coast retains the last inferred observation."""
    scene = SceneState("camera-sampling")
    observed = FrameObservation(detections=((_tracked_box(),), ()), track_ids=(0,))

    assert scene.observe(observed, track_ids=(0,)) is observed
    for _ in range(4):
        assert scene.coast() is observed
    assert scene.latest_observation is observed
    assert scene.track_ids == (0,)


def test_non_inferred_frame_does_not_age_tracks_or_advance_scene_state_as_empty() -> None:
    """The explicit perception coast API is neutral until todo 8 wires it."""
    tracker = GreedyIouTracker(max_misses=3)
    scene = SceneState("camera-sampling")
    box = _tracked_box()
    observation = FrameObservation(detections=((box,), ()), track_ids=(0,))

    assert tracker.observe((box,)) == (0,)
    _ = scene.observe(observation, track_ids=(0,))
    for _ in range(15):
        tracker.coast()
        assert scene.coast() is observation

    assert tracker.live_ids == frozenset({0})
    assert scene.track_ids == (0,)
    assert tracker.observe((box,)) == (0,)


def test_bed_exit_missing_person_evidence_is_never_reported_as_empty() -> None:
    """A person-inference gap reports COVERED and cannot claim EMPTY."""
    bed = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
    monitor = BedExitMonitor(
        config=BedExitConfig(
            camera_id="camera-bed-sampling",
            facility_id="facility-1",
            min_containment=0.5,
            hold_frames=1,
            grace_frames=0,
        ),
        clock=lambda: datetime(2026, 8, 17, tzinfo=UTC),
    )

    def decision(persons: tuple[BoundingBox, ...], frame_index: int) -> DecisionInput:
        return DecisionInput(
            observation=FrameObservation(detections=(persons, ()), regions=((bed,), ())),
            frame_width=200,
            frame_height=200,
            live_track_ids=(0,) if persons else (),
            time_sec=float(frame_index),
            frame_index=frame_index,
            bed_region=BedRegionDebugSnapshot(source=BedRegionCacheState.FRESH),
        )

    _ = monitor.update(decision((BoundingBox(10, 10, 90, 90, 0.9),), 0))
    occupied = monitor.last_debug_snapshot
    assert occupied is not None
    assert tuple(status.occupancy for status in occupied.statuses) == ("occupied",)

    _ = monitor.coast(frame_index=1)
    unobserved = monitor.last_debug_snapshot
    assert unobserved is not None
    assert tuple(status.occupancy for status in unobserved.statuses) == ("covered",)
    assert unobserved.events == ()
