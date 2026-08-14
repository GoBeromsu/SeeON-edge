from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import final

import pytest
import yaml

from shared.detection_policies import default_policy_bundle
from worker.domains.fall import FallEventLatch
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
from worker.pipeline.analytics.merge import result_merger_names
from worker.runtime.model_composition import SharedComponentPool
from worker.runtime.profile.registry import PROFILE_REGISTRY, runtime_descriptor_for
from worker.runtime.worker import WorkerRuntime

_PACKAGED_FALL_METADATA = (
    Path(__file__).resolve().parents[1] / "models" / "fall" / "lstm" / "metadata.yaml"
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

    def predict(self, _features: object) -> float:
        return 0.0


def _context(camera_id: str, model: _FallModel | None = None) -> CameraModuleContext:
    return CameraModuleContext(
        camera_id=camera_id,
        facility_id="facility-1",
        shared_components={"fall-classifier": model or _FallModel()},
        camera_components={"person-tracker": object()},
        detection_window=None,
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        diagnostics=None,
        policy=default_policy_bundle().resolve(camera_id, "fall", 1),
    )


def test_default_registry_compiles_versioned_multi_component_modules() -> None:
    fall = DETECTION_MODULE_REGISTRY.get("fall")
    bed_exit = DETECTION_MODULE_REGISTRY.get("bed_exit")

    assert fall.qualified_id == "fall.v1"
    assert bed_exit.qualified_id == "bed_exit.v1"
    assert fall.policy_schema.qualified_id == "fall.policy.v1"
    assert bed_exit.policy_schema.qualified_id == "bed_exit.policy.v1"
    assert fall.event_types == frozenset({"fall"})
    assert bed_exit.event_types == frozenset({"bed-exit"})

    fall_components = {binding.component_id for binding in fall.component_bindings}
    bed_exit_components = {binding.component_id for binding in bed_exit.component_bindings}
    assert fall_components == {
        "pose",
        "person-tracker",
        "fall-window",
        "fall-classifier",
        "fall-latch",
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
        for binding in DETECTION_MODULE_REGISTRY.get("fall", 1).shared_bindings
        if binding.component_id == "fall-classifier"
    )

    assert binding.artifact_digest == metadata["artifact_digest"]
    assert binding.preprocessing_identity == metadata["preprocessing_identity"]


def test_camera_module_factories_isolate_temporal_state_while_sharing_model() -> None:
    definition = DETECTION_MODULE_REGISTRY.get("fall")
    model = _FallModel()
    first = definition.create_camera_module(_context("camera-a", model))
    second = definition.create_camera_module(_context("camera-b", model))

    assert first.state is not second.state
    assert first.decider is not second.decider
    assert first.camera_component_ids == second.camera_component_ids
    assert isinstance(first.decider, FallEventLatch)
    assert isinstance(second.decider, FallEventLatch)
    assert first.decider.classifier.model is second.decider.classifier.model


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

    for profile_name in ("cpu-host", "nvidia-host-bridge"):
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
        )

        assert activation.qualified_ids == identities
        assert activation.schedule == {"pose": 5, "person": 5, "bed": 30}
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
        )
