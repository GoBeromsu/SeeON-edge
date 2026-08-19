from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import final

import numpy as np
import pytest

from contracts.frame import Frame
from contracts.runner import Image, RunnerResult, pose_result
from worker.domains.base import DomainAuditSnapshot
from worker.domains.module_compiler import (
    CompiledDetectionModuleRegistry,
    compile_detection_module_registry,
)
from worker.domains.module_definition import (
    CameraModuleContext,
    ComponentBinding,
    ComponentKind,
    DetectionModuleActivationError,
    DetectionModuleCompilationError,
    DetectionModuleDefinition,
    PolicySchemaIdentity,
    ScheduleRule,
)
from worker.domains.registry import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_DEFINITIONS,
    DETECTION_MODULE_REGISTRY,
)
from worker.pipeline.analytics.merge import merge_module_results, result_merger_names
from worker.runtime.config import WorkerConfig
from worker.runtime.config.domain_models import DomainsConfig
from worker.runtime.model_composition import SharedComponentPool, compose_shared_components
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.worker import WorkerRuntime
from worker.types import CURRENT_TEMPORAL_PROFILE, BusinessEvent, DecisionInput, FramePacket

ServingOption = str | int | float | bool | None


class _Runner:
    artifact_digest = "a" * 64
    preprocessing_identity = "numeric-v1"

    def __call__(self, _image: Image) -> RunnerResult:
        return pose_result((), ())

    def warmup(self) -> None:
        return None


class _Serving:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, ServingOption]]] = []

    def create(self, task: str, **options: ServingOption) -> _Runner:
        self.calls.append((task, options))
        return _Runner()


class _Decider:
    def update(self, input_value: DecisionInput) -> tuple[BusinessEvent, ...]:
        del input_value
        return ()


def _third_definition(*, version: int = 3) -> DetectionModuleDefinition:
    def component(component_id: str, kind: ComponentKind, value: object) -> ComponentBinding:
        return ComponentBinding(
            component_id=component_id,
            component_kind=kind,
            camera_factory=lambda _context, value=value: value,
        )

    source_audit_adapter = DETECTION_MODULE_DEFINITIONS[0].audit_adapter
    assert source_audit_adapter is not None

    def audit_adapter(_context: CameraModuleContext) -> DomainAuditSnapshot:
        return replace(
            source_audit_adapter(
                CameraModuleContext(
                    "audit",
                    "facility",
                    {},
                    {},
                    None,
                    lambda: datetime.now(UTC),
                    None,
                )
            ),
            model_version=None,
            operating_threshold=None,
        )

    return DetectionModuleDefinition(
        module_id="mobility_risk",
        version=version,
        required_observation_channels=frozenset({"poses", "bed_regions"}),
        component_bindings=(
            ComponentBinding(
                "mobility-pose",
                "extractor",
                model_family="pose-family",
                provisioner="serving-client",
                serving_task="pose",
                artifact_digest="a" * 64,
                preprocessing_identity="numeric-v1",
                output_adapter="pose",
                warmup_required=True,
            ),
            ComponentBinding(
                "mobility-bed",
                "extractor",
                model_family="bed-family",
                provisioner="serving-client",
                serving_task="bed",
                artifact_digest="a" * 64,
                preprocessing_identity="numeric-v1",
                output_adapter="bed",
                warmup_required=True,
            ),
            component("mobility-window", "state", object()),
            component("mobility-rule", "rule", object()),
        ),
        schedule_rules=(
            ScheduleRule("mobility-pose", "camera-frame-stride"),
            ScheduleRule("mobility-bed", "fixed", interval=11),
        ),
        policy_schema=PolicySchemaIdentity("mobility.policy", 1),
        camera_state_factory=lambda context: (
            context.camera_components["mobility-window"],
            context.camera_components["mobility-rule"],
        ),
        decider_factory=lambda _state: _Decider(),
        audit_adapter=audit_adapter,
        trace_adapter=lambda _decider: {},
        debug_adapter=lambda _decider, _frame: None,
        event_types=frozenset({"mobility-risk"}),
        input_view="mobility",
        window_mode="external",
    )


def _registry(
    *definitions: DetectionModuleDefinition,
) -> CompiledDetectionModuleRegistry:
    return compile_detection_module_registry(
        definitions,
        available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
        output_adapter_ids=result_merger_names(),
        temporal_profile=CURRENT_TEMPORAL_PROFILE,
    )


def test_registry_selects_qualified_versions_and_config_rejects_drift() -> None:
    third_v2 = _third_definition(version=2)
    third_v3 = _third_definition(version=3)
    registry = _registry(third_v2, third_v3)

    assert registry.get("mobility_risk", 2) is third_v2
    assert registry.get("mobility_risk", 3) is third_v3
    assert registry.selected({"mobility_risk": 2}) == (third_v2,)

    config = DomainsConfig(versions={"mobility_risk": 3})
    assert config.selected_versions(registry) == MappingProxyType({"mobility_risk": 3})

    with pytest.raises(DetectionModuleActivationError, match="configured detection module"):
        DomainsConfig(versions={"mobility_risk": 4}).selected_versions(registry)


def _packet() -> FramePacket:
    return FramePacket(
        camera_id="camera-1",
        frame=Frame(
            index=0,
            time_sec=0.0,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        pts=0.0,
        seq=0,
        width=8,
        height=8,
        decode_time_ms=0.0,
    )


def test_third_multi_component_module_is_composed_from_bindings() -> None:
    definition = _third_definition()
    registry = _registry(definition)
    serving = _Serving()

    shared = compose_shared_components(
        registry,
        module_versions={"mobility_risk": 3},
        serving_client=serving,
        runtime="cpu",
        device="cpu",
        flags={},
        pool=SharedComponentPool(),
    )
    context = CameraModuleContext(
        camera_id="camera-1",
        facility_id="facility-1",
        shared_components=shared.components,
        camera_components={},
        detection_window=None,
        clock=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        diagnostics=None,
    )
    camera_module = definition.create_camera_module(context)

    assert serving.calls == [
        ("pose", {"device": "cpu"}),
        ("bed", {"device": "cpu"}),
    ]
    assert tuple(extractor.module_name for extractor in shared.extractors) == (
        "mobility-pose",
        "mobility-bed",
    )
    assert camera_module.camera_component_ids == frozenset({"mobility-window", "mobility-rule"})
    assert len(shared.identities) == 2
    assert all(identity.runtime == "cpu" for identity in shared.identities)
    assert all(identity.device == "cpu" for identity in shared.identities)


def test_declared_output_adapter_routes_mobility_component_to_pose_merger() -> None:
    pose = tuple((index, index + 1, 0.9) for index in range(17))
    flattened = tuple(value for point in pose for value in point)

    class _PoseRunner(_Runner):
        def __call__(self, _image: Image) -> RunnerResult:
            return pose_result((flattened,), ((1, 1, 4, 5, 0.8),))

    class _PoseServing(_Serving):
        def create(self, task: str, **options: ServingOption) -> _Runner:
            self.calls.append((task, options))
            return _PoseRunner() if task == "pose" else _Runner()

    shared = compose_shared_components(
        _registry(_third_definition()),
        module_versions={"mobility_risk": 3},
        serving_client=_PoseServing(),
        runtime="cpu",
        device="cpu",
        flags={},
        pool=SharedComponentPool(),
    )

    mobility_pose = shared.extractors[0]
    result = mobility_pose.extract(_packet())
    merged = merge_module_results((result,))

    assert result.module_name == "mobility-pose"
    assert result.output_adapter == "pose"
    assert merged.poses == (pose,)


def test_compiler_rejects_two_output_writers_inside_one_module() -> None:
    definition = _third_definition()
    conflicting = replace(
        definition,
        component_bindings=tuple(
            replace(binding, output_adapter="pose")
            if binding.component_id == "mobility-bed"
            else binding
            for binding in definition.component_bindings
        ),
    )

    with pytest.raises(DetectionModuleCompilationError, match="multiple active writers"):
        _registry(conflicting)


def test_activation_rejects_distinct_active_writers_to_one_output_adapter() -> None:
    registry = _registry(DETECTION_MODULE_DEFINITIONS[0], _third_definition())
    selection = {"fall": 1, "mobility_risk": 3}
    components = registry.shared_component_ids(selection, flags={})

    with pytest.raises(DetectionModuleActivationError, match="output adapter.*pose"):
        registry.activation(
            module_versions=selection,
            available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
            available_component_ids=components,
            warmed_component_ids=components,
            output_adapter_ids=result_merger_names(),
            camera_frame_stride=1,
            flags={},
            temporal_profile=CURRENT_TEMPORAL_PROFILE,
        )


@pytest.mark.parametrize(
    ("profile_name", "device"),
    (("cpu-host", "cpu"), ("nvidia-host-bridge", "cuda")),
)
def test_worker_runtime_preflights_third_module_without_name_dispatch(
    tmp_path: Path,
    profile_name: str,
    device: str,
) -> None:
    definition = _third_definition()
    registry = _registry(definition)
    config = WorkerConfig.model_validate(
        {
            "relay": {"url": "http://relay.test", "token": "token"},
            "domains": {"versions": {"mobility_risk": 3}},
            "clip": {"enabled": False},
            "cameras": [
                {
                    "camera_id": "camera-1",
                    "facility_id": "facility-1",
                    "rtsp_url": "rtsp://example.test/camera-1",
                    "frame_stride": 7,
                }
            ],
        }
    )
    runtime = WorkerRuntime(
        config,
        serving_client=_Serving(),
        module_registry=registry,
        state_dir=tmp_path,
        clip_store_dir=tmp_path,
    )
    profile = PROFILE_REGISTRY[profile_name]
    boot = BootContext(profile, profile.device, profile.decode, profile.encode)

    graph = runtime._initialize_models(boot)  # noqa: SLF001
    runtime._warmed_component_ids = frozenset(graph.components)  # noqa: SLF001
    plan = runtime._preflight_camera_graph(config.cameras[0])  # noqa: SLF001

    assert plan.schedule == {"mobility-pose": 7, "mobility-bed": 11}
    assert tuple(plan.definitions) == ("mobility_risk",)
    assert tuple(plan.domain_deciders) == ("mobility_risk",)
    assert {identity.device for identity in graph.identities} == {device}


def test_production_shared_component_semantics_are_equal_on_cpu_and_nvidia() -> None:
    selection = {"fall": 1, "bed_exit": 1}
    bindings = DETECTION_MODULE_REGISTRY.shared_bindings(selection, flags={})
    by_task = {
        binding.serving_task: binding for binding in bindings if binding.serving_task is not None
    }

    @final
    class _CompiledIdentityRunner:
        def __init__(self, binding: ComponentBinding) -> None:
            artifact_digest = binding.artifact_digest
            preprocessing_identity = binding.preprocessing_identity
            assert isinstance(artifact_digest, str)
            assert isinstance(preprocessing_identity, str)
            self.artifact_digest: str = artifact_digest
            self.preprocessing_identity: str = preprocessing_identity

        def __call__(self, _image: Image) -> RunnerResult:
            return pose_result((), ())

        def warmup(self) -> None:
            return None

    @final
    class _ProductionServing:
        def create(self, task: str, **_options: ServingOption) -> _CompiledIdentityRunner:
            return _CompiledIdentityRunner(by_task[task])

    semantics_by_profile: list[tuple[tuple[str, str, str], ...]] = []
    execution_by_profile: list[tuple[tuple[str, str], ...]] = []
    for runtime, device in (("cpu", "cpu"), ("cuda", "cuda")):
        graph = compose_shared_components(
            DETECTION_MODULE_REGISTRY,
            module_versions=selection,
            serving_client=_ProductionServing(),
            runtime=runtime,
            device=device,
            flags={},
            pool=SharedComponentPool(),
            provisioners={
                "fall-model-family-registry": lambda binding, _device: _CompiledIdentityRunner(
                    binding
                )
            },
        )
        semantics_by_profile.append(
            tuple(
                (
                    identity.component_id,
                    identity.artifact_digest,
                    identity.preprocessing_identity,
                )
                for identity in graph.identities
            )
        )
        execution_by_profile.append(
            tuple((identity.runtime, identity.device) for identity in graph.identities)
        )

    assert semantics_by_profile[0] == semantics_by_profile[1]
    assert set(component_id for component_id, _, _ in semantics_by_profile[0]) == {
        "pose",
        "bed",
        "fall-classifier",
    }
    assert set(execution_by_profile[0]) == {("cpu", "cpu")}
    assert set(execution_by_profile[1]) == {("cuda", "cuda")}


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("artifact_digest", "artifact identity mismatch"),
        ("preprocessing_identity", "preprocessing identity mismatch"),
    ),
)
def test_production_pose_binding_rejects_reported_identity_mismatch_before_pooling(
    field: str,
    message: str,
) -> None:
    pose = DETECTION_MODULE_REGISTRY.get("fall", 1).shared_bindings[0]

    class _MismatchedPose(_Runner):
        artifact_digest = pose.artifact_digest
        preprocessing_identity = pose.preprocessing_identity

    setattr(_MismatchedPose, field, "mismatch")

    class _ProductionServing(_Serving):
        def create(self, task: str, **options: ServingOption) -> _MismatchedPose:
            self.calls.append((task, options))
            return _MismatchedPose()

    pool = SharedComponentPool()
    with pytest.raises(DetectionModuleActivationError, match=message):
        compose_shared_components(
            DETECTION_MODULE_REGISTRY,
            module_versions={"fall": 1},
            serving_client=_ProductionServing(),
            runtime="cpu",
            device="cpu",
            flags={},
            pool=pool,
            provisioners={"fall-model-family-registry": lambda _binding, _device: _Runner()},
        )

    assert pool.identities == ()


def test_missing_runner_preprocessing_identity_never_reaches_pool() -> None:
    pool = SharedComponentPool()

    class _MissingPreprocessing(_Runner):
        preprocessing_identity = None

    class _MissingPreprocessingServing(_Serving):
        def create(self, task: str, **options: ServingOption) -> _Runner:
            self.calls.append((task, options))
            return _MissingPreprocessing() if task == "pose" else _Runner()

    with pytest.raises(DetectionModuleActivationError, match="preprocessing identity"):
        compose_shared_components(
            _registry(_third_definition()),
            module_versions={"mobility_risk": 3},
            serving_client=_MissingPreprocessingServing(),
            runtime="cpu",
            device="cpu",
            flags={},
            pool=pool,
        )

    assert pool.identities == ()


def test_unresolved_artifact_identity_never_reaches_applied_graph() -> None:
    definition = _third_definition()
    broken = replace(
        definition,
        component_bindings=tuple(
            replace(binding, artifact_digest=None)
            if binding.component_id == "mobility-pose"
            else binding
            for binding in definition.component_bindings
        ),
    )
    registry = _registry(broken)

    class _UnidentifiedRunner:
        def __call__(self, _image: Image) -> RunnerResult:
            return pose_result((), ())

    class _UnidentifiedServing:
        def create(self, task: str, **_options: ServingOption) -> _UnidentifiedRunner:
            return _UnidentifiedRunner()

    with pytest.raises(DetectionModuleActivationError, match="artifact identity"):
        compose_shared_components(
            registry,
            module_versions={"mobility_risk": 3},
            serving_client=_UnidentifiedServing(),
            runtime="cuda",
            device="cuda",
            flags={},
            pool=SharedComponentPool(),
        )
