from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields, replace

from shared.detection_policies import default_policy_bundle, make_effective_policy
from worker.domains.module_compiler import (
    CompiledDetectionModuleRegistry,
    compile_detection_module_registry,
)
from worker.domains.module_definition import (
    DetectionModuleDefinition,
    RuntimeResolvedArtifactDigest,
    RuntimeResolvedPreprocessingIdentity,
    SharedComponentIdentity,
)
from worker.domains.registry import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_DEFINITIONS,
    DETECTION_MODULE_REGISTRY,
)
from worker.pipeline.analytics.merge import result_merger_names
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.provenance.content import module_content
from worker.runtime.provenance.manifest import (
    RuntimeEnvironmentFacts,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.runtime.provenance.models import AppliedRuntimeManifest, JsonValue, canonical_json
from worker.types import CURRENT_TEMPORAL_PROFILE

_CAMERA_ID = "camera:module-catalog/a"
_BUILD_REVISION = "1" * 40
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _compile(
    definitions: Sequence[DetectionModuleDefinition],
) -> CompiledDetectionModuleRegistry:
    return compile_detection_module_registry(
        definitions,
        available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
        output_adapter_ids=result_merger_names(),
        temporal_profile=CURRENT_TEMPORAL_PROFILE,
    )


def _registry_with(
    *,
    module_change: bool = False,
    component_change: bool = False,
    model_change: bool = False,
) -> CompiledDetectionModuleRegistry:
    definitions: list[DetectionModuleDefinition] = []
    for definition in DETECTION_MODULE_DEFINITIONS:
        changed = definition
        if module_change and definition.module_id == "fall":
            changed = replace(changed, input_view="fall_window.v2")
        if component_change or model_change:
            changed = replace(
                changed,
                component_bindings=tuple(
                    replace(binding, preprocessing_identity="rgb24-to-coco17.v2")
                    if component_change and binding.component_id == "pose"
                    else replace(
                        binding,
                        artifact_digest="d" * 64,
                        preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
                    )
                    if model_change and binding.component_id == "fall-classifier"
                    else binding
                    for binding in changed.component_bindings
                ),
            )
        definitions.append(changed)
    return _compile(definitions)


def _boot(profile_name: str) -> BootContext:
    profile = PROFILE_REGISTRY[profile_name]
    return BootContext(
        profile=profile,
        device=profile.device,
        decode=profile.decode,
        encode=profile.encode,
        requested_profile=profile_name,
    )


def _environment(profile_name: str) -> RuntimeEnvironmentFacts:
    nvidia = profile_name == "flow"
    return RuntimeEnvironmentFacts(
        worker_build_revision=_BUILD_REVISION,
        os_name="Linux",
        architecture="x86_64",
        python_version="3.12.11",
        model_runtime="torch",
        model_runtime_version="2.13.0",
        accelerator_runtime="CUDA 13.0" if nvidia else None,
        driver_version="580.65" if nvidia else None,
        device_name="NVIDIA RTX" if nvidia else None,
    )


def _manifest(
    *,
    registry: CompiledDetectionModuleRegistry = DETECTION_MODULE_REGISTRY,
    profile_name: str = "flow",
    fall_threshold: float | None = None,
    reordered: bool = False,
) -> AppliedRuntimeManifest:
    selection = {"fall": 2, "bed_exit": 1}
    boot = _boot(profile_name)
    bindings = registry.shared_bindings(selection, flags={"person-box-source": True})
    identities = tuple(
        SharedComponentIdentity(
            component_id=binding.component_id,
            artifact_digest="c" * 64,
            runtime=boot.runtime_profile.effective_inference_backend,
            device=boot.device,
            preprocessing_identity="coco17-xyc-plus-pose-head-xyxy-valid-f32-v1",
        )
        if isinstance(binding.artifact_digest, RuntimeResolvedArtifactDigest)
        else binding.identity(
            runtime=boot.runtime_profile.effective_inference_backend,
            device=boot.device,
        )
        for binding in bindings
    )
    bundle = default_policy_bundle((_CAMERA_ID,))
    fall_policy = bundle.resolve(_CAMERA_ID, "fall", 2)
    if fall_threshold is not None:
        fall_policy = make_effective_policy(
            module_id="fall",
            module_version=2,
            values=replace(
                fall_policy.values,
                transition_threshold=fall_threshold,
            ),
            source=fall_policy.source,
            facility_revision_id=fall_policy.facility_revision_id,
            camera_revision_id=fall_policy.camera_revision_id,
        )
    policies = {
        "fall": fall_policy,
        "bed_exit": bundle.resolve(_CAMERA_ID, "bed_exit", 1),
    }
    windows = {"fall": None, "bed_exit": None}
    schedule = {"pose": 2, "person": 2, "bed": 30}
    module_ids = ("fall.v2", "bed_exit.v1")
    if reordered:
        selection = dict(reversed(tuple(selection.items())))
        identities = tuple(reversed(identities))
        policies = dict(reversed(tuple(policies.items())))
        windows = dict(reversed(tuple(windows.items())))
        schedule = dict(reversed(tuple(schedule.items())))
        module_ids = tuple(reversed(module_ids))
    camera = build_applied_camera_state(
        camera_id=_CAMERA_ID,
        effective_decode_backend=boot.decode,
        ingest_target_fps=5.0,
        module_qualified_ids=module_ids,
        schedule=schedule,
        detection_windows=windows,
        policies=policies,
        bed_zone_polygon=None,
        bed_zone_image_width=None,
        bed_zone_image_height=None,
    )
    return build_applied_runtime_manifest(
        boot=boot,
        module_registry=registry,
        module_versions=selection,
        component_identities=identities,
        cameras=(camera,),
        config_version=9,
        restart_generation=3,
        detector_version="worker-domain-detectors-v1",
        environment=_environment(profile_name),
        edge_database_schema_version=5,
    )


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, member in value.items():
            yield str(key)
            yield from _strings(member)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for member in value:
            yield from _strings(member)


def test_catalog_qualifies_production_modules_components_and_model_bindings() -> None:
    assert DETECTION_MODULE_REGISTRY.qualified_ids == ("fall.v2", "bed_exit.v1")
    assert dict(DETECTION_MODULE_REGISTRY.latest_versions) == {"fall": 2, "bed_exit": 1}

    expected_components = {
        "fall.v2": (
            ("pose", "extractor"),
            ("person-tracker", "state"),
            ("fall-classifier", "model"),
            ("fall-v2", "state"),
        ),
        "bed_exit.v1": (
            ("pose", "extractor"),
            ("person", "extractor"),
            ("bed", "extractor"),
            ("person-tracker", "state"),
            ("containment", "rule"),
            ("bed-assignment", "state"),
            ("bed-exit-state", "state"),
        ),
    }
    expected_models = {
        "fall.v2": {
            "pose": ("yolo-pose", "serving-client", "pose"),
            "fall-classifier": (
                "configured-fall-family",
                "fall-model-family-registry",
                None,
            ),
        },
        "bed_exit.v1": {
            "pose": ("yolo-pose", "serving-client", "pose"),
            "person": ("yolo-person", "serving-client", "person"),
            "bed": ("yolo-bed-segmentation", "serving-client", "bed"),
        },
    }

    for definition in DETECTION_MODULE_REGISTRY.definitions:
        assert definition.qualified_id == f"{definition.module_id}.v{definition.version}"
        assert (
            tuple(
                (binding.component_id, binding.component_kind)
                for binding in definition.component_bindings
            )
            == expected_components[definition.qualified_id]
        )
        model_bindings = {
            binding.component_id: (
                binding.model_family,
                binding.provisioner,
                binding.serving_task,
            )
            for binding in definition.component_bindings
            if binding.model_family is not None
        }
        assert model_bindings == expected_models[definition.qualified_id]
        for binding in definition.shared_bindings:
            assert isinstance(binding.artifact_digest, (str, RuntimeResolvedArtifactDigest))
            if isinstance(binding.artifact_digest, str):
                assert _SHA256.fullmatch(binding.artifact_digest)
            assert isinstance(
                binding.preprocessing_identity,
                (str, RuntimeResolvedPreprocessingIdentity),
            )
            assert binding.warmup_required is True

    fall_pose = DETECTION_MODULE_REGISTRY.get("fall", 2).shared_bindings[0]
    bed_exit_pose = DETECTION_MODULE_REGISTRY.get("bed_exit", 1).shared_bindings[0]
    assert fall_pose == bed_exit_pose


def test_pose_binding_identity_is_byte_identical_across_fall_v2_and_bed_exit_v1() -> None:
    """Cross-camera batched pose (nvidia-multistream-serving) rests on this.

    A single inference owner may batch every camera's pose frames into ONE
    forward only because fall.v2 and bed_exit.v1 declare the SAME pose
    component: same artifact, same preprocessing, same serving task. If the
    two modules ever diverge on any identity field, ``SharedComponentPool``
    correctly produces two runners and one batched pose lane becomes wrong,
    not merely slower. The equality above compares whole bindings; this pins
    the individual fields so a failure names which one drifted, and pins the
    derived ``SharedComponentIdentity`` that the pool actually keys on.
    """
    fall_pose = next(
        binding
        for binding in DETECTION_MODULE_REGISTRY.get("fall", 2).shared_bindings
        if binding.component_id == "pose"
    )
    bed_exit_pose = next(
        binding
        for binding in DETECTION_MODULE_REGISTRY.get("bed_exit", 1).shared_bindings
        if binding.component_id == "pose"
    )

    identity_fields = (
        "component_id",
        "model_family",
        "provisioner",
        "serving_task",
        "artifact_digest",
        "preprocessing_identity",
    )
    assert {field_name: getattr(fall_pose, field_name) for field_name in identity_fields} == {
        field_name: getattr(bed_exit_pose, field_name) for field_name in identity_fields
    }
    assert fall_pose.serving_task == "pose"
    assert fall_pose.provisioner == "serving-client"

    runtime, device = "onnxruntime", "cuda:0"
    assert fall_pose.identity(runtime=runtime, device=device) == bed_exit_pose.identity(
        runtime=runtime, device=device
    )
    assert (
        len(
            {
                fall_pose.identity(runtime=runtime, device=device),
                bed_exit_pose.identity(runtime=runtime, device=device),
            }
        )
        == 1
    )


def test_catalog_order_and_projected_identity_are_deterministic() -> None:
    first = _compile(DETECTION_MODULE_DEFINITIONS)
    second = _compile(tuple(replace(definition) for definition in DETECTION_MODULE_DEFINITIONS))

    first_content: list[JsonValue] = [module_content(item) for item in first.definitions]
    second_content: list[JsonValue] = [module_content(item) for item in second.definitions]
    assert first.qualified_ids == second.qualified_ids == ("fall.v2", "bed_exit.v1")
    assert canonical_json(first_content) == canonical_json(second_content)
    assert first.get("fall") is first.definitions[0]
    assert first.get("bed_exit") is first.definitions[1]


def test_catalog_preserves_qualified_identity_for_enabled_and_disabled_modules() -> None:
    fall, bed_exit = DETECTION_MODULE_DEFINITIONS
    registry = _compile((fall, replace(bed_exit, enabled=False)))

    qualification = {
        state: tuple(
            definition.qualified_id
            for definition in registry.definitions
            if definition.enabled is state
        )
        for state in (True, False)
    }

    assert qualification == {True: ("fall.v2",), False: ("bed_exit.v1",)}
    assert registry.get("fall", 2).enabled is True
    assert registry.get("bed_exit", 1).enabled is False


def test_manifest_identity_changes_iff_effective_catalog_or_runtime_state_changes() -> None:
    baseline = _manifest()
    equivalent = _manifest(
        registry=_compile(tuple(replace(item) for item in DETECTION_MODULE_DEFINITIONS)),
        reordered=True,
    )
    changed = {
        "module": _manifest(registry=_registry_with(module_change=True)),
        "component": _manifest(registry=_registry_with(component_change=True)),
        "model": _manifest(registry=_registry_with(model_change=True)),
        "policy": _manifest(fall_threshold=0.72),
    }

    assert equivalent.sha256 == baseline.sha256
    assert equivalent.canonical_json == baseline.canonical_json
    assert all(manifest.sha256 != baseline.sha256 for manifest in changed.values())
    assert len({manifest.sha256 for manifest in changed.values()}) == len(changed)

    baseline_content = json.loads(baseline.canonical_json)
    assert json.loads(changed["module"].canonical_json)["modules"] != baseline_content["modules"]
    assert (
        json.loads(changed["component"].canonical_json)["components"]
        != baseline_content["components"]
    )
    assert (
        json.loads(changed["model"].canonical_json)["components"] != baseline_content["components"]
    )
    assert (
        json.loads(changed["policy"].canonical_json)["cameras"][0]["policies"]
        != baseline_content["cameras"][0]["policies"]
    )


def test_module_definitions_are_profile_independent_and_emit_no_secrets_or_local_paths() -> None:
    definition_fields = {field.name for field in fields(DetectionModuleDefinition)}
    assert definition_fields.isdisjoint(
        {
            "profile",
            "device",
            "decode_backend",
            "encode_backend",
            "inference_backend",
            "memory_path",
            "converter_chain",
            "copy_count",
        }
    )

    flow_content = json.loads(_manifest().canonical_json)

    serialized_catalog = canonical_json(
        [module_content(definition) for definition in DETECTION_MODULE_REGISTRY.definitions]
    )
    serialized_manifest = canonical_json(flow_content)
    forbidden_fragments = (
        "://",
        "/Users/",
        "/var/",
        "C:\\",
        "rtsp",
        "password",
        "secret",
        "token",
        "facility_id",
    )
    assert all(
        fragment.casefold() not in payload.casefold()
        for payload in (serialized_catalog, serialized_manifest)
        for fragment in forbidden_fragments
    )
    assert all(
        not value.startswith(("/", "\\\\")) and re.match(r"[A-Za-z]:[\\/]", value) is None
        for value in _strings((flow_content["modules"], flow_content["components"]))
    )
