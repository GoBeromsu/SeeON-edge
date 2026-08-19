from __future__ import annotations

import json
from dataclasses import replace

import pytest

from shared.detection_policies import FallPolicyV1, default_policy_bundle, make_effective_policy
from worker.domains.module_compiler import (
    CompiledDetectionModuleRegistry,
    compile_detection_module_registry,
)
from worker.domains.module_definition import SharedComponentIdentity
from worker.domains.registry import (
    AVAILABLE_OBSERVATION_CHANNELS,
    DETECTION_MODULE_DEFINITIONS,
    DETECTION_MODULE_REGISTRY,
)
from worker.pipeline.analytics.merge import result_merger_names
from worker.runtime.profile.boot import BootContext
from worker.runtime.profile.registry import PROFILE_REGISTRY
from worker.runtime.provenance.manifest import (
    AppliedRuntimeManifestError,
    RuntimeEnvironmentFacts,
    build_applied_camera_state,
    build_applied_runtime_manifest,
)
from worker.types import CURRENT_TEMPORAL_PROFILE

_CAMERA_ID = "camera/module-status"
_BUILD_REVISION = "1" * 40


def _facts(*, nvidia: bool = False, os_name: str = "Linux") -> RuntimeEnvironmentFacts:
    return RuntimeEnvironmentFacts(
        worker_build_revision=_BUILD_REVISION,
        os_name=os_name,
        architecture="x86_64",
        python_version="3.12.11",
        model_runtime="torch",
        model_runtime_version="2.13.0",
        accelerator_runtime="CUDA 13.0" if nvidia else None,
        driver_version="580.65" if nvidia else None,
        device_name="NVIDIA RTX" if nvidia else None,
    )


def _policy(threshold: float):
    default = default_policy_bundle((_CAMERA_ID,)).resolve(_CAMERA_ID, "fall", 1)
    return make_effective_policy(
        module_id="fall",
        module_version=1,
        values=FallPolicyV1(operating_threshold=threshold),
        source=default.source,
        facility_revision_id=default.facility_revision_id,
        camera_revision_id=default.camera_revision_id,
    )


def _compiled_with(*, module_change: bool = False, model_change: bool = False):
    definitions = []
    for definition in DETECTION_MODULE_DEFINITIONS:
        changed = definition
        if definition.module_id == "fall" and module_change:
            changed = replace(
                changed,
                event_types=changed.event_types | {"fall-warning"},
            )
        if definition.module_id == "fall" and model_change:
            changed = replace(
                changed,
                component_bindings=tuple(
                    replace(binding, artifact_digest="d" * 64)
                    if binding.component_id == "fall-classifier"
                    else binding
                    for binding in changed.component_bindings
                ),
            )
        definitions.append(changed)
    return compile_detection_module_registry(
        definitions,
        available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
        output_adapter_ids=result_merger_names(),
        temporal_profile=CURRENT_TEMPORAL_PROFILE,
    )


def _identities(
    registry: CompiledDetectionModuleRegistry,
    *,
    runtime: str,
    device: str,
) -> tuple[SharedComponentIdentity, ...]:
    return tuple(
        binding.identity(runtime=runtime, device=device)
        for binding in registry.shared_bindings({"fall": 1}, flags={})
    )


def _manifest(
    *,
    registry: CompiledDetectionModuleRegistry = DETECTION_MODULE_REGISTRY,
    profile_name: str = "cpu-host",
    threshold: float = 0.5,
    facts: RuntimeEnvironmentFacts | None = None,
    reverse_identities: bool = False,
):
    profile = PROFILE_REGISTRY[profile_name]
    boot = BootContext(
        profile=profile,
        device=profile.device,
        decode=profile.decode,
        encode=profile.encode,
        requested_profile=profile_name,
    )
    policy = _policy(threshold)
    identities = _identities(
        registry,
        runtime=boot.runtime_profile.effective_inference_backend,
        device=boot.device,
    )
    if reverse_identities:
        identities = tuple(reversed(identities))
    camera = build_applied_camera_state(
        camera_id=_CAMERA_ID,
        effective_decode_backend=boot.decode,
        ingest_target_fps=5.0,
        module_qualified_ids=("fall.v1",),
        schedule={"pose": 2},
        detection_windows={"fall": None},
        policies={"fall": policy},
        bed_zone_polygon=None,
        bed_zone_image_width=None,
        bed_zone_image_height=None,
    )
    return build_applied_runtime_manifest(
        boot=boot,
        module_registry=registry,
        module_versions={"fall": 1},
        component_identities=identities,
        cameras=(camera,),
        config_version=9,
        restart_generation=3,
        detector_version="worker-domain-detectors-v1",
        environment=facts or _facts(nvidia=profile_name.startswith("nvidia-")),
        edge_database_schema_version=5,
    )


def test_runtime_manifest_identity_tracks_effective_module_profile_model_and_policy() -> None:
    baseline = _manifest()
    equivalent = _manifest(reverse_identities=True)
    module_changed = _manifest(registry=_compiled_with(module_change=True))
    model_changed = _manifest(registry=_compiled_with(model_change=True))
    profile_changed = _manifest(profile_name="nvidia-host-bridge")
    policy_changed = _manifest(threshold=0.72)

    assert equivalent.sha256 == baseline.sha256
    assert equivalent.canonical_json == baseline.canonical_json
    assert (
        len(
            {
                baseline.sha256,
                module_changed.sha256,
                model_changed.sha256,
                profile_changed.sha256,
                policy_changed.sha256,
            }
        )
        == 5
    )

    baseline_content = json.loads(baseline.canonical_json)
    assert json.loads(module_changed.canonical_json)["modules"] != baseline_content["modules"]
    assert json.loads(model_changed.canonical_json)["components"] != baseline_content["components"]
    assert json.loads(profile_changed.canonical_json)["profile"] != baseline_content["profile"]
    assert (
        json.loads(policy_changed.canonical_json)["cameras"][0]["policies"]
        != baseline_content["cameras"][0]["policies"]
    )


def test_runtime_manifest_status_contains_no_secret_url_or_filesystem_path() -> None:
    manifest = _manifest()
    content = manifest.canonical_json

    assert "rtsp://" not in content
    assert "facility-secret" not in content
    assert "/var/lib/eldercare" not in content
    assert "facility_id" not in content

    with pytest.raises(AppliedRuntimeManifestError, match="unsafe provenance value"):
        _manifest(facts=_facts(os_name="/var/lib/eldercare/worker"))
