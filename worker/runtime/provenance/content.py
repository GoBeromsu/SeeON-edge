from __future__ import annotations

from worker.domains.module_definition import (
    ComponentBinding,
    DetectionModuleDefinition,
    SharedComponentIdentity,
)
from worker.runtime.profile.boot import BootContext
from worker.runtime.provenance.models import JsonValue


def profile_content(boot: BootContext) -> dict[str, JsonValue]:
    descriptor = boot.runtime_profile
    graph = boot.capability_graph
    return {
        "requested": descriptor.requested_profile,
        "canonical": descriptor.canonical_profile,
        "backends": {
            "decode": _requested_effective(
                descriptor.requested_decode_backend, descriptor.effective_decode_backend
            ),
            "preprocess": _requested_effective(
                descriptor.requested_preprocess_backend,
                descriptor.effective_preprocess_backend,
            ),
            "inference": _requested_effective(
                descriptor.requested_inference_backend,
                descriptor.effective_inference_backend,
            ),
            "overlay": _requested_effective(
                descriptor.requested_overlay_backend, descriptor.effective_overlay_backend
            ),
            "encode": _requested_effective(
                descriptor.requested_encode_backend, descriptor.effective_encode_backend
            ),
        },
        "requested_memory_path": list(descriptor.requested_memory_path),
        "effective_memory_path": list(descriptor.memory_path),
        "requested_converters": list(descriptor.requested_converter_chain),
        "effective_converters": list(descriptor.converter_chain),
        "validated_edges": [
            {
                "source_stage": edge.source_stage,
                "target_stage": edge.target_stage,
                "converter_names": list(edge.converter_names),
                "validated": edge.validated,
            }
            for edge in graph.edges
        ],
        "full_frame_copy_counts": {
            "h2d": descriptor.full_frame_h2d_count,
            "d2h": descriptor.full_frame_d2h_count,
        },
        "degraded_reasons": list(descriptor.degraded_reasons),
        "device_resident_after_decode": descriptor.device_resident_after_decode,
    }


def module_content(definition: DetectionModuleDefinition) -> dict[str, JsonValue]:
    observation_channels: list[JsonValue] = []
    observation_channels.extend(sorted(definition.required_observation_channels))
    event_types: list[JsonValue] = []
    event_types.extend(sorted(definition.event_types))
    return {
        "module_id": definition.module_id,
        "version": definition.version,
        "qualified_id": definition.qualified_id,
        "required_observation_channels": observation_channels,
        "policy_schema": definition.policy_schema.qualified_id,
        "event_types": event_types,
        "input_view": definition.input_view,
        "window_mode": definition.window_mode,
        "component_bindings": [
            _binding_content(binding)
            for binding in sorted(
                definition.component_bindings, key=lambda binding: binding.component_id
            )
        ],
    }


def component_content(identity: SharedComponentIdentity) -> dict[str, JsonValue]:
    return {
        "component_id": identity.component_id,
        "artifact_sha256": identity.artifact_digest,
        "preprocessing_identity": identity.preprocessing_identity,
        "runtime": identity.runtime,
        "device": identity.device,
    }


def _requested_effective(requested: str, effective: str) -> dict[str, JsonValue]:
    return {"requested": requested, "effective": effective}


def _binding_content(binding: ComponentBinding) -> dict[str, JsonValue]:
    return {
        "component_id": binding.component_id,
        "kind": binding.component_kind,
        "model_family": binding.model_family,
        "provisioner": binding.provisioner,
        "serving_task": binding.serving_task,
        "output_adapter": binding.output_adapter,
        "activation_flag": binding.activation_flag,
        "warmup_required": binding.warmup_required,
    }


__all__ = ["component_content", "module_content", "profile_content"]
