from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import cast

from contracts.runner import RunnerProtocol
from worker.domains.module_compiler import (
    CompiledDetectionModuleRegistry,
    ModuleVersionSelection,
)
from worker.domains.module_definition import (
    ComponentBinding,
    DetectionModuleActivationError,
    SharedComponentIdentity,
)
from worker.interfaces.serving import ServingClient
from worker.pipeline.analytics import Clock, NamedExtractor

SharedProvisioner = Callable[[ComponentBinding, str], object]


@dataclass(frozen=True, slots=True)
class ProvisionedSharedComponent:
    component: object
    artifact_digest: str | None = None
    preprocessing_identity: str | None = None
    runtime: str | None = None


@dataclass(frozen=True, slots=True)
class SharedComponentGraph:
    components: Mapping[str, object]
    extractors: tuple[NamedExtractor, ...]
    identities: tuple[SharedComponentIdentity, ...]


class SharedComponentPool:
    """Process-local pool keyed by immutable execution and artifact identity."""

    def __init__(self) -> None:
        self._components: dict[SharedComponentIdentity, object] = {}

    def get_or_create(
        self,
        identity: SharedComponentIdentity,
        factory: Callable[[], object],
    ) -> object:
        component = self._components.get(identity)
        if component is None:
            component = factory()
            self._components[identity] = component
        return component

    @property
    def identities(self) -> tuple[SharedComponentIdentity, ...]:
        return tuple(self._components)


def compose_shared_components(
    registry: CompiledDetectionModuleRegistry,
    *,
    module_versions: ModuleVersionSelection,
    serving_client: ServingClient,
    runtime: str,
    device: str,
    flags: Mapping[str, bool],
    pool: SharedComponentPool,
    provisioners: Mapping[str, SharedProvisioner] | None = None,
    clock: Clock = perf_counter,
) -> SharedComponentGraph:
    """Materialize the selected graph through its declared provisioners.

    Provisioner names resolve only through this fixed, injected mapping; no
    module path or arbitrary Python symbol is ever loaded from configuration.
    """
    configured_provisioners: dict[str, SharedProvisioner] = {
        "serving-client": lambda binding, selected_device: serving_client.create(
            _serving_task(binding), device=selected_device
        )
    }
    if provisioners is not None:
        configured_provisioners.update(provisioners)

    components: dict[str, object] = {}
    extractors: list[NamedExtractor] = []
    identities: list[SharedComponentIdentity] = []
    bindings = registry.shared_bindings(module_versions, flags=flags)
    ordered_bindings = tuple(
        binding for binding in bindings if binding.component_kind == "extractor"
    ) + tuple(binding for binding in bindings if binding.component_kind != "extractor")
    for binding in ordered_bindings:
        provisioner_id = binding.provisioner
        provisioner = (
            None if provisioner_id is None else configured_provisioners.get(provisioner_id)
        )
        if provisioner is None:
            raise DetectionModuleActivationError(
                "detection module requires unavailable component(s): "
                + binding.component_id
                + f" (unknown provisioner {provisioner_id!r})"
            )
        provisioned = provisioner(binding, device)
        resolved = (
            provisioned
            if isinstance(provisioned, ProvisionedSharedComponent)
            else ProvisionedSharedComponent(provisioned)
        )
        component = resolved.component
        artifact_digest = _verified_identity_field(
            binding,
            component,
            "artifact_digest",
            resolved.artifact_digest,
            "artifact",
        )
        preprocessing_identity = _verified_identity_field(
            binding,
            component,
            "preprocessing_identity",
            resolved.preprocessing_identity,
            "preprocessing",
        )
        identity = SharedComponentIdentity(
            component_id=binding.component_id,
            artifact_digest=artifact_digest,
            runtime=resolved.runtime or runtime,
            device=device,
            preprocessing_identity=preprocessing_identity,
        )
        component = pool.get_or_create(identity, lambda component=component: component)
        identities.append(identity)
        if binding.component_kind == "extractor":
            runner = _require_runner(component, binding.component_id)
            extractor = NamedExtractor(
                module_name=binding.component_id,
                runner=runner,
                _call=runner if callable(runner) else runner.run,
                _clock=clock,
                output_adapter=binding.output_adapter,
            )
            components[binding.component_id] = extractor
            extractors.append(extractor)
        else:
            components[binding.component_id] = component
    return SharedComponentGraph(
        MappingProxyType(components),
        tuple(extractors),
        tuple(identities),
    )


def _serving_task(binding: ComponentBinding) -> str:
    if not binding.serving_task:
        raise DetectionModuleActivationError(
            f"component {binding.component_id!r} has no serving task"
        )
    return binding.serving_task


def _verified_identity_field(
    binding: ComponentBinding,
    component: object,
    field: str,
    provisioned: str | None,
    label: str,
) -> str:
    expected = getattr(binding, field)
    if not isinstance(expected, str) or not expected or "runtime-resolved" in expected:
        raise DetectionModuleActivationError(
            f"component {binding.component_id!r} has no compiled {label} identity"
        )
    actual = getattr(component, field, None)
    if actual is None:
        actual = getattr(component, f"_{field}", None)
    resolved = actual if actual is not None else provisioned
    if not isinstance(resolved, str) or not resolved or "runtime-resolved" in resolved:
        raise DetectionModuleActivationError(
            f"component {binding.component_id!r} has no resolved {label} identity"
        )
    if resolved != expected:
        raise DetectionModuleActivationError(
            f"component {binding.component_id!r} {label} identity mismatch: "
            f"compiled {expected!r}, resolved {resolved!r}"
        )
    return resolved


def _require_runner(component: object, component_id: str) -> RunnerProtocol:
    if callable(component) or callable(getattr(component, "run", None)):
        return cast("RunnerProtocol", component)
    raise DetectionModuleActivationError(f"extractor component {component_id!r} is not a runner")


@dataclass(frozen=True, slots=True)
class SharedYoloExtractors:
    """Hold the process-shared YOLO extractors provisioned by runtime.

    ``person`` is ``None`` whenever ``box_source`` (issue #44) is "pose" --
    the person model is then never provisioned at all, not merely unused.
    """

    pose: NamedExtractor
    person: NamedExtractor | None
    bed: NamedExtractor

    @property
    def extractors(self) -> tuple[NamedExtractor, ...]:
        return tuple(
            extractor for extractor in (self.pose, self.person, self.bed) if extractor is not None
        )


__all__ = [
    "ProvisionedSharedComponent",
    "SharedComponentGraph",
    "SharedComponentPool",
    "SharedProvisioner",
    "SharedYoloExtractors",
    "compose_shared_components",
]
