"""Validation and activation for qualified detection-module definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.detection_policies import (
    LATEST_POLICY_VERSIONS,
    PolicyDocumentError,
    policy_definition,
)
from worker.domains.module_definition import (
    ComponentBinding,
    DetectionModuleActivationError,
    DetectionModuleCompilationError,
    DetectionModuleDefinition,
    RuntimeResolvedArtifactDigest,
)
from worker.types import TemporalProfile

ModuleVersionSelection = Mapping[str, int]


@dataclass(frozen=True, slots=True)
class DetectionModuleActivation:
    definitions: tuple[DetectionModuleDefinition, ...]
    schedule: Mapping[str, int]

    @property
    def qualified_ids(self) -> tuple[str, ...]:
        return tuple(definition.qualified_id for definition in self.definitions)


@dataclass(frozen=True, slots=True)
class CompiledDetectionModuleRegistry:
    definitions: tuple[DetectionModuleDefinition, ...]
    by_id: Mapping[str, DetectionModuleDefinition]
    by_qualified_id: Mapping[tuple[str, int], DetectionModuleDefinition]
    latest_versions: Mapping[str, int]

    @property
    def qualified_ids(self) -> tuple[str, ...]:
        return tuple(definition.qualified_id for definition in self.definitions)

    def get(self, module_id: str, version: int | None = None) -> DetectionModuleDefinition:
        selected_version = self.latest_versions.get(module_id) if version is None else version
        if selected_version is None:
            raise DetectionModuleActivationError(f"unknown detection module {module_id!r}")
        try:
            if selected_version == self.latest_versions.get(module_id):
                return self.by_id[module_id]
            return self.by_qualified_id[(module_id, selected_version)]
        except KeyError as exc:
            raise DetectionModuleActivationError(
                f"configured detection module {module_id!r} version "
                f"{selected_version} is not compiled"
            ) from exc

    def selected(
        self, selection: ModuleVersionSelection | Iterable[str]
    ) -> tuple[DetectionModuleDefinition, ...]:
        if isinstance(selection, Mapping):
            return tuple(self.get(module_id, version) for module_id, version in selection.items())
        return tuple(self.get(module_id) for module_id in selection)

    def shared_bindings(
        self,
        selection: ModuleVersionSelection | Iterable[str],
        *,
        flags: Mapping[str, bool],
    ) -> tuple[ComponentBinding, ...]:
        bindings: dict[str, ComponentBinding] = {}
        owners: dict[str, str] = {}
        for definition in self.selected(selection):
            for binding in definition.shared_bindings:
                if binding.activation_flag is not None and not flags.get(
                    binding.activation_flag, False
                ):
                    continue
                previous = bindings.setdefault(binding.component_id, binding)
                owner = owners.setdefault(binding.component_id, definition.qualified_id)
                if previous != binding:
                    raise DetectionModuleActivationError(
                        f"component binding conflict for {binding.component_id!r} "
                        f"between {owner} and {definition.qualified_id}"
                    )
        return tuple(bindings.values())

    def shared_component_ids(
        self,
        selection: ModuleVersionSelection | Iterable[str],
        *,
        flags: Mapping[str, bool],
    ) -> frozenset[str]:
        return frozenset(
            binding.component_id for binding in self.shared_bindings(selection, flags=flags)
        )

    def activation(
        self,
        *,
        module_versions: ModuleVersionSelection | None = None,
        module_ids: Iterable[str] | None = None,
        available_observation_channels: Iterable[str],
        available_component_ids: Iterable[str],
        warmed_component_ids: Iterable[str],
        output_adapter_ids: Iterable[str],
        camera_frame_stride: int,
        flags: Mapping[str, bool],
        temporal_profile: TemporalProfile,
    ) -> DetectionModuleActivation:
        if (module_versions is None) == (module_ids is None):
            raise DetectionModuleActivationError(
                "activation requires exactly one module version selection"
            )
        selection: ModuleVersionSelection | Iterable[str] = (
            module_versions if module_versions is not None else module_ids or ()
        )
        selected = self.selected(selection)
        available_channels = frozenset(available_observation_channels)
        available_components = frozenset(available_component_ids)
        warmed = frozenset(warmed_component_ids)
        output_adapters = frozenset(output_adapter_ids)
        required_bindings = self.shared_bindings(selection, flags=flags)
        required_components = frozenset(binding.component_id for binding in required_bindings)
        missing_channels = sorted(
            {
                channel
                for definition in selected
                for channel in definition.required_observation_channels - available_channels
            }
        )
        missing_components = sorted(required_components - available_components)
        missing_warmup = sorted(
            binding.component_id
            for binding in required_bindings
            if binding.warmup_required and binding.component_id not in warmed
        )
        missing_outputs = sorted(
            {
                binding.output_adapter
                for binding in required_bindings
                if binding.component_kind == "extractor"
                and binding.output_adapter is not None
                and binding.output_adapter not in output_adapters
            }
        )
        _reject_output_writer_conflicts(required_bindings)
        failures = (
            ("observation channel", missing_channels),
            ("component binding", missing_components),
            ("component warmup", missing_warmup),
            ("output adapter", missing_outputs),
        )
        for label, values in failures:
            if values:
                raise DetectionModuleActivationError(
                    f"detection-module activation missing {label}(s): {', '.join(values)}"
                )
        schedule: dict[str, int] = {}
        for definition in selected:
            for rule in definition.schedule_rules:
                binding = _binding_by_id(definition, rule.component_id)
                if binding.activation_flag is not None and not flags.get(
                    binding.activation_flag, False
                ):
                    continue
                if rule.skip_when_flag is not None and flags.get(rule.skip_when_flag, False):
                    continue
                interval = rule.resolve(camera_frame_stride, temporal_profile)
                if interval is None:
                    continue
                previous = schedule.setdefault(rule.component_id, interval)
                if previous != interval:
                    raise DetectionModuleActivationError(
                        f"schedule conflict for component {rule.component_id!r}"
                    )
        return DetectionModuleActivation(selected, MappingProxyType(schedule))


def compile_detection_module_registry(
    definitions: Iterable[DetectionModuleDefinition],
    *,
    available_observation_channels: Iterable[str],
    output_adapter_ids: Iterable[str],
    temporal_profile: TemporalProfile,
) -> CompiledDetectionModuleRegistry:
    frozen = tuple(definitions)
    channels = frozenset(available_observation_channels)
    outputs = frozenset(output_adapter_ids)
    by_qualified: dict[tuple[str, int], DetectionModuleDefinition] = {}
    latest_versions: dict[str, int] = {}
    event_owner: dict[str, str] = {}
    shared_bindings: dict[tuple[str, str], ComponentBinding] = {}
    for definition in frozen:
        _validate_definition(definition, channels, outputs, temporal_profile)
        qualified = (definition.module_id, definition.version)
        if qualified in by_qualified:
            raise DetectionModuleCompilationError(
                f"duplicate qualified module id {definition.qualified_id!r}"
            )
        by_qualified[qualified] = definition
        latest_versions[definition.module_id] = max(
            definition.version,
            latest_versions.get(definition.module_id, 0),
        )
        for event_type in definition.event_types:
            owner = event_owner.setdefault(event_type, definition.module_id)
            if owner != definition.module_id:
                raise DetectionModuleCompilationError(
                    f"event type {event_type!r} conflicts between {owner!r} "
                    f"and {definition.module_id!r}"
                )
        for binding in definition.shared_bindings:
            key = (definition.module_id, binding.component_id)
            previous = shared_bindings.setdefault(key, binding)
            if previous != binding and definition.version == latest_versions[definition.module_id]:
                # Different versions may intentionally change a binding. Conflicts
                # across simultaneously selected modules are checked at activation.
                shared_bindings[key] = binding
    latest_by_id = {
        module_id: by_qualified[(module_id, version)]
        for module_id, version in latest_versions.items()
    }
    return CompiledDetectionModuleRegistry(
        frozen,
        MappingProxyType(latest_by_id),
        MappingProxyType(by_qualified),
        MappingProxyType(latest_versions),
    )


def _validate_definition(
    definition: DetectionModuleDefinition,
    channels: frozenset[str],
    outputs: frozenset[str],
    temporal_profile: TemporalProfile,
) -> None:
    if not definition.module_id or definition.version < 1:
        raise DetectionModuleCompilationError("module id/version must be valid")
    if not definition.policy_schema.schema_id or definition.policy_schema.version < 1:
        raise DetectionModuleCompilationError("policy schema identity must be valid")
    if definition.module_id in LATEST_POLICY_VERSIONS:
        try:
            compiled_policy = policy_definition(definition.module_id, definition.version)
        except PolicyDocumentError as error:
            raise DetectionModuleCompilationError(str(error)) from error
        if (
            definition.policy_schema.schema_id,
            definition.policy_schema.version,
        ) != (compiled_policy.schema_id, compiled_policy.schema_version):
            raise DetectionModuleCompilationError(
                f"module {definition.qualified_id!r} policy schema drift: "
                f"{definition.policy_schema.qualified_id!r}"
            )
    if not definition.event_types or any(not value for value in definition.event_types):
        raise DetectionModuleCompilationError("event types must be non-empty")
    missing_channels = definition.required_observation_channels - channels
    if missing_channels:
        raise DetectionModuleCompilationError(
            "required observation channel is unavailable: " + ", ".join(sorted(missing_channels))
        )
    if any(
        factory is None
        for factory in (
            definition.camera_state_factory,
            definition.decider_factory,
            definition.audit_adapter,
            definition.trace_adapter,
            definition.debug_adapter,
        )
    ):
        raise DetectionModuleCompilationError("state/decider/audit/trace adapter is missing")
    ids = tuple(binding.component_id for binding in definition.component_bindings)
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise DetectionModuleCompilationError("component binding ids must be unique and non-empty")
    schedule_ids = {rule.component_id for rule in definition.schedule_rules}
    _reject_compiled_output_writer_conflicts(definition)
    for binding in definition.component_bindings:
        if binding.shared:
            if not all((binding.model_family, binding.provisioner)):
                raise DetectionModuleCompilationError(
                    f"component binding {binding.component_id!r} is missing provisioner provenance"
                )
            if (
                binding.artifact_digest is None
                or not isinstance(
                    binding.artifact_digest,
                    (str, RuntimeResolvedArtifactDigest),
                )
                or binding.artifact_digest == ""
                or binding.preprocessing_identity == ""
            ):
                raise DetectionModuleCompilationError(
                    f"component binding {binding.component_id!r} has invalid provenance"
                )
            if not binding.warmup_required:
                raise DetectionModuleCompilationError(
                    f"component binding {binding.component_id!r} must require warmup"
                )
            if binding.component_kind == "extractor":
                if not binding.serving_task:
                    raise DetectionModuleCompilationError(
                        f"extractor {binding.component_id!r} has no serving task"
                    )
                if binding.component_id not in schedule_ids:
                    raise DetectionModuleCompilationError(
                        f"extractor {binding.component_id!r} is missing a schedule rule"
                    )
                if binding.output_adapter is None or binding.output_adapter not in outputs:
                    raise DetectionModuleCompilationError(
                        f"extractor {binding.component_id!r} has no registered output adapter"
                    )
        elif binding.camera_factory is None:
            raise DetectionModuleCompilationError(
                f"camera component {binding.component_id!r} has no factory"
            )
    for rule in definition.schedule_rules:
        binding = _binding_by_id(definition, rule.component_id)
        if binding.component_kind != "extractor":
            raise DetectionModuleCompilationError(
                f"schedule rule {rule.component_id!r} does not target an extractor"
            )
        interval = rule.resolve(1, temporal_profile)
        # An on-demand extractor resolves to no interval: it is provisioned but
        # never scheduled per frame.
        if interval is not None and interval <= 0:
            raise DetectionModuleCompilationError("schedule intervals must be positive")


def _reject_output_writer_conflicts(bindings: Iterable[ComponentBinding]) -> None:
    owners: dict[str, str] = {}
    for binding in bindings:
        if binding.component_kind != "extractor" or binding.output_adapter is None:
            continue
        owner = owners.setdefault(binding.output_adapter, binding.component_id)
        # One provisioned component shared by modules is one semantic writer.
        # Different components writing the same adapter are ambiguous and fatal.
        if owner != binding.component_id:
            raise DetectionModuleActivationError(
                f"output adapter {binding.output_adapter!r} has multiple active writers: "
                f"{owner!r}, {binding.component_id!r}"
            )


def _reject_compiled_output_writer_conflicts(
    definition: DetectionModuleDefinition,
) -> None:
    try:
        _reject_output_writer_conflicts(definition.shared_bindings)
    except DetectionModuleActivationError as exc:
        raise DetectionModuleCompilationError(f"module {definition.qualified_id!r} {exc}") from exc


def _binding_by_id(definition: DetectionModuleDefinition, component_id: str) -> ComponentBinding:
    for binding in definition.component_bindings:
        if binding.component_id == component_id:
            return binding
    raise DetectionModuleCompilationError(
        f"schedule references missing component binding {component_id!r}"
    )


__all__ = [
    "CompiledDetectionModuleRegistry",
    "DetectionModuleActivation",
    "ModuleVersionSelection",
    "compile_detection_module_registry",
]
