"""Hardware-neutral contracts for compiled detection-module composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Literal, TypeAlias

from shared.detection_policies import EffectivePolicy
from worker.domains.base import DomainAuditSnapshot
from worker.interfaces.decision import Decider
from worker.types import TemporalProfile

ObservationChannel: TypeAlias = str
ComponentKind: TypeAlias = Literal["extractor", "model", "state", "rule"]
IntervalSource: TypeAlias = Literal["camera-frame-stride", "fixed", "on-demand", "temporal-profile"]
WindowMode: TypeAlias = Literal["external", "internal"]


class DetectionModuleCompilationError(ValueError):
    pass


class DetectionModuleActivationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeResolvedIdentityField:
    """Marker for a shared model identity field supplied by its verified bundle."""


# Field-specific aliases retain readable binding annotations while making all
# runtime-resolved identity fields one explicit, non-string marker type.
RuntimeResolvedArtifactDigest = RuntimeResolvedIdentityField
RuntimeResolvedPreprocessingIdentity = RuntimeResolvedIdentityField


RUNTIME_RESOLVED_ARTIFACT_DIGEST = RuntimeResolvedIdentityField()
RUNTIME_RESOLVED_PREPROCESSING_IDENTITY = RuntimeResolvedIdentityField()


@dataclass(frozen=True, slots=True)
class PolicySchemaIdentity:
    schema_id: str
    version: int

    @property
    def qualified_id(self) -> str:
        return f"{self.schema_id}.v{self.version}"


@dataclass(frozen=True, slots=True)
class SharedComponentIdentity:
    """Process-sharing identity; changing any field creates another runner."""

    component_id: str
    artifact_digest: str
    runtime: str
    device: str
    preprocessing_identity: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.component_id,
                self.artifact_digest,
                self.runtime,
                self.device,
                self.preprocessing_identity,
            )
        ):
            raise ValueError("shared component identity fields must be non-empty")


CameraComponentFactory: TypeAlias = Callable[["CameraModuleContext"], object]


@dataclass(frozen=True, slots=True)
class ComponentBinding:
    component_id: str
    component_kind: ComponentKind
    model_family: str | None = None
    provisioner: str | None = None
    serving_task: str | None = None
    artifact_digest: str | RuntimeResolvedIdentityField | None = None
    preprocessing_identity: str | RuntimeResolvedIdentityField | None = None
    output_adapter: str | None = None
    activation_flag: str | None = None
    warmup_required: bool = False
    camera_factory: CameraComponentFactory | None = None

    @property
    def shared(self) -> bool:
        return self.component_kind in ("extractor", "model")

    def identity(self, *, runtime: str, device: str) -> SharedComponentIdentity:
        if not self.shared:
            raise DetectionModuleCompilationError(
                f"camera-local binding {self.component_id!r} has no shared identity"
            )
        if not isinstance(self.artifact_digest, str) or not isinstance(
            self.preprocessing_identity, str
        ):
            raise DetectionModuleCompilationError(
                f"component binding {self.component_id!r} is missing provenance"
            )
        return SharedComponentIdentity(
            component_id=self.component_id,
            artifact_digest=self.artifact_digest,
            runtime=runtime,
            device=device,
            preprocessing_identity=self.preprocessing_identity,
        )


@dataclass(frozen=True, slots=True)
class ScheduleRule:
    component_id: str
    interval_source: IntervalSource
    interval: int | None = None
    skip_when_flag: str | None = None

    def resolve(self, camera_frame_stride: int, temporal_profile: TemporalProfile) -> int | None:
        # temporal_profile is required: compile-time validation and live
        # activation must name the same owner. A missing argument used to
        # fall through to CURRENT_TEMPORAL_PROFILE, so a 15fps activation
        # still validated CURRENT's 30-frame bed interval.
        if self.interval_source == "on-demand":
            # Provisioned for an explicit operator request, never per frame.
            return None
        if self.interval_source == "camera-frame-stride":
            return camera_frame_stride
        if self.interval_source == "temporal-profile":
            return temporal_profile.decision_interval_frames(self.component_id)
        if self.interval is None:
            raise DetectionModuleCompilationError(
                f"fixed schedule for {self.component_id!r} is missing an interval"
            )
        return self.interval


@dataclass(frozen=True, slots=True)
class CameraModuleContext:
    camera_id: str
    facility_id: str
    shared_components: Mapping[str, object]
    camera_components: Mapping[str, object]
    detection_window: object | None
    clock: Callable[[], datetime]
    diagnostics: object | None
    policy: EffectivePolicy | None = None


@dataclass(frozen=True, slots=True)
class CameraDetectionModule:
    module_id: str
    version: int
    state: object
    decider: Decider
    camera_component_ids: frozenset[str]
    camera_components: Mapping[str, object]


StateFactory: TypeAlias = Callable[[CameraModuleContext], object]
DeciderFactory: TypeAlias = Callable[[object], Decider]
AuditAdapter: TypeAlias = Callable[[CameraModuleContext], DomainAuditSnapshot]
TraceAdapter: TypeAlias = Callable[[Decider], object]
DebugAdapter: TypeAlias = Callable[[object, int], object | None]


@dataclass(frozen=True, slots=True)
class DetectionModuleDefinition:
    module_id: str
    version: int
    required_observation_channels: frozenset[ObservationChannel]
    component_bindings: tuple[ComponentBinding, ...]
    schedule_rules: tuple[ScheduleRule, ...]
    policy_schema: PolicySchemaIdentity
    camera_state_factory: StateFactory | None
    decider_factory: DeciderFactory | None
    audit_adapter: AuditAdapter | None
    trace_adapter: TraceAdapter | None
    debug_adapter: DebugAdapter | None
    event_types: frozenset[str]
    input_view: str
    window_mode: WindowMode
    enabled: bool = True
    requires: frozenset[str] = frozenset()
    compatibility_factory: Callable[[object], Decider] | None = None
    compatibility_audit_adapter: Callable[[object], object] | None = None

    @property
    def qualified_id(self) -> str:
        return f"{self.module_id}.v{self.version}"

    @property
    def factory(self) -> Callable[[object], Decider]:
        if self.compatibility_factory is None:
            raise DetectionModuleActivationError(
                f"module {self.qualified_id!r} has no compatibility factory"
            )
        return self.compatibility_factory

    @property
    def audit_metadata_provider(self) -> Callable[[object], object] | None:
        return self.compatibility_audit_adapter

    @property
    def debug_snapshot_adapter(self) -> DebugAdapter | None:
        return self.debug_adapter

    @property
    def shared_bindings(self) -> tuple[ComponentBinding, ...]:
        return tuple(binding for binding in self.component_bindings if binding.shared)

    @property
    def camera_component_ids(self) -> frozenset[str]:
        return frozenset(
            binding.component_id for binding in self.component_bindings if not binding.shared
        )

    def create_camera_module(self, context: CameraModuleContext) -> CameraDetectionModule:
        if self.camera_state_factory is None or self.decider_factory is None:
            raise DetectionModuleActivationError(
                f"module {self.qualified_id!r} has no camera state/decider factory"
            )
        components = dict(context.camera_components)
        for binding in self.component_bindings:
            if binding.shared or binding.component_id in components:
                continue
            factory = binding.camera_factory
            if factory is None:
                raise DetectionModuleActivationError(
                    f"module {self.qualified_id!r} camera component "
                    f"{binding.component_id!r} has no factory"
                )
            factory_context = replace(
                context,
                camera_components=MappingProxyType(dict(components)),
            )
            components[binding.component_id] = factory(factory_context)
        resolved_context = replace(
            context,
            camera_components=MappingProxyType(components),
        )
        state = self.camera_state_factory(resolved_context)
        return CameraDetectionModule(
            self.module_id,
            self.version,
            state,
            self.decider_factory(state),
            self.camera_component_ids,
            resolved_context.camera_components,
        )


__all__ = [
    "CameraDetectionModule",
    "CameraModuleContext",
    "ComponentBinding",
    "DetectionModuleActivationError",
    "DetectionModuleCompilationError",
    "DetectionModuleDefinition",
    "PolicySchemaIdentity",
    "ScheduleRule",
    "SharedComponentIdentity",
]
