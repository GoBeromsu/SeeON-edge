"""Compiled, versioned detection-module registry.

The definitions in this module describe detection semantics only. Infrastructure
profiles decide where the declared components run; they never change this graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Literal, NamedTuple, Protocol, cast, runtime_checkable

from shared.detection_policies import (
    FALL_POLICY_V2_DEFAULT,
    BedExitPolicyV1,
    EffectivePolicy,
    FallPolicyV2,
)
from worker.domains.base import AuditContext, DomainAuditSnapshot
from worker.domains.bed_exit import (
    BedExitConfig,
    BedExitDebugSnapshot,
    BedExitMonitor,
    BedExitScoringRecorder,
)
from worker.domains.bed_exit.geometry import containment_ratio
from worker.domains.detection_window import DetectionWindow
from worker.domains.fall import FallPolicyDeciderV2, FallV2DomainDecider, FallWindowClassifierV2
from worker.domains.module_compiler import compile_detection_module_registry
from worker.domains.module_definition import (
    RUNTIME_RESOLVED_ARTIFACT_DIGEST,
    RUNTIME_RESOLVED_PREPROCESSING_IDENTITY,
    CameraModuleContext,
    ComponentBinding,
    DetectionModuleDefinition,
    PolicySchemaIdentity,
    ScheduleRule,
)
from worker.domains.tracker import GreedyIouTracker
from worker.interfaces.decision import Decider
from worker.interfaces.fall_model import FallV2ModelProtocol
from worker.pipeline.analytics.merge import result_merger_names
from worker.types import CURRENT_TEMPORAL_PROFILE

AVAILABLE_OBSERVATION_CHANNELS = frozenset({"person_boxes", "poses", "track_ids", "bed_regions"})
# person/bed identities are the official ultralytics/assets release v8.4.0
# artifacts `yolo26n.pt` and `yolo26m-seg.pt`; re-derive with sha256sum.
# Concrete compiled expectations; runtime provisioning resolves and verifies
# the applied identity before any camera graph can activate.
_COMPONENT_ARTIFACT_DIGESTS = MappingProxyType(
    {
        "pose": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
        "person": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        "bed": "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5",
    }
)
_COMPONENT_PREPROCESSING = MappingProxyType(
    {
        "pose": "rgb24-to-coco17.v1",
        "person": "rgb24-to-person-boxes.v1",
        "bed": "rgb24-to-bed-regions.v1",
    }
)

# Compatibility aliases are explicit data, not composition-root dispatch.
EXTERNAL_DOMAIN_MODULE_IDS: Mapping[str, str] = MappingProxyType(
    {
        "fall": "fall",
        "bed_exit": "bed_exit",
    }
)


@dataclass(frozen=True, slots=True)
class DomainRegistration:
    """Legacy view kept for source compatibility during registry migration."""

    domain: str
    input_view: str
    event_types: frozenset[str]
    factory: Callable[[object], Decider]
    requires: frozenset[str]
    enabled: bool = True
    audit_metadata_provider: Callable[[object], object] | None = None
    debug_snapshot_adapter: Callable[[object, int], object | None] | None = None


@dataclass(frozen=True, slots=True)
class BedExitDomainDependencies:
    config: BedExitConfig
    clock: Callable[[], datetime]
    boot_id: str
    stream_epoch: str
    source_generation: int
    scoring_recorder: BedExitScoringRecorder | None = None


@runtime_checkable
class _ArtifactProvenance(Protocol):
    artifact_digest: str


@runtime_checkable
class _ThresholdReceipt(Protocol):
    receipt_threshold: float | None
    promotion_eligible: bool


@runtime_checkable
class _ConfirmationRuleReceipt(Protocol):
    receipt_transition_votes: int
    receipt_transition_window: int


class _EffectiveFallPolicy(NamedTuple):
    transition_threshold: float
    threshold_source: str
    receipt_threshold: float | None
    unapplied_policy_threshold: float | None
    transition_votes: int
    transition_window: int
    confirmation_rule_source: str
    receipt_transition_votes: int | None
    receipt_transition_window: int | None
    unapplied_transition_votes: int | None
    unapplied_transition_window: int | None


def _extractor(
    component_id: str,
    model_family: str,
    *,
    activation_flag: str | None = None,
) -> ComponentBinding:
    return ComponentBinding(
        component_id=component_id,
        component_kind="extractor",
        model_family=model_family,
        provisioner="serving-client",
        serving_task=component_id,
        artifact_digest=_COMPONENT_ARTIFACT_DIGESTS[component_id],
        preprocessing_identity=_COMPONENT_PREPROCESSING[component_id],
        output_adapter=component_id,
        activation_flag=activation_flag,
        warmup_required=True,
    )


def _camera_component(
    component_id: str,
    factory: Callable[[CameraModuleContext], object],
    kind: Literal["state", "rule"] = "state",
) -> ComponentBinding:
    return ComponentBinding(component_id, kind, camera_factory=factory)


def _fall_classifier_binding() -> ComponentBinding:
    return ComponentBinding(
        component_id="fall-classifier",
        component_kind="model",
        model_family="configured-fall-family",
        provisioner="fall-model-family-registry",
        artifact_digest=RUNTIME_RESOLVED_ARTIFACT_DIGEST,
        preprocessing_identity=RUNTIME_RESOLVED_PREPROCESSING_IDENTITY,
        warmup_required=True,
    )


def _build_bed_exit_compat(dependencies: object) -> BedExitMonitor:
    if not isinstance(dependencies, BedExitDomainDependencies):
        raise TypeError("bed_exit.v1 received invalid dependencies")
    return BedExitMonitor(
        config=dependencies.config,
        clock=dependencies.clock,
        boot_id=dependencies.boot_id,
        stream_epoch=dependencies.stream_epoch,
        source_generation=dependencies.source_generation,
        scoring_recorder=dependencies.scoring_recorder,
    )


def _fall_v2_compat(dependencies: object) -> FallV2DomainDecider:
    if not isinstance(dependencies, Mapping):
        raise TypeError("fall.v2 compatibility dependencies must be a mapping")
    model = dependencies.get("model")
    camera_id = dependencies.get("camera_id")
    facility_id = dependencies.get("facility_id")
    boot_id = dependencies.get("boot_id")
    stream_epoch = dependencies.get("stream_epoch")
    source_generation = dependencies.get("source_generation")
    if (
        not isinstance(model, FallV2ModelProtocol)
        or not isinstance(camera_id, str)
        or not isinstance(facility_id, str)
        or not isinstance(boot_id, str)
        or not isinstance(stream_epoch, str)
        or not isinstance(source_generation, int)
    ):
        raise TypeError("fall.v2 compatibility dependencies are invalid")
    return FallV2DomainDecider(
        classifier=FallWindowClassifierV2(model),
        policy=FallPolicyDeciderV2(
            camera_id=camera_id,
            facility_id=facility_id,
            boot_id=boot_id,
            stream_epoch=stream_epoch,
            source_generation=source_generation,
            policy=FallPolicyV2(),
        ),
    )


def _audit_snapshot_compat(context: object) -> DomainAuditSnapshot:
    if not isinstance(context, AuditContext):
        raise TypeError("audit adapter requires AuditContext")
    return DomainAuditSnapshot(context.model_version, context.operating_threshold)


def _shared_fall_model(context: CameraModuleContext) -> FallV2ModelProtocol:
    model = context.shared_components.get("fall-classifier")
    if not isinstance(model, FallV2ModelProtocol):
        raise TypeError("fall.v2 requires a FallV2ModelProtocol fall-classifier binding")
    return model


def _person_tracker(_context: CameraModuleContext) -> GreedyIouTracker:
    return GreedyIouTracker()


def _bed_exit_policy(context: CameraModuleContext) -> BedExitPolicyV1:
    policy = None if context.policy is None else context.policy.values
    if not isinstance(policy, BedExitPolicyV1):
        raise TypeError("bed_exit.v1 requires a typed bed_exit.policy.v1 effective policy")
    return policy


def _fall_v2(context: CameraModuleContext) -> FallV2DomainDecider:
    resolved = None if context.policy is None else context.policy.values
    if not isinstance(resolved, FallPolicyV2):
        raise TypeError("fall.v2 requires a typed fall.policy.v2 effective policy")
    model = _shared_fall_model(context)
    effective = _effective_transition_threshold(model, context.policy)
    identity = context.camera_components.get("episode-identity")
    if not isinstance(identity, tuple) or len(identity) != 3:
        raise TypeError("fall.v2 requires runtime boot and source identities")
    boot_id, stream_epoch, source_generation = identity
    if (
        not isinstance(boot_id, str)
        or not isinstance(stream_epoch, str)
        or not isinstance(source_generation, int)
    ):
        raise TypeError("fall.v2 received invalid runtime source identities")
    return FallV2DomainDecider(
        classifier=FallWindowClassifierV2(model),
        policy=FallPolicyDeciderV2(
            camera_id=context.camera_id,
            facility_id=context.facility_id,
            boot_id=boot_id,
            stream_epoch=stream_epoch,
            source_generation=source_generation,
            policy=replace(
                resolved,
                transition_threshold=effective.transition_threshold,
                transition_votes=effective.transition_votes,
                transition_window=effective.transition_window,
            ),
        ),
    )


def _effective_transition_threshold(
    model: object, effective_policy: EffectivePolicy
) -> _EffectiveFallPolicy:
    """Resolve receipt operating parameters under one precedence rule."""
    policy = effective_policy.values
    if not isinstance(policy, FallPolicyV2):
        raise TypeError("fall.v2 requires a typed fall.policy.v2 effective policy")
    receipt_threshold = model.receipt_threshold if isinstance(model, _ThresholdReceipt) else None
    receipt_votes = (
        model.receipt_transition_votes if isinstance(model, _ConfirmationRuleReceipt) else None
    )
    receipt_window = (
        model.receipt_transition_window if isinstance(model, _ConfirmationRuleReceipt) else None
    )
    promotion_eligible = model.promotion_eligible if isinstance(model, _ThresholdReceipt) else False
    if promotion_eligible and receipt_threshold is not None:
        threshold = receipt_threshold
        threshold_source = "receipt"
        unapplied_threshold = None
    elif effective_policy.source == "image-default":
        threshold = policy.transition_threshold
        threshold_source = "default"
        unapplied_threshold = None
    else:
        threshold = FALL_POLICY_V2_DEFAULT.transition_threshold
        threshold_source = "default"
        unapplied_threshold = policy.transition_threshold
    if promotion_eligible and receipt_votes is not None and receipt_window is not None:
        transition_votes = receipt_votes
        transition_window = receipt_window
        confirmation_source = "receipt"
        unapplied_votes = None
        unapplied_window = None
    else:
        transition_votes = FALL_POLICY_V2_DEFAULT.transition_votes
        transition_window = FALL_POLICY_V2_DEFAULT.transition_window
        confirmation_source = "default"
        unapplied_votes = receipt_votes
        unapplied_window = receipt_window
    return _EffectiveFallPolicy(
        threshold,
        threshold_source,
        receipt_threshold,
        unapplied_threshold,
        transition_votes,
        transition_window,
        confirmation_source,
        receipt_votes,
        receipt_window,
        unapplied_votes,
        unapplied_window,
    )


def _fall_state(context: CameraModuleContext) -> object:
    return context.camera_components["fall-v2"]


def _identity_decider(state: object) -> Decider:
    if not isinstance(state, Decider):
        raise TypeError("camera state factory did not produce a Decider")
    return state


def _bed_exit_monitor(context: CameraModuleContext) -> BedExitMonitor:
    policy = _bed_exit_policy(context)
    identity = context.camera_components.get("episode-identity")
    if not isinstance(identity, tuple) or len(identity) != 3:
        raise TypeError("bed_exit.v1 requires runtime boot and source identities")
    boot_id, stream_epoch, source_generation = identity
    if (
        not isinstance(boot_id, str)
        or not isinstance(stream_epoch, str)
        or not isinstance(source_generation, int)
    ):
        raise TypeError("bed_exit.v1 received invalid runtime source identities")
    return BedExitMonitor(
        config=BedExitConfig(
            camera_id=context.camera_id,
            facility_id=context.facility_id,
            min_containment=policy.min_containment,
            hold_frames=policy.hold_frames,
            grace_frames=policy.grace_frames,
            night_window=cast("DetectionWindow | None", context.detection_window),
        ),
        clock=context.clock,
        boot_id=boot_id,
        stream_epoch=stream_epoch,
        source_generation=source_generation,
        scoring_recorder=cast("BedExitScoringRecorder | None", context.diagnostics),
    )


def _bed_exit_state(context: CameraModuleContext) -> object:
    return context.camera_components["bed-exit-state"]


def _audit_snapshot(context: CameraModuleContext) -> DomainAuditSnapshot:
    model = context.shared_components.get("fall-classifier")
    if model is None:
        return DomainAuditSnapshot(model_version=None, operating_threshold=None)
    model_version = model.artifact_digest if isinstance(model, _ArtifactProvenance) else None
    effective_policy = context.policy
    if effective_policy is None or not isinstance(effective_policy.values, FallPolicyV2):
        raise TypeError("fall.v2 requires a typed fall.policy.v2 effective policy")
    effective = _effective_transition_threshold(model, effective_policy)
    return DomainAuditSnapshot(
        model_version=model_version,
        operating_threshold=effective.transition_threshold,
        threshold_source=effective.threshold_source,
        receipt_threshold=effective.receipt_threshold,
        unapplied_policy_threshold=effective.unapplied_policy_threshold,
        transition_votes=effective.transition_votes,
        transition_window=effective.transition_window,
        confirmation_rule_source=effective.confirmation_rule_source,
        receipt_transition_votes=effective.receipt_transition_votes,
        receipt_transition_window=effective.receipt_transition_window,
        unapplied_transition_votes=effective.unapplied_transition_votes,
        unapplied_transition_window=effective.unapplied_transition_window,
    )


def _bed_exit_audit(_context: CameraModuleContext) -> DomainAuditSnapshot:
    return DomainAuditSnapshot(model_version=None, operating_threshold=None)


def _trace_adapter(decider: object) -> object:
    snapshots = getattr(decider, "last_trace_snapshots", None)
    return {} if snapshots is None else snapshots


def _no_debug_snapshot(_detector: object, _frame_index: int) -> None:
    return None


def _bed_exit_debug_snapshot(
    detector: object,
    frame_index: int,
) -> BedExitDebugSnapshot | None:
    if not isinstance(detector, BedExitMonitor):
        raise TypeError("bed_exit.v1 debug adapter requires BedExitMonitor")
    snapshot = detector.last_debug_snapshot
    if snapshot is None:
        return None
    return replace(snapshot, frame_index=frame_index)


_FALL_V2 = DetectionModuleDefinition(
    module_id="fall",
    version=2,
    required_observation_channels=frozenset({"person_boxes", "poses", "track_ids"}),
    component_bindings=(
        _extractor("pose", "yolo-pose"),
        _camera_component("person-tracker", _person_tracker),
        _fall_classifier_binding(),
        _camera_component("fall-v2", _fall_v2),
    ),
    schedule_rules=(ScheduleRule("pose", "camera-frame-stride"),),
    policy_schema=PolicySchemaIdentity("fall.policy", 2),
    camera_state_factory=_fall_state,
    decider_factory=_identity_decider,
    audit_adapter=_audit_snapshot,
    trace_adapter=_trace_adapter,
    debug_adapter=_no_debug_snapshot,
    event_types=frozenset({"fall"}),
    input_view="fall_window",
    window_mode="external",
    requires=frozenset({"pose"}),
    compatibility_factory=_fall_v2_compat,
)

_BED_EXIT_V1 = DetectionModuleDefinition(
    module_id="bed_exit",
    version=1,
    required_observation_channels=frozenset({"person_boxes", "poses", "track_ids", "bed_regions"}),
    component_bindings=(
        _extractor("pose", "yolo-pose"),
        _extractor("person", "yolo-person", activation_flag="person-box-source"),
        _extractor("bed", "yolo-bed-segmentation"),
        _camera_component("person-tracker", _person_tracker),
        _camera_component("containment", lambda _context: containment_ratio, "rule"),
        _camera_component("bed-assignment", lambda _context: {}),
        _camera_component("bed-exit-state", _bed_exit_monitor),
    ),
    schedule_rules=(
        ScheduleRule("pose", "camera-frame-stride"),
        ScheduleRule("person", "camera-frame-stride"),
        # The persisted polygon is the only runtime bed truth, so bed
        # segmentation is never scheduled per frame: the extractor stays
        # provisioned only for the on-demand recognize route, whose result an
        # operator persists explicitly.
        ScheduleRule("bed", "on-demand"),
    ),
    policy_schema=PolicySchemaIdentity("bed_exit.policy", 1),
    camera_state_factory=_bed_exit_state,
    decider_factory=_identity_decider,
    audit_adapter=_bed_exit_audit,
    trace_adapter=_trace_adapter,
    debug_adapter=_bed_exit_debug_snapshot,
    event_types=frozenset({"bed-exit"}),
    input_view="bed_regions",
    window_mode="internal",
    requires=frozenset({"pose", "bed"}),
    compatibility_factory=_build_bed_exit_compat,
    compatibility_audit_adapter=_audit_snapshot_compat,
)

DETECTION_MODULE_DEFINITIONS = (_FALL_V2, _BED_EXIT_V1)
DETECTION_MODULE_REGISTRY = compile_detection_module_registry(
    DETECTION_MODULE_DEFINITIONS,
    available_observation_channels=AVAILABLE_OBSERVATION_CHANNELS,
    output_adapter_ids=result_merger_names(),
    temporal_profile=CURRENT_TEMPORAL_PROFILE,
)
# Temporary source-compatible view while external callers move to the compiled registry.
DOMAIN_REGISTRY: Mapping[str, DomainRegistration] = MappingProxyType(
    {
        definition.module_id: DomainRegistration(
            domain=definition.module_id,
            input_view=definition.input_view,
            event_types=definition.event_types,
            factory=definition.factory,
            requires=definition.requires,
            enabled=definition.enabled,
            audit_metadata_provider=definition.audit_metadata_provider,
            debug_snapshot_adapter=definition.debug_snapshot_adapter,
        )
        for definition in DETECTION_MODULE_REGISTRY.definitions
    }
)


def list_domains(*, enabled: bool = True) -> tuple[str, ...]:
    return tuple(
        name for name, registration in DOMAIN_REGISTRY.items() if registration.enabled is enabled
    )


def enabled_domains() -> tuple[str, ...]:
    return list_domains(enabled=True)


__all__ = [
    "AVAILABLE_OBSERVATION_CHANNELS",
    "DETECTION_MODULE_DEFINITIONS",
    "DETECTION_MODULE_REGISTRY",
    "DOMAIN_REGISTRY",
    "EXTERNAL_DOMAIN_MODULE_IDS",
    "BedExitDomainDependencies",
    "DomainRegistration",
    "enabled_domains",
    "list_domains",
]
