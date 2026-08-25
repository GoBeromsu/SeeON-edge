from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from types import MappingProxyType

from shared.detection_policies import EffectivePolicy, parse_effective_policy, policy_values_dict
from worker.domains.module_compiler import CompiledDetectionModuleRegistry
from worker.domains.module_definition import (
    ComponentBinding,
    DetectionModuleDefinition,
    SharedComponentIdentity,
)
from worker.runtime.profile.boot import BootContext
from worker.runtime.provenance.content import (
    component_content,
    module_content,
    profile_content,
)
from worker.runtime.provenance.models import (
    AppliedBedZone,
    AppliedCameraState,
    AppliedDetectionWindow,
    AppliedRuntimeManifest,
    AppliedRuntimeManifestError,
    JsonValue,
    RuntimeEnvironmentFacts,
    require_sha256,
)

MANIFEST_SCHEMA_VERSION = 1
BED_ZONE_COORDINATE_SCHEMA_VERSION = 1
BED_ZONE_COORDINATE_SPACE = "source-image-pixels"
SCHEDULE_INTERVAL_BASIS = "ingested-frame-index"
_EFFECTIVE_DECODE_BACKENDS = frozenset({"cpu", "opencv", "nvdec", "vaapi"})


def build_applied_camera_state(
    *,
    camera_id: str,
    effective_decode_backend: str,
    ingest_target_fps: float,
    module_qualified_ids: Sequence[str],
    schedule: Mapping[str, int],
    detection_windows: Mapping[str, AppliedDetectionWindow | None],
    policies: Mapping[str, EffectivePolicy],
    bed_zone_polygon: Sequence[tuple[int, int]] | None,
    bed_zone_image_width: int | None,
    bed_zone_image_height: int | None,
) -> AppliedCameraState:
    """Freeze the effective camera-local state consumed by the runtime plan."""
    if bed_zone_polygon is None:
        bed_zone = AppliedBedZone(
            authority="live-segmentation",
            coordinate_schema_version=BED_ZONE_COORDINATE_SCHEMA_VERSION,
            coordinate_space=None,
            polygon=None,
            source_width=None,
            source_height=None,
        )
    else:
        if bed_zone_image_width is None or bed_zone_image_height is None:
            raise AppliedRuntimeManifestError(
                "persisted bed-zone provenance requires source dimensions"
            )
        bed_zone = AppliedBedZone(
            authority="persisted-polygon",
            coordinate_schema_version=BED_ZONE_COORDINATE_SCHEMA_VERSION,
            coordinate_space=BED_ZONE_COORDINATE_SPACE,
            polygon=tuple(bed_zone_polygon),
            source_width=bed_zone_image_width,
            source_height=bed_zone_image_height,
        )
    return AppliedCameraState(
        camera_id=camera_id,
        effective_decode_backend=effective_decode_backend,
        ingest_target_fps=ingest_target_fps,
        module_qualified_ids=tuple(module_qualified_ids),
        schedule=MappingProxyType(dict(schedule)),
        detection_windows=MappingProxyType(dict(detection_windows)),
        policies=MappingProxyType(dict(policies)),
        bed_zone=bed_zone,
    )


def build_applied_runtime_manifest(
    *,
    boot: BootContext,
    module_registry: CompiledDetectionModuleRegistry,
    module_versions: Mapping[str, int],
    component_identities: Sequence[SharedComponentIdentity],
    cameras: Sequence[AppliedCameraState],
    config_version: int,
    restart_generation: int,
    detector_version: str,
    environment: RuntimeEnvironmentFacts,
    edge_database_schema_version: int,
) -> AppliedRuntimeManifest:
    """Freeze only independently resolved state after profile, model, and policy gates."""
    if config_version < 0 or restart_generation < 0:
        raise AppliedRuntimeManifestError("config and restart generations must be non-negative")
    if edge_database_schema_version < 1 or not detector_version:
        raise AppliedRuntimeManifestError("detector or database schema identity is unresolved")
    definitions = tuple(
        sorted(
            module_registry.selected(module_versions),
            key=lambda definition: definition.qualified_id,
        )
    )
    if not definitions:
        raise AppliedRuntimeManifestError("no applied detection module identity resolved")
    identities = _verified_component_identities(definitions, component_identities, boot)
    selected_qualified_ids = tuple(definition.qualified_id for definition in definitions)
    profile = boot.runtime_profile
    facts = environment.validated(nvidia=profile.canonical_profile == "nvidia")
    content: dict[str, JsonValue] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "build": {
            "detector_version": detector_version,
            "edge_database_schema_version": edge_database_schema_version,
            **facts.as_dict(),
        },
        "configuration": {
            "config_version": config_version,
            "restart_generation": restart_generation,
        },
        "profile": profile_content(boot),
        "modules": [module_content(definition) for definition in definitions],
        "components": [component_content(identity) for identity in identities],
        "cameras": [
            _camera_content(camera, selected_qualified_ids)
            for camera in sorted(cameras, key=lambda camera: camera.camera_id)
        ],
    }
    return AppliedRuntimeManifest.from_content(content)


def _verified_component_identities(
    definitions: Sequence[DetectionModuleDefinition],
    identities: Sequence[SharedComponentIdentity],
    boot: BootContext,
) -> tuple[SharedComponentIdentity, ...]:
    by_id: dict[str, SharedComponentIdentity] = {}
    for identity in identities:
        if identity.component_id in by_id:
            raise AppliedRuntimeManifestError(
                f"duplicate applied identity for component {identity.component_id!r}"
            )
        require_sha256(identity.artifact_digest, f"component {identity.component_id!r} artifact")
        if not all((identity.runtime, identity.device, identity.preprocessing_identity)):
            raise AppliedRuntimeManifestError(
                f"unresolved applied identity for component {identity.component_id!r}"
            )
        cpu_policy = (
            boot.profile.name == "nvidia"
            and identity.device == "cpu"
            and identity.runtime == "cpu-policy"
        )
        if identity.device != boot.device and not cpu_policy:
            raise AppliedRuntimeManifestError(
                f"contradictory device identity for component {identity.component_id!r}"
            )
        by_id[identity.component_id] = identity

    bindings: dict[str, ComponentBinding] = {}
    required: set[str] = set()
    for definition in definitions:
        for binding in definition.shared_bindings:
            previous = bindings.setdefault(binding.component_id, binding)
            if previous != binding:
                raise AppliedRuntimeManifestError(
                    f"contradictory binding for component {binding.component_id!r}"
                )
            if binding.activation_flag is None:
                required.add(binding.component_id)
    missing = sorted(required - set(by_id))
    if missing:
        raise AppliedRuntimeManifestError(
            "unresolved applied identity for component(s): " + ", ".join(missing)
        )
    unknown = sorted(set(by_id) - set(bindings))
    if unknown:
        raise AppliedRuntimeManifestError(
            "applied identity has no selected component binding: " + ", ".join(unknown)
        )
    for component_id, identity in by_id.items():
        binding = bindings[component_id]
        if binding.artifact_digest != identity.artifact_digest:
            raise AppliedRuntimeManifestError(
                f"contradictory artifact identity for component {component_id!r}"
            )
        if binding.preprocessing_identity != identity.preprocessing_identity:
            raise AppliedRuntimeManifestError(
                f"contradictory preprocessing identity for component {component_id!r}"
            )
    return tuple(by_id[key] for key in sorted(by_id))


def _camera_content(
    camera: AppliedCameraState,
    selected_qualified_ids: tuple[str, ...],
) -> dict[str, JsonValue]:
    if not camera.camera_id.strip():
        raise AppliedRuntimeManifestError("camera opaque identity is unresolved")
    qualified_ids = tuple(sorted(camera.module_qualified_ids))
    if qualified_ids != selected_qualified_ids:
        raise AppliedRuntimeManifestError(
            f"camera {camera.camera_id!r} module identity contradicts process selection"
        )
    if camera.effective_decode_backend not in _EFFECTIVE_DECODE_BACKENDS:
        raise AppliedRuntimeManifestError(
            f"camera {camera.camera_id!r} has an invalid effective decode backend"
        )
    if (
        isinstance(camera.ingest_target_fps, bool)
        or not isinstance(camera.ingest_target_fps, (int, float))
        or not isfinite(camera.ingest_target_fps)
        or camera.ingest_target_fps <= 0
    ):
        raise AppliedRuntimeManifestError(
            f"camera {camera.camera_id!r} has an invalid ingest timing basis"
        )
    if any(
        isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0
        for interval in camera.schedule.values()
    ):
        raise AppliedRuntimeManifestError(
            f"camera {camera.camera_id!r} has an invalid applied schedule"
        )
    policies: dict[str, JsonValue] = {}
    for module_id, policy in sorted(camera.policies.items()):
        verified = parse_effective_policy(
            policy.as_dict(),
            expected_module_id=module_id,
            expected_module_version=policy.module_version,
        )
        policies[module_id] = {
            "module_version": verified.module_version,
            "policy_schema": f"{verified.schema_id}.v{verified.schema_version}",
            "source": verified.source,
            "facility_revision_id": verified.facility_revision_id,
            "camera_revision_id": verified.camera_revision_id,
            "values": _numeric_policy_content(policy_values_dict(verified.values)),
            "effective_policy_id": verified.effective_policy_id,
        }
    selected_module_ids = {qualified.rsplit(".v", 1)[0] for qualified in qualified_ids}
    if set(policies) != selected_module_ids:
        raise AppliedRuntimeManifestError(
            f"camera {camera.camera_id!r} effective policy set is incomplete"
        )
    if set(camera.detection_windows) != selected_module_ids:
        raise AppliedRuntimeManifestError(
            f"camera {camera.camera_id!r} applied detection-window set is incomplete"
        )
    modules: list[JsonValue] = list(qualified_ids)
    schedule: dict[str, JsonValue] = {key: camera.schedule[key] for key in sorted(camera.schedule)}
    windows: dict[str, JsonValue] = {
        module_id: _window_content(camera.detection_windows[module_id])
        for module_id in sorted(camera.detection_windows)
    }
    return {
        "camera_id": camera.camera_id,
        "effective_decode_backend": camera.effective_decode_backend,
        "timing": {
            "ingest_target_fps": camera.ingest_target_fps,
            "schedule_interval_basis": SCHEDULE_INTERVAL_BASIS,
        },
        "modules": modules,
        "schedule": schedule,
        "detection_windows": windows,
        "bed_zone": _bed_zone_content(camera.bed_zone, camera.camera_id),
        "policies": policies,
    }


def _window_content(window: AppliedDetectionWindow | None) -> JsonValue:
    if window is None:
        return None
    return {
        "start": window.start,
        "end": window.end,
        "timezone": window.timezone,
    }


def _bed_zone_content(bed_zone: AppliedBedZone, camera_id: str) -> dict[str, JsonValue]:
    if bed_zone.coordinate_schema_version < 1:
        raise AppliedRuntimeManifestError(
            f"camera {camera_id!r} has an invalid bed-zone coordinate schema version"
        )
    if bed_zone.authority == "live-segmentation":
        if any(
            value is not None
            for value in (
                bed_zone.coordinate_space,
                bed_zone.polygon,
                bed_zone.source_width,
                bed_zone.source_height,
            )
        ):
            raise AppliedRuntimeManifestError(
                f"camera {camera_id!r} has contradictory absent bed-zone semantics"
            )
        dimensions: JsonValue = None
        polygon_content: JsonValue = None
    else:
        polygon = bed_zone.polygon
        open_polygon = () if polygon is None else _without_closing_vertex(polygon)
        if (
            bed_zone.coordinate_space != BED_ZONE_COORDINATE_SPACE
            or len(open_polygon) < 3
            or bed_zone.source_width is None
            or bed_zone.source_height is None
            or bed_zone.source_width <= 0
            or bed_zone.source_height <= 0
            or any(
                isinstance(coordinate, bool) or not isinstance(coordinate, int)
                for point in open_polygon
                for coordinate in point
            )
        ):
            raise AppliedRuntimeManifestError(
                f"camera {camera_id!r} has invalid persisted bed-zone semantics"
            )
        dimensions = {
            "width": bed_zone.source_width,
            "height": bed_zone.source_height,
        }
        polygon_content = [[x, y] for x, y in _canonical_polygon(open_polygon)]
    return {
        "authority": bed_zone.authority,
        "coordinate_schema_version": bed_zone.coordinate_schema_version,
        "coordinate_space": bed_zone.coordinate_space,
        "polygon": polygon_content,
        "source_dimensions": dimensions,
    }


def _without_closing_vertex(
    polygon: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    while len(polygon) > 1 and polygon[-1] == polygon[0]:
        polygon = polygon[:-1]
    return polygon


def _canonical_polygon(
    polygon: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Normalize only cyclic start/winding; edge-changing permutations remain distinct."""
    rotations = tuple(polygon[index:] + polygon[:index] for index in range(len(polygon)))
    reversed_polygon = tuple(reversed(polygon))
    reversed_rotations = tuple(
        reversed_polygon[index:] + reversed_polygon[:index]
        for index in range(len(reversed_polygon))
    )
    return min((*rotations, *reversed_rotations))


def _numeric_policy_content(values: Mapping[str, int | float]) -> dict[str, JsonValue]:
    return {key: value for key, value in values.items()}


__all__ = [
    "BED_ZONE_COORDINATE_SCHEMA_VERSION",
    "BED_ZONE_COORDINATE_SPACE",
    "MANIFEST_SCHEMA_VERSION",
    "SCHEDULE_INTERVAL_BASIS",
    "AppliedCameraState",
    "AppliedDetectionWindow",
    "AppliedRuntimeManifestError",
    "RuntimeEnvironmentFacts",
    "build_applied_camera_state",
    "build_applied_runtime_manifest",
]
