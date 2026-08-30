"""Typed, versioned numeric policies shared by the edge API and worker.

The wire parser is deliberately closed: qualified module identity, policy schema,
field set, numeric type/range, cross-field constraints, and content identity are
parsed together. Unknown or drifted documents never degrade into defaults.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

PolicySource: TypeAlias = Literal["image-default", "facility-default", "camera-override"]


class PolicyDocumentError(ValueError):
    """A policy boundary could not be parsed into a supported typed policy."""


@dataclass(frozen=True, slots=True)
class FallPolicyV1:
    operating_threshold: float


@dataclass(frozen=True, slots=True)
class FallPolicyV2:
    """Frozen inactive fall-candidate temporal policy, not a policy document."""

    transition_threshold: float = 0.7
    transition_votes: int = 3
    transition_window: int = 5
    fallen_threshold: float = 0.8
    fallen_consecutive: int = 3
    recovery_transition_max: float = 0.4
    recovery_fallen_max: float = 0.5
    recovery_consecutive: int = 5
    track_ttl_frames: int = 45
    cooldown_frames: int = 90

    def __post_init__(self) -> None:
        for name in (
            "transition_threshold",
            "fallen_threshold",
            "recovery_transition_max",
            "recovery_fallen_max",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability in [0, 1]")
        for name in (
            "transition_votes",
            "transition_window",
            "fallen_consecutive",
            "recovery_consecutive",
            "track_ttl_frames",
            "cooldown_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.transition_votes > self.transition_window:
            raise ValueError("transition_votes must not exceed transition_window")


@dataclass(frozen=True, slots=True)
class BedExitPolicyV1:
    min_containment: float
    hold_frames: int
    grace_frames: int


NumericPolicy: TypeAlias = FallPolicyV1 | BedExitPolicyV1

FALL_POLICY_V1_DEFAULT: Final = FallPolicyV1(operating_threshold=0.5)
BED_EXIT_POLICY_V1_DEFAULT: Final = BedExitPolicyV1(
    min_containment=0.35,
    hold_frames=2,
    grace_frames=3,
)


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    module_id: str
    module_version: int
    schema_id: str
    schema_version: int
    units: Mapping[str, str]
    image_default: NumericPolicy

    @property
    def qualified_module_id(self) -> str:
        return f"{self.module_id}.v{self.module_version}"

    @property
    def qualified_schema_id(self) -> str:
        return f"{self.schema_id}.v{self.schema_version}"


_POLICY_DEFINITIONS: Final = (
    PolicyDefinition(
        "fall",
        1,
        "fall.policy",
        1,
        MappingProxyType({"operating_threshold": "probability [0,1]"}),
        FALL_POLICY_V1_DEFAULT,
    ),
    PolicyDefinition(
        "bed_exit",
        1,
        "bed_exit.policy",
        1,
        MappingProxyType(
            {
                "min_containment": "ratio (0,1]",
                "hold_frames": "frames",
                "grace_frames": "frames",
            }
        ),
        BED_EXIT_POLICY_V1_DEFAULT,
    ),
)
POLICY_DEFINITIONS: Final[Mapping[tuple[str, int], PolicyDefinition]] = MappingProxyType(
    {
        (definition.module_id, definition.module_version): definition
        for definition in _POLICY_DEFINITIONS
    }
)
LATEST_POLICY_VERSIONS: Final[Mapping[str, int]] = MappingProxyType(
    {definition.module_id: definition.module_version for definition in _POLICY_DEFINITIONS}
)


def policy_definition(module_id: str, module_version: int) -> PolicyDefinition:
    try:
        return POLICY_DEFINITIONS[(module_id, module_version)]
    except KeyError as exc:
        known_version = LATEST_POLICY_VERSIONS.get(module_id)
        if known_version is None:
            raise PolicyDocumentError(f"unknown policy module {module_id!r}") from exc
        raise PolicyDocumentError(
            f"unsupported policy module version for {module_id!r}: "
            f"received {module_version}, supported {known_version}"
        ) from exc


def parse_policy_values(
    *,
    module_id: str,
    module_version: int,
    schema_id: str,
    schema_version: int,
    values: object,
) -> NumericPolicy:
    definition = policy_definition(module_id, module_version)
    if (schema_id, schema_version) != (
        definition.schema_id,
        definition.schema_version,
    ):
        raise PolicyDocumentError(
            f"policy schema drift for {definition.qualified_module_id}: "
            f"received {schema_id}.v{schema_version}, "
            f"supported {definition.qualified_schema_id}"
        )
    mapping = _mapping(values, "policy values")
    if module_id == "fall":
        _require_fields(mapping, {"operating_threshold"}, "fall policy")
        threshold = _finite_number(mapping["operating_threshold"], "operating_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise PolicyDocumentError("operating_threshold must be in [0, 1]")
        return FallPolicyV1(threshold)
    if module_id == "bed_exit":
        _require_fields(
            mapping,
            {"min_containment", "hold_frames", "grace_frames"},
            "bed-exit policy",
        )
        containment = _finite_number(mapping["min_containment"], "min_containment")
        hold_frames = _integer(mapping["hold_frames"], "hold_frames")
        grace_frames = _integer(mapping["grace_frames"], "grace_frames")
        if not 0.0 < containment <= 1.0:
            raise PolicyDocumentError("min_containment must be in (0, 1]")
        if not 1 <= hold_frames <= 300:
            raise PolicyDocumentError("hold_frames must be in [1, 300]")
        if not 0 <= grace_frames <= 300:
            raise PolicyDocumentError("grace_frames must be in [0, 300]")
        if hold_frames + grace_frames > 300:
            raise PolicyDocumentError(
                "combined hold_frames + grace_frames must not exceed 300 frames"
            )
        return BedExitPolicyV1(containment, hold_frames, grace_frames)
    raise PolicyDocumentError(f"unknown policy module {module_id!r}")


def policy_values_dict(policy: NumericPolicy) -> dict[str, int | float]:
    if isinstance(policy, FallPolicyV1):
        return {"operating_threshold": policy.operating_threshold}
    return {
        "min_containment": policy.min_containment,
        "hold_frames": policy.hold_frames,
        "grace_frames": policy.grace_frames,
    }


@dataclass(frozen=True, slots=True)
class EffectivePolicy:
    module_id: str
    module_version: int
    schema_id: str
    schema_version: int
    source: PolicySource
    facility_revision_id: int | None
    camera_revision_id: int | None
    values: NumericPolicy
    effective_policy_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "module_version": self.module_version,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "facility_revision_id": self.facility_revision_id,
            "camera_revision_id": self.camera_revision_id,
            "values": policy_values_dict(self.values),
            "effective_policy_id": self.effective_policy_id,
        }


def make_effective_policy(
    *,
    module_id: str,
    module_version: int,
    values: NumericPolicy,
    source: PolicySource,
    facility_revision_id: int | None,
    camera_revision_id: int | None,
) -> EffectivePolicy:
    definition = policy_definition(module_id, module_version)
    parsed = parse_policy_values(
        module_id=module_id,
        module_version=module_version,
        schema_id=definition.schema_id,
        schema_version=definition.schema_version,
        values=policy_values_dict(values),
    )
    if source == "image-default" and (
        facility_revision_id is not None or camera_revision_id is not None
    ):
        raise PolicyDocumentError("image-default policy cannot carry revision ids")
    if source == "facility-default" and facility_revision_id is None:
        raise PolicyDocumentError("facility-default policy requires a facility revision")
    if source == "camera-override" and camera_revision_id is None:
        raise PolicyDocumentError("camera-override policy requires a camera revision")
    identity_payload = {
        "module_id": module_id,
        "module_version": module_version,
        "schema_id": definition.schema_id,
        "schema_version": definition.schema_version,
        "source": source,
        "facility_revision_id": facility_revision_id,
        "camera_revision_id": camera_revision_id,
        "values": policy_values_dict(parsed),
    }
    identity = hashlib.sha256(_canonical_json(identity_payload).encode()).hexdigest()
    return EffectivePolicy(
        module_id,
        module_version,
        definition.schema_id,
        definition.schema_version,
        source,
        facility_revision_id,
        camera_revision_id,
        parsed,
        identity,
    )


def _policy_source(value: object) -> PolicySource:
    """Narrow an untyped document field to PolicySource, or reject it.

    Returning each literal directly is what actually narrows the type. A set
    membership test does not narrow, and a cast would only assert the narrowing
    instead of establishing it -- mypy flagged that cast as redundant while the
    value stayed a plain str at the call site.
    """
    if value == "image-default":
        return "image-default"
    if value == "facility-default":
        return "facility-default"
    if value == "camera-override":
        return "camera-override"
    raise PolicyDocumentError("effective policy source is unknown")


def parse_effective_policy(
    value: object,
    *,
    expected_module_id: str | None = None,
    expected_module_version: int | None = None,
) -> EffectivePolicy:
    mapping = _mapping(value, "effective policy")
    _require_fields(
        mapping,
        {
            "module_id",
            "module_version",
            "schema_id",
            "schema_version",
            "source",
            "facility_revision_id",
            "camera_revision_id",
            "values",
            "effective_policy_id",
        },
        "effective policy",
    )
    module_id = _text(mapping["module_id"], "module_id")
    module_version = _integer(mapping["module_version"], "module_version")
    if expected_module_id is not None and module_id != expected_module_id:
        raise PolicyDocumentError("effective policy module id does not match its map key")
    if expected_module_version is not None and module_version != expected_module_version:
        raise PolicyDocumentError("effective policy module version does not match selection")
    schema_id = _text(mapping["schema_id"], "schema_id")
    schema_version = _integer(mapping["schema_version"], "schema_version")
    source = _policy_source(mapping["source"])
    facility_revision_id = _optional_revision(
        mapping["facility_revision_id"], "facility_revision_id"
    )
    camera_revision_id = _optional_revision(mapping["camera_revision_id"], "camera_revision_id")
    parsed_values = parse_policy_values(
        module_id=module_id,
        module_version=module_version,
        schema_id=schema_id,
        schema_version=schema_version,
        values=mapping["values"],
    )
    parsed = make_effective_policy(
        module_id=module_id,
        module_version=module_version,
        values=parsed_values,
        source=source,
        facility_revision_id=facility_revision_id,
        camera_revision_id=camera_revision_id,
    )
    identity = _text(mapping["effective_policy_id"], "effective_policy_id")
    if identity != parsed.effective_policy_id:
        raise PolicyDocumentError("effective policy identity does not match its content")
    return parsed


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    schema_version: int
    defaults: Mapping[str, EffectivePolicy]
    cameras: Mapping[str, Mapping[str, EffectivePolicy]]

    def resolve(self, camera_id: str, module_id: str, module_version: int) -> EffectivePolicy:
        camera = self.cameras.get(camera_id)
        selected = None if camera is None else camera.get(module_id)
        if selected is None:
            selected = self.defaults.get(module_id)
        if selected is None:
            raise PolicyDocumentError(f"effective policy missing for module {module_id!r}")
        if selected.module_version != module_version:
            raise PolicyDocumentError(f"effective policy module version drift for {module_id!r}")
        return selected

    def with_camera(self, camera_id: str, policies: Mapping[str, EffectivePolicy]) -> PolicyBundle:
        cameras = {key: MappingProxyType(dict(value)) for key, value in self.cameras.items()}
        cameras[camera_id] = MappingProxyType(dict(policies))
        return PolicyBundle(
            self.schema_version,
            MappingProxyType(dict(self.defaults)),
            MappingProxyType(cameras),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "defaults": {
                module_id: policy.as_dict() for module_id, policy in self.defaults.items()
            },
            "cameras": {
                camera_id: {module_id: policy.as_dict() for module_id, policy in policies.items()}
                for camera_id, policies in self.cameras.items()
            },
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()


def default_policy_bundle(camera_ids: Sequence[str] = ()) -> PolicyBundle:
    defaults = {
        definition.module_id: make_effective_policy(
            module_id=definition.module_id,
            module_version=definition.module_version,
            values=definition.image_default,
            source="image-default",
            facility_revision_id=None,
            camera_revision_id=None,
        )
        for definition in _POLICY_DEFINITIONS
    }
    camera_defaults = {camera_id: MappingProxyType(dict(defaults)) for camera_id in camera_ids}
    return PolicyBundle(
        1,
        MappingProxyType(defaults),
        MappingProxyType(camera_defaults),
    )


def parse_policy_bundle(value: object) -> PolicyBundle:
    mapping = _mapping(value, "detection policy bundle")
    _require_fields(mapping, {"schema_version", "defaults", "cameras"}, "policy bundle")
    if _integer(mapping["schema_version"], "schema_version") != 1:
        raise PolicyDocumentError("unsupported detection policy bundle schema version")
    raw_defaults = _mapping(mapping["defaults"], "policy defaults")
    expected_modules = set(LATEST_POLICY_VERSIONS)
    _require_fields(raw_defaults, expected_modules, "policy defaults")
    defaults = {
        module_id: parse_effective_policy(
            raw_defaults[module_id],
            expected_module_id=module_id,
            expected_module_version=LATEST_POLICY_VERSIONS[module_id],
        )
        for module_id in sorted(expected_modules)
    }
    raw_cameras = _mapping(mapping["cameras"], "camera policies")
    cameras: dict[str, Mapping[str, EffectivePolicy]] = {}
    for raw_camera_id, raw_policies in raw_cameras.items():
        camera_id = _text(raw_camera_id, "camera policy id")
        policies = _mapping(raw_policies, f"camera policies for {camera_id!r}")
        _require_fields(policies, expected_modules, f"camera policies for {camera_id!r}")
        cameras[camera_id] = MappingProxyType(
            {
                module_id: parse_effective_policy(
                    policies[module_id],
                    expected_module_id=module_id,
                    expected_module_version=LATEST_POLICY_VERSIONS[module_id],
                )
                for module_id in sorted(expected_modules)
            }
        )
    return PolicyBundle(1, MappingProxyType(defaults), MappingProxyType(cameras))


def _mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise PolicyDocumentError(f"{label} must be an object")
    return value


def _require_fields(mapping: Mapping[object, object], expected: set[str], label: str) -> None:
    actual = set(mapping)
    unknown = sorted(str(key) for key in actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise PolicyDocumentError(f"{label} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise PolicyDocumentError(f"{label} is missing field(s): {', '.join(missing)}")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyDocumentError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PolicyDocumentError(f"{field_name} must be finite")
    return parsed


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyDocumentError(f"{field_name} must be an integer")
    return value


def _optional_revision(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = _integer(value, field_name)
    if parsed < 1:
        raise PolicyDocumentError(f"{field_name} must be positive or null")
    return parsed


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyDocumentError(f"{field_name} must be a non-empty string")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "BED_EXIT_POLICY_V1_DEFAULT",
    "FALL_POLICY_V1_DEFAULT",
    "LATEST_POLICY_VERSIONS",
    "POLICY_DEFINITIONS",
    "BedExitPolicyV1",
    "EffectivePolicy",
    "FallPolicyV1",
    "NumericPolicy",
    "PolicyBundle",
    "PolicyDefinition",
    "PolicyDocumentError",
    "PolicySource",
    "default_policy_bundle",
    "make_effective_policy",
    "parse_effective_policy",
    "parse_policy_bundle",
    "parse_policy_values",
    "policy_definition",
    "policy_values_dict",
]
