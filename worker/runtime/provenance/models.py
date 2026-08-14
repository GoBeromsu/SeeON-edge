from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from shared.detection_policies import EffectivePolicy

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class AppliedRuntimeManifestError(RuntimeError):
    """Applied runtime identity is unresolved, contradictory, or unsafe."""


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentFacts:
    worker_build_revision: str
    os_name: str
    architecture: str
    python_version: str
    model_runtime: str
    model_runtime_version: str
    accelerator_runtime: str | None = None
    driver_version: str | None = None
    device_name: str | None = None

    def validated(self, *, nvidia: bool) -> RuntimeEnvironmentFacts:
        if (
            not _SOURCE_REVISION.fullmatch(self.worker_build_revision)
            or self.worker_build_revision == "0" * 40
        ):
            raise AppliedRuntimeManifestError(
                "worker build identity must be an applied 40-character source revision"
            )
        required = (
            self.os_name,
            self.architecture,
            self.python_version,
            self.model_runtime,
            self.model_runtime_version,
        )
        if any(not value.strip() for value in required):
            raise AppliedRuntimeManifestError("runtime environment identity is unresolved")
        if nvidia and not all(
            value is not None and value.strip()
            for value in (self.accelerator_runtime, self.driver_version, self.device_name)
        ):
            raise AppliedRuntimeManifestError(
                "NVIDIA applied identity requires accelerator runtime, driver, and device facts"
            )
        _reject_unsafe_values(self.as_dict())
        return self

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "worker_build_revision": self.worker_build_revision,
            "os_name": self.os_name,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "model_runtime": self.model_runtime,
            "model_runtime_version": self.model_runtime_version,
            "accelerator_runtime": self.accelerator_runtime,
            "driver_version": self.driver_version,
            "device_name": self.device_name,
        }


@dataclass(frozen=True, slots=True)
class AppliedDetectionWindow:
    start: str
    end: str
    timezone: str

    def __post_init__(self) -> None:
        for value in (self.start, self.end):
            try:
                parsed = datetime.strptime(value, "%H:%M")
            except ValueError as error:
                raise AppliedRuntimeManifestError(
                    "applied detection-window time must use HH:MM"
                ) from error
            if parsed.strftime("%H:%M") != value:
                raise AppliedRuntimeManifestError("applied detection-window time must use HH:MM")
        if self.start == self.end:
            raise AppliedRuntimeManifestError("applied detection window must not be empty")
        try:
            _ = ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise AppliedRuntimeManifestError(
                "applied detection-window timezone must be an IANA name"
            ) from error


@dataclass(frozen=True, slots=True)
class AppliedBedZone:
    authority: Literal["persisted-polygon", "live-segmentation"]
    coordinate_schema_version: int
    coordinate_space: Literal["source-image-pixels"] | None
    polygon: tuple[tuple[int, int], ...] | None
    source_width: int | None
    source_height: int | None


@dataclass(frozen=True, slots=True)
class AppliedCameraState:
    camera_id: str
    effective_decode_backend: str
    ingest_target_fps: float
    module_qualified_ids: tuple[str, ...]
    schedule: Mapping[str, int]
    detection_windows: Mapping[str, AppliedDetectionWindow | None]
    policies: Mapping[str, EffectivePolicy]
    bed_zone: AppliedBedZone


@dataclass(frozen=True, slots=True)
class AppliedRuntimeManifest:
    schema_version: int
    canonical_json: str
    sha256: str

    @classmethod
    def from_content(cls, content: Mapping[str, JsonValue]) -> AppliedRuntimeManifest:
        canonical = canonical_json(content)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        version = content.get("manifest_schema_version")
        if not isinstance(version, int):
            raise AppliedRuntimeManifestError("manifest schema version is unresolved")
        return cls(version, canonical, digest)

    @classmethod
    def parse(cls, canonical: str, expected_sha256: str) -> AppliedRuntimeManifest:
        if not _SHA256.fullmatch(expected_sha256):
            raise AppliedRuntimeManifestError("stored runtime manifest hash is invalid")
        try:
            value = json.loads(canonical)
        except json.JSONDecodeError as error:
            raise AppliedRuntimeManifestError("stored runtime manifest is invalid JSON") from error
        if not isinstance(value, dict) or canonical_json(value) != canonical:
            raise AppliedRuntimeManifestError("stored runtime manifest is not canonical")
        parsed = cls.from_content(value)
        if parsed.sha256 != expected_sha256:
            raise AppliedRuntimeManifestError("stored runtime manifest hash is contradictory")
        return parsed


def canonical_json(value: object) -> str:
    _reject_unsafe_values(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def require_sha256(value: str, label: str) -> str:
    if not _SHA256.fullmatch(value):
        raise AppliedRuntimeManifestError(f"{label} must be a lowercase SHA-256 identity")
    return value


def _reject_unsafe_values(value: object) -> None:
    if isinstance(value, Mapping):
        for key, member in value.items():
            if not isinstance(key, str):
                raise AppliedRuntimeManifestError("manifest object keys must be strings")
            _reject_unsafe_values(member)
        return
    if isinstance(value, (list, tuple)):
        for member in value:
            _reject_unsafe_values(member)
        return
    if isinstance(value, str) and (
        "://" in value
        or value.startswith(("/", "\\\\"))
        or re.match(r"[A-Za-z]:[\\/]", value) is not None
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise AppliedRuntimeManifestError("unsafe provenance value contains a URL or path")


__all__ = [
    "AppliedBedZone",
    "AppliedCameraState",
    "AppliedDetectionWindow",
    "AppliedRuntimeManifest",
    "AppliedRuntimeManifestError",
    "JsonValue",
    "RuntimeEnvironmentFacts",
    "canonical_json",
    "require_sha256",
]
