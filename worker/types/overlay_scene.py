from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

Color: TypeAlias = tuple[int, int, int]
Point: TypeAlias = tuple[float, float]
OVERLAY_SCENE_SCHEMA_VERSION: Final = 1


class ObservationSemantics(StrEnum):
    PRESENT = "present"
    STALE = "stale"
    MISSING = "missing"
    NOT_EVALUATED = "not-evaluated"


@dataclass(frozen=True, slots=True)
class SceneValue:
    value: int | float | None
    semantics: ObservationSemantics
    reason: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and (
            isinstance(self.value, bool) or not math.isfinite(float(self.value))
        ):
            raise ValueError("overlay numeric values must be finite")
        if (self.value is None) != (self.semantics is not ObservationSemantics.PRESENT):
            if self.semantics is not ObservationSemantics.STALE or self.value is None:
                raise ValueError("overlay value and observation semantics disagree")
        if self.semantics is ObservationSemantics.PRESENT and self.reason is not None:
            raise ValueError("present overlay values cannot have a missing reason")
        if self.semantics is not ObservationSemantics.PRESENT and not self.reason:
            raise ValueError("non-present overlay values require an explicit reason")


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def point(self, value: Point) -> tuple[int, int]:
        return (
            round(value[0] * self.scale_x + self.offset_x),
            round(value[1] * self.scale_y + self.offset_y),
        )


@dataclass(frozen=True, slots=True)
class SceneFrameIdentity:
    worker_boot_id: str
    camera_id: str
    stream_epoch: int
    seq: int
    pts: SceneValue
    source_time: SceneValue
    camera_configuration_id: str


@dataclass(frozen=True, slots=True)
class SceneKeypoint:
    index: int
    point: Point | None
    confidence: float | None
    semantics: ObservationSemantics
    reason: str | None


@dataclass(frozen=True, slots=True)
class ScenePerson:
    ordinal: int
    track_id: SceneValue
    box: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[SceneKeypoint, ...]
    color: Color
    z_order: int


@dataclass(frozen=True, slots=True)
class SceneContainment:
    track_id: SceneValue
    ratio: SceneValue
    threshold: SceneValue
    state: str
    reason: str


@dataclass(frozen=True, slots=True)
class SceneBed:
    ordinal: int
    box: tuple[float, float, float, float]
    polygon: tuple[Point, ...]
    confidence: float
    provenance: str
    semantics: ObservationSemantics
    containments: tuple[SceneContainment, ...]
    color: Color
    z_order: int


@dataclass(frozen=True, slots=True)
class SceneDecision:
    module_qualified_id: str
    policy_qualified_id: str
    effective_policy_id: str
    runtime_manifest_sha256: str
    track_id: SceneValue
    bed_id: SceneValue
    previous_state: str
    current_state: str
    triggered: bool
    reason: str
    score: SceneValue
    threshold: SceneValue
    counters: Mapping[str, int | float]
    semantics: ObservationSemantics
    color: Color
    z_order: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "counters", MappingProxyType(dict(sorted(self.counters.items()))))


@dataclass(frozen=True, slots=True)
class SceneComponent:
    qualified_id: str
    semantics: ObservationSemantics
    reason: str | None


@dataclass(frozen=True, slots=True)
class SceneLabel:
    text: str
    anchor: Point
    color: Color
    z_order: int


@dataclass(frozen=True, slots=True)
class OverlayScene:
    scene_id: str
    frame: SceneFrameIdentity
    source_dimensions: tuple[int, int]
    coordinate_space: Literal["source-pixels"]
    transform: CoordinateTransform
    persons: tuple[ScenePerson, ...]
    beds: tuple[SceneBed, ...]
    decisions: tuple[SceneDecision, ...]
    components: tuple[SceneComponent, ...]
    labels: tuple[SceneLabel, ...]
    schema_version: int = OVERLAY_SCENE_SCHEMA_VERSION

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            scene_data(self), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")


def fit_scene_transform(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    mode: Literal["stretch", "contain"] = "stretch",
) -> CoordinateTransform:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("overlay transform dimensions must be positive")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    if mode == "contain":
        scale_x = scale_y = min(scale_x, scale_y)
    return CoordinateTransform(
        source_width,
        source_height,
        target_width,
        target_height,
        scale_x,
        scale_y,
        (target_width - source_width * scale_x) / 2,
        (target_height - source_height * scale_y) / 2,
    )


def scene_data(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: scene_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): scene_data(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [scene_data(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def scene_content_id(payload: object) -> str:
    encoded = json.dumps(
        scene_data(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "Color",
    "CoordinateTransform",
    "ObservationSemantics",
    "OverlayScene",
    "Point",
    "SceneBed",
    "SceneComponent",
    "SceneContainment",
    "SceneDecision",
    "SceneFrameIdentity",
    "SceneKeypoint",
    "SceneLabel",
    "ScenePerson",
    "SceneValue",
    "fit_scene_transform",
    "scene_content_id",
    "scene_data",
]
