"""Parse and latest-only latency helpers for API runtime status."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from backend.app.features.status.detection_health import (
    DetectionHealth,
    accept_detection_sample,
    detection_health_fields,
    parse_detection_counters,
)

JsonObject: TypeAlias = dict[str, object]
HealthMap: TypeAlias = dict[tuple[str, str], DetectionHealth]


@dataclass(slots=True)  # noqa: MUTABLE_OK
class LatestLatency:
    """Process-local first-attempt latency totals. Mutation is the purpose."""

    _by_facility: dict[str, JsonObject] = field(default_factory=dict)

    def record(self, facility_id: str, detected_at: str, received_at: float) -> None:
        try:
            detected = datetime.fromisoformat(detected_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return
        elapsed = received_at - detected
        previous = self._by_facility.get(facility_id, {})
        if elapsed < 0 or not math.isfinite(elapsed):
            self._by_facility[facility_id] = {**previous, "last_error": "invalid_elapsed"}
            return
        samples = _as_int(previous.get("first_attempt_samples"), default=0) + 1
        max_sec = max(_as_float(previous.get("max_sec"), default=elapsed), elapsed)
        self._by_facility[facility_id] = {
            **previous,
            "first_attempt_samples": samples,
            "max_sec": max_sec,
            "since_sec": _as_float(previous.get("since_sec"), default=received_at),
        }

    def get(self, facility_id: str) -> JsonObject | None:
        latency = self._by_facility.get(facility_id)
        return None if latency is None else deepcopy(latency)


def require_int(value: object, *, field: str) -> int:  # noqa: OBJECT_OK, GENERIC_ERR_OK
    parsed = optional_int(value, field=field)
    if parsed is None:
        raise TypeError(f"runtime status payload has invalid {field}")  # noqa: GENERIC_ERR_OK
    return parsed


def optional_int(value: object, *, field: str) -> int | None:  # noqa: OBJECT_OK, GENERIC_ERR_OK
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"runtime status payload has invalid {field}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise TypeError(f"runtime status payload has invalid {field}") from exc
    raise TypeError(f"runtime status payload has invalid {field}")


def object_dicts(values: list[object]) -> list[JsonObject]:
    cameras: list[JsonObject] = []
    for item in values:
        if not isinstance(item, dict):
            raise TypeError("runtime status payload has invalid telemetry fields")
        cameras.append(object_dict(item))
    return cameras


def object_dict(value: Mapping[object, object]) -> JsonObject:
    return {str(field_key): field_value for field_key, field_value in value.items()}


def accept_camera_samples(
    health: HealthMap,
    facility_id: str,
    cameras: list[JsonObject],
    *,
    accepted_at: float,
) -> None:
    active: set[tuple[str, str]] = set()
    for camera in cameras:
        camera_id = camera.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id:
            continue
        key = (facility_id, camera_id)
        counters = parse_detection_counters(camera.get("detection"))
        if counters is None:
            _ = health.pop(key, None)
            continue
        active.add(key)
        health[key] = accept_detection_sample(
            health.get(key), counters, accepted_at=accepted_at
        )
    for key in tuple(health):
        if key[0] == facility_id and key not in active:
            del health[key]


def prune_health(health: HealthMap, active: set[tuple[str, str]]) -> None:
    for key in tuple(health):
        if key not in active:
            del health[key]


def project_cameras(
    health: HealthMap,
    facility_id: str,
    cameras: tuple[JsonObject, ...],
    *,
    now: float,
    stale: bool,
) -> list[JsonObject]:
    projected: list[JsonObject] = []
    for camera in cameras:
        row = deepcopy(camera)
        camera_id = row.get("camera_id")
        raw_detection = row.get("detection")
        missing = parse_detection_counters(raw_detection) is None
        sample = health.get((facility_id, camera_id)) if isinstance(camera_id, str) else None
        derived = detection_health_fields(sample, now=now, stale=stale, missing=missing)
        if isinstance(raw_detection, dict):
            row["detection"] = {**raw_detection, **derived}
        else:
            row["detection"] = derived
        projected.append(row)
    return projected


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_float(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


__all__ = [
    "JsonObject",
    "LatestLatency",
    "accept_camera_samples",
    "object_dict",
    "object_dicts",
    "optional_int",
    "project_cameras",
    "prune_health",
    "require_int",
]
