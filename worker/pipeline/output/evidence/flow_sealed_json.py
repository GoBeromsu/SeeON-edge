"""Strict JSON primitives for the sealed-observation codec."""

from __future__ import annotations

import math

from worker.types import BusinessEvent


def event_payload(event: BusinessEvent, *, exact: bool) -> dict[str, object]:
    """Keep only immutable publication facts, preserving established READY numbers."""
    probability = event.probability
    return {
        "domain": event.domain,
        "event_type": event.event_type,
        "identity": str(event.identity),
        "camera_id": event.camera_id,
        "facility_id": event.facility_id,
        "time_sec": number_as_float(event.time_sec) if exact else event.time_sec,
        "probability": (
            number_as_float(probability) if exact and probability is not None else probability
        ),
    }


def text_field(value: object, key: str) -> str:
    if not isinstance(value, dict) or type(value.get(key)) is not str:
        raise ValueError(f"{key} is not a string")
    return value[key]


def finite_number(value: object) -> float:
    """Validate a finite JSON number, returning it exactly as read."""
    if type(value) not in (int, float) or not math.isfinite(value):  # type: ignore[arg-type]
        raise ValueError("value is not a finite number")
    return value  # type: ignore[return-value]


def number_as_float(value: object) -> float:
    return float(finite_number(value))


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")
