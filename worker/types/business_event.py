from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BusinessEvent:
    domain: str
    event_type: str
    identity: str | int
    camera_id: str
    facility_id: str
    time_sec: float
    probability: float
    person_id: int | None = None
    bed_id: int | None = None
    audit: Mapping[str, object] | None = None
    snapshot_jpeg: bytes | None = None
    snapshot_unavailable_reason: str | None = None


__all__ = ["BusinessEvent"]
