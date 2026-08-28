"""Event schemas for ML-emitted events and backend Event API payloads.

ML emits typed events. Backend owns severity/channel/policy/final-dedup;
ML runtime incident management owns only idempotency and cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

from contracts.event import DetectionEventType, Level, Severity, front_event_type
from contracts.relay import AlertEventType, EventApiPayload

EventLifecycle: TypeAlias = Literal["detected", "updated", "resolved"]
CLOCK_SOURCE_EDGE_WALL: Final = "edge_wall_clock"




@dataclass(frozen=True, slots=True)
class EmittedEvent:
    facility: str
    camera: str
    domain: str
    event_type: str
    lifecycle: EventLifecycle
    severity: Severity
    front_event_type: DetectionEventType
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "facility": self.facility,
            "camera": self.camera,
            "domain": self.domain,
            "event_type": self.event_type,
            "lifecycle": self.lifecycle,
            "severity": self.severity.value,
            "front_event_type": self.front_event_type.value,
            "evidence": dict(self.evidence),
        }


def build_emitted_event(
    *,
    facility: str,
    camera: str,
    domain: str,
    event_type: str,
    lifecycle: EventLifecycle = "detected",
    severity: Level | str,
    evidence: dict[str, Any] | None = None,
) -> EmittedEvent:
    normalized_event_type = event_type.strip()
    if normalized_event_type == "":
        raise ValueError("event_type must be non-empty")
    return EmittedEvent(
        facility=facility,
        camera=camera,
        domain=domain,
        event_type=normalized_event_type,
        lifecycle=lifecycle,
        severity=Level(severity),
        front_event_type=front_event_type(normalized_event_type),
        evidence={} if evidence is None else dict(evidence),
    )
def build_audit_envelope(
    *,
    model_version: str | None,
    detector_version: str | None,
    operating_threshold: float | None,
) -> dict[str, object]:
    envelope: dict[str, object] = {"clock_source": CLOCK_SOURCE_EDGE_WALL}
    if model_version is not None:
        envelope["model_version"] = model_version
    if detector_version is not None:
        envelope["detector_version"] = detector_version
    if operating_threshold is not None:
        envelope["operating_threshold"] = operating_threshold
    return envelope




__all__ = [
    "CLOCK_SOURCE_EDGE_WALL",
    "AlertEventType",
    "EmittedEvent",
    "EventApiPayload",
    "EventLifecycle",
    "build_audit_envelope",
    "build_emitted_event",
]
