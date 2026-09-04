"""Flow Smart Record admission and sealed-clip completion binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublisher
from worker.pipeline.output.evidence.smart_record_actor import ClipSealed, SmartRecordActor
from worker.types import BusinessEvent, NativeEvidenceTrigger


class FlowEvidenceStager(Protocol):
    """The durable methods required by the Flow evidence bridge."""

    def stage(self, event: dict[str, object]) -> None: ...

    def complete(self, edge_event_id: str, clip_id: str | None) -> None: ...


@dataclass(slots=True)
class FlowEvidenceBinding:
    """Stage admitted alerts and complete every Smart Record contributor together.

    Flow cannot claim a clip at admission: the recorder assigns one only after
    the Smart Record callback seals the shared recording.  Keeping contributor
    references on the actor-owned clip makes a single sealed receipt complete
    all incidents that extended that recording.
    """

    actor: SmartRecordActor
    stager: FlowEvidenceStager
    publisher: FlowClipPublisher
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    _events: dict[str, BusinessEvent] = field(default_factory=dict, init=False)

    def emit_for_frame(self, event: BusinessEvent, trigger: NativeEvidenceTrigger) -> None:
        if event.camera_id != trigger.camera_id:
            raise ValueError("event camera does not match Flow trigger")
        detected = self.now()
        if detected.tzinfo is None or detected.utcoffset() is None:
            raise ValueError("Flow detected_at must be timezone-aware")
        detected_at = detected.isoformat().replace("+00:00", "Z")
        event_ref = str(event.identity)
        self.stager.stage(
            {
                "edge_event_id": event_ref,
                "event_type": event.event_type,
                "probability": event.probability,
                "detected_at": detected_at,
                "camera_id": event.camera_id,
                "facility_id": event.facility_id,
                "evidence": {
                    "domain": event.domain,
                    "identity": event_ref,
                    "time_sec": event.time_sec,
                },
            }
        )
        self._events[event_ref] = event
        self.actor.admit(event_ref, detected_at)

    def on_sealed(self, sealed: ClipSealed) -> None:
        """Publish before completing every incident bound to the shared clip."""
        for contributor in sealed.contributors:
            if contributor.event_ref not in self._events:
                raise ValueError(
                    f"sealed Flow clip has unknown contributor {contributor.event_ref}"
                )
        published = self.publisher.publish(sealed, self._events)
        for contributor in sealed.contributors:
            self.stager.complete(contributor.event_ref, str(published.clip_id))
            del self._events[contributor.event_ref]


__all__ = ["FlowEvidenceBinding", "FlowEvidenceStager"]
