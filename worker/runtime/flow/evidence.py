"""Flow Smart Record admission and sealed-clip completion binding."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from worker.pipeline.output.evidence.flow_clip_publication import FlowClipPublisher
from worker.pipeline.output.evidence.flow_sealed_sidecar import (
    FlowSealedRecovery,
    FlowSealedSidecars,
)
from worker.pipeline.output.evidence.smart_record_actor import ClipSealed, SmartRecordActor
from worker.types import BusinessEvent, NativeEvidenceTrigger

LOGGER = logging.getLogger(__name__)


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
    sidecars: FlowSealedSidecars
    camera_id: str
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    _events: dict[str, BusinessEvent] = field(default_factory=dict, init=False)
    sealed_recovery_missing_media_total: int = field(default=0, init=False)

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
        sidecar_path = self.sidecars.persist(sealed, self._events)
        recovery = FlowSealedRecovery(sealed, dict(self._events), self.camera_id, sidecar_path)
        self._publish_recovery(recovery)
        for contributor in sealed.contributors:
            del self._events[contributor.event_ref]

    def replay_sealed(self) -> None:
        """Retry sealed clips before Flow activates any camera sources."""
        for recovery in self.sidecars.pending_for_camera(self.camera_id):
            media_path = Path(recovery.sealed.path)
            if not media_path.is_file():
                error = self.sidecars.discard_missing_media(recovery)
                self.sealed_recovery_missing_media_total += 1
                LOGGER.error("%s", error)
                continue
            self._publish_recovery(recovery)

    def _publish_recovery(self, recovery: FlowSealedRecovery) -> None:
        for contributor in recovery.sealed.contributors:
            if contributor.event_ref not in recovery.events:
                raise ValueError(
                    f"sealed Flow clip has unknown contributor {contributor.event_ref}"
                )
        published = self.publisher.publish(recovery.sealed, recovery.events)
        for contributor in recovery.sealed.contributors:
            self.stager.complete(contributor.event_ref, str(published.clip_id))
        self.sidecars.remove(recovery)


__all__ = ["FlowEvidenceBinding", "FlowEvidenceStager"]
