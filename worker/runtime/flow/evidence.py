"""Flow Smart Record admission and sealed-clip completion binding."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from shared.events.delivery_queue import AdmissionResult
from worker.interfaces.execution_records import ExecutionRecordSink
from worker.pipeline.diagnostics.emit_delivery import event_delivery_record
from worker.pipeline.diagnostics.record_builder import try_emit
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

    def stage(self, event: dict[str, object]) -> AdmissionResult: ...

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
    execution_records: ExecutionRecordSink | None = None
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
        payload = {
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

        def _delivery_record(admitted: bool, reason: str | None):
            return event_delivery_record(
                camera_id=event.camera_id,
                worker_boot_id=trigger.worker_boot_id,
                source_generation=trigger.source_generation,
                stream_epoch=trigger.stream_epoch,
                frame_seq=trigger.seq,
                source_pts_ns=trigger.source_pts,
                edge_event_id=event_ref,
                event_type=event.event_type,
                domain=event.domain,
                admitted=admitted,
                reason=reason,
            )

        try:
            result = self.stager.stage(payload)
        except Exception as error:
            try_emit(
                self.execution_records,
                _delivery_record(False, _admission_reason(error)),
            )
            raise
        admitted, reason = _admission_from_stage_result(result)
        if not admitted:
            try_emit(
                self.execution_records,
                _delivery_record(False, reason),
            )
            raise RuntimeError(f"event delivery admission failed: {reason}")
        try_emit(
            self.execution_records,
            _delivery_record(True, reason),
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
        """Retry sealed clips before Flow activates any camera sources.

        Each sidecar is isolated: a clip this replay cannot safely resume (a
        genuine identity mismatch, or any other publish failure) is logged and
        left in place rather than aborting every other sidecar queued behind it.
        """
        for recovery in self.sidecars.pending_for_camera(self.camera_id):
            media_path = Path(recovery.sealed.path)
            if not media_path.is_file():
                error = self.sidecars.discard_missing_media(recovery)
                self.sealed_recovery_missing_media_total += 1
                LOGGER.error("%s", error)
                continue
            try:
                self._publish_recovery(recovery)
            except Exception:  # noqa: BLE001 - one bad sidecar must not block the rest
                LOGGER.exception(
                    "sealed Flow clip replay failed clip_id=%s camera_id=%s",
                    recovery.sealed.clip_id,
                    self.camera_id,
                )

    def _publish_recovery(self, recovery: FlowSealedRecovery) -> None:
        for contributor in recovery.sealed.contributors:
            if contributor.event_ref not in recovery.events:
                raise ValueError(
                    f"sealed Flow clip has unknown contributor {contributor.event_ref}"
                )
        published = self.publisher.publish(recovery.sealed, recovery.events)
        for contributor in recovery.sealed.contributors:
            self.stager.complete(contributor.event_ref, str(published.clip_id))
        # ponytail: publish is idempotent (FlowClipPublisher resumes from the
        # existing manifest on a collision) and replay_sealed() isolates each
        # sidecar, so retiring here is just cleanup -- a crash before this line
        # leaves a sidecar that the next replay_sealed() resumes and retires.
        self.sidecars.remove(recovery)


def _admission_from_stage_result(result: object) -> tuple[bool, str | None]:
    """Read try_admit proof from a stager return.

    The stager contract is ``stage() -> AdmissionResult``. Only a real
    ``AdmissionResult`` with ``accepted`` True proves admission; anything else
    (``None``, a duck with an ``accepted`` attribute, an unrelated object) is an
    unproven admission and is recorded as refused. An "admitted" record must
    never be emitted without the queue's own proof.
    """
    if not isinstance(result, AdmissionResult):
        return False, f"unproven-admission:{type(result).__name__}"
    reason = None if result.fault is None else str(result.fault)
    return bool(result.accepted), reason


def _admission_reason(error: BaseException) -> str:
    text = str(error)
    marker = "event delivery admission failed: "
    if text.startswith(marker):
        fault = text[len(marker) :].strip()
        if fault and fault != "None":
            return fault
    name = type(error).__name__
    return name if not text else f"{name}: {text}"


__all__ = ["FlowEvidenceBinding", "FlowEvidenceStager"]
