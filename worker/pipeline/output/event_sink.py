from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol

from contracts.event import EventEvidence, EventScalar
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_metadata import (
    runtime_manifest_sha256_from_audit,
)
from worker.pipeline.output.evidence.snapshot_store import (
    SnapshotCapacityError,
    SnapshotStore,
    StoredSnapshot,
)
from worker.types import EvidenceTrigger
from worker.types.business_event import BusinessEvent

LOGGER: Final = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceStager(Protocol):
    def stage(self, event: WorkerEventPayload) -> None: ...

    def attach_snapshot(
        self,
        edge_event_id: str,
        snapshot: EventEvidence,
    ) -> None: ...

    def record_snapshot_disposition(
        self,
        edge_event_id: str,
        snapshot_id: str,
        disposition: str,
        reason: str,
    ) -> None: ...

    def complete(self, edge_event_id: str, clip_id: str | None) -> None: ...


class EventClipRecorder(Protocol):
    def on_event(
        self,
        trigger_packet: EvidenceTrigger,
        event: BusinessEvent,
        *,
        allow_new_clip: bool = True,
        detected_at: datetime,
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class EvidenceEventSink:
    """Stages an admitted event before binding its optional durable clip."""

    stager: EvidenceStager
    recorder: EventClipRecorder
    now: Callable[[], datetime] = _utc_now
    snapshot_store: SnapshotStore | None = None

    def emit(self, event: BusinessEvent) -> None:
        """Reject the legacy event-only path because clip identity would be lossy."""
        del event
        raise ValueError("trigger packet is required for evidence emission")

    def emit_for_frame(self, event: BusinessEvent, trigger_packet: EvidenceTrigger) -> None:
        """Persist an event with its authoritative triggering frame packet."""
        if trigger_packet.camera_id != event.camera_id:
            raise ValueError("event camera does not match trigger packet")
        audit = _event_audit(event.audit)
        _ = runtime_manifest_sha256_from_audit(audit)
        edge_event_id = str(event.identity)
        evidence: dict[str, str | int | float] = {
            "domain": event.domain,
            "identity": edge_event_id,
            "time_sec": event.time_sec,
        }
        if event.person_id is not None:
            evidence["person_id"] = event.person_id
        if event.bed_id is not None:
            evidence["bed_id"] = event.bed_id
        detected_at_value = self.now()
        detected_at = detected_at_value.isoformat().replace("+00:00", "Z")
        payload: WorkerEventPayload = {
            "edge_event_id": edge_event_id,
            "event_type": event.event_type,
            "probability": event.probability,
            "detected_at": detected_at,
            "camera_id": event.camera_id,
            "facility_id": event.facility_id,
            "evidence": evidence,
        }
        if audit is not None:
            payload["audit"] = audit
        # This is deliberately before every optional media operation. The
        # detector has no durable replay above this call, so failure to admit
        # its decision envelope must stop processing rather than leave media
        # or a missing alert behind.
        self.stager.stage(payload)
        snapshot_id = edge_event_id
        staged_snapshot: StoredSnapshot | None = None
        snapshot_payload: dict[str, EventScalar] | None = None
        snapshot_store = self.snapshot_store
        if event.snapshot_jpeg is not None:
            if snapshot_store is None:
                self._record_snapshot_disposition(
                    edge_event_id, snapshot_id, "UNAVAILABLE", "snapshot_store_unconfigured"
                )
            else:
                try:
                    staged_snapshot = snapshot_store.stage(
                        event.snapshot_jpeg,
                        snapshot_id=edge_event_id,
                        captured_at=detected_at,
                        camera_id=event.camera_id,
                        edge_event_id=edge_event_id,
                    )
                except SnapshotCapacityError as error:
                    LOGGER.warning(
                        (
                            "snapshot dropped by event sink backpressure: "
                            "camera_id=%s edge_event_id=%s reason=%s"
                        ),
                        event.camera_id,
                        edge_event_id,
                        error.reason,
                        extra={
                            "camera_id": event.camera_id,
                            "edge_event_id": edge_event_id,
                            "reason": error.reason,
                        },
                    )
                    self._record_snapshot_disposition(
                        edge_event_id, snapshot_id, "UNAVAILABLE", "stage_capacity"
                    )
                except Exception:  # noqa: BLE001 - optional media must not affect the event
                    LOGGER.exception(
                        "snapshot staging failed: camera_id=%s edge_event_id=%s",
                        event.camera_id,
                        edge_event_id,
                    )
                    self._record_snapshot_disposition(
                        edge_event_id, snapshot_id, "UNAVAILABLE", "stage_failed"
                    )
                else:
                    snapshot_payload = _snapshot_payload(staged_snapshot)
        if staged_snapshot is not None and snapshot_payload is not None:
            assert snapshot_store is not None
            try:
                snapshot_store.publish(staged_snapshot)
                self.stager.attach_snapshot(edge_event_id, snapshot_payload)
                snapshot_store.commit(staged_snapshot)
            except Exception:  # noqa: BLE001 - durable transition resumes at startup
                LOGGER.exception(
                    (
                        "snapshot publication remains staged for reconciliation: "
                        "camera_id=%s edge_event_id=%s"
                    ),
                    event.camera_id,
                    edge_event_id,
                    extra={"camera_id": event.camera_id, "edge_event_id": edge_event_id},
                )
                self._record_snapshot_disposition(
                    edge_event_id, snapshot_id, "UNAVAILABLE", "publish_or_attachment_failed"
                )
        elif event.snapshot_jpeg is None:
            self._record_snapshot_disposition(
                edge_event_id, snapshot_id, "UNAVAILABLE", "snapshot_not_provided"
            )
        clip_id = self.recorder.on_event(
            trigger_packet,
            event,
            detected_at=detected_at_value,
        )
        self.stager.complete(edge_event_id, clip_id)

    def _record_snapshot_disposition(
        self, edge_event_id: str, snapshot_id: str, disposition: str, reason: str
    ) -> None:
        try:
            self.stager.record_snapshot_disposition(
                edge_event_id, snapshot_id, disposition, reason
            )
        except Exception:  # noqa: BLE001 - event remains authoritative
            LOGGER.exception(
                "snapshot disposition admission failed: edge_event_id=%s", edge_event_id
            )


def _snapshot_payload(snapshot: StoredSnapshot) -> dict[str, EventScalar]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "path": snapshot.path,
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "mime_type": snapshot.mime_type,
        "captured_at": snapshot.captured_at,
        "camera_id": snapshot.camera_id,
        "edge_event_id": snapshot.edge_event_id,
    }


def _event_audit(audit: Mapping[str, object] | None) -> EventEvidence | None:
    if audit is None:
        return None
    parsed: dict[str, EventScalar] = {}
    for key, value in audit.items():
        if value is not None and not isinstance(value, str | int | float | bool):
            raise ValueError(f"event audit {key!r} must be scalar")
        parsed[key] = value
    return parsed


__all__ = ["EvidenceEventSink"]
