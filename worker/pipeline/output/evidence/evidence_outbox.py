"""Queue-backed evidence admission compatibility seam.

The delivery queue is the only durable authority.  This class deliberately
contains no database handle or transaction state.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Self

from shared.events.delivery_queue import DeliveryQueue, EventEntry
from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClaimedClip,
    ClaimedEvent,
    ClaimLease,
    ClipId,
    ClipLocalState,
    ClipOutcome,
    ClipOutcomeConflictError,
    DatabaseSettings,
    EdgeEventId,
    EventClipConflictError,
    EvidenceReasonCode,
    MissingStagedEventError,
    NewerSchemaVersionError,
    StagedEvent,
    StagedEventConflictError,
)


class EvidenceOutbox:
    """Admit evidence events to a publish-once queue."""

    def __init__(self, queue_directory: Path) -> None:
        self._queue = DeliveryQueue(queue_directory)

    @classmethod
    def open(cls, queue_directory: Path) -> Self:
        return cls(queue_directory)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    def close(self) -> None:
        """The filesystem queue owns no per-instance resource."""

    def stage(
        self,
        event: StagedEvent,
        *,
        required_runtime_manifest_sha256: str | None = None,
        required_decision_trace_id: str | None = None,
    ) -> None:
        del required_runtime_manifest_sha256, required_decision_trace_id
        payload = json.loads(event.payload_json)
        result = self._queue.try_admit(
            EventEntry(
                edge_event_id=str(event.edge_event_id),
                event_type=str(payload["event_type"]),
                detected_at=event.detected_at,
                camera_id=str(payload["camera_id"]),
                facility_id=str(payload["facility_id"]),
                decision_trace=b"{}",
                values=event.payload_json.encode("ascii"),
            )
        )
        if not result.accepted:
            raise RuntimeError(f"event delivery admission failed: {result.fault}")

    def has_event(self, edge_event_id: EdgeEventId) -> bool:
        return any(
            entry["entry_id"] == f"event-{edge_event_id}"
            for entry in self._queue.entries()
        )

    def pending_count(self) -> int:
        return self._queue.accepted_count

__all__ = [
    "ClaimLease",
    "ClaimedClip",
    "ClaimedEvent",
    "ClipId",
    "ClipLocalState",
    "ClipOutcome",
    "ClipOutcomeConflictError",
    "DatabaseSettings",
    "EdgeEventId",
    "EventClipConflictError",
    "EvidenceOutbox",
    "EvidenceReasonCode",
    "MissingStagedEventError",
    "NewerSchemaVersionError",
    "StagedEvent",
    "StagedEventConflictError",
]
