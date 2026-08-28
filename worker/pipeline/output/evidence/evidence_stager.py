"""Durably admit event and snapshot envelopes to the publish-once delivery queue."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import shared.events.envelope_limits as limits
from contracts.event import EventEvidence
from contracts.worker_config import CONFIG_VERSION_KEY
from shared.events.delivery_queue import (
    DeliveryQueue,
    EventEntry,
    SnapshotAttachmentEntry,
    SnapshotDispositionEntry,
)
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_metadata import (
    RUNTIME_MANIFEST_SHA256_KEY,
    validate_runtime_manifest_sha256,
)
from worker.pipeline.output.evidence.runtime_manifest_reference import (
    RuntimeManifestReferenceError,
    RuntimeManifestReferenceFailure,
)


@dataclass(frozen=True, slots=True, init=False)
class DurableEvidenceStager:
    """Queue-only durable boundary for an incident and optional snapshot facts.

    The event is admitted independently before callers perform any media work.
    A failed admission is raised to the detector path: silently continuing would
    drop an unreplayable safety event.
    """

    queue_directory: Path
    camera_id: str
    facility_id: str
    resident_id: str | None
    config_version: int
    clock: Callable[[], float]
    runtime_manifest_sha256: str | None = None
    _queue: DeliveryQueue = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        queue_directory: Path,
        camera_id: str = "",
        facility_id: str = "",
        resident_id: str | None = None,
        config_version: int = 0,
        clock: Callable[[], float] = lambda: 0.0,
        runtime_manifest_sha256: str | None = None,
    ) -> None:
        """Construct the queue boundary."""
        object.__setattr__(self, "queue_directory", queue_directory)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "facility_id", facility_id)
        object.__setattr__(self, "resident_id", resident_id)
        object.__setattr__(self, "config_version", config_version)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "runtime_manifest_sha256", runtime_manifest_sha256)
        validate_runtime_manifest_sha256(runtime_manifest_sha256)
        object.__setattr__(self, "_queue", DeliveryQueue(queue_directory))

    @property
    def queue(self) -> DeliveryQueue:
        return self._queue

    def stage(self, event: WorkerEventPayload) -> None:
        edge_event_id = _required_text(event, "edge_event_id")
        detected_at = _required_text(event, "detected_at")
        event_type = _required_text(event, "event_type")
        values, trace, shed_detail_keys = self._envelope(event)
        result = self._queue.try_admit(
            EventEntry(
                edge_event_id=edge_event_id,
                event_type=event_type,
                detected_at=detected_at,
                camera_id=self.camera_id,
                facility_id=self.facility_id,
                decision_trace=trace,
                values=values,
                shed_detail_keys=shed_detail_keys,
            )
        )
        if not result.accepted:
            raise RuntimeError(f"event delivery admission failed: {result.fault}")

    def attach_snapshot(self, edge_event_id: str, snapshot: EventEvidence) -> None:
        result = self._queue.try_admit(
            SnapshotAttachmentEntry(
                edge_event_id=edge_event_id,
                snapshot_id=_snapshot_text(snapshot, "snapshot_id"),
                sha256=_snapshot_text(snapshot, "sha256"),
                media_reference=_snapshot_text(snapshot, "path"),
                size_bytes=_snapshot_size(snapshot),
                mime_type=_snapshot_text(snapshot, "mime_type"),
            )
        )
        if not result.accepted:
            raise RuntimeError(f"snapshot attachment admission failed: {result.fault}")

    def record_snapshot_disposition(
        self, edge_event_id: str, snapshot_id: str, disposition: str, reason: str
    ) -> None:
        result = self._queue.try_admit(
            SnapshotDispositionEntry(
                edge_event_id=edge_event_id,
                snapshot_id=snapshot_id,
                disposition=disposition,
                reason=reason,
            )
        )
        if not result.accepted:
            raise RuntimeError(f"snapshot disposition admission failed: {result.fault}")

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        """Clips are optional media and do not alter delivery of the event."""
        del edge_event_id, clip_id

    def _envelope(self, event: WorkerEventPayload) -> tuple[bytes, bytes, tuple[str, ...]]:
        values = dict(event)
        values.pop("snapshot_jpeg", None)
        values.pop("snapshot", None)
        audit = values.pop("audit", None)
        values["camera_id"] = self.camera_id
        values["facility_id"] = self.facility_id
        if self.resident_id is not None:
            values["resident_id"] = self.resident_id
        shed_audit_keys: tuple[str, ...] = ()
        if isinstance(audit, Mapping) or self.runtime_manifest_sha256 is not None:
            raw_trace = dict(audit) if isinstance(audit, Mapping) else {}
            dropped_audit_keys = sorted(set(raw_trace) - limits.RELAY_AUDIT_FIELDS)
            trace = {
                key: value
                for key, value in raw_trace.items()
                if key in limits.RELAY_AUDIT_FIELDS
            }
            shed_audit_keys = tuple(f"audit.{key}" for key in dropped_audit_keys)
            validate_runtime_manifest_sha256(trace.get(RUNTIME_MANIFEST_SHA256_KEY))
            trace[CONFIG_VERSION_KEY] = self.config_version
            if self.runtime_manifest_sha256 is not None:
                trace[RUNTIME_MANIFEST_SHA256_KEY] = self.runtime_manifest_sha256
        else:
            trace = {}
        encoded_values, shed_detail_keys = _shed_to_limit(
            values, limits.VALUES_BYTES_MAX, _PROTECTED_VALUE_KEYS
        )
        encoded_trace, _ = _shed_to_limit(
            trace, limits.DECISION_TRACE_BYTES_MAX, _PROTECTED_TRACE_KEYS
        )
        return encoded_values, encoded_trace, tuple(sorted((*shed_detail_keys, *shed_audit_keys)))


#: The relay-required core is defined at the shared wire boundary and asserted
#: against ``RelayAlertRequest`` in backend contract coverage.
_PROTECTED_VALUE_KEYS: Final = limits.REQUIRED_ALERT_FIELDS
_PROTECTED_TRACE_KEYS: Final = frozenset({CONFIG_VERSION_KEY, RUNTIME_MANIFEST_SHA256_KEY})

def _shed_to_limit(
    payload: dict[str, object], limit: int, protected: frozenset[str]
) -> tuple[bytes, tuple[str, ...]]:
    """Serialize ``payload``, shedding bulk detail rather than losing the event.

    An oversized envelope used to raise out of :meth:`DurableEvidenceStager.stage`
    from ``EventEntry.__post_init__``, before ``try_admit`` was ever reached. That
    bypassed the queue's fail-closed admission contract entirely: the fall event
    was destroyed with no ``AdmissionFault`` and no durable record. A legitimate
    event carrying many keypoints was enough to trigger it.

    Detail is therefore shed largest-key-first until the canonical form fits,
    while relay-required fields are protected. Shed key names are returned for
    durable queue metadata, never injected into the wire payload.
    """
    encoded = _canonical_bytes(payload)
    if len(encoded) <= limit:
        return encoded, ()

    remaining = dict(payload)
    shed: list[str] = []
    sheddable = sorted(
        (key for key in remaining if key not in protected),
        key=lambda key: len(_canonical_bytes(remaining[key])),
        reverse=True,
    )
    for key in sheddable:
        del remaining[key]
        shed.append(key)
        encoded = _canonical_bytes(remaining)
        if len(encoded) <= limit:
            return encoded, tuple(sorted(shed))

    # Every sheddable field is gone and the protected core still does not fit.
    # Refusing here would destroy the event, so surface the overflow to the
    # caller as a hard error only in this genuinely unrepresentable case.
    raise ValueError(
        f"evidence envelope protected core cannot fit within {limit} bytes: "
        f"protected fields alone serialize to {len(encoded)} bytes"
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _required_text(event: WorkerEventPayload, key: str) -> str:
    value = event.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"event {key} must be set")
    return text


def _snapshot_text(snapshot: EventEvidence, key: str) -> str:
    value = snapshot.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"snapshot {key} must be a non-empty string")
    return value


def _snapshot_size(snapshot: EventEvidence) -> int:
    value = snapshot.get("size_bytes")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("snapshot size_bytes must be a non-negative integer")
    return value


__all__ = [
    "DurableEvidenceStager",
    "RuntimeManifestReferenceError",
    "RuntimeManifestReferenceFailure",
]
