"""Durably stage canonical relay alerts before recorder side effects."""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contracts.event import EventEvidence
from contracts.worker_config import CONFIG_VERSION_KEY
from shared.edge_db.compatibility import EdgeDatabaseError
from worker.pipeline.output.evidence.decision_trace_reference import (
    DECISION_TRACE_ID_KEY,
    validate_decision_trace_id,
)
from worker.pipeline.output.evidence.event_payload import WorkerEventPayload
from worker.pipeline.output.evidence.evidence_metadata import (
    RUNTIME_MANIFEST_SHA256_KEY,
    validate_runtime_manifest_sha256,
)
from worker.pipeline.output.evidence.evidence_outbox import (
    ClipId,
    EdgeEventId,
    EvidenceOutbox,
    NewerSchemaVersionError,
    StagedEvent,
)
from worker.pipeline.output.evidence.runtime_manifest_reference import (
    RuntimeManifestReferenceError,
    RuntimeManifestReferenceFailure,
)


@dataclass(frozen=True, slots=True)
class DurableEvidenceStager:
    database_path: Path
    camera_id: str
    facility_id: str
    resident_id: str | None
    config_version: int
    clock: Callable[[], float]
    runtime_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_runtime_manifest_sha256(self.runtime_manifest_sha256)

    def stage(self, event: WorkerEventPayload) -> None:
        edge_event_id = EdgeEventId(_required_text(event, "edge_event_id"))
        detected_at = _required_text(event, "detected_at")
        payload = self._canonical_payload(event, edge_event_id, detected_at)
        runtime_manifest_sha256 = _runtime_manifest_reference(payload)
        decision_trace_id = _decision_trace_reference(payload)
        if decision_trace_id is not None and self.database_path.name != "edge.sqlite3":
            raise ValueError("decision trace reference requires the central edge database")
        if runtime_manifest_sha256 is not None and self.database_path.name != "edge.sqlite3":
            raise RuntimeManifestReferenceError(
                runtime_manifest_sha256,
                RuntimeManifestReferenceFailure.UNAVAILABLE,
            )
        staged_event = StagedEvent(
            edge_event_id=edge_event_id,
            detected_at=detected_at,
            payload_json=json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=True,
            ),
            queued_at=self.clock(),
        )
        try:
            with EvidenceOutbox.open(self.database_path) as outbox:
                outbox.stage(
                    staged_event,
                    required_runtime_manifest_sha256=runtime_manifest_sha256,
                    required_decision_trace_id=decision_trace_id,
                )
        except RuntimeManifestReferenceError:
            raise
        except (OSError, sqlite3.Error, EdgeDatabaseError, NewerSchemaVersionError) as error:
            if runtime_manifest_sha256 is None:
                raise
            raise RuntimeManifestReferenceError(
                runtime_manifest_sha256,
                RuntimeManifestReferenceFailure.UNAVAILABLE,
            ) from error

    def attach_snapshot(
        self,
        edge_event_id: str,
        snapshot: EventEvidence,
    ) -> None:
        with EvidenceOutbox.open(self.database_path) as outbox:
            outbox.attach_snapshot(EdgeEventId(edge_event_id), dict(snapshot))

    def complete(self, edge_event_id: str, clip_id: str | None) -> None:
        event_id = EdgeEventId(edge_event_id)
        with EvidenceOutbox.open(self.database_path) as outbox:
            if clip_id is None:
                outbox.mark_ready(event_id)
                return
            outbox.bind_clip(event_id, ClipId(clip_id))

    def _canonical_payload(
        self,
        event: WorkerEventPayload,
        edge_event_id: EdgeEventId,
        detected_at: str,
    ) -> dict[str, object]:
        event_type = _required_text(event, "event_type")
        evidence = dict(event)
        event_audit = evidence.pop("audit", None)
        snapshot_jpeg = evidence.pop("snapshot_jpeg", None)
        snapshot = evidence.pop("snapshot", None)
        payload: dict[str, object] = {
            "edge_event_id": edge_event_id,
            "event_type": "bed-exit" if event_type == "bed-exit" else "fall",
            "probability": _event_probability(event),
            "detected_at": detected_at,
            "camera_id": self.camera_id,
            "facility_id": self.facility_id,
            "evidence": evidence,
        }
        if self.resident_id is not None:
            payload["resident_id"] = self.resident_id
        if isinstance(event_audit, Mapping) or self.runtime_manifest_sha256 is not None:
            audit = dict(event_audit) if isinstance(event_audit, Mapping) else {}
            audit[CONFIG_VERSION_KEY] = self.config_version
            if self.runtime_manifest_sha256 is not None:
                audit["runtime_manifest_sha256"] = self.runtime_manifest_sha256
            payload["audit"] = audit
        if isinstance(snapshot_jpeg, bytes):
            payload["snapshot_jpeg_base64"] = base64.b64encode(snapshot_jpeg).decode(
                "ascii"
            )
        if isinstance(snapshot, Mapping):
            payload["snapshot"] = dict(snapshot)
        return payload


def _decision_trace_reference(payload: Mapping[str, object]) -> str | None:
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        return None
    return validate_decision_trace_id(audit.get(DECISION_TRACE_ID_KEY))


def _runtime_manifest_reference(payload: Mapping[str, object]) -> str | None:
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        return None
    return validate_runtime_manifest_sha256(audit.get(RUNTIME_MANIFEST_SHA256_KEY))


def _required_text(event: WorkerEventPayload, key: str) -> str:
    value = event.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"event {key} must be set")
    return text


def _event_probability(event: WorkerEventPayload) -> float:
    value = event.get("probability", event.get("confidence", 1.0))
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    return 1.0


__all__ = [
    "DurableEvidenceStager",
    "RuntimeManifestReferenceError",
    "RuntimeManifestReferenceFailure",
]
