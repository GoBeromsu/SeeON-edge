"""API-owned projection of relay-delivered evidence queue entries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction


class RelayEvidenceProjectionError(RuntimeError):
    """A delivered relay entry cannot be represented by central evidence."""


class RelayEvidenceProjectionConflict(RelayEvidenceProjectionError):
    """A replay conflicts with an immutable evidence fact."""


class RelayEvidenceProjectionMissingEvent(RelayEvidenceProjectionError):
    """A media companion arrived before its event projection."""


@dataclass(frozen=True, slots=True)
class RelayEvent:
    edge_event_id: str
    event_type: str
    probability: float
    detected_at: str
    camera_id: str
    facility_id: str
    resident_id: str | None
    evidence: dict[str, Any] | None
    audit: dict[str, Any] | None


class RelayEvidenceProjection:
    """Write queue facts once, under the API runtime actor."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def project_event(self, event: RelayEvent) -> None:
        payload = json.dumps(
            {
                "edge_event_id": event.edge_event_id,
                "event_type": event.event_type,
                "probability": event.probability,
                "detected_at": event.detected_at,
                "camera_id": event.camera_id,
                "facility_id": event.facility_id,
                "resident_id": event.resident_id,
                "evidence": event.evidence,
                "audit": event.audit,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                connection.execute(
                    """
                    INSERT INTO evidence_events (
                        edge_event_id, detected_at, payload_json, state, queued_at,
                        next_attempt_at, delivery_state
                    ) VALUES (?, ?, ?, 'ACKED', 0, 0, 'ACKED')
                    ON CONFLICT(edge_event_id) DO NOTHING
                    """,
                    (event.edge_event_id, event.detected_at, payload),
                )
                connection.execute(
                    """
                    INSERT INTO evidence_incidents (
                        incident_id, edge_event_id, camera_id, event_type, detected_at,
                        provenance_missing_reason, lifecycle_state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'NOT_RECORDED', 'STAGING', ?, ?)
                    ON CONFLICT(edge_event_id) DO NOTHING
                    """,
                    (
                        _incident_id(event.edge_event_id),
                        event.edge_event_id,
                        event.camera_id,
                        event.event_type,
                        event.detected_at,
                        event.detected_at,
                        event.detected_at,
                    ),
                )
        finally:
            connection.close()

    def attach_snapshot(
        self,
        *,
        edge_event_id: str,
        snapshot_id: str,
        sha256: str,
        media_reference: str,
        size_bytes: int,
        mime_type: str,
    ) -> None:
        if size_bytes <= 0:
            raise RelayEvidenceProjectionError("snapshot attachment size_bytes must be positive")
        _validate_media_reference(media_reference)
        media_identity = f"{sha256}\0{media_reference}".encode()
        media_id = f"snapshot:{hashlib.sha256(media_identity).hexdigest()}"
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                incident_id = _incident_for_event(connection, edge_event_id)
                connection.execute(
                    """
                    INSERT INTO evidence_media_objects (
                        media_id, content_sha256, size_bytes, mime_type, contained_relpath,
                        basename, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(media_id) DO NOTHING
                    """,
                    (
                        media_id,
                        sha256,
                        size_bytes,
                        mime_type,
                        media_reference,
                        media_reference.rsplit("/", 1)[-1],
                        snapshot_id,
                    ),
                )
                slot = connection.execute(
                    "SELECT state, media_id FROM evidence_artifact_slots "
                    "WHERE incident_id = ? AND slot_name = 'SNAPSHOT'",
                    (incident_id,),
                ).fetchone()
                if slot is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_artifact_slots (
                            incident_id, slot_name, state, media_id, created_at, updated_at
                        ) VALUES (?, 'SNAPSHOT', 'AVAILABLE', ?, ?, ?)
                        """,
                        (incident_id, media_id, snapshot_id, snapshot_id),
                    )
                elif tuple(slot) != ("AVAILABLE", media_id):
                    raise RelayEvidenceProjectionConflict(
                        "snapshot attachment conflicts with terminal evidence fact"
                    )
        finally:
            connection.close()

    def record_snapshot_disposition(
        self, *, edge_event_id: str, snapshot_id: str, disposition: str, reason: str
    ) -> None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                incident_id = _incident_for_event(connection, edge_event_id)
                terminal_reason = f"{disposition}:{reason}"
                slot = connection.execute(
                    "SELECT state, reason FROM evidence_artifact_slots "
                    "WHERE incident_id = ? AND slot_name = 'SNAPSHOT'",
                    (incident_id,),
                ).fetchone()
                if slot is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_artifact_slots (
                            incident_id, slot_name, state, reason, created_at, updated_at
                        ) VALUES (?, 'SNAPSHOT', 'UNAVAILABLE', ?, ?, ?)
                        """,
                        (incident_id, terminal_reason, snapshot_id, snapshot_id),
                    )
                elif tuple(slot) != ("UNAVAILABLE", terminal_reason):
                    raise RelayEvidenceProjectionConflict(
                        "snapshot disposition conflicts with existing evidence fact"
                    )
        finally:
            connection.close()


def _incident_id(edge_event_id: str) -> str:
    return f"incident:{edge_event_id}"


def _incident_for_event(connection: sqlite3.Connection, edge_event_id: str) -> str:
    row = connection.execute(
        "SELECT incident_id FROM evidence_incidents WHERE edge_event_id = ?", (edge_event_id,)
    ).fetchone()
    if row is None:
        raise RelayEvidenceProjectionMissingEvent(
            "snapshot companion requires an already-projected event"
        )
    return str(row[0])


def _validate_media_reference(value: str) -> None:
    if (
        value.startswith("/")
        or "\\" in value
        or value in {".", ".."}
        or "/../" in f"/{value}/"
        or not value.rsplit("/", 1)[-1]
    ):
        raise RelayEvidenceProjectionError("snapshot media_reference is not a contained path")


__all__ = [
    "RelayEvent",
    "RelayEvidenceProjection",
    "RelayEvidenceProjectionConflict",
    "RelayEvidenceProjectionError",
    "RelayEvidenceProjectionMissingEvent",
]
