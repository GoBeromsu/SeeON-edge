"""Atomic schema-18 projection of relay incidents and artifact facts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction


class RelayEvidenceProjectionError(RuntimeError):
    """A relay fact cannot be represented by compact evidence."""


class RelayEvidenceProjectionConflict(RelayEvidenceProjectionError):
    """An idempotency key was replayed with different immutable facts."""


class RelayEvidenceProjectionMissingEvent(RelayEvidenceProjectionError):
    """An artifact arrived before its incident."""


@dataclass(frozen=True, slots=True)
class RelayEvent:
    edge_event_id: str
    event_type: str
    probability: float
    detected_at: str
    camera_id: str
    facility_id: str
    resident_id: str | None
    evidence: dict[str, JsonValue] | None
    audit: dict[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class RelaySnapshot:
    snapshot_id: str
    path: str
    sha256: str
    size_bytes: int
    mime_type: str
    captured_at: str


class RelayEvidenceProjection:
    """Own the transaction that acknowledges incident and optional snapshot."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def project_event(self, event: RelayEvent, snapshot: RelaySnapshot | None = None) -> None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                existing = connection.execute(
                    """
                    SELECT incident_id, facility_id, camera_id, event_type, probability, detected_at
                    FROM incidents WHERE edge_event_id = ?
                    """,
                    (event.edge_event_id,),
                ).fetchone()
                expected = (
                    _incident_id(event.edge_event_id), event.facility_id, event.camera_id,
                    event.event_type, event.probability, event.detected_at,
                )
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO incidents (
                            incident_id, edge_event_id, facility_id, camera_id, event_type,
                            probability, detected_at, lifecycle_state, provenance_state,
                            provenance_missing_reason, review_version, revision,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', 'MISSING',
                                  'NOT_RECORDED', 0, 1, ?, ?)
                        """,
                        (
                            expected[0], event.edge_event_id, event.facility_id,
                            event.camera_id, event.event_type, event.probability,
                            event.detected_at, event.detected_at, event.detected_at,
                        ),
                    )
                elif tuple(existing) != expected:
                    raise RelayEvidenceProjectionConflict(
                        "edge_event_id conflicts with existing incident identity"
                    )
                if snapshot is not None:
                    _put_snapshot(connection, expected[0], snapshot)
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
        snapshot = RelaySnapshot(
            snapshot_id=snapshot_id,
            path=media_reference,
            sha256=sha256,
            size_bytes=size_bytes,
            mime_type=mime_type,
            captured_at=_snapshot_timestamp(snapshot_id),
        )
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                _put_snapshot(connection, _incident_for_event(connection, edge_event_id), snapshot)
        finally:
            connection.close()

    def record_snapshot_disposition(
        self, *, edge_event_id: str, snapshot_id: str, disposition: str, reason: str
    ) -> None:
        terminal_reason = _bounded_reason(disposition, reason)
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                incident_id = _incident_for_event(connection, edge_event_id)
                existing = connection.execute(
                    "SELECT artifact_id, state, reason FROM artifacts "
                    "WHERE incident_id = ? AND kind = 'SNAPSHOT'",
                    (incident_id,),
                ).fetchone()
                expected = (None, "UNAVAILABLE", terminal_reason)
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO artifacts (
                            incident_id, kind, state, reason, captured_at,
                            revision, created_at, updated_at
                        ) VALUES (?, 'SNAPSHOT', 'UNAVAILABLE', ?, ?, 1, ?, ?)
                        """,
                        (incident_id, terminal_reason, _snapshot_timestamp(snapshot_id),
                         _snapshot_timestamp(snapshot_id), _snapshot_timestamp(snapshot_id)),
                    )
                elif tuple(existing) != expected:
                    raise RelayEvidenceProjectionConflict(
                        "snapshot disposition conflicts with existing terminal fact"
                    )
        finally:
            connection.close()


def _put_snapshot(
    connection: sqlite3.Connection, incident_id: str, snapshot: RelaySnapshot
) -> None:
    _validate_snapshot(snapshot)
    existing = connection.execute(
        """
        SELECT artifact_id, state, contained_relpath, content_sha256, size_bytes,
               mime_type, captured_at
        FROM artifacts WHERE incident_id = ? AND kind = 'SNAPSHOT'
        """,
        (incident_id,),
    ).fetchone()
    expected = (
        snapshot.snapshot_id, "AVAILABLE", snapshot.path, snapshot.sha256,
        snapshot.size_bytes, snapshot.mime_type, snapshot.captured_at,
    )
    if existing is None:
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, artifact_id, state, contained_relpath,
                content_sha256, size_bytes, mime_type, captured_at,
                revision, created_at, updated_at
            ) VALUES (?, 'SNAPSHOT', ?, 'AVAILABLE', ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (incident_id, snapshot.snapshot_id, snapshot.path, snapshot.sha256,
             snapshot.size_bytes, snapshot.mime_type, snapshot.captured_at,
             snapshot.captured_at, snapshot.captured_at),
        )
    elif tuple(existing) != expected:
        raise RelayEvidenceProjectionConflict(
            "snapshot attachment conflicts with existing content identity"
        )


def _validate_snapshot(snapshot: RelaySnapshot) -> None:
    if snapshot.size_bytes <= 0:
        raise RelayEvidenceProjectionError("snapshot size_bytes must be positive")
    if (
        not snapshot.path or snapshot.path.startswith("/") or "\\" in snapshot.path
        or snapshot.path in {".", ".."} or "/../" in f"/{snapshot.path}/"
    ):
        raise RelayEvidenceProjectionError("snapshot path is not contained")


def _incident_id(edge_event_id: str) -> str:
    return f"incident:{edge_event_id}"


def _incident_for_event(connection: sqlite3.Connection, edge_event_id: str) -> str:
    row = connection.execute(
        "SELECT incident_id FROM incidents WHERE edge_event_id = ?", (edge_event_id,)
    ).fetchone()
    if row is None:
        raise RelayEvidenceProjectionMissingEvent(
            "snapshot companion requires an already-projected incident"
        )
    return str(row[0])


def _snapshot_timestamp(snapshot_id: str) -> str:
    # Separate companion routes predate captured_at on their wire model. Keep a
    # deterministic, schema-valid boundary value until that contract is retired.
    del snapshot_id
    return "1970-01-01T00:00:00Z"


def _bounded_reason(disposition: str, reason: str) -> str:
    value = f"{disposition}:{reason}"
    return value[:64]


__all__ = [
    "RelayEvent", "RelayEvidenceProjection", "RelayEvidenceProjectionConflict",
    "RelayEvidenceProjectionError", "RelayEvidenceProjectionMissingEvent", "RelaySnapshot",
]
