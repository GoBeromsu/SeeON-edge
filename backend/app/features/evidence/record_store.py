"""Native incident query and incident-local review authority."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import assert_never

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.edge_db.reviews import EvidenceReview, ReviewDisposition


@dataclass(frozen=True, slots=True)
class CentralEvidenceSummary:
    incident_id: str
    edge_event_id: str
    schema_version: int
    camera_id: str
    event_type: str
    detected_at: str
    lifecycle_state: str
    revision: int
    failure_reason: str | None
    runtime_manifest_sha256: str | None
    decision_trace_id: str | None
    module_qualified_id: str | None
    policy_qualified_id: str | None
    primary_clip_id: str | None
    primary_artifact_state: str | None
    snapshot_artifact_state: str | None
    event_delivery_state: str
    clip_publish_state: str | None
    retention_state: str | None
    review: EvidenceReview | None


@dataclass(slots=True)
class EvidenceReviewConflictError(RuntimeError):
    incident_id: str
    expected_version: int

    def __str__(self) -> str:
        return (
            f"incident review {self.incident_id}: expected version {self.expected_version} changed"
        )


class EvidenceProjectionUnavailable(RuntimeError):
    """An incident lacks a required durable delivery obligation."""


class CentralEvidenceReviewStore:
    """Compare-and-swap the review columns owned by one incident row."""

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native incident reviews require a PostgreSQL owner")
        if not isinstance(authority, AuthorityToken):
            raise TypeError("native incident reviews require deployment authority")
        self.database, self.authority = database, authority

    def update(
        self,
        *,
        incident_id: str,
        expected_version: int,
        actor_id: str,
        reviewed_at: str,
        disposition: ReviewDisposition,
        notes: str | None,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> EvidenceReview:
        _validate_review_input(incident_id, expected_version, actor_id, reviewed_at, notes)
        match disposition:
            case ReviewDisposition.TRUE_POSITIVE:
                database_disposition = "TP"
            case ReviewDisposition.FALSE_POSITIVE:
                database_disposition = "FP"
            case unreachable:
                assert_never(unreachable)

        def write(connection: psycopg.Connection) -> EvidenceReview:
            require_authority(connection, self.authority)
            changed = connection.execute(
                "UPDATE incidents SET review_version=review_version+1,"
                "review_disposition=%s,review_actor=%s,review_at=%s,review_notes=%s,"
                "revision=revision+1,updated_at=%s WHERE incident_id=%s AND review_version=%s",
                (
                    database_disposition,
                    actor_id,
                    reviewed_at,
                    notes,
                    reviewed_at,
                    incident_id,
                    expected_version,
                ),
            ).rowcount
            if changed != 1:
                raise EvidenceReviewConflictError(incident_id, expected_version)
            clip_row = connection.execute(
                "SELECT clip_id FROM artifacts WHERE incident_id=%s AND kind='PRIMARY_CLIP'",
                (incident_id,),
            ).fetchone()
            if after_write is not None:
                after_write(connection)
            version = expected_version + 1
            return EvidenceReview(
                review_id=f"{incident_id}:review:{version}",
                incident_id=incident_id,
                clip_id=None if clip_row is None else _text(clip_row[0]),
                version=version,
                actor_id=actor_id,
                reviewed_at=reviewed_at,
                disposition=disposition,
                notes=notes,
            )

        return self.database.transact(write)


class CentralEvidenceQuery:
    """Read privacy-bounded incident projections from native authorities."""

    def __init__(self, database: PostgresDatabase) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native incident queries require a PostgreSQL owner")
        self.database = database

    def get(self, identity: str) -> CentralEvidenceSummary | None:
        def read(connection: psycopg.Connection) -> CentralEvidenceSummary | None:
            row = connection.execute(
                _SUMMARY_SELECT + " WHERE incident.incident_id = %s OR incident.edge_event_id = %s",
                (identity, identity),
            ).fetchone()
            return None if row is None else _summary_from_row(row)

        return self.database.read(read)

    def list(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[tuple[CentralEvidenceSummary, ...], str | None]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        params: list[str | int] = []
        where = ""
        if cursor is not None:
            detected_at, incident_id = _parse_cursor(cursor)
            where = (
                " WHERE incident.detected_at < %s OR "
                "(incident.detected_at = %s AND incident.incident_id < %s)"
            )
            params.extend((detected_at, detected_at, incident_id))
        params.append(limit + 1)

        def read(connection: psycopg.Connection) -> list[tuple[object, ...]]:
            return connection.execute(
                _SUMMARY_SELECT
                + where
                + " ORDER BY incident.detected_at DESC, incident.incident_id DESC LIMIT %s",
                tuple(params),
            ).fetchall()

        rows = self.database.read(read)
        page = rows[:limit]
        summaries = tuple(_summary_from_row(row) for row in page)
        next_cursor = None
        if len(rows) > limit:
            last = summaries[-1]
            next_cursor = _format_cursor(last.detected_at, last.incident_id)
        return summaries, next_cursor


_SUMMARY_SELECT = """
SELECT incident.incident_id, incident.edge_event_id, incident.camera_id,
       incident.event_type, incident.detected_at, incident.lifecycle_state,
       incident.revision, incident.failure_reason, incident.runtime_manifest_sha256,
       incident.module_qualified_id, incident.policy_qualified_id,
       primary_artifact.clip_id, primary_artifact.state, snapshot_artifact.state,
       clip.publish_state, clip.retention_state,
       incident.review_version, incident.review_actor, incident.review_at,
       incident.review_disposition, incident.review_notes, delivery.state
FROM incidents AS incident
LEFT JOIN artifacts AS primary_artifact
  ON primary_artifact.incident_id = incident.incident_id
 AND primary_artifact.kind = 'PRIMARY_CLIP'
LEFT JOIN artifacts AS snapshot_artifact
  ON snapshot_artifact.incident_id = incident.incident_id
 AND snapshot_artifact.kind = 'SNAPSHOT'
LEFT JOIN clips AS clip ON clip.clip_id = primary_artifact.clip_id
LEFT JOIN event_outbox AS delivery ON delivery.edge_event_id = incident.edge_event_id
"""


def _summary_from_row(row: tuple[object, ...]) -> CentralEvidenceSummary:
    if row[21] is None:
        raise EvidenceProjectionUnavailable("incident delivery obligation is missing")
    review_version = _integer(row[16])
    review = None
    if review_version > 0:
        disposition = (
            ReviewDisposition.TRUE_POSITIVE if row[19] == "TP" else ReviewDisposition.FALSE_POSITIVE
        )
        review = EvidenceReview(
            review_id=f"{row[0]}:review:{review_version}",
            incident_id=str(row[0]),
            clip_id=_text(row[11]),
            version=review_version,
            actor_id=str(row[17]),
            reviewed_at=str(row[18]),
            disposition=disposition,
            notes=_text(row[20]),
        )
    return CentralEvidenceSummary(
        incident_id=str(row[0]),
        edge_event_id=str(row[1]),
        schema_version=18,
        camera_id=str(row[2]),
        event_type=str(row[3]),
        detected_at=str(row[4]),
        lifecycle_state=str(row[5]),
        revision=_integer(row[6]),
        failure_reason=_text(row[7]),
        runtime_manifest_sha256=_text(row[8]),
        decision_trace_id=None,
        module_qualified_id=_text(row[9]),
        policy_qualified_id=_text(row[10]),
        primary_clip_id=_text(row[11]),
        primary_artifact_state=_text(row[12]),
        snapshot_artifact_state=_text(row[13]),
        event_delivery_state=str(row[21]),
        clip_publish_state=_text(row[14]),
        retention_state=_text(row[15]),
        review=review,
    )


def _validate_review_input(
    incident_id: str, expected_version: int, actor_id: str, reviewed_at: str, notes: str | None
) -> None:
    if not incident_id or len(incident_id) > 128 or "\x00" in incident_id:
        raise ValueError("invalid incident_id")
    if expected_version < 0:
        raise ValueError("expected_version must be non-negative")
    if not actor_id or len(actor_id) > 128 or "\x00" in actor_id:
        raise ValueError("invalid actor_id")
    try:
        parsed = datetime.fromisoformat(reviewed_at)
    except ValueError as error:
        raise ValueError("invalid reviewed_at") from error
    if parsed.tzinfo is None or len(reviewed_at) > 30:
        raise ValueError("invalid reviewed_at")
    if notes is not None and (not notes or len(notes) > 1000 or "\x00" in notes):
        raise ValueError("invalid notes")


def _format_cursor(detected_at: str, incident_id: str) -> str:
    return base64.urlsafe_b64encode(f"{detected_at}\0{incident_id}".encode()).decode()


def _parse_cursor(cursor: str) -> tuple[str, str]:
    try:
        decoded = base64.b64decode(cursor, altchars=b"-_", validate=True).decode()
        detected_at, incident_id = decoded.split("\0", 1)
    except (ValueError, UnicodeDecodeError, binascii.Error) as error:
        raise ValueError("invalid cursor") from error
    if not detected_at or not incident_id or len(detected_at) > 30 or len(incident_id) > 128:
        raise ValueError("invalid cursor")
    return detected_at, incident_id


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored integer is invalid")
    return value


def _text(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "CentralEvidenceQuery",
    "CentralEvidenceReviewStore",
    "CentralEvidenceSummary",
    "EvidenceProjectionUnavailable",
    "EvidenceReview",
    "EvidenceReviewConflictError",
    "ReviewDisposition",
]
