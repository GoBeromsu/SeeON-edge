"""Schema-18 incident query and incident-local review authority."""

from __future__ import annotations

import base64
import binascii
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import assert_never

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
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
    derivative_state: str | None
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
            f"incident review {self.incident_id}: "
            f"expected version {self.expected_version} changed"
        )


class CentralEvidenceReviewStore:
    """Compare-and-swap the review columns owned by one incident row."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def update(
        self,
        *,
        incident_id: str,
        expected_version: int,
        actor_id: str,
        reviewed_at: str,
        disposition: ReviewDisposition,
        notes: str | None,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> EvidenceReview:
        _validate_review_input(incident_id, expected_version, actor_id, reviewed_at, notes)
        match disposition:
            case ReviewDisposition.TRUE_POSITIVE:
                database_disposition = "TP"
            case ReviewDisposition.FALSE_POSITIVE:
                database_disposition = "FP"
            case unreachable:
                assert_never(unreachable)
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                changed = connection.execute(
                    """
                    UPDATE incidents
                    SET review_version = review_version + 1,
                        review_disposition = ?, review_actor = ?, review_at = ?,
                        review_notes = ?, revision = revision + 1, updated_at = ?
                    WHERE incident_id = ? AND review_version = ?
                    """,
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
                    "SELECT clip_id FROM artifacts "
                    "WHERE incident_id = ? AND kind = 'PRIMARY_CLIP'",
                    (incident_id,),
                ).fetchone()
                if after_write is not None:
                    after_write(connection)
        finally:
            connection.close()
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


class CentralEvidenceQuery:
    """Read privacy-bounded incident projections from compact authorities."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def get(self, identity: str) -> CentralEvidenceSummary | None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(
                _SUMMARY_SELECT
                + " WHERE incident.incident_id = ? OR incident.edge_event_id = ?",
                (identity, identity),
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else _summary_from_row(row)

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
                " WHERE incident.detected_at < ? OR "
                "(incident.detected_at = ? AND incident.incident_id < ?)"
            )
            params.extend((detected_at, detected_at, incident_id))
        params.append(limit + 1)
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            rows = connection.execute(
                _SUMMARY_SELECT
                + where
                + " ORDER BY incident.detected_at DESC, incident.incident_id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        finally:
            connection.close()
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
       incident.review_disposition, incident.review_notes
FROM incidents AS incident
LEFT JOIN artifacts AS primary_artifact
  ON primary_artifact.incident_id = incident.incident_id
 AND primary_artifact.kind = 'PRIMARY_CLIP'
LEFT JOIN artifacts AS snapshot_artifact
  ON snapshot_artifact.incident_id = incident.incident_id
 AND snapshot_artifact.kind = 'SNAPSHOT'
LEFT JOIN clips AS clip ON clip.clip_id = primary_artifact.clip_id
"""


def _summary_from_row(row: sqlite3.Row | tuple[object, ...]) -> CentralEvidenceSummary:
    review_version = _integer(row[16])
    review = None
    if review_version > 0:
        disposition = (
            ReviewDisposition.TRUE_POSITIVE
            if row[19] == "TP"
            else ReviewDisposition.FALSE_POSITIVE
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
        incident_id=str(row[0]), edge_event_id=str(row[1]), schema_version=18,
        camera_id=str(row[2]), event_type=str(row[3]), detected_at=str(row[4]),
        lifecycle_state=str(row[5]), revision=_integer(row[6]), failure_reason=_text(row[7]),
        runtime_manifest_sha256=_text(row[8]), decision_trace_id=None,
        module_qualified_id=_text(row[9]), policy_qualified_id=_text(row[10]),
        primary_clip_id=_text(row[11]), primary_artifact_state=_text(row[12]),
        snapshot_artifact_state=_text(row[13]), derivative_state=None,
        event_delivery_state="ACKED", clip_publish_state=_text(row[14]),
        retention_state=_text(row[15]), review=review,
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
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
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
    "CentralEvidenceQuery", "CentralEvidenceReviewStore", "CentralEvidenceSummary",
    "EvidenceReview", "EvidenceReviewConflictError", "ReviewDisposition",
]
