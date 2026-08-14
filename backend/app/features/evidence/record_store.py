"""Read-only privacy-bounded central evidence query seam."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import token_hex

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from shared.edge_db.reviews import EvidenceReview, ReviewDisposition


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
            f"central evidence review {self.incident_id}: "
            f"expected version {self.expected_version} changed"
        )


class CentralEvidenceReviewStore:
    """API-owned DDL-free CAS writer for versioned operator review state."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def update(
        self,
        *,
        incident_id: str,
        clip_id: str,
        expected_version: int,
        actor_id: str,
        reviewed_at: str,
        disposition: ReviewDisposition,
        notes: str | None,
    ) -> EvidenceReview:
        _validate_review_input(
            incident_id=incident_id,
            clip_id=clip_id,
            expected_version=expected_version,
            actor_id=actor_id,
            reviewed_at=reviewed_at,
            disposition=disposition,
            notes=notes,
        )
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        review_id = f"review:{token_hex(16)}"
        next_version = expected_version + 1
        try:
            try:
                with write_transaction(connection):
                    relation = connection.execute(
                        "SELECT 1 FROM evidence_primary_clips "
                        "WHERE incident_id = ? AND clip_id = ?",
                        (incident_id, clip_id),
                    ).fetchone()
                    if relation is None:
                        raise ValueError(
                            "review must reference an existing central evidence relation"
                        )
                    current = connection.execute(
                        "SELECT current_version FROM control_evidence_review_state "
                        "WHERE incident_id = ? AND clip_id = ?",
                        (incident_id, clip_id),
                    ).fetchone()
                    current_version = 0 if current is None else int(current[0])
                    if current_version != expected_version:
                        raise EvidenceReviewConflictError(incident_id, expected_version)
                    connection.execute(
                        """
                        INSERT INTO control_evidence_review_revisions (
                            review_id, incident_id, clip_id, review_version, actor_id,
                            reviewed_at, disposition, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            review_id,
                            incident_id,
                            clip_id,
                            next_version,
                            actor_id,
                            reviewed_at,
                            disposition.value,
                            notes,
                        ),
                    )
                    if expected_version == 0:
                        connection.execute(
                            "INSERT INTO control_evidence_review_state "
                            "(incident_id, clip_id, current_version) VALUES (?, ?, 1)",
                            (incident_id, clip_id),
                        )
                    else:
                        changed = connection.execute(
                            "UPDATE control_evidence_review_state "
                            "SET current_version = ? "
                            "WHERE incident_id = ? AND clip_id = ? AND current_version = ?",
                            (next_version, incident_id, clip_id, expected_version),
                        ).rowcount
                        if changed != 1:
                            raise EvidenceReviewConflictError(incident_id, expected_version)
            except sqlite3.IntegrityError as error:
                raise EvidenceReviewConflictError(incident_id, expected_version) from error
        finally:
            connection.close()
        return EvidenceReview(
            review_id=review_id,
            incident_id=incident_id,
            clip_id=clip_id,
            version=next_version,
            actor_id=actor_id,
            reviewed_at=reviewed_at,
            disposition=disposition,
            notes=notes,
        )


class CentralEvidenceQuery:
    """Project stable service values instead of exposing worker table rows."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def get(self, identity: str) -> CentralEvidenceSummary | None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(
                _SUMMARY_SELECT
                + """
                WHERE incident.incident_id = ? OR incident.edge_event_id = ?
                """,
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
        """Return a newest-first page and an opaque continuation cursor.

        ``cursor`` is ``detected_at\\x1fincident_id`` of the last row from the
        previous page. ``limit`` is clamped by the operator router.
        """

        if limit < 1:
            raise ValueError("limit must be >= 1")
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            params: list[object] = []
            where = ""
            if cursor is not None:
                detected_at, incident_id = _parse_list_cursor(cursor)
                where = (
                    " WHERE (incident.detected_at < ?)"
                    " OR (incident.detected_at = ? AND incident.incident_id < ?)"
                )
                params.extend((detected_at, detected_at, incident_id))
            # Fetch one extra row to decide has_more without a second query.
            params.append(limit + 1)
            rows = connection.execute(
                _SUMMARY_SELECT
                + where
                + " ORDER BY incident.detected_at DESC, incident.incident_id DESC"
                + " LIMIT ?",
                tuple(params),
            ).fetchall()
        finally:
            connection.close()
        page_rows = rows[:limit]
        summaries = tuple(_summary_from_row(row) for row in page_rows)
        next_cursor = None
        if len(rows) > limit and page_rows:
            last = summaries[-1]
            next_cursor = _format_list_cursor(last.detected_at, last.incident_id)
        return summaries, next_cursor


_SUMMARY_SELECT = """
                SELECT incident.incident_id, incident.edge_event_id,
                       incident.record_schema_version, incident.camera_id,
                       incident.event_type, incident.detected_at,
                       incident.lifecycle_state, incident.revision,
                       incident.failure_reason, incident.runtime_manifest_sha256,
                       incident.decision_trace_id, incident.module_qualified_id,
                       incident.policy_qualified_id, incident.primary_clip_id,
                       primary_slot.state, snapshot_slot.state, derivative.state,
                       event.delivery_state, clip.publish_state, retention.state,
                       review.review_id, review.clip_id,
                       review.review_version, review.actor_id, review.reviewed_at,
                       review.disposition, review.notes
                FROM evidence_incidents AS incident
                JOIN evidence_events AS event USING (edge_event_id)
                LEFT JOIN evidence_artifact_slots AS primary_slot
                  ON primary_slot.incident_id = incident.incident_id
                 AND primary_slot.slot_name = 'PRIMARY_CLIP'
                LEFT JOIN evidence_artifact_slots AS snapshot_slot
                  ON snapshot_slot.incident_id = incident.incident_id
                 AND snapshot_slot.slot_name = 'SNAPSHOT'
                LEFT JOIN derivative_evidence_slots AS derivative
                  ON derivative.incident_id = incident.incident_id
                 AND derivative.derivative_kind = 'ANNOTATED_CLIP'
                LEFT JOIN evidence_clips AS clip
                  ON clip.clip_id = incident.primary_clip_id
                LEFT JOIN evidence_retention_states AS retention
                  ON retention.clip_id = incident.primary_clip_id
                LEFT JOIN control_evidence_review_state AS review_state
                  ON review_state.incident_id = incident.incident_id
                LEFT JOIN control_evidence_review_revisions AS review
                  ON review.incident_id = review_state.incident_id
                 AND review.clip_id = review_state.clip_id
                 AND review.review_version = review_state.current_version
"""


def _summary_from_row(row: sqlite3.Row | tuple[object, ...]) -> CentralEvidenceSummary:
    return CentralEvidenceSummary(
        incident_id=str(row[0]),
        edge_event_id=str(row[1]),
        schema_version=_required_int(row[2]),
        camera_id=str(row[3]),
        event_type=str(row[4]),
        detected_at=str(row[5]),
        lifecycle_state=str(row[6]),
        revision=_required_int(row[7]),
        failure_reason=_text(row[8]),
        runtime_manifest_sha256=_text(row[9]),
        decision_trace_id=_text(row[10]),
        module_qualified_id=_text(row[11]),
        policy_qualified_id=_text(row[12]),
        primary_clip_id=_text(row[13]),
        primary_artifact_state=_text(row[14]),
        snapshot_artifact_state=_text(row[15]),
        derivative_state=_text(row[16]),
        event_delivery_state=str(row[17]),
        clip_publish_state=_text(row[18]),
        retention_state=_text(row[19]),
        review=_review_from_row(row, 20, str(row[0])),
    )


def _review_from_row(
    row: sqlite3.Row | tuple[object, ...], offset: int, incident_id: str
) -> EvidenceReview | None:
    if row[offset] is None:
        return None
    return EvidenceReview(
        review_id=str(row[offset]),
        incident_id=incident_id,
        clip_id=str(row[offset + 1]),
        version=_required_int(row[offset + 2]),
        actor_id=str(row[offset + 3]),
        reviewed_at=str(row[offset + 4]),
        disposition=ReviewDisposition(str(row[offset + 5])),
        notes=_text(row[offset + 6]),
    )


def _validate_review_input(
    *,
    incident_id: str,
    clip_id: str,
    expected_version: int,
    actor_id: str,
    reviewed_at: str,
    disposition: ReviewDisposition,
    notes: str | None,
) -> None:
    if not incident_id or len(incident_id) > 256 or "\x00" in incident_id:
        raise ValueError("invalid incident_id")
    if not clip_id or len(clip_id) > 256 or "\x00" in clip_id:
        raise ValueError("invalid clip_id")
    if expected_version < 0:
        raise ValueError("expected_version must be non-negative")
    if not actor_id or len(actor_id) > 128 or "\x00" in actor_id:
        raise ValueError("invalid actor_id")
    if len(reviewed_at) > 64 or "\x00" in reviewed_at:
        raise ValueError("invalid reviewed_at")
    try:
        parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid reviewed_at") from error
    if parsed_time.tzinfo is None:
        raise ValueError("invalid reviewed_at")
    if notes is not None and (not notes or len(notes) > 1000 or "\x00" in notes):
        raise ValueError("invalid notes")


def _required_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored review version is invalid")
    return value


def _text(value: object) -> str | None:
    return None if value is None else str(value)


_CURSOR_SEPARATOR = "\x1f"


def _format_list_cursor(detected_at: str, incident_id: str) -> str:
    return f"{detected_at}{_CURSOR_SEPARATOR}{incident_id}"


def _parse_list_cursor(cursor: str) -> tuple[str, str]:
    if not cursor or "\x00" in cursor or _CURSOR_SEPARATOR not in cursor:
        raise ValueError("invalid cursor")
    detected_at, incident_id = cursor.split(_CURSOR_SEPARATOR, 1)
    if not detected_at or not incident_id or "\x00" in detected_at or "\x00" in incident_id:
        raise ValueError("invalid cursor")
    if len(detected_at) > 64 or len(incident_id) > 256:
        raise ValueError("invalid cursor")
    return detected_at, incident_id


__all__ = [
    "CentralEvidenceQuery",
    "CentralEvidenceReviewStore",
    "CentralEvidenceSummary",
    "EvidenceReview",
    "EvidenceReviewConflictError",
    "ReviewDisposition",
]
