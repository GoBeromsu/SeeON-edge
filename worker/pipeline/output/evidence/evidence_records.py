"""Worker-owned authoritative central evidence record service."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from shared.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from shared.edge_db.reviews import EvidenceReview, ReviewDisposition
from worker.pipeline.output.evidence.evidence_record_models import (
    ArtifactState,
    EvidenceLifecycle,
    EvidenceRecord,
    EvidenceRecordConflictError,
    PrimaryEvidence,
)


class EvidenceRecordStore:
    """DDL-free lifecycle writer and privacy-bounded incident reader."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def transition(
        self,
        incident_id: str,
        *,
        expected_revision: int,
        target: EvidenceLifecycle,
        updated_at: str,
        failure_reason: str | None = None,
    ) -> EvidenceRecord:
        if (target is EvidenceLifecycle.FAILED) != (failure_reason is not None):
            raise ValueError("FAILED transitions require one explicit failure reason")
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            try:
                with write_transaction(connection):
                    changed = connection.execute(
                        """
                        UPDATE evidence_incidents
                        SET lifecycle_state = ?, revision = revision + 1,
                            failure_reason = ?, updated_at = ?
                        WHERE incident_id = ? AND revision = ?
                        """,
                        (
                            target.value,
                            failure_reason,
                            updated_at,
                            incident_id,
                            expected_revision,
                        ),
                    ).rowcount
                    if changed != 1:
                        raise EvidenceRecordConflictError(
                            incident_id, "revision changed or incident is absent"
                        )
            except sqlite3.IntegrityError as error:
                detail = str(error)
                if "illegal central evidence lifecycle" in detail:
                    detail = "illegal evidence lifecycle transition"
                raise EvidenceRecordConflictError(incident_id, detail) from error
        finally:
            connection.close()
        record = self.get(incident_id)
        if record is None:
            raise EvidenceRecordConflictError(incident_id, "incident disappeared")
        return record

    def request_annotated_derivative(
        self,
        incident_id: str,
        *,
        expected_revision: int,
        updated_at: str,
    ) -> EvidenceRecord:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            with write_transaction(connection):
                existing = connection.execute(
                    "SELECT state FROM derivative_evidence_slots "
                    "WHERE incident_id = ? AND derivative_kind = 'ANNOTATED_CLIP'",
                    (incident_id,),
                ).fetchone()
                if existing == ("PENDING",):
                    pass
                elif existing is not None:
                    raise EvidenceRecordConflictError(
                        incident_id, "annotated derivative slot is already resolved"
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO derivative_evidence_slots (
                            incident_id, derivative_kind, state, created_at, updated_at
                        ) VALUES (?, 'ANNOTATED_CLIP', 'PENDING', ?, ?)
                        """,
                        (incident_id, updated_at, updated_at),
                    )
                    changed = connection.execute(
                        """
                        UPDATE evidence_incidents
                        SET lifecycle_state = 'DERIVATIVE_PENDING',
                            revision = revision + 1, updated_at = ?
                        WHERE incident_id = ? AND revision = ?
                          AND lifecycle_state = 'PUBLISHED'
                        """,
                        (updated_at, incident_id, expected_revision),
                    ).rowcount
                    if changed != 1:
                        raise EvidenceRecordConflictError(
                            incident_id, "revision changed or incident is not published"
                        )
        finally:
            connection.close()
        record = self.get(incident_id)
        if record is None:
            raise EvidenceRecordConflictError(incident_id, "incident disappeared")
        return record

    def get(self, identity: str) -> EvidenceRecord | None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.WORKER)
        try:
            row = connection.execute(_RECORD_QUERY, (identity, identity)).fetchone()
            if row is None:
                return None
            primary = _primary_from_row(row)
            return EvidenceRecord(
                incident_id=str(row[0]),
                schema_version=int(row[1]),
                edge_event_id=str(row[2]),
                camera_id=str(row[3]),
                event_type=str(row[4]),
                detected_at=str(row[5]),
                runtime_manifest_sha256=_text(row[6]),
                decision_trace_id=_text(row[7]),
                module_qualified_id=_text(row[8]),
                policy_qualified_id=_text(row[9]),
                effective_policy_id=_text(row[10]),
                provenance_state=str(row[11]),
                provenance_missing_reason=_text(row[12]),
                lifecycle=EvidenceLifecycle(str(row[13])),
                revision=int(row[14]),
                failure_reason=_text(row[15]),
                primary_state=ArtifactState(str(row[16] or "PENDING")),
                snapshot_state=ArtifactState(str(row[17] or "PENDING")),
                derivative_state=(None if row[18] is None else ArtifactState(str(row[18]))),
                event_delivery_state=str(row[19]),
                event_attempt_count=int(row[20]),
                clip_publish_state=_text(row[21]),
                clip_publish_attempt_count=(None if row[22] is None else int(row[22])),
                retention_state=_text(row[39]),
                primary=primary,
                review=_review_from_row(row),
            )
        finally:
            connection.close()


_RECORD_QUERY = """
SELECT incident.incident_id, incident.record_schema_version, incident.edge_event_id,
       incident.camera_id, incident.event_type, incident.detected_at,
       incident.runtime_manifest_sha256, incident.decision_trace_id,
       incident.module_qualified_id, incident.policy_qualified_id,
       incident.effective_policy_id, incident.provenance_state,
       incident.provenance_missing_reason, incident.lifecycle_state,
       incident.revision, incident.failure_reason,
       primary_slot.state, snapshot_slot.state, derivative.state,
       event.delivery_state, event.attempt_count,
       clip.publish_state, clip.publish_attempt_count,
       primary_record.clip_id, media.content_sha256, media.size_bytes,
       media.contained_relpath, primary_record.manifest_sha256,
       primary_record.manifest_size_bytes, primary_record.manifest_relpath,
       primary_record.codec, primary_record.audio_codec, primary_record.duration_ms,
       primary_record.source_packet_preserved, primary_record.source_missing_reason,
       primary_record.source_media_json, primary_record.time_origin_json,
       primary_record.truncation_json, primary_record.unavailable_reason,
       retention.state, review.review_id, review.clip_id, review.review_version,
       review.actor_id, review.reviewed_at, review.disposition, review.notes
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
LEFT JOIN evidence_primary_clips AS primary_record USING (incident_id)
LEFT JOIN evidence_media_objects AS media ON media.media_id = primary_record.media_id
LEFT JOIN evidence_clips AS clip ON clip.clip_id = incident.primary_clip_id
LEFT JOIN evidence_retention_states AS retention
  ON retention.clip_id = incident.primary_clip_id
LEFT JOIN control_evidence_review_state AS review_state
  ON review_state.incident_id = incident.incident_id
LEFT JOIN control_evidence_review_revisions AS review
  ON review.incident_id = review_state.incident_id
 AND review.clip_id = review_state.clip_id
 AND review.review_version = review_state.current_version
WHERE incident.incident_id = ? OR incident.edge_event_id = ?
"""


def _review_from_row(row: sqlite3.Row | tuple[object, ...]) -> EvidenceReview | None:
    if row[40] is None:
        return None
    return EvidenceReview(
        review_id=str(row[40]),
        incident_id=str(row[0]),
        clip_id=str(row[41]),
        version=_required_int(row[42]),
        actor_id=str(row[43]),
        reviewed_at=str(row[44]),
        disposition=ReviewDisposition(str(row[45])),
        notes=_text(row[46]),
    )


def _primary_from_row(row: sqlite3.Row | tuple[object, ...]) -> PrimaryEvidence | None:
    if row[23] is None:
        return None
    source_media = _json_object(row[35])
    time_origin = _json_object(row[36])
    truncations = json.loads(str(row[37]))
    if not isinstance(truncations, list) or not all(
        isinstance(value, str) for value in truncations
    ):
        raise ValueError("stored evidence truncation facts are invalid")
    return PrimaryEvidence(
        clip_id=str(row[23]),
        media_sha256=_text(row[24]),
        media_size_bytes=_optional_int(row[25]),
        media_relpath=_text(row[26]),
        manifest_sha256=_text(row[27]),
        manifest_size_bytes=_optional_int(row[28]),
        manifest_relpath=_text(row[29]),
        codec=_text(row[30]),
        audio_codec=_text(row[31]),
        duration_ms=_optional_int(row[32]),
        source_packet_preserved=bool(row[33]),
        source_missing_reason=_text(row[34]),
        source_media=source_media,
        time_origin=time_origin,
        truncation_reasons=tuple(truncations),
        unavailable_reason=_text(row[38]),
    )


def _json_object(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise TypeError("stored evidence object facts are invalid")
    return parsed


def _required_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored review version is invalid")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("stored evidence integer fact is invalid")
    return value


def _text(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ArtifactState",
    "EvidenceLifecycle",
    "EvidenceRecord",
    "EvidenceRecordConflictError",
    "EvidenceRecordStore",
    "PrimaryEvidence",
]
