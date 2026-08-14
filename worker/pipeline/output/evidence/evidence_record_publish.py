"""Atomic immutable primary-media linkage and resumable lifecycle recovery."""

from __future__ import annotations

import sqlite3
from pathlib import PurePosixPath

from worker.pipeline.output.evidence.evidence_outbox_types import (
    ClipLocalState,
    ClipOutcome,
    EdgeEventId,
)
from worker.pipeline.output.evidence.evidence_record_stage import (
    central_records_available,
    ensure_media_object,
)


def reconcile_primary_records(
    connection: sqlite3.Connection,
    event_ids: tuple[EdgeEventId, ...],
    outcome: ClipOutcome,
) -> None:
    if not central_records_available(connection):
        return
    for event_id in event_ids:
        incident = connection.execute(
            """
            SELECT incident_id, lifecycle_state, revision, primary_clip_id,
                   runtime_manifest_sha256
            FROM evidence_incidents WHERE edge_event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if incident is None:
            continue
        incident_id = str(incident[0])
        if incident[3] is not None and str(incident[3]) != outcome.clip_id:
            raise ValueError("central evidence incident is bound to another primary clip")
        if (
            outcome.runtime_manifest_sha256 is not None
            and incident[4] is not None
            and str(incident[4]) != outcome.runtime_manifest_sha256
        ):
            raise ValueError("primary media runtime manifest differs from incident")
        if outcome.local_state is ClipLocalState.VERIFIED:
            _record_verified(connection, incident_id, outcome)
            _advance_media_ready(connection, incident_id, outcome)
        elif outcome.local_state in {ClipLocalState.UNAVAILABLE, ClipLocalState.CORRUPT}:
            _record_unavailable_if_absent(connection, incident_id, outcome)
            _fail_incident(connection, incident_id, outcome)


def mark_clip_published(
    connection: sqlite3.Connection,
    clip_id: str,
    *,
    updated_at: str,
) -> None:
    if not central_records_available(connection):
        return
    rows = connection.execute(
        "SELECT incident_id, lifecycle_state, revision FROM evidence_incidents "
        "WHERE primary_clip_id = ? ORDER BY incident_id",
        (clip_id,),
    ).fetchall()
    for incident_id, lifecycle, revision in rows:
        if str(lifecycle) != "MEDIA_READY":
            continue
        changed = connection.execute(
            """
            UPDATE evidence_incidents
            SET lifecycle_state = 'PUBLISHED', revision = revision + 1, updated_at = ?
            WHERE incident_id = ? AND revision = ? AND lifecycle_state = 'MEDIA_READY'
            """,
            (updated_at, str(incident_id), int(revision)),
        ).rowcount
        if changed != 1:
            raise sqlite3.IntegrityError("central evidence publication revision changed")


def complete_published_records(connection: sqlite3.Connection, *, updated_at: str) -> int:
    if not central_records_available(connection):
        return 0
    rows = connection.execute(
        """
        SELECT incident.incident_id, incident.revision
        FROM evidence_incidents AS incident
        JOIN evidence_events AS event USING (edge_event_id)
        JOIN evidence_clips AS clip ON clip.clip_id = incident.primary_clip_id
        WHERE incident.lifecycle_state = 'PUBLISHED'
          AND event.delivery_state = 'ACKED'
          AND clip.publish_state = 'PUBLISHED'
          AND NOT EXISTS (
              SELECT 1 FROM derivative_evidence_slots AS derivative
              WHERE derivative.incident_id = incident.incident_id
                AND derivative.state = 'PENDING'
          )
        ORDER BY incident.incident_id
        """
    ).fetchall()
    completed = 0
    for incident_id, revision in rows:
        completed += connection.execute(
            """
            UPDATE evidence_incidents
            SET lifecycle_state = 'COMPLETE', revision = revision + 1, updated_at = ?
            WHERE incident_id = ? AND revision = ? AND lifecycle_state = 'PUBLISHED'
            """,
            (updated_at, str(incident_id), int(revision)),
        ).rowcount
    return completed


def _record_verified(
    connection: sqlite3.Connection,
    incident_id: str,
    outcome: ClipOutcome,
) -> None:
    required = (
        outcome.manifest_path,
        outcome.manifest_sha256,
        outcome.manifest_size_bytes,
        outcome.media_relpath,
        outcome.sha256,
        outcome.size_bytes,
        outcome.mime_type,
    )
    if any(value is None for value in required):
        raise ValueError("verified central evidence is missing immutable media facts")
    assert outcome.media_relpath is not None
    assert outcome.sha256 is not None
    assert outcome.size_bytes is not None
    assert outcome.mime_type is not None
    media_id = ensure_media_object(
        connection,
        sha256=outcome.sha256,
        size_bytes=outcome.size_bytes,
        mime_type=outcome.mime_type,
        relpath=_contained_relpath(outcome.media_relpath),
        created_at=outcome.finalized_at or "UNKNOWN",
    )
    source_preserved = outcome.source_media_json is not None
    source_missing = (
        None
        if source_preserved
        else outcome.source_missing_reason or "SOURCE_PACKET_FACTS_UNAVAILABLE"
    )
    values = (
        incident_id,
        outcome.clip_id,
        outcome.manifest_path,
        outcome.manifest_sha256,
        outcome.manifest_size_bytes,
        media_id,
        outcome.codec,
        outcome.audio_codec,
        outcome.duration_ms,
        outcome.clip_start_at,
        outcome.clip_end_at,
        outcome.finalized_at,
        int(source_preserved),
        source_missing,
        outcome.source_media_json,
        outcome.time_origin_json,
        outcome.truncation_json,
        outcome.finalized_at or "UNKNOWN",
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO evidence_primary_clips (
            incident_id, clip_id, manifest_relpath, manifest_sha256,
            manifest_size_bytes, media_id, codec, audio_codec, duration_ms,
            clip_start_at, clip_end_at, finalized_at, source_packet_preserved,
            source_missing_reason, source_media_json, time_origin_json,
            truncation_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    actual = connection.execute(
        """
        SELECT incident_id, clip_id, manifest_relpath, manifest_sha256,
               manifest_size_bytes, media_id, codec, audio_codec, duration_ms,
               clip_start_at, clip_end_at, finalized_at, source_packet_preserved,
               source_missing_reason, source_media_json, time_origin_json,
               truncation_json, created_at
        FROM evidence_primary_clips WHERE incident_id = ?
        """,
        (incident_id,),
    ).fetchone()
    if actual != values:
        raise ValueError("immutable primary evidence differs from resumed publication")
    _transition_slot(connection, incident_id, "AVAILABLE", media_id, None, outcome.finalized_at)


def _record_unavailable_if_absent(
    connection: sqlite3.Connection,
    incident_id: str,
    outcome: ClipOutcome,
) -> None:
    if (
        connection.execute(
            "SELECT 1 FROM evidence_primary_clips WHERE incident_id = ?", (incident_id,)
        ).fetchone()
        is not None
    ):
        return
    reason = (
        "UNAVAILABLE" if outcome.unavailable_reason is None else outcome.unavailable_reason.value
    )
    connection.execute(
        """
        INSERT INTO evidence_primary_clips (
            incident_id, clip_id, source_packet_preserved, source_missing_reason,
            truncation_json, unavailable_reason, created_at
        ) VALUES (?, ?, 0, ?, ?, ?, ?)
        """,
        (
            incident_id,
            outcome.clip_id,
            outcome.source_missing_reason or reason,
            outcome.truncation_json,
            reason,
            outcome.finalized_at or "UNKNOWN",
        ),
    )


def _advance_media_ready(
    connection: sqlite3.Connection,
    incident_id: str,
    outcome: ClipOutcome,
) -> None:
    row = connection.execute(
        "SELECT lifecycle_state, revision, primary_clip_id FROM evidence_incidents "
        "WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if row is None:
        return
    if str(row[0]) != "STAGING":
        return
    connection.execute(
        """
        UPDATE evidence_incidents
        SET primary_clip_id = ?, lifecycle_state = 'MEDIA_READY',
            revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND revision = ? AND lifecycle_state = 'STAGING'
        """,
        (outcome.clip_id, outcome.finalized_at or "UNKNOWN", incident_id, int(row[1])),
    )


def _fail_incident(
    connection: sqlite3.Connection,
    incident_id: str,
    outcome: ClipOutcome,
) -> None:
    corrupt = outcome.local_state is ClipLocalState.CORRUPT
    reason = _failure_reason(outcome)
    slot_state = "CORRUPT" if corrupt else "UNAVAILABLE"
    _transition_slot(
        connection,
        incident_id,
        slot_state,
        None,
        reason,
        outcome.finalized_at,
    )
    row = connection.execute(
        "SELECT lifecycle_state, revision FROM evidence_incidents WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if row is None or str(row[0]) == "FAILED":
        return
    connection.execute(
        """
        UPDATE evidence_incidents
        SET primary_clip_id = COALESCE(primary_clip_id, ?), lifecycle_state = 'FAILED',
            failure_reason = ?, revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND revision = ?
        """,
        (outcome.clip_id, reason, outcome.finalized_at or "UNKNOWN", incident_id, int(row[1])),
    )


def _transition_slot(
    connection: sqlite3.Connection,
    incident_id: str,
    state: str,
    media_id: str | None,
    reason: str | None,
    updated_at: str | None,
) -> None:
    row = connection.execute(
        "SELECT state, media_id, reason, revision FROM evidence_artifact_slots "
        "WHERE incident_id = ? AND slot_name = 'PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    if row is None:
        raise ValueError("central primary artifact slot is absent")
    target_media = row[1] if media_id is None else media_id
    if (str(row[0]), row[1], row[2]) == (state, target_media, reason):
        return
    connection.execute(
        """
        UPDATE evidence_artifact_slots
        SET state = ?, media_id = ?, reason = ?, revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND slot_name = 'PRIMARY_CLIP' AND revision = ?
        """,
        (state, target_media, reason, updated_at or "UNKNOWN", incident_id, int(row[3])),
    )


def _failure_reason(outcome: ClipOutcome) -> str:
    value = None if outcome.unavailable_reason is None else outcome.unavailable_reason.value
    if value == "MISSING":
        return "MISSING"
    if value == "CORRUPT" or outcome.local_state is ClipLocalState.CORRUPT:
        return "CORRUPT"
    if value == "INTERRUPTED_FINALIZE":
        return "INTERRUPTED"
    return "UNAVAILABLE"


def _contained_relpath(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("central primary media path is not contained")
    return path.as_posix()


__all__ = [
    "complete_published_records",
    "mark_clip_published",
    "reconcile_primary_records",
]
