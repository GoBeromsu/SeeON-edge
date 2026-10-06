"""Native clip/primary projection inside a caller-owned, authority-checked transaction.

These helpers do not verify files, acquire a pool lease, publish audit work or
acknowledge a receipt. The receipt owner must serialize clip work before taking
incident locks, and lock multiple incidents in stable order before projecting
artifacts. All locks last through the caller's actual transaction outcome.
"""

from __future__ import annotations

import psycopg

from backend.app.features.clips.listing import effective_event_type
from backend.app.features.evidence.receipt_store import (
    ArtifactReceiptConflictError,
    ClipProjection,
    primary_artifact_id,
)


def lock_clip(connection: psycopg.Connection, clip_id: str) -> None:
    # A missing row cannot be locked. Namespace a clip-key advisory lock by this
    # schema's table OID; hash collisions only add serialization, never identity.
    connection.execute(
        "SELECT pg_advisory_xact_lock('clips'::regclass::oid::integer, hashtext(%s))",
        (clip_id,),
    )


def commit_clip(connection: psycopg.Connection, projection: ClipProjection) -> None:
    receipt, verified, manifest = projection.receipt, projection.verified, projection.manifest
    lock_clip(connection, receipt.artifact_id)
    existing = connection.execute(
        "SELECT media_sha256,media_size_bytes,publish_state FROM clips WHERE clip_id=%s FOR UPDATE",
        (receipt.artifact_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO clips (clip_id,camera_id,event_facet,started_at,duration_ms,"
            "codec,mime_type,manifest_relpath,media_relpath,manifest_sha256,media_sha256,"
            "manifest_size_bytes,media_size_bytes,local_state,publish_state,published_at,"
            "retention_state,revision,created_at,updated_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,'video/mp4',%s,%s,%s,%s,%s,%s,"
            "'AVAILABLE','PUBLISHED',%s,'RETAINED',1,%s,%s)",
            (
                receipt.artifact_id,
                manifest.camera_id,
                effective_event_type(manifest),
                manifest.started_at,
                max(1, round(manifest.duration_s * 1000)),
                manifest.codec or None,
                projection.manifest_relpath,
                projection.media_relpath,
                projection.manifest_hash,
                verified.sha256,
                projection.manifest_size,
                verified.size_bytes,
                manifest.started_at,
                manifest.started_at,
                manifest.started_at,
            ),
        )
        return
    if (existing[0], existing[1]) != (verified.sha256, verified.size_bytes):
        raise ArtifactReceiptConflictError("immutable artifact receipt fields conflict")
    if existing[2] != "PUBLISHED":
        connection.execute(
            "UPDATE clips SET publish_state='PUBLISHED',published_at=%s,"
            "last_publish_error_code=NULL,revision=revision+1,updated_at=%s "
            "WHERE clip_id=%s",
            (manifest.started_at, manifest.started_at, receipt.artifact_id),
        )


def commit_primary_artifact(
    connection: psycopg.Connection,
    incident_id: str,
    edge_event_id: str,
    projection: ClipProjection,
    *,
    timestamp: str,
) -> None:
    _lock_incident(connection, incident_id)
    if not projection.manifest.video_available:
        _commit_primary_failure(
            connection,
            incident_id,
            projection.manifest.video_error or "PRIMARY_UNAVAILABLE",
            timestamp,
        )
        return
    clip_id, verified = projection.receipt.artifact_id, projection.verified
    existing = connection.execute(
        "SELECT clip_id,state,content_sha256,size_bytes FROM artifacts "
        "WHERE incident_id=%s AND kind='PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    expected = (clip_id, "AVAILABLE", verified.sha256, verified.size_bytes)
    if existing is not None:
        if tuple(existing) != expected:
            raise ArtifactReceiptConflictError("primary clip artifact conflicts")
        _complete_incident(connection, incident_id, timestamp)
        return
    artifact_id = primary_artifact_id(clip_id, edge_event_id)
    owner = connection.execute(
        "SELECT incident_id FROM artifacts WHERE artifact_id=%s", (artifact_id,)
    ).fetchone()
    if owner is not None:
        raise ArtifactReceiptConflictError("primary clip artifact identity conflicts")
    try:
        connection.execute(
            "INSERT INTO artifacts (incident_id,kind,artifact_id,clip_id,state,contained_relpath,"
            "content_sha256,size_bytes,mime_type,codec,revision,created_at,updated_at) "
            "VALUES (%s,'PRIMARY_CLIP',%s,%s,'AVAILABLE',%s,%s,%s,'video/mp4',%s,1,%s,%s)",
            (
                incident_id,
                artifact_id,
                clip_id,
                projection.media_relpath,
                verified.sha256,
                verified.size_bytes,
                projection.manifest.codec or None,
                projection.manifest.started_at,
                projection.manifest.started_at,
            ),
        )
    except psycopg.errors.UniqueViolation:
        # Another incident can claim this identity after the read above.
        # Let the outer owner roll back; do not retry or create a savepoint.
        raise ArtifactReceiptConflictError("primary clip artifact identity conflicts") from None
    _complete_incident(connection, incident_id, timestamp)


def commit_unavailable_primary(
    connection: psycopg.Connection, incident_id: str, reason: str, timestamp: str
) -> None:
    _lock_incident(connection, incident_id)
    _commit_primary_failure(connection, incident_id, reason, timestamp)


def _lock_incident(connection: psycopg.Connection, incident_id: str) -> None:
    connection.execute(
        "SELECT incident_id FROM incidents WHERE incident_id=%s FOR UPDATE", (incident_id,)
    ).fetchone()


def _complete_incident(connection: psycopg.Connection, incident_id: str, timestamp: str) -> None:
    connection.execute(
        "UPDATE incidents SET lifecycle_state='COMPLETE',failure_reason=NULL,"
        "revision=revision+1,updated_at=%s WHERE incident_id=%s AND lifecycle_state='OPEN'",
        (timestamp, incident_id),
    )


def _commit_primary_failure(
    connection: psycopg.Connection, incident_id: str, reason: str, timestamp: str
) -> None:
    failure_reason = reason[:64]
    existing = connection.execute(
        "SELECT clip_id,state,reason FROM artifacts WHERE incident_id=%s AND kind='PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    expected = (None, "UNAVAILABLE", failure_reason)
    if existing is None:
        connection.execute(
            "INSERT INTO artifacts "
            "(incident_id,kind,clip_id,state,reason,revision,created_at,updated_at) "
            "VALUES (%s,'PRIMARY_CLIP',NULL,'UNAVAILABLE',%s,1,%s,%s)",
            (incident_id, failure_reason, timestamp, timestamp),
        )
    elif tuple(existing) != expected:
        raise ArtifactReceiptConflictError("primary clip artifact conflicts")
    connection.execute(
        "UPDATE incidents SET lifecycle_state='FAILED',failure_reason=%s,revision=revision+1,"
        "updated_at=%s WHERE incident_id=%s AND lifecycle_state='OPEN'",
        (failure_reason, timestamp, incident_id),
    )


__all__ = ["commit_clip", "commit_primary_artifact", "commit_unavailable_primary", "lock_clip"]
