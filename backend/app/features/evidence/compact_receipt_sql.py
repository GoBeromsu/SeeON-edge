"""Schema-18 SQL projection for descriptor-verified clip receipts."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

from backend.app.edge_db.configuration import utc_now
from backend.app.features.clips.listing import effective_event_type
from backend.app.features.clips.manifest import ClipManifest
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    VerifiedArtifact,
)


@dataclass(frozen=True, slots=True)
class ClipProjection:
    receipt: ArtifactReceipt
    verified: VerifiedArtifact
    manifest: ClipManifest
    manifest_relpath: str
    media_relpath: str
    manifest_hash: str
    manifest_size: int


def commit_clip(connection: sqlite3.Connection, projection: ClipProjection) -> None:
    receipt = projection.receipt
    verified = projection.verified
    manifest = projection.manifest
    existing = connection.execute(
        "SELECT media_sha256, media_size_bytes, publish_state FROM clips WHERE clip_id = ?",
        (receipt.artifact_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            """
            INSERT INTO clips (
                clip_id, camera_id, event_facet, started_at, duration_ms,
                codec, mime_type, manifest_relpath, media_relpath,
                manifest_sha256, media_sha256, manifest_size_bytes,
                media_size_bytes, local_state, publish_state, published_at,
                retention_state, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'video/mp4', ?, ?, ?, ?, ?, ?,
                      'AVAILABLE', 'PUBLISHED', ?, 'RETAINED', 1, ?, ?)
            """,
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
    if (str(existing[0]), int(existing[1])) != (verified.sha256, verified.size_bytes):
        raise ArtifactReceiptConflictError("immutable artifact receipt fields conflict")
    if existing[2] != "PUBLISHED":
        connection.execute(
            "UPDATE clips SET publish_state = 'PUBLISHED', published_at = ?, "
            "last_publish_error_code = NULL, revision = revision + 1, updated_at = ? "
            "WHERE clip_id = ?",
            (manifest.started_at, manifest.started_at, receipt.artifact_id),
        )


def commit_primary_artifact(
    connection: sqlite3.Connection,
    incident_id: str,
    edge_event_id: str,
    projection: ClipProjection,
    *,
    timestamp: str | None = None,
) -> None:
    timestamp = utc_now() if timestamp is None else timestamp
    clip_id = projection.receipt.artifact_id
    verified = projection.verified
    if not projection.manifest.video_available:
        _commit_primary_failure(
            connection,
            incident_id,
            projection.manifest.video_error or "PRIMARY_UNAVAILABLE",
            timestamp,
        )
        return
    existing = connection.execute(
        "SELECT clip_id, state, content_sha256, size_bytes "
        "FROM artifacts WHERE incident_id = ? AND kind = 'PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    artifact_id = _primary_artifact_id(clip_id, edge_event_id)
    expected = (
        clip_id,
        "AVAILABLE",
        verified.sha256,
        verified.size_bytes,
    )
    if existing is not None:
        if tuple(existing) != expected:
            raise ArtifactReceiptConflictError("primary clip artifact conflicts")
        _complete_incident(connection, incident_id, timestamp)
        return
    owner = connection.execute(
        "SELECT incident_id FROM artifacts WHERE artifact_id = ?", (artifact_id,)
    ).fetchone()
    if owner is not None:
        raise ArtifactReceiptConflictError("primary clip artifact identity conflicts")
    connection.execute(
        """
        INSERT INTO artifacts (
            incident_id, kind, artifact_id, clip_id, state, contained_relpath,
            content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at
        ) VALUES (?, 'PRIMARY_CLIP', ?, ?, 'AVAILABLE', ?, ?, ?, 'video/mp4', ?, 1, ?, ?)
        """,
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
    _complete_incident(connection, incident_id, timestamp)


def commit_unavailable_primary(
    connection: sqlite3.Connection,
    incident_id: str,
    reason: str,
    timestamp: str,
) -> None:
    _commit_primary_failure(connection, incident_id, reason, timestamp)


def _primary_artifact_id(clip_id: str, edge_event_id: str) -> str:
    digest = hashlib.sha256(f"{clip_id}\x1f{edge_event_id}".encode()).hexdigest()[:32]
    return f"primary:{digest}"


def _complete_incident(connection: sqlite3.Connection, incident_id: str, timestamp: str) -> None:
    connection.execute(
        """
        UPDATE incidents
        SET lifecycle_state = 'COMPLETE', failure_reason = NULL,
            revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND lifecycle_state = 'OPEN'
        """,
        (timestamp, incident_id),
    )


def _commit_primary_failure(
    connection: sqlite3.Connection,
    incident_id: str,
    reason: str,
    timestamp: str,
) -> None:
    failure_reason = reason[:64]
    existing = connection.execute(
        "SELECT clip_id, state, reason FROM artifacts "
        "WHERE incident_id = ? AND kind = 'PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    expected = (None, "UNAVAILABLE", failure_reason)
    if existing is None:
        connection.execute(
            """
            INSERT INTO artifacts (
                incident_id, kind, clip_id, state, reason, revision, created_at, updated_at
            ) VALUES (?, 'PRIMARY_CLIP', NULL, 'UNAVAILABLE', ?, 1, ?, ?)
            """,
            (incident_id, failure_reason, timestamp, timestamp),
        )
    elif tuple(existing) != expected:
        raise ArtifactReceiptConflictError("primary clip artifact conflicts")
    connection.execute(
        """
        UPDATE incidents
        SET lifecycle_state = 'FAILED', failure_reason = ?,
            revision = revision + 1, updated_at = ?
        WHERE incident_id = ? AND lifecycle_state = 'OPEN'
        """,
        (failure_reason, timestamp, incident_id),
    )


__all__ = [
    "ClipProjection",
    "commit_clip",
    "commit_primary_artifact",
    "commit_unavailable_primary",
]
