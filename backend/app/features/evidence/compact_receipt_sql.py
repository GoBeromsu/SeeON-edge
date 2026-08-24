"""Schema-18 SQL projection for descriptor-verified clip receipts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

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
    projection: ClipProjection,
) -> None:
    clip_id = projection.receipt.artifact_id
    verified = projection.verified
    existing = connection.execute(
        "SELECT artifact_id, clip_id, state, content_sha256, size_bytes "
        "FROM artifacts WHERE incident_id = ? AND kind = 'PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    expected = (
        f"primary:{clip_id}",
        clip_id,
        "AVAILABLE",
        verified.sha256,
        verified.size_bytes,
    )
    if existing is not None:
        if tuple(existing) != expected:
            raise ArtifactReceiptConflictError("primary clip artifact conflicts")
        return
    connection.execute(
        """
        INSERT INTO artifacts (
            incident_id, kind, artifact_id, clip_id, state, contained_relpath,
            content_sha256, size_bytes, mime_type, codec, revision, created_at, updated_at
        ) VALUES (?, 'PRIMARY_CLIP', ?, ?, 'AVAILABLE', ?, ?, ?, 'video/mp4', ?, 1, ?, ?)
        """,
        (
            incident_id,
            expected[0],
            clip_id,
            projection.media_relpath,
            verified.sha256,
            verified.size_bytes,
            projection.manifest.codec or None,
            projection.manifest.started_at,
            projection.manifest.started_at,
        ),
    )


__all__ = ["ClipProjection", "commit_clip", "commit_primary_artifact"]
