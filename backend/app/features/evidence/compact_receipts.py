"""Descriptor-bound schema-18 clip receipt persistence."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.clips.descriptor_files import (
    OpenedRegularFile,
    open_contained_regular_file,
)
from backend.app.features.clips.listing import effective_event_type
from backend.app.features.clips.manifest import ClipManifest
from backend.app.features.clips.store import ClipStore
from backend.app.features.evidence.receipt_store import (
    ArtifactReceipt,
    ArtifactReceiptConflictError,
    ArtifactReceiptVerificationError,
    VerifiedArtifact,
    verified_artifact,
)


@dataclass(frozen=True, slots=True)
class _ClipProjection:
    receipt: ArtifactReceipt
    verified: VerifiedArtifact
    manifest: ClipManifest
    manifest_relpath: str
    media_relpath: str
    manifest_hash: str
    manifest_size: int


class CompactArtifactReceiptStore:
    """Bind compact publication facts to one verified media descriptor."""

    def __init__(self, database_path: Path, clip_root: Path) -> None:
        self._database_path = database_path
        self._clip_store = ClipStore(clip_root)

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt:
        located = self._clip_store.locate_manifest(receipt.artifact_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        opened = self._open_media(located.manifest_path.parent / "clip.mp4")
        try:
            return self.commit_verified(receipt, verified_artifact(opened.handle))
        finally:
            opened.handle.close()

    def commit_verified(
        self,
        receipt: ArtifactReceipt,
        route_verified: VerifiedArtifact,
    ) -> ArtifactReceipt:
        descriptor_verified = verified_artifact(route_verified.handle)
        if descriptor_verified.identity != route_verified.identity:
            raise ArtifactReceiptVerificationError("verified descriptor identity changed")
        if (descriptor_verified.sha256, descriptor_verified.size_bytes) != (
            receipt.sha256,
            receipt.size_bytes,
        ):
            raise ArtifactReceiptVerificationError("declared receipt differs from media bytes")
        located = self._clip_store.locate_manifest(receipt.artifact_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        current = self._open_media(located.manifest_path.parent / "clip.mp4")
        try:
            current_stat = os.fstat(current.handle.fileno())
            if (current_stat.st_dev, current_stat.st_ino) != descriptor_verified.identity:
                raise ArtifactReceiptVerificationError("clip media inode changed")
        finally:
            current.handle.close()
        manifest_opened = open_contained_regular_file(
            self._clip_store.root,
            located.manifest_path,
        )
        try:
            manifest_bytes = manifest_opened.handle.read()
        finally:
            manifest_opened.handle.close()
        manifest = located.manifest
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_relpath = located.manifest_path.relative_to(self._clip_store.root).as_posix()
        media_relpath = current.path.relative_to(self._clip_store.root).as_posix()
        projection = _ClipProjection(
            receipt=receipt,
            verified=descriptor_verified,
            manifest=manifest,
            manifest_relpath=manifest_relpath,
            media_relpath=media_relpath,
            manifest_hash=manifest_hash,
            manifest_size=len(manifest_bytes),
        )
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                self._commit_clip(connection, projection)
                incident = connection.execute(
                    "SELECT incident_id FROM incidents WHERE edge_event_id = ?",
                    (manifest.event_ref,),
                ).fetchone()
                if incident is not None:
                    _commit_primary_artifact(connection, str(incident[0]), projection)
        finally:
            connection.close()
        return ArtifactReceipt(
            receipt.artifact_id,
            descriptor_verified.sha256,
            descriptor_verified.size_bytes,
        )

    def _open_media(self, path: Path) -> OpenedRegularFile:
        try:
            path_stat = os.lstat(path)
            if not stat.S_ISREG(path_stat.st_mode):
                raise ArtifactReceiptVerificationError("clip media pathname is not regular")
            opened = open_contained_regular_file(self._clip_store.root, path)
            opened_stat = os.fstat(opened.handle.fileno())
        except (FileNotFoundError, OSError, ValueError) as error:
            raise ArtifactReceiptVerificationError("clip media is missing") from error
        if (path_stat.st_dev, path_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            opened.handle.close()
            raise ArtifactReceiptVerificationError("clip media pathname changed")
        return opened

    def _commit_clip(
        self,
        connection: sqlite3.Connection,
        projection: _ClipProjection,
    ) -> None:
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
        if (str(existing[0]), int(existing[1])) != (
            verified.sha256,
            verified.size_bytes,
        ):
            raise ArtifactReceiptConflictError("immutable artifact receipt fields conflict")
        if existing[2] != "PUBLISHED":
            connection.execute(
                "UPDATE clips SET publish_state = 'PUBLISHED', published_at = ?, "
                "last_publish_error_code = NULL, revision = revision + 1, updated_at = ? "
                "WHERE clip_id = ?",
                (manifest.started_at, manifest.started_at, receipt.artifact_id),
            )

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(
                "SELECT media_sha256, media_size_bytes FROM clips "
                "WHERE clip_id = ? AND publish_state = 'PUBLISHED'",
                (artifact_id,),
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else ArtifactReceipt(artifact_id, str(row[0]), int(row[1]))


def _commit_primary_artifact(
    connection: sqlite3.Connection,
    incident_id: str,
    projection: _ClipProjection,
) -> None:
    clip_id = projection.receipt.artifact_id
    verified = projection.verified
    existing = connection.execute(
        "SELECT artifact_id, clip_id, state, content_sha256, size_bytes "
        "FROM artifacts WHERE incident_id = ? AND kind = 'PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    expected = (f"primary:{clip_id}", clip_id, "AVAILABLE", verified.sha256,
                verified.size_bytes)
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


__all__ = ["CompactArtifactReceiptStore"]
