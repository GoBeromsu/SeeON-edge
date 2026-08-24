"""Backend receipt persistence contract for locally produced clip artifacts.

The schema migrator owns the production table.  This module deliberately owns
no DDL: deployments inject the migrated persistence implementation through
``app.state.artifact_receipt_store``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.clips.catalog import CatalogConflictError, CatalogStore
from backend.app.features.clips.descriptor_files import open_contained_regular_file
from backend.app.features.clips.listing import effective_event_type
from backend.app.features.clips.store import ClipStore

if TYPE_CHECKING:
    from fastapi import FastAPI

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactReceiptConflictError(RuntimeError):
    """A retry changed immutable identity, hash, or size fields."""


class ArtifactReceiptVerificationError(RuntimeError):
    """Declared artifact identity does not match the local regular file."""


class ArtifactReceiptPersistenceError(RuntimeError):
    """No durable backend receipt store is available."""


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    artifact_id: str
    sha256: str
    size_bytes: int
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.artifact_id or "\x00" in self.artifact_id:
            raise ValueError("invalid artifact identity")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("invalid artifact hash")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("invalid artifact size")


@runtime_checkable
class ArtifactReceiptStore(Protocol):
    """Durable compare-or-insert receipt port.

    ``commit`` must return only after its transaction is durable.  It inserts a
    first receipt, returns an identical existing receipt, and raises
    ``ArtifactReceiptConflictError`` when immutable fields differ.
    """

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt: ...

    def get(self, artifact_id: str) -> ArtifactReceipt | None: ...


class CompactArtifactReceiptStore:
    """Commit verified receipt and primary projection to schema-18 authorities."""

    def __init__(self, database_path: Path, clip_root: Path) -> None:
        self._database_path = database_path
        self._clip_store = ClipStore(clip_root)

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt:
        located = self._clip_store.locate_manifest(receipt.artifact_id)
        if located is None:
            raise ArtifactReceiptVerificationError("clip manifest is missing")
        opened = open_contained_regular_file(self._clip_store.root, located.manifest_path)
        try:
            manifest_bytes = opened.handle.read()
        finally:
            opened.handle.close()
        manifest = located.manifest
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_relpath = located.manifest_path.relative_to(self._clip_store.root).as_posix()
        media_path = self._clip_store.resolve_located_video_path(located)
        media_relpath = media_path.relative_to(self._clip_store.root).as_posix()
        connection = open_runtime_database(self._database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                existing = connection.execute(
                    "SELECT media_sha256, media_size_bytes, publish_state FROM clips "
                    "WHERE clip_id = ?",
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
                            receipt.artifact_id, manifest.camera_id,
                            effective_event_type(manifest), manifest.started_at,
                            max(1, round(manifest.duration_s * 1000)), manifest.codec or None,
                            manifest_relpath, media_relpath, manifest_hash, receipt.sha256,
                            len(manifest_bytes), receipt.size_bytes, manifest.started_at,
                            manifest.started_at, manifest.started_at,
                        ),
                    )
                elif (str(existing[0]), int(existing[1])) != (
                    receipt.sha256, receipt.size_bytes
                ):
                    raise ArtifactReceiptConflictError(
                        "immutable artifact receipt fields conflict"
                    )
                elif existing[2] != "PUBLISHED":
                    connection.execute(
                        "UPDATE clips SET publish_state = 'PUBLISHED', published_at = ?, "
                        "last_publish_error_code = NULL, revision = revision + 1, updated_at = ? "
                        "WHERE clip_id = ?",
                        (manifest.started_at, manifest.started_at, receipt.artifact_id),
                    )
                incident = connection.execute(
                    "SELECT incident_id FROM incidents WHERE edge_event_id = ?",
                    (manifest.event_ref,),
                ).fetchone()
                if incident is not None:
                    _commit_primary_artifact(
                        connection, str(incident[0]), receipt, media_relpath,
                        manifest.started_at, manifest.codec or None,
                    )
        finally:
            connection.close()
        return receipt

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


class CatalogArtifactReceiptStore:
    """Receipt adapter over the existing API-owned ``clips`` catalog table."""

    def __init__(self, catalog: CatalogStore) -> None:
        self._catalog = catalog

    @classmethod
    def from_app(cls, app: FastAPI) -> CatalogArtifactReceiptStore:
        from backend.app.features.clips.catalog import get_catalog_store

        catalog = get_catalog_store(app)
        if catalog is None:
            raise ArtifactReceiptPersistenceError("clip catalog is unavailable")
        return cls(catalog)

    def commit(self, receipt: ArtifactReceipt) -> ArtifactReceipt:
        try:
            sha256, size_bytes, accepted = self._catalog.commit_artifact_receipt(
                receipt.artifact_id, receipt.sha256, receipt.size_bytes
            )
        except CatalogConflictError as exc:
            raise ArtifactReceiptConflictError(str(exc)) from exc
        return ArtifactReceipt(receipt.artifact_id, sha256, size_bytes, accepted)

    def get(self, artifact_id: str) -> ArtifactReceipt | None:
        row = self._catalog.artifact_receipt(artifact_id)
        return None if row is None else ArtifactReceipt(artifact_id, *row)


def _commit_primary_artifact(
    connection,
    incident_id: str,
    receipt: ArtifactReceipt,
    media_relpath: str,
    timestamp: str,
    codec: str | None,
) -> None:
    existing = connection.execute(
        "SELECT artifact_id, clip_id, state, content_sha256, size_bytes "
        "FROM artifacts WHERE incident_id = ? AND kind = 'PRIMARY_CLIP'",
        (incident_id,),
    ).fetchone()
    expected = (f"primary:{receipt.artifact_id}", receipt.artifact_id, "AVAILABLE",
                receipt.sha256, receipt.size_bytes)
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
        (incident_id, expected[0], receipt.artifact_id, media_relpath, receipt.sha256,
         receipt.size_bytes, codec, timestamp, timestamp),
    )


def verify_artifact(path: Path, receipt: ArtifactReceipt) -> None:
    """Require a current regular-file size and SHA-256 match before use."""
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact is missing") from exc
    if not path.is_file() or stat_result.st_size != receipt.size_bytes:
        raise ArtifactReceiptVerificationError("artifact size does not match receipt")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactReceiptVerificationError("artifact cannot be verified") from exc
    if digest.hexdigest() != receipt.sha256:
        raise ArtifactReceiptVerificationError("artifact hash does not match receipt")


__all__ = [
    "ArtifactReceipt",
    "ArtifactReceiptConflictError",
    "CatalogArtifactReceiptStore",
    "CompactArtifactReceiptStore",
    "ArtifactReceiptPersistenceError",
    "ArtifactReceiptStore",
    "ArtifactReceiptVerificationError",
    "verify_artifact",
]
