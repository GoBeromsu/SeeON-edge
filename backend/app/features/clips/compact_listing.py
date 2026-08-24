"""Manifest rebuild and keyset reads backed only by schema-18 ``clips``."""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database, write_transaction
from backend.app.features.clips.descriptor_files import open_contained_regular_file
from backend.app.features.clips.listing import effective_event_type
from backend.app.features.clips.manifest import ClipManifest
from backend.app.features.clips.store import ClipStore, DuplicateClipIdError


class CompactClipConflictError(RuntimeError):
    """A clip ID resolves to changed immutable manifest/media identity."""


@dataclass(frozen=True, slots=True)
class CompactClipPage:
    manifests: tuple[ClipManifest, ...]
    total: int
    has_more: bool
    next_cursor: str | None
    event_type_counts: dict[str, int]


class CompactClipListing:
    """Rebuild valid manifest facts and page the compact clip authority."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def rebuild_and_page(
        self,
        store: ClipStore,
        *,
        camera_id: str | None,
        event_type: str | None,
        limit: int,
        cursor: str | None,
    ) -> CompactClipPage:
        manifests = self._rebuild(store)
        params: list[str | int] = []
        predicates: list[str] = []
        if camera_id is not None:
            predicates.append("camera_id = ?")
            params.append(camera_id)
        if event_type is not None:
            predicates.append("event_facet = ?")
            params.append(event_type)
        if cursor is not None:
            started_at, clip_id = _parse_cursor(cursor)
            predicates.append("(started_at < ? OR (started_at = ? AND clip_id < ?))")
            params.extend((started_at, started_at, clip_id))
        where = "" if not predicates else " WHERE " + " AND ".join(predicates)
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            rows = connection.execute(
                "SELECT clip_id, started_at FROM clips" + where
                + " ORDER BY started_at DESC, clip_id DESC LIMIT ?",
                (*params, limit + 1),
            ).fetchall()
            count_params: list[str] = []
            count_where = ""
            if camera_id is not None:
                count_where = " WHERE camera_id = ?"
                count_params.append(camera_id)
            total = int(connection.execute(
                "SELECT count(*) FROM clips" + count_where, tuple(count_params)
            ).fetchone()[0])
            facets = dict(connection.execute(
                "SELECT event_facet, count(*) FROM clips" + count_where
                + " GROUP BY event_facet ORDER BY event_facet",
                tuple(count_params),
            ).fetchall())
        finally:
            connection.close()
        by_id = {manifest.clip_id: manifest for manifest in manifests}
        visible = tuple(by_id[str(row[0])] for row in rows[:limit] if str(row[0]) in by_id)
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = _format_cursor(str(last[1]), str(last[0]))
        return CompactClipPage(visible, total, len(rows) > limit, next_cursor, facets)

    def _rebuild(self, store: ClipStore) -> tuple[ClipManifest, ...]:
        manifests = tuple(store.list_manifests())
        prepared: list[tuple[ClipManifest, tuple[str | int | None, ...]]] = []
        seen: set[str] = set()
        for manifest in manifests:
            if manifest.clip_id in seen:
                store.locate_manifest(manifest.clip_id)
                raise DuplicateClipIdError(manifest.clip_id, ())
            seen.add(manifest.clip_id)
            located = store.locate_manifest(manifest.clip_id)
            if located is None or not _valid_timestamp(manifest.started_at):
                continue
            manifest_path = located.manifest_path
            manifest_hash, manifest_size = _hash_regular(store.root, manifest_path)
            manifest_relpath = manifest_path.relative_to(store.root).as_posix()
            try:
                media_path = store.resolve_located_video_path(located)
                media_hash, media_size = _hash_regular(store.root, media_path)
            except (FileNotFoundError, ValueError):
                media_path = None
                media_hash = None
                media_size = None
            local_state = "AVAILABLE" if media_path is not None else "CORRUPT"
            local_reason = None if media_path is not None else "MEDIA_MISSING"
            media_relpath = (
                None if media_path is None else media_path.relative_to(store.root).as_posix()
            )
            thumbnail_path = manifest_path.parent / "thumbnail.jpg"
            try:
                thumbnail_hash, thumbnail_size = _hash_regular(store.root, thumbnail_path)
                thumbnail_relpath = thumbnail_path.relative_to(store.root).as_posix()
            except FileNotFoundError:
                thumbnail_hash = None
                thumbnail_size = None
                thumbnail_relpath = None
            prepared.append((manifest, (
                manifest.clip_id, manifest.camera_id, effective_event_type(manifest),
                manifest.started_at, max(1, round(manifest.duration_s * 1000)),
                manifest.codec or None, "video/mp4", manifest_relpath, media_relpath,
                thumbnail_relpath, manifest_hash, media_hash, thumbnail_hash,
                manifest_size, media_size, thumbnail_size, local_state,
                local_reason, manifest.started_at, manifest.started_at,
            )))
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            with write_transaction(connection):
                for manifest, values in prepared:
                    existing = connection.execute(
                        "SELECT manifest_sha256, media_sha256, media_size_bytes FROM clips "
                        "WHERE clip_id = ?", (manifest.clip_id,),
                    ).fetchone()
                    identity = (values[10], values[11], values[14])
                    if existing is not None and tuple(existing) != identity:
                        raise CompactClipConflictError(
                            f"clip {manifest.clip_id} immutable content changed"
                        )
                    if existing is None:
                        connection.execute(_INSERT_CLIP, values)
        finally:
            connection.close()
        return tuple(manifest for manifest, _values in prepared)


_INSERT_CLIP = """
INSERT INTO clips (
    clip_id, camera_id, event_facet, started_at, duration_ms, codec, mime_type,
    manifest_relpath, media_relpath, thumbnail_relpath, manifest_sha256,
    media_sha256, thumbnail_sha256, manifest_size_bytes, media_size_bytes,
    thumbnail_size_bytes, local_state, local_reason, publish_state,
    retention_state, revision, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          'WAITING', 'RETAINED', 1, ?, ?)
"""


def _hash_regular(root: Path, path: Path) -> tuple[str, int]:
    opened = open_contained_regular_file(root, path)
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    finally:
        opened.handle.close()
    return digest.hexdigest(), size


def _valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and value.endswith("Z") and 20 <= len(value) <= 30


def _format_cursor(started_at: str, clip_id: str) -> str:
    return base64.urlsafe_b64encode(f"{started_at}\0{clip_id}".encode()).decode()


def _parse_cursor(cursor: str) -> tuple[str, str]:
    try:
        decoded = base64.b64decode(cursor, altchars=b"-_", validate=True).decode()
        started_at, clip_id = decoded.split("\0", 1)
    except (ValueError, UnicodeDecodeError, binascii.Error) as error:
        raise ValueError("invalid cursor") from error
    if not started_at or not clip_id or len(started_at) > 30 or len(clip_id) > 128:
        raise ValueError("invalid cursor")
    return started_at, clip_id


__all__ = ["CompactClipConflictError", "CompactClipListing", "CompactClipPage"]
