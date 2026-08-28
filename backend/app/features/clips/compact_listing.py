"""Transactional manifest reconciliation over schema-18 ``clips``."""

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
class CompactClipQuery:
    camera_id: str | None
    event_type: str | None
    limit: int
    cursor: str | None


@dataclass(frozen=True, slots=True)
class CompactClipPage:
    manifests: tuple[ClipManifest, ...]
    total: int
    has_more: bool
    next_cursor: str | None
    event_type_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _PreparedClip:
    manifest: ClipManifest
    values: tuple[str | int | None, ...]
    identity: tuple[str, str | None, int | None]


class CompactClipListing:
    """Reconcile verified filesystem facts and read one matching keyset page."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def rebuild_and_page(
        self,
        store: ClipStore,
        query: CompactClipQuery,
    ) -> CompactClipPage:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            prepared = (
                None if query.cursor is not None
                else _prepare(store, _known_media(connection))
            )
            with write_transaction(connection):
                if prepared is not None:
                    _reconcile(connection, prepared)
                page_rows, total, facets = _page_rows(connection, query)
        finally:
            connection.close()
        prepared_by_id = (
            {} if prepared is None
            else {item.manifest.clip_id: item.manifest for item in prepared}
        )
        visible: list[ClipManifest] = []
        for row in page_rows[: query.limit]:
            clip_id = str(row[0])
            manifest = prepared_by_id.get(clip_id)
            if manifest is None:
                located = store.locate_manifest(clip_id)
                if located is None:
                    raise CompactClipConflictError(
                        f"visible clip {clip_id} disappeared during pagination"
                    )
                manifest = located.manifest
            visible.append(manifest)
        next_cursor = None
        if len(page_rows) > query.limit:
            last = page_rows[query.limit - 1]
            next_cursor = _format_cursor(str(last[1]), str(last[0]))
        return CompactClipPage(
            manifests=tuple(visible),
            total=total,
            has_more=len(page_rows) > query.limit,
            next_cursor=next_cursor,
            event_type_counts=facets,
        )


_KnownMedia = dict[str, tuple[str | None, str | None, int | None]]


def _known_media(connection) -> _KnownMedia:
    """Catalogued ``(media_relpath, media_sha256, media_size_bytes)`` per clip.

    Read once per rebuild so that media already verified by an earlier
    reconcile is not re-hashed on every listing: with thousands of clips the
    full re-read costs minutes of I/O per ``GET /clips``, which starved the
    dashboard of any listing at all. A clip whose media path or byte size
    differs from the catalogued row is still hashed in full below, so a
    changed or replaced file keeps tripping the immutable-content conflict.
    """
    rows = connection.execute(
        "SELECT clip_id, media_relpath, media_sha256, media_size_bytes FROM clips "
        "WHERE media_sha256 IS NOT NULL"
    ).fetchall()
    return {
        str(row[0]): (row[1], row[2], None if row[3] is None else int(row[3]))
        for row in rows
    }


def _prepare(store: ClipStore, known: _KnownMedia | None = None) -> tuple[_PreparedClip, ...]:
    prepared: list[_PreparedClip] = []
    seen: set[str] = set()
    known = {} if known is None else known
    for manifest in store.list_manifests():
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
            media_hash, media_size = _media_identity(
                store.root, media_path, known.get(manifest.clip_id)
            )
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
        values: tuple[str | int | None, ...] = (
            manifest.clip_id, manifest.camera_id, effective_event_type(manifest),
            manifest.started_at, max(1, round(manifest.duration_s * 1000)),
            manifest.codec or None, "video/mp4", manifest_relpath, media_relpath,
            thumbnail_relpath, manifest_hash, media_hash, thumbnail_hash,
            manifest_size, media_size, thumbnail_size, local_state, local_reason,
            manifest.started_at, manifest.started_at,
        )
        prepared.append(
            _PreparedClip(manifest, values, (manifest_hash, media_hash, media_size))
        )
    return tuple(prepared)


def _reconcile(connection, prepared: tuple[_PreparedClip, ...]) -> None:
    present_ids = {item.manifest.clip_id for item in prepared}
    existing_ids = {
        str(row[0]) for row in connection.execute("SELECT clip_id FROM clips").fetchall()
    }
    for clip_id in existing_ids - present_ids:
        referenced = connection.execute(
            "SELECT 1 FROM artifacts WHERE clip_id = ? LIMIT 1",
            (clip_id,),
        ).fetchone()
        if referenced is None:
            connection.execute("DELETE FROM clips WHERE clip_id = ?", (clip_id,))
        else:
            connection.execute(
                """
                UPDATE clips SET
                    manifest_relpath=NULL, media_relpath=NULL, thumbnail_relpath=NULL,
                    manifest_sha256=NULL, media_sha256=NULL, thumbnail_sha256=NULL,
                    manifest_size_bytes=NULL, media_size_bytes=NULL, thumbnail_size_bytes=NULL,
                    local_state='UNAVAILABLE', local_reason='MANIFEST_MISSING',
                    revision=revision+1
                WHERE clip_id=? AND (
                    local_state != 'UNAVAILABLE' OR local_reason != 'MANIFEST_MISSING'
                )
                """,
                (clip_id,),
            )
    for item in prepared:
        existing = connection.execute(
            "SELECT manifest_sha256, media_sha256, media_size_bytes FROM clips WHERE clip_id = ?",
            (item.manifest.clip_id,),
        ).fetchone()
        if existing is not None and tuple(existing) != item.identity:
            raise CompactClipConflictError(
                f"clip {item.manifest.clip_id} immutable content changed"
            )
        if existing is None:
            connection.execute(_INSERT_CLIP, item.values)


def _page_rows(connection, query: CompactClipQuery):
    visible = "local_state != 'UNAVAILABLE' AND manifest_relpath IS NOT NULL"
    scope_predicates = [visible]
    scope_params: list[str] = []
    if query.camera_id is not None:
        scope_predicates.append("camera_id = ?")
        scope_params.append(query.camera_id)
    page_predicates = list(scope_predicates)
    page_params: list[str | int] = list(scope_params)
    if query.event_type is not None:
        page_predicates.append("event_facet = ?")
        page_params.append(query.event_type)
    count_predicates = list(page_predicates)
    count_params = tuple(page_params)
    if query.cursor is not None:
        started_at, clip_id = _parse_cursor(query.cursor)
        page_predicates.append("(started_at < ? OR (started_at = ? AND clip_id < ?))")
        page_params.extend((started_at, started_at, clip_id))
    rows = connection.execute(
        "SELECT clip_id, started_at FROM clips WHERE " + " AND ".join(page_predicates)
        + " ORDER BY started_at DESC, clip_id DESC LIMIT ?",
        (*page_params, query.limit + 1),
    ).fetchall()
    total = int(connection.execute(
        "SELECT count(*) FROM clips WHERE " + " AND ".join(count_predicates),
        count_params,
    ).fetchone()[0])
    facets = dict(connection.execute(
        "SELECT event_facet, count(*) FROM clips WHERE " + " AND ".join(scope_predicates)
        + " GROUP BY event_facet ORDER BY event_facet",
        tuple(scope_params),
    ).fetchall())
    return rows, total, facets


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


def _media_identity(
    root: Path,
    media_path: Path,
    catalogued: tuple[str | None, str | None, int | None] | None,
) -> tuple[str, int]:
    """Reuse the catalogued media hash when the file is byte-for-byte the same size.

    The catalogue row is the durable receipt of a hash this process already
    computed over these bytes; re-reading every clip on every listing is what
    made ``GET /clips`` unusable on a populated store. Anything not matching the
    receipt (unknown clip, moved path, different size) is hashed in full.
    """
    if catalogued is not None:
        relpath, sha256, size_bytes = catalogued
        if sha256 is not None and relpath == media_path.relative_to(root).as_posix():
            opened = open_contained_regular_file(root, media_path)
            opened.handle.close()
            if opened.size_bytes == size_bytes:
                return sha256, opened.size_bytes
    return _hash_regular(root, media_path)


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


__all__ = [
    "CompactClipConflictError",
    "CompactClipListing",
    "CompactClipPage",
    "CompactClipQuery",
]
