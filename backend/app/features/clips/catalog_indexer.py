"""PostgreSQL clip catalogue: one background indexer writes, listings only read.

Clip recording is always on, so the API always keeps the catalogue. The indexer
reconciles the clip store once at startup (a failure refuses startup) and then
every ``API_CLIP_CATALOG_INTERVAL_SEC``. Each pass walks the store once, trusts
clips whose catalogued manifest and media path/size still match, and examines
at most ``EXAMINE_BUDGET`` others (newest first); the remainder is picked up by
the next pass without waiting. Every clip is applied under its own savepoint,
so one malformed clip is counted and skipped instead of failing the pass.
``GET /clips`` reads one keyset page from the catalogue and never writes it.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import math
import os
import re
import stat
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase
from backend.app.features.clips.descriptor_files import open_contained_regular_file
from backend.app.features.clips.listing import effective_event_type
from backend.app.features.clips.manifest import read_manifest_file
from backend.app.features.clips.store import ClipStore, LocatedClip, ScannedManifest

logger = logging.getLogger(__name__)

API_CLIP_CATALOG_INTERVAL_SEC_ENV = "API_CLIP_CATALOG_INTERVAL_SEC"
DEFAULT_CLIP_CATALOG_INTERVAL_SEC = 5.0
MAX_CLIP_CATALOG_INTERVAL_SEC = 300.0
CLIP_CATALOG_SHUTDOWN_WAIT_SEC = 1.0
EXAMINE_BUDGET = 200
"""Upper bound on clips read and hashed per reconcile pass."""
APPLY_CHUNK = 64
"""Clips applied per transaction, so one pass never holds many row locks."""

_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    r"T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]([.][0-9]{1,6})?Z"
)
_VISIBLE = (
    "local_state <> 'UNAVAILABLE' AND manifest_relpath IS NOT NULL "
    "AND local_reason IS DISTINCT FROM 'MANIFEST_MISSING'"
)
# Same key as the receipt path's clip lock, so indexer and receipts serialize.
_LOCK_CLIP = "SELECT pg_advisory_xact_lock('clips'::regclass::oid::integer, hashtext(%s))"
_LOCKED_ROW = """
SELECT publish_state, retention_state,
    camera_id, event_facet, started_at, duration_ms, codec, mime_type,
    manifest_relpath, media_relpath, thumbnail_relpath,
    manifest_sha256, media_sha256, thumbnail_sha256,
    manifest_size_bytes, media_size_bytes, thumbnail_size_bytes,
    local_state, local_reason, revision
FROM clips WHERE clip_id = %s FOR UPDATE
"""
_INSERT_CLIP = """
INSERT INTO clips (
    clip_id, camera_id, event_facet, started_at, duration_ms, codec, mime_type,
    manifest_relpath, media_relpath, thumbnail_relpath,
    manifest_sha256, media_sha256, thumbnail_sha256,
    manifest_size_bytes, media_size_bytes, thumbnail_size_bytes,
    local_state, local_reason, publish_state, retention_state, revision,
    created_at, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
          'WAITING', 'RETAINED', 1, %s, %s)
"""
_UPDATE_CLIP = """
UPDATE clips SET
    camera_id=%s, event_facet=%s, started_at=%s, duration_ms=%s, codec=%s, mime_type=%s,
    manifest_relpath=%s, media_relpath=%s, thumbnail_relpath=%s,
    manifest_sha256=%s, media_sha256=%s, thumbnail_sha256=%s,
    manifest_size_bytes=%s, media_size_bytes=%s, thumbnail_size_bytes=%s,
    local_state=%s, local_reason=%s, revision=revision+1, updated_at=%s
WHERE clip_id=%s
"""
_MARK_CLIP = """
UPDATE clips SET local_state='CORRUPT', local_reason=%s, revision=revision+1, updated_at=%s
WHERE clip_id=%s
"""
_PARK_CLIP = """
UPDATE clips SET
    manifest_relpath=NULL, media_relpath=NULL, thumbnail_relpath=NULL,
    manifest_sha256=NULL, media_sha256=NULL, thumbnail_sha256=NULL,
    manifest_size_bytes=NULL, media_size_bytes=NULL, thumbnail_size_bytes=NULL,
    local_state='UNAVAILABLE', local_reason='MANIFEST_MISSING',
    revision=revision+1, updated_at=%s
WHERE clip_id=%s
"""


class InvalidClipCatalogIntervalError(ValueError):
    """``API_CLIP_CATALOG_INTERVAL_SEC`` is not a finite number in (0, 300]."""


def clip_catalog_interval_sec() -> float:
    raw = os.environ.get(API_CLIP_CATALOG_INTERVAL_SEC_ENV)
    if raw is None:
        return DEFAULT_CLIP_CATALOG_INTERVAL_SEC
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or not 0 < value <= MAX_CLIP_CATALOG_INTERVAL_SEC:
        raise InvalidClipCatalogIntervalError(
            f"{API_CLIP_CATALOG_INTERVAL_SEC_ENV} must be a finite number in (0, 300], got {raw!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ClipCatalogQuery:
    camera_id: str | None
    event_type: str | None
    limit: int
    cursor: str | None


@dataclass(frozen=True, slots=True)
class ClipCatalogPage:
    clips: tuple[LocatedClip, ...]
    total: int
    has_more: bool
    next_cursor: str | None
    event_type_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReconcileOutcome:
    examined: int
    remaining: int
    isolated: int


@dataclass(frozen=True, slots=True)
class _CataloguedClip:
    manifest_relpath: str | None
    manifest_size_bytes: int | None
    media_relpath: str | None
    media_sha256: str | None
    media_size_bytes: int | None
    local_state: str
    local_reason: str | None
    retention_state: str
    revision: int


@dataclass(frozen=True, slots=True)
class _PreparedClip:
    located: LocatedClip
    values: tuple[str | int | None, ...]


class ClipCatalogIndexer:
    """The catalogue's only writer; borrows transactions from the API pool."""

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority
        self._lock = threading.Lock()
        self._settled: dict[str, tuple[object, ...]] = {}

    def reconcile(self, store: ClipStore) -> ReconcileOutcome:
        with self._lock:
            return self._reconcile(store)

    def _reconcile(self, store: ClipStore) -> ReconcileOutcome:
        root = store.root
        if not root.is_dir():
            # An unmounted or missing store is not evidence that clips were deleted.
            return ReconcileOutcome(0, 0, 0)
        catalogued = self.database.read(_catalogued)
        scanned, duplicates = store.scan_manifest_partition()
        scanned_ids = {item.clip_id for item in scanned}
        self._settled = {
            clip_id: fingerprint
            for clip_id, fingerprint in self._settled.items()
            if clip_id in scanned_ids
        }
        candidates: list[tuple[ScannedManifest, int | None]] = []
        for item in scanned:
            row = catalogued.get(item.clip_id)
            if row is not None and row.retention_state != "RETAINED":
                continue
            if _unchanged(root, item, row):
                continue
            dir_mtime_ns = _dir_mtime(item)
            if self._settled.get(item.clip_id) == _fingerprint(root, item, dir_mtime_ns, row):
                continue
            candidates.append((item, dir_mtime_ns))
        candidates.sort(key=lambda pair: (pair[0].mtime_ns, pair[0].clip_id), reverse=True)
        batch = candidates[:EXAMINE_BUDGET]
        ambiguous = {duplicate.clip_id for duplicate in duplicates}
        removals = {
            clip_id
            for clip_id, row in catalogued.items()
            if clip_id not in scanned_ids
            and clip_id not in ambiguous
            and row.retention_state == "RETAINED"
            and row.manifest_relpath is not None
            and row.local_reason != "MANIFEST_MISSING"
            and _manifest_gone(root / row.manifest_relpath)
        }
        prepared: list[_PreparedClip] = []
        isolated = 0
        for item, _dir_mtime_ns in batch:
            try:
                located = item.located()
                if located is None or not _valid_timestamp(located.manifest.started_at):
                    # Not (or no longer) a finalized, well-formed clip: filesystem
                    # truth says there is nothing to list, as if the manifest were gone.
                    if item.clip_id in catalogued:
                        removals.add(item.clip_id)
                    continue
                prepared.append(_prepare_clip(store, located, catalogued.get(item.clip_id)))
            except (OSError, ValueError, ArithmeticError):
                logger.exception("clip %s could not be examined", item.clip_id)
                isolated += 1
        operations: list[str | _PreparedClip] = [*sorted(removals), *prepared]
        for start in range(0, len(operations), APPLY_CHUNK):
            isolated += self.database.transact(
                partial(
                    _apply_chunk,
                    token=self.authority,
                    operations=operations[start : start + APPLY_CHUNK],
                )
            )
        if batch:
            examined_ids = [item.clip_id for item, _ in batch]
            rows = self.database.read(partial(_catalogued, clip_ids=examined_ids))
            for item, dir_mtime_ns in batch:
                self._settled[item.clip_id] = _fingerprint(
                    root, item, dir_mtime_ns, rows.get(item.clip_id)
                )
        if isolated:
            logger.warning("clip catalog isolated %d clips this pass", isolated)
        return ReconcileOutcome(len(batch), len(candidates) - len(batch), isolated)


class PostgresClipCatalog:
    """Read-only keyset pages over the catalogue for ``GET /clips``."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def page(self, store: ClipStore, query: ClipCatalogQuery) -> ClipCatalogPage:
        cursor = None if query.cursor is None else _parse_cursor(query.cursor)
        rows, total, facets = self.database.read_snapshot(
            partial(_page_rows, query=query, cursor=cursor)
        )
        visible: list[LocatedClip] = []
        for clip_id, _started_at, manifest_relpath in rows[: query.limit]:
            located = _located_from_row(store, str(clip_id), manifest_relpath)
            if located is not None:
                visible.append(located)
        has_more = len(rows) > query.limit
        next_cursor = None
        if has_more:
            last = rows[query.limit - 1]
            next_cursor = _format_cursor(str(last[1]), str(last[0]))
        return ClipCatalogPage(
            clips=tuple(visible),
            total=total,
            has_more=has_more,
            next_cursor=next_cursor,
            event_type_counts=facets,
        )


def _catalogued(
    connection: psycopg.Connection, clip_ids: Sequence[str] | None = None
) -> dict[str, _CataloguedClip]:
    sql = (
        "SELECT clip_id, manifest_relpath, manifest_size_bytes, media_relpath, media_sha256, "
        "media_size_bytes, local_state, local_reason, retention_state, revision FROM clips"
    )
    if clip_ids is None:
        rows = connection.execute(sql).fetchall()
    else:
        rows = connection.execute(sql + " WHERE clip_id = ANY(%s)", (list(clip_ids),)).fetchall()
    return {str(row[0]): _CataloguedClip(*row[1:]) for row in rows}


def _unchanged(root: Path, item: ScannedManifest, row: _CataloguedClip | None) -> bool:
    """Trust a catalogued clip when manifest and media path/size still match."""
    if row is None or row.local_state != "AVAILABLE" or row.media_relpath is None:
        return False
    if row.manifest_relpath != item.manifest_path.relative_to(root).as_posix():
        return False
    if row.manifest_size_bytes != item.size_bytes:
        return False
    return _regular_size(root / row.media_relpath) == row.media_size_bytes


def _fingerprint(
    root: Path, item: ScannedManifest, dir_mtime_ns: int | None, row: _CataloguedClip | None
) -> tuple[object, ...]:
    """Everything whose change could alter the outcome of examining ``item`` again."""
    media_size = (
        None
        if row is None or row.media_relpath is None
        else _regular_size(root / row.media_relpath)
    )
    return (
        item.manifest_path,
        item.size_bytes,
        item.mtime_ns,
        dir_mtime_ns,
        None if row is None else (row.local_state, row.local_reason, row.revision),
        media_size,
    )


def _dir_mtime(item: ScannedManifest) -> int | None:
    try:
        return os.stat(item.manifest_path.parent).st_mtime_ns
    except OSError:
        return None


def _manifest_gone(path: Path) -> bool:
    try:
        path_stat = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return not stat.S_ISREG(path_stat.st_mode)


def _regular_size(path: Path) -> int | None:
    try:
        path_stat = path.stat()
    except OSError:
        return None
    return path_stat.st_size if stat.S_ISREG(path_stat.st_mode) else None


def _relative(root: Path, path: Path) -> str:
    """``path`` relative to the store root, whether or not ``path`` was resolved."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.relative_to(root.resolve()).as_posix()


def _prepare_clip(
    store: ClipStore, located: LocatedClip, row: _CataloguedClip | None
) -> _PreparedClip:
    manifest = located.manifest
    manifest_path = located.manifest_path
    manifest_hash, manifest_size = _hash_regular(store.root, manifest_path)
    known = None if row is None else (row.media_relpath, row.media_sha256, row.media_size_bytes)
    media: tuple[str, str, int] | None
    try:
        media_path = store.resolve_located_video_path(located)
        media_relpath = _relative(store.root, media_path)
        media_hash, media_size = _media_identity(store.root, media_path, media_relpath, known)
        media = (media_relpath, media_hash, media_size) if media_size > 0 else None
    except (FileNotFoundError, ValueError):
        media = None
    thumbnail_path = manifest_path.parent / "thumbnail.jpg"
    thumbnail: tuple[str, str, int] | None
    try:
        thumbnail_hash, thumbnail_size = _hash_regular(store.root, thumbnail_path)
        thumbnail = (
            (thumbnail_path.relative_to(store.root).as_posix(), thumbnail_hash, thumbnail_size)
            if thumbnail_size > 0
            else None
        )
    except FileNotFoundError:
        thumbnail = None
    values: tuple[str | int | None, ...] = (
        manifest.camera_id,
        effective_event_type(manifest),
        manifest.started_at,
        max(1, round(manifest.duration_s * 1000)),
        manifest.codec or None,
        "video/mp4",
        manifest_path.relative_to(store.root).as_posix(),
        None if media is None else media[0],
        None if thumbnail is None else thumbnail[0],
        manifest_hash,
        None if media is None else media[1],
        None if thumbnail is None else thumbnail[1],
        manifest_size,
        None if media is None else media[2],
        None if thumbnail is None else thumbnail[2],
        "AVAILABLE" if media is not None else "CORRUPT",
        None if media is not None else "MEDIA_MISSING",
    )
    return _PreparedClip(located, values)


def _media_identity(
    root: Path,
    media_path: Path,
    media_relpath: str,
    catalogued: tuple[str | None, str | None, int | None] | None,
) -> tuple[str, int]:
    """Reuse the catalogued media hash when the file is byte-for-byte the same size.

    The catalogue row is the durable receipt of a hash computed over these
    bytes; anything not matching it (unknown clip, moved path, different size)
    is hashed in full.
    """
    if catalogued is not None:
        relpath, sha256, size_bytes = catalogued
        if sha256 is not None and relpath == media_relpath:
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
        for chunk in iter(partial(opened.handle.read, 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    finally:
        opened.handle.close()
    return digest.hexdigest(), size


def _apply_chunk(
    connection: psycopg.Connection,
    *,
    token: AuthorityToken,
    operations: Sequence[str | _PreparedClip],
) -> int:
    """Apply removals (clip ids) and upserts, each under its own savepoint."""
    require_authority(connection, token)
    now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    isolated = 0
    for operation in operations:
        clip_id = operation if isinstance(operation, str) else operation.located.manifest.clip_id
        connection.execute("SAVEPOINT clip_item")
        try:
            connection.execute(_LOCK_CLIP, (clip_id,))
            locked = connection.execute(_LOCKED_ROW, (clip_id,)).fetchone()
            if isinstance(operation, str):
                _remove(connection, clip_id, locked, now)
            else:
                _upsert(connection, clip_id, operation.values, locked, now)
        except (psycopg.IntegrityError, psycopg.DataError):
            logger.exception("clip %s rejected by the catalogue", clip_id)
            connection.execute("ROLLBACK TO SAVEPOINT clip_item")
            isolated += 1
        else:
            connection.execute("RELEASE SAVEPOINT clip_item")
    return isolated


def _remove(connection: psycopg.Connection, clip_id: str, locked: tuple | None, now: str) -> None:
    if locked is None or locked[1] != "RETAINED":
        return
    stored = locked[2:19]
    if stored[6] is None or stored[16] == "MANIFEST_MISSING":
        return
    if locked[0] != "WAITING":
        # Already delivered: keep the durable identity, hide it until it returns.
        connection.execute(_MARK_CLIP, ("MANIFEST_MISSING", now, clip_id))
        return
    referenced = connection.execute(
        "SELECT 1 FROM artifacts WHERE clip_id = %s LIMIT 1", (clip_id,)
    ).fetchone()
    if referenced is None:
        connection.execute("DELETE FROM clips WHERE clip_id = %s", (clip_id,))
    else:
        connection.execute(_PARK_CLIP, (now, clip_id))


def _upsert(
    connection: psycopg.Connection,
    clip_id: str,
    values: tuple[str | int | None, ...],
    locked: tuple | None,
    now: str,
) -> None:
    if locked is None:
        connection.execute(_INSERT_CLIP, (clip_id, *values, values[2], now))
        return
    if locked[1] != "RETAINED":
        return
    stored = tuple(locked[2:19])
    merged = list(values)
    if merged[7] is None and stored[7] is not None:
        # Missing media keeps the identity already catalogued (and possibly receipted).
        merged[7], merged[10], merged[13] = stored[7], stored[10], stored[13]
    observed = tuple(merged)
    if observed == stored:
        return
    if _identity_compatible(
        (stored[9], stored[10], stored[13]), (observed[9], observed[10], observed[13])
    ):
        connection.execute(_UPDATE_CLIP, (*observed, now, clip_id))
        return
    if stored[15:17] == ("CORRUPT", "IDENTITY_CONFLICT"):
        return
    connection.execute(_MARK_CLIP, ("IDENTITY_CONFLICT", now, clip_id))


def _identity_compatible(catalogued: tuple[object, ...], observed: tuple[object, ...]) -> bool:
    """True when every identity fact the catalogue holds matches what is on disk.

    A NULL catalogued fact is absence of knowledge, not a claim: only a known
    hash or size that differs from the observed one is a content change.
    """
    return all(
        known is None or known == seen for known, seen in zip(catalogued, observed, strict=True)
    )


def _page_rows(
    connection: psycopg.Connection,
    *,
    query: ClipCatalogQuery,
    cursor: tuple[str, str] | None,
) -> tuple[list[tuple], int, dict[str, int]]:
    scope_predicates = [_VISIBLE]
    scope_params: list[str] = []
    if query.camera_id is not None:
        scope_predicates.append("camera_id = %s")
        scope_params.append(query.camera_id)
    page_predicates = list(scope_predicates)
    page_params: list[str | int] = list(scope_params)
    if query.event_type is not None:
        page_predicates.append("event_facet = %s")
        page_params.append(query.event_type)
    count_predicates = list(page_predicates)
    count_params = tuple(page_params)
    if cursor is not None:
        started_at, clip_id = cursor
        page_predicates.append(
            '(started_at COLLATE "C" < %s OR (started_at = %s AND clip_id COLLATE "C" < %s))'
        )
        page_params.extend((started_at, started_at, clip_id))
    rows = connection.execute(
        "SELECT clip_id, started_at, manifest_relpath FROM clips WHERE "
        + " AND ".join(page_predicates)
        + ' ORDER BY started_at COLLATE "C" DESC, clip_id COLLATE "C" DESC LIMIT %s',
        (*page_params, query.limit + 1),
    ).fetchall()
    total_row = connection.execute(
        "SELECT count(*) FROM clips WHERE " + " AND ".join(count_predicates),
        count_params,
    ).fetchone()
    facets = connection.execute(
        "SELECT event_facet, count(*) FROM clips WHERE "
        + " AND ".join(scope_predicates)
        + " GROUP BY event_facet ORDER BY event_facet",
        tuple(scope_params),
    ).fetchall()
    total = 0 if total_row is None else int(total_row[0])
    return rows, total, {str(facet): int(count) for facet, count in facets}


def _located_from_row(
    store: ClipStore, clip_id: str, manifest_relpath: object
) -> LocatedClip | None:
    """Read a visible row's manifest at its catalogued path (no store walk).

    A manifest that vanished since the last pass is skipped; the indexer
    reconciles the row on its next pass.
    """
    if not isinstance(manifest_relpath, str):
        return None
    manifest_path = store.root / manifest_relpath
    manifest = read_manifest_file(manifest_path)
    if (
        manifest is None
        or not manifest.finalized
        or manifest.clip_id != clip_id
        or manifest_path.parent.name != clip_id
    ):
        return None
    return LocatedClip(manifest, manifest_path)


def _valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and 20 <= len(value) <= 30
        and _UTC_TIMESTAMP_RE.fullmatch(value) is not None
    )


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


async def run_clip_catalog_indexer(
    indexer: ClipCatalogIndexer,
    store: ClipStore,
    stop: asyncio.Event,
    executor: ThreadPoolExecutor,
    interval: float,
    remaining: int,
) -> None:
    while not stop.is_set():
        if remaining == 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
        if stop.is_set():
            break
        try:
            outcome = await asyncio.get_running_loop().run_in_executor(
                executor, indexer.reconcile, store
            )
        except Exception:
            logger.exception("clip catalog reconcile failed")
            remaining = 0
        else:
            remaining = outcome.remaining


__all__ = [
    "API_CLIP_CATALOG_INTERVAL_SEC_ENV",
    "CLIP_CATALOG_SHUTDOWN_WAIT_SEC",
    "DEFAULT_CLIP_CATALOG_INTERVAL_SEC",
    "EXAMINE_BUDGET",
    "MAX_CLIP_CATALOG_INTERVAL_SEC",
    "ClipCatalogIndexer",
    "ClipCatalogPage",
    "ClipCatalogQuery",
    "InvalidClipCatalogIntervalError",
    "PostgresClipCatalog",
    "ReconcileOutcome",
    "clip_catalog_interval_sec",
    "run_clip_catalog_indexer",
]
