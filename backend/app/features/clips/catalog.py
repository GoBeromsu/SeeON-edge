"""ml-api-owned SQLite catalog for clip-store metadata.

The edge worker never opens this database; it supplies metadata through relay.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from backend.app.features.clips.store import ClipManifest, is_valid_clip_id
from backend.app.shared.state_dir import resolve_state_dir

if TYPE_CHECKING:
    from fastapi import FastAPI

SCHEMA_VERSION = 3
logger = logging.getLogger(__name__)
_CAMERA_PAYLOAD_FIELDS = (
    "id",
    "backend_camera_id",
    "label",
    "decode_backend",
    "created_at",
    "mapping_pending",
)
_CAMERA_SCALAR_TYPES = (str, int, float, bool, type(None))
_MANIFEST_SCHEMA_VERSION = 2
_MANIFEST_STATES = {"READY", "UNAVAILABLE"}
_UNAVAILABLE_REASON_CODES = {
    "ENCODER_FAILED",
    "NO_FRAMES",
    "FINALIZE_FAILED",
    "INTERRUPTED_FINALIZE",
    "MISSING",
    "CORRUPT",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "state",
    "clip_id",
    "camera_id",
    "event_refs",
    "clip_start_at",
    "clip_end_at",
    "finalized_at",
    "sha256",
    "size_bytes",
    "mime_type",
    "codec",
    "duration_ms",
    "state_version",
    "reason_code",
    "event_ref",
    "event_type",
    "started_at",
    "duration_s",
    "encoder",
    "path",
    "finalized",
    "video_available",
    "video_error",
}

_TABLE_KEYS = {
    "clips": "clip_id",
    "snapshots": "snapshot_id",
    "events": "edge_event_id",
    "labels": "clip_id",
    "audit": "audit_id",
    "cameras": "camera_id",
}
_TABLE_COLUMNS = {
    "clips": (
        "camera_id",
        "event_type",
        "state",
        "started_at",
        "path",
        "sha256",
        "size_bytes",
        "mime_type",
        "encoder",
    ),
    "snapshots": (
        "camera_id",
        "edge_event_id",
        "captured_at",
        "path",
        "sha256",
        "size_bytes",
        "mime_type",
    ),
    "events": ("camera_id", "event_type", "detected_at", "clip_id"),
    "labels": ("label", "reviewer", "reviewed_at"),
    "cameras": ("label", "decode_backend"),
    "audit": ("occurred_at", "action"),
}
_CREATE_STATEMENTS = (
    (
        "CREATE TABLE clips (clip_id TEXT PRIMARY KEY, camera_id TEXT, event_type TEXT, "
        "state TEXT, started_at TEXT, path TEXT, sha256 TEXT, size_bytes INTEGER, "
        "mime_type TEXT, encoder TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE snapshots (snapshot_id TEXT PRIMARY KEY, camera_id TEXT, "
        "edge_event_id TEXT, captured_at TEXT, path TEXT, sha256 TEXT, "
        "size_bytes INTEGER, mime_type TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE events (edge_event_id TEXT PRIMARY KEY, camera_id TEXT, "
        "event_type TEXT, detected_at TEXT, clip_id TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE labels (clip_id TEXT PRIMARY KEY, label TEXT, reviewer TEXT, "
        "reviewed_at TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE cameras (camera_id TEXT PRIMARY KEY, label TEXT, "
        "decode_backend TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE audit (audit_id TEXT PRIMARY KEY, occurred_at TEXT, action TEXT, "
        "payload_json TEXT NOT NULL) STRICT"
    ),
)
# Schema-version-3 tables: ml-api-owned state that used to live in standalone
# JSON files (dashboard_credentials.json, cameras.json, runtime-latency.json).
# Deliberately NOT added to _TABLE_KEYS/_TABLE_COLUMNS -- those two dicts back
# the payload_json compare-or-insert convention used by record()/records(),
# which does not apply here. `camera_registry` is a distinct table from the
# pre-existing `cameras` clip/audit denormalization cache above: `cameras` is
# a derived read cache of catalog-relevant camera fields, while
# `camera_registry` is the ml-api camera registry's own source of truth.
#
# IF NOT EXISTS (unlike every other statement in _CREATE_STATEMENTS): the
# owning stores (DashboardCredentialsStore, CameraRegistryStore,
# RuntimeStatusStore) each independently bootstrap their own single table on
# first use via the same statement text, since two of the three live in
# backend/app/shared and backend/app/features/cameras -- CatalogStore itself
# cannot be a dependency there (see "backend base (core/shared) does not
# import upper layers" in pyproject.toml's import-linter contracts) or a
# forward reference (cameras is a sibling feature, not a base layer). Whoever
# opens the database file first (a catalog request, or one of the three
# stores) must not fail when the others have already created their table.
_V3_TABLE_STATEMENTS = (
    (
        "CREATE TABLE IF NOT EXISTS credentials (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "username TEXT NOT NULL, algorithm TEXT NOT NULL, salt BLOB NOT NULL, "
        "password_hash BLOB NOT NULL, updated_at TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE IF NOT EXISTS camera_registry (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "registry_version INTEGER NOT NULL, cameras_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE IF NOT EXISTS runtime_latency (facility_id TEXT PRIMARY KEY, "
        "payload_json TEXT NOT NULL) STRICT"
    ),
)
_CREATE_STATEMENTS = (*_CREATE_STATEMENTS, *_V3_TABLE_STATEMENTS)
_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS clips_camera_started_at_idx ON clips(camera_id, started_at)",
    "CREATE INDEX IF NOT EXISTS clips_event_type_idx ON clips(event_type)",
    (
        "CREATE INDEX IF NOT EXISTS snapshots_camera_captured_at_idx "
        "ON snapshots(camera_id, captured_at)"
    ),
    "CREATE INDEX IF NOT EXISTS events_camera_detected_at_idx ON events(camera_id, detected_at)",
)


def _catalog_path() -> Path:
    return resolve_state_dir("ml-api") / "catalog.sqlite3"


def get_catalog_store(app: FastAPI) -> CatalogStore | None:
    """Open the optional catalog only when a catalog read or write is requested."""
    store = getattr(app.state, "catalog_store", None)
    if isinstance(store, CatalogStore):
        return store
    try:
        store = CatalogStore.open(_catalog_path())
    except (OSError, sqlite3.Error, CatalogSchemaNewerThanSupportedError) as exc:
        message = f"catalog unavailable at {_catalog_path()}: {exc}"
        app.state.catalog_error = message
        logger.warning(message)
        return None
    app.state.catalog_store = store
    app.state.catalog_error = None
    return store


@dataclass(frozen=True)
class StrictManifest:
    """A raw sidecar accepted by the migration and verification boundary."""

    manifest: ClipManifest
    path: Path
    payload: dict[str, Any]


def _require_regular_contained_file(path: Path, root: Path) -> None:
    root_resolved = root.resolve(strict=True)
    current = path.parent
    while current != root:
        if current.is_symlink():
            raise ValueError(f"manifest path contains a symlink: {path}")
        if root not in current.parents:
            raise ValueError(f"manifest path escapes clip store: {path}")
        current = current.parent
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"unable to inspect manifest: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"manifest is not a regular file: {path}")
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"manifest path escapes clip store: {path}")


def strict_manifest_records(clip_store: Any) -> list[StrictManifest]:
    """Read finalized evidence sidecars without inheriting serving parser leniency."""
    clips_dir = Path(clip_store.root) / "clips"
    if clips_dir.is_symlink():
        raise ValueError(f"clips directory must not be a symlink: {clips_dir}")
    if not clips_dir.is_dir():
        return []

    records: list[StrictManifest] = []
    for path in sorted(clips_dir.glob("*/manifest.json")):
        _require_regular_contained_file(path, clips_dir)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read manifest: {path}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"invalid manifest: {path}")
        manifest = _strict_manifest_from_payload(payload, path)
        if manifest.clip_id != path.parent.name:
            raise ValueError(f"manifest clip_id does not match directory: {path}")
        records.append(StrictManifest(manifest=manifest, path=path, payload=payload))
    return records


def _strict_manifest_from_payload(payload: dict[str, Any], path: Path) -> ClipManifest:
    """Validate evidence schema v2 sidecars."""
    if "manifest_schema_version" not in payload:
        raise ValueError(f"missing manifest schema: {path}")
    if set(payload) - _MANIFEST_FIELDS:
        raise ValueError(f"unexpected manifest fields: {path}")
    if payload.get("manifest_schema_version") != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {path}")
    if payload.get("finalized") is not True or payload.get("state_version") != 2:
        raise ValueError(f"manifest is not finalized: {path}")

    state = payload.get("state")
    if state not in _MANIFEST_STATES:
        raise ValueError(f"invalid manifest state: {path}")
    clip_id, camera_id, event_ref = _manifest_identity(payload, path)
    event_refs = payload.get("event_refs")
    if not isinstance(event_refs, list) or not event_refs:
        raise ValueError(f"invalid manifest event_refs: {path}")
    if len(set(event_refs)) != len(event_refs) or any(
        not _is_canonical_uuid4(value) for value in event_refs
    ):
        raise ValueError(f"invalid manifest event_refs: {path}")
    if event_ref not in event_refs:
        raise ValueError(f"manifest event_ref missing from event_refs: {path}")

    timestamps = (
        _utc_timestamp(payload.get("clip_start_at"), path),
        _utc_timestamp(payload.get("clip_end_at"), path),
        _utc_timestamp(payload.get("finalized_at"), path),
    )
    _utc_timestamp(payload.get("started_at"), path)
    if timestamps != tuple(sorted(timestamps)):
        raise ValueError(f"manifest timestamps are unordered: {path}")
    duration_s = payload.get("duration_s")
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, int | float)
        or not math.isfinite(duration_s)
        or duration_s < 0
    ):
        raise ValueError(f"invalid manifest duration: {path}")
    if state == "READY":
        _validate_ready_media(payload, clip_id, path)
    else:
        _validate_unavailable_media(payload, path)

    event_type = payload.get("event_type")
    if event_type is not None and (not isinstance(event_type, str) or not event_type.strip()):
        raise ValueError(f"invalid manifest event_type: {path}")
    codec = payload.get("codec")
    if state == "READY" and (not isinstance(codec, str) or not codec.strip()):
        raise ValueError(f"invalid manifest codec: {path}")
    if state == "UNAVAILABLE":
        codec = payload.get("encoder") if isinstance(payload.get("encoder"), str) else ""
    return ClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_ref=event_ref,
        event_type=event_type,
        started_at=str(payload["started_at"]),
        duration_s=float(duration_s),
        codec=codec,
        path=payload.get("path"),
        video_available=payload["video_available"],
        video_error=payload.get("video_error")
        if isinstance(payload.get("video_error"), str)
        else None,
        finalized=True,
    )


def _manifest_identity(payload: dict[str, Any], path: Path) -> tuple[str, str, str]:
    clip_id = payload.get("clip_id")
    camera_id = payload.get("camera_id")
    event_ref = payload.get("event_ref")
    if (
        not isinstance(clip_id, str)
        or not is_valid_clip_id(clip_id)
        or not isinstance(camera_id, str)
        or not camera_id.strip()
        or not isinstance(event_ref, str)
        or not event_ref.strip()
        or not _is_canonical_uuid4(event_ref)
    ):
        raise ValueError(f"invalid manifest identity: {path}")
    return clip_id, camera_id, event_ref


def _validate_ready_media(payload: dict[str, Any], clip_id: str, path: Path) -> None:
    media_path = payload.get("path")
    if (
        not isinstance(media_path, str)
        or not media_path.startswith(f"clips/{clip_id}/")
        or not Path(media_path).name
        or Path(media_path).suffix.lower() != ".mp4"
        or Path(media_path).is_absolute()
        or ".." in Path(media_path).parts
        or not _SHA256_RE.fullmatch(str(payload.get("sha256", "")))
        or isinstance(payload.get("size_bytes"), bool)
        or not isinstance(payload.get("size_bytes"), int)
        or payload["size_bytes"] <= 0
        or payload.get("mime_type") != "video/mp4"
        or payload.get("codec") != "h264"
        or payload.get("video_available") is not True
        or isinstance(payload.get("duration_ms"), bool)
        or not isinstance(payload.get("duration_ms"), int)
        or not 1 <= payload["duration_ms"] <= 120_000
        or payload.get("reason_code") is not None
    ):
        raise ValueError(f"invalid ready media declaration: {path}")


def _validate_unavailable_media(payload: dict[str, Any], path: Path) -> None:
    if (
        payload.get("reason_code") not in _UNAVAILABLE_REASON_CODES
        or payload.get("path") is not None
        or payload.get("video_available") is not False
        or any(
            payload.get(field) is not None
            for field in ("sha256", "size_bytes", "mime_type", "codec", "duration_ms")
        )
    ):
        raise ValueError(f"invalid unavailable media declaration: {path}")


def _is_canonical_uuid4(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return UUID(value).version == 4 and str(UUID(value)) == value
    except ValueError:
        return False


def _utc_timestamp(value: object, path: Path) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"invalid manifest timestamp: {path}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid manifest timestamp: {path}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"invalid manifest timestamp: {path}")
    return parsed.astimezone(UTC)


def strict_camera_snapshot(path: Path) -> dict[str, Any] | None:
    """Read a camera registry for catalog tooling without serving-path fallbacks."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read camera source: {path}") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"invalid camera source: {path}")
    cameras = raw.get("cameras")
    if not isinstance(cameras, list):
        raise TypeError(f"invalid camera source: {path}")
    registry_version = raw.get("registry_version")
    if (
        isinstance(registry_version, bool)
        or not isinstance(registry_version, int)
        or registry_version < 0
    ):
        raise ValueError(f"invalid camera registry_version: {path}")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    field_types: dict[str, tuple[type, ...]] = {
        "label": (str,),
        "backend_camera_id": (str, type(None)),
        "decode_backend": (str, type(None)),
        "created_at": (str,),
        "mapping_pending": (bool,),
    }
    for index, record in enumerate(cameras):
        if not isinstance(record, dict):
            raise TypeError(f"invalid camera record at index {index}: {path}")
        camera_id = record.get("id")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError(f"invalid camera id at index {index}: {path}")
        if camera_id in seen_ids:
            raise ValueError(f"duplicate camera id {camera_id}: {path}")
        seen_ids.add(camera_id)
        for field, accepted_types in field_types.items():
            if field in record and not isinstance(record[field], accepted_types):
                raise TypeError(f"invalid camera field {field} at index {index}: {path}")
        validated.append(record)
    return {"registry_version": registry_version, "cameras": validated}


def sanitized_camera_payload(camera: dict[str, Any]) -> dict[str, Any]:
    """Project a camera record to the scalar catalog fields safe to persist."""
    return {
        key: camera[key]
        for key in _CAMERA_PAYLOAD_FIELDS
        if key in camera and isinstance(camera[key], _CAMERA_SCALAR_TYPES)
    }


@dataclass(frozen=True)
class CatalogRecord:
    """A catalog payload accompanied by its primary and promoted SQL values."""

    key: str
    columns: dict[str, Any]
    payload: dict[str, Any]


class CatalogConflictError(ValueError):
    """A stable id was submitted with content different from its first write."""


@dataclass(slots=True)
class CatalogSchemaNewerThanSupportedError(Exception):
    """A downgraded ml-api binary opened a catalog written by a newer schema.

    Mirrors the worker-side ``NewerSchemaVersionError``
    (``worker/pipeline/output/evidence/evidence_outbox_types.py``), which
    ``get_catalog_store`` treats as "storage unavailable, degrade" the same
    way it treats ``OSError``/``sqlite3.Error`` -- a schema downgrade must not
    crash ml-api startup.
    """

    found: int
    supported: int

    def __str__(self) -> str:
        return f"catalog schema {self.found} is newer than supported {self.supported}"


def _raise_conflict(table: str, key: str) -> None:
    raise CatalogConflictError(f"conflicting {table} record: {key}")


def _column_values(table: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in _TABLE_COLUMNS[table]:
        value = payload.get(column)
        if column == "size_bytes" and (isinstance(value, bool) or not isinstance(value, int)):
            value = None
        elif column != "size_bytes" and not isinstance(value, str):
            value = None
        values.append(value)
    return tuple(values)


class CatalogStore:
    """Single-process API catalog with explicit crash-safe SQLite policy."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()

    @property
    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect(self._path)
            self._local.connection = connection
            with self._connections_lock:
                self._connections.append(connection)
        return connection

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            path, timeout=5.0, isolation_level=None, check_same_thread=False
        )
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @classmethod
    def from_env(cls) -> CatalogStore:
        return cls.open(_catalog_path())

    @classmethod
    def open(cls, path: Path | str) -> CatalogStore:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = cls._connect(path)
        try:
            cls._migrate(connection)
            os.chmod(path, 0o600)
        finally:
            connection.close()
        return cls(path)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise CatalogSchemaNewerThanSupportedError(found=version, supported=SCHEMA_VERSION)
        if version == SCHEMA_VERSION:
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            if version == 0:
                for statement in _CREATE_STATEMENTS:
                    connection.execute(statement)
            elif version == 1:
                for table, columns in _TABLE_COLUMNS.items():
                    for column in columns:
                        kind = "INTEGER" if column == "size_bytes" else "TEXT"
                        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                for statement in _V3_TABLE_STATEMENTS:
                    connection.execute(statement)
            elif version == 2:
                for statement in _V3_TABLE_STATEMENTS:
                    connection.execute(statement)
            else:
                raise CatalogSchemaNewerThanSupportedError(  # noqa: TRY301
                    found=version, supported=SCHEMA_VERSION
                )
            for table, key in _TABLE_KEYS.items():
                rows = connection.execute(f"SELECT {key}, payload_json FROM {table}").fetchall()
                columns = _TABLE_COLUMNS[table]
                assignments = ", ".join(f"{column} = ?" for column in columns)
                for record_key, encoded in rows:
                    payload = json.loads(encoded)
                    if isinstance(payload, dict):
                        connection.execute(
                            f"UPDATE {table} SET {assignments} WHERE {key} = ?",
                            (*_column_values(table, payload), record_key),
                        )
            for statement in _INDEX_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._connections_lock:
            for connection in self._connections:
                connection.close()
            self._connections.clear()

    def record(self, table: str, key: str, payload: dict[str, Any]) -> bool:
        """Compare-or-insert a canonical payload, returning True only on insertion."""
        return self.record_many(((table, key, payload),))[0]

    def record_many(self, records: tuple[tuple[str, str, dict[str, Any]], ...]) -> tuple[bool, ...]:
        """Atomically compare-or-insert all records in one transaction."""
        connection = self._connection
        encoded_records: list[tuple[str, str, dict[str, Any], str]] = []
        for table, key, payload in records:
            if table not in _TABLE_KEYS:
                raise ValueError("unknown catalog table")
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            encoded_records.append((table, key, payload, encoded))
        connection.execute("BEGIN IMMEDIATE")
        try:
            inserted: list[bool] = []
            for table, key, payload, encoded in encoded_records:
                key_column = _TABLE_KEYS[table]
                row = connection.execute(
                    f"SELECT payload_json FROM {table} WHERE {key_column} = ?", (key,)
                ).fetchone()
                if row is not None:
                    if row[0] != encoded:
                        _raise_conflict(table, key)
                    inserted.append(False)
                    continue
                columns = _TABLE_COLUMNS[table]
                insert_columns = (key_column, *columns, "payload_json")
                placeholders = ", ".join("?" for _ in insert_columns)
                connection.execute(
                    f"INSERT INTO {table} ({', '.join(insert_columns)}) VALUES ({placeholders})",
                    (key, *_column_values(table, payload), encoded),
                )
                inserted.append(True)
            connection.execute("COMMIT")
            return tuple(inserted)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def list_clips(
        self,
        camera_id: str | None = None,
        *,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[str] = []
        for column, value, operator in (
            ("camera_id", camera_id, "="),
            ("started_at", started_at_from, ">="),
            ("started_at", started_at_to, "<="),
            ("event_type", event_type, "="),
        ):
            if value is not None:
                clauses.append(f"{column} {operator} ?")
                values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT payload_json FROM clips{where} ORDER BY started_at DESC, clip_id", values
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def records_with_columns(self, table: str) -> list[CatalogRecord]:
        """Return canonical payloads with the persisted primary/promoted SQL values."""
        if table not in _TABLE_KEYS:
            raise ValueError("unknown catalog table")
        key_column = _TABLE_KEYS[table]
        columns = _TABLE_COLUMNS[table]
        rows = self._connection.execute(
            f"SELECT {key_column}, {', '.join(columns)}, payload_json FROM {table}"
        ).fetchall()
        return [
            CatalogRecord(
                key=str(row[0]),
                columns=dict(zip(columns, row[1:-1], strict=True)),
                payload=json.loads(row[-1]),
            )
            for row in rows
        ]

    def records(self, table: str) -> list[dict[str, Any]]:
        return [record.payload for record in self.records_with_columns(table)]

    def integrity_check(self) -> str:
        return str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])

    def backfill(
        self,
        clip_store: Any,
        camera_registry: Any | None = None,
        audit_log: Any | None = None,
        label_store: Any | None = None,
    ) -> None:
        for record in strict_manifest_records(clip_store):
            self.record("clips", record.manifest.clip_id, record.payload)
            if label_store is not None:
                label = label_store.get(record.manifest.clip_id)
                if label is not None:
                    self.record("labels", record.manifest.clip_id, label.as_response())
        if camera_registry is not None:
            for camera in camera_registry.snapshot().get("cameras", []):
                if isinstance(camera, dict) and isinstance(camera.get("id"), str) and camera["id"]:
                    self.record("cameras", camera["id"], sanitized_camera_payload(camera))
        if audit_log is not None:
            for entry in audit_log.list_entries():
                audit_id = str(
                    entry.get("audit_id")
                    or entry.get("id")
                    or hashlib.sha256(
                        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                )
                self.record("audit", audit_id, entry)


__all__ = [
    "CatalogConflictError",
    "CatalogSchemaNewerThanSupportedError",
    "StrictManifest",
    "CatalogRecord",
    "sanitized_camera_payload",
    "strict_manifest_records",
    "strict_camera_snapshot",
    "CatalogStore",
    "get_catalog_store",
]
