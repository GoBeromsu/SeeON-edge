"""ml-api-owned SQLite catalog for clip-store metadata.

The edge worker never opens this database; it supplies metadata through relay.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database
from backend.app.features.clips.manifest import ClipExtension, ExtensionContributor
from backend.app.features.clips.store import ClipManifest, is_valid_clip_id

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
    "STREAM_EPOCH_MISMATCH",
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
    "audio_codec",
    "duration_ms",
    "state_version",
    "reason_code",
    "event_ref",
    "event_type",
    "domain",
    "started_at",
    "duration_s",
    "encoder",
    "path",
    "finalized",
    "video_available",
    "video_error",
    "runtime_manifest_sha256",
    "decision_trace_id",
    "recovery_state",
    "source_media",
    "source_error_reason",
    "truncation_reasons",
    "time_origin",
    "detected_at",
    "extension",
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
_CATALOG_TABLE_STATEMENTS = (
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
)
_CREATE_STATEMENTS = (*_CATALOG_TABLE_STATEMENTS, *_V3_TABLE_STATEMENTS)
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
    return EDGE_DATABASE_PATH


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
    runtime_manifest_sha256 = payload.get("runtime_manifest_sha256")
    if runtime_manifest_sha256 is not None and (
        not isinstance(runtime_manifest_sha256, str)
        or _SHA256_RE.fullmatch(runtime_manifest_sha256) is None
    ):
        raise ValueError(f"invalid runtime manifest identity: {path}")

    state = payload.get("state")
    if state not in _MANIFEST_STATES:
        raise ValueError(f"invalid manifest state: {path}")
    _validate_source_remux_metadata(payload, path, state)
    extension = _validate_extension(payload.get("extension"), path)
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
    # Reader tolerance staged ahead of the worker writer (P0-AC7): an optional
    # RFC3339-Z event time; older manifests without it stay valid.
    detected_at = payload.get("detected_at")
    if detected_at is not None:
        _utc_timestamp(detected_at, path)
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
    if state == "READY":
        codec = payload.get("codec")
        if not isinstance(codec, str) or not codec.strip():
            raise ValueError(f"invalid manifest codec: {path}")
    else:
        encoder = payload.get("encoder")
        codec = encoder if isinstance(encoder, str) else ""
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
        detected_at=detected_at if isinstance(detected_at, str) else None,
        truncation_reasons=tuple(payload.get("truncation_reasons") or ()),
        extension=extension,
    )


def _validate_source_remux_metadata(
    payload: dict[str, Any],
    path: Path,
    state: str,
) -> None:
    recovery_state = payload.get("recovery_state")
    expected_recovery_state = "MEDIA_VERIFIED" if state == "READY" else "UNAVAILABLE"
    if recovery_state is not None and recovery_state != expected_recovery_state:
        raise ValueError(f"invalid manifest recovery state: {path}")
    for field in ("audio_codec", "source_error_reason"):
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"invalid manifest {field}: {path}")
    decision_trace_id = payload.get("decision_trace_id")
    if decision_trace_id is not None and (
        not isinstance(decision_trace_id, str) or _SHA256_RE.fullmatch(decision_trace_id) is None
    ):
        raise ValueError(f"invalid manifest decision trace identity: {path}")
    source_media = payload.get("source_media")
    if source_media is not None:
        if not isinstance(source_media, dict):
            raise ValueError(f"invalid manifest source media: {path}")
        _validate_source_media_translation(source_media, path)
    time_origin = payload.get("time_origin")
    if time_origin is not None and not isinstance(time_origin, dict):
        raise ValueError(f"invalid manifest time origin: {path}")
    truncations = payload.get("truncation_reasons")
    if truncations is not None and (
        not isinstance(truncations, list)
        or any(not isinstance(value, str) or not value for value in truncations)
        or len(set(truncations)) != len(truncations)
    ):
        raise ValueError(f"invalid manifest truncation reasons: {path}")


def _validate_extension(value: Any, path: Path) -> ClipExtension | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "contributors",
        "duration_s",
        "boundary",
    }:
        raise ValueError(f"invalid manifest extension: {path}")
    boundary = value["boundary"]
    duration_s = value["duration_s"]
    contributors = value["contributors"]
    if (
        not isinstance(boundary, str)
        or boundary not in {"none", "extension_bounded", "extension_raced"}
        or isinstance(duration_s, bool)
        or not isinstance(duration_s, int | float)
        or not math.isfinite(duration_s)
        or duration_s < 0
        or not isinstance(contributors, list)
        or not contributors
    ):
        raise ValueError(f"invalid manifest extension: {path}")
    parsed: list[ExtensionContributor] = []
    for contributor in contributors:
        if not isinstance(contributor, dict) or set(contributor) != {"event_ref", "detected_at"}:
            raise ValueError(f"invalid manifest extension contributor: {path}")
        event_ref = contributor["event_ref"]
        detected_at = contributor["detected_at"]
        if not isinstance(event_ref, str) or not event_ref.strip():
            raise ValueError(f"invalid manifest extension contributor: {path}")
        _utc_timestamp(detected_at, path)
        parsed.append(ExtensionContributor(event_ref=event_ref, detected_at=detected_at))
    return ClipExtension(
        contributors=tuple(parsed),
        duration_s=float(duration_s),
        boundary=boundary,
    )


def _validate_source_media_translation(
    source_media: dict[str, Any],
    path: Path,
) -> None:
    raw_translation = source_media.get("timestamp_translation_seconds")
    streams = source_media.get("streams")
    if not isinstance(raw_translation, str) or not isinstance(streams, list) or not streams:
        raise ValueError(f"invalid remux timestamp translation: {path}")
    try:
        translation = Fraction(raw_translation)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid remux timestamp translation: {path}") from exc
    if raw_translation != f"{translation.numerator}/{translation.denominator}":
        raise ValueError(f"noncanonical remux timestamp translation: {path}")
    for stream in streams:
        if not isinstance(stream, dict):
            raise TypeError(f"invalid remux stream facts: {path}")
        raw_time_base = stream.get("time_base")
        packet_count = stream.get("packet_count")
        ticks = stream.get("timestamp_translation_ticks")
        if (
            not isinstance(raw_time_base, str)
            or isinstance(packet_count, bool)
            or not isinstance(packet_count, int)
            or packet_count < 0
            or isinstance(ticks, bool)
            or not isinstance(ticks, int | None)
            or (packet_count == 0) != (ticks is None)
        ):
            raise TypeError(f"invalid remux stream translation: {path}")
        try:
            time_base = Fraction(raw_time_base)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"invalid remux stream time base: {path}") from exc
        if time_base <= 0 or (ticks is not None and ticks * time_base != translation):
            raise ValueError(f"nonuniform remux timestamp translation: {path}")


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
    if not isinstance(value, str) or not value.endswith("Z"):
        raise TypeError(f"invalid manifest timestamp: {path}")
    try:
        parsed = datetime.fromisoformat(value)
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


class _CatalogStoreLifecycle(Enum):
    OPEN, CLOSING, CLOSED = auto(), auto(), auto()


def _raise_conflict(table: str, key: str) -> None:
    raise CatalogConflictError(f"conflicting {table} record: {key}")


def _column_values(table: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in _TABLE_COLUMNS[table]:
        value = payload.get(column)
        well_typed = (
            isinstance(value, int) and not isinstance(value, bool)
            if column == "size_bytes"
            else isinstance(value, str)
        )
        if not well_typed:
            value = None
        values.append(value)
    return tuple(values)


class CatalogStore:
    """Single-process API catalog with explicit crash-safe SQLite policy."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self._path = path
        self._connection = connection
        self._operation_lock = threading.Lock()
        self._state_condition = threading.Condition()
        self._state = _CatalogStoreLifecycle.OPEN

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        if path.name == "edge.sqlite3":
            return open_runtime_database(path, actor=RuntimeActor.API, check_same_thread=False)
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
        with ExitStack() as cleanup:
            connection = cls._connect(path)
            cleanup.callback(connection.close)
            if path.name != "edge.sqlite3":
                cls._migrate(connection)
            os.chmod(path, 0o600)
            store = cls(path, connection)
            _ = cleanup.pop_all()
            return store

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
        with self._state_condition:
            if self._state is _CatalogStoreLifecycle.CLOSED:
                return
            if self._state is _CatalogStoreLifecycle.CLOSING:
                _ = self._state_condition.wait_for(
                    lambda: self._state is _CatalogStoreLifecycle.CLOSED
                )
                return
            self._state = _CatalogStoreLifecycle.CLOSING
        try:
            with self._operation_lock:
                self._connection.close()
        finally:
            with self._state_condition:
                self._state = _CatalogStoreLifecycle.CLOSED
                self._state_condition.notify_all()

    def record(self, table: str, key: str, payload: dict[str, Any]) -> bool:
        """Compare-or-insert a canonical payload, returning True only on insertion."""
        with self._serialized_operation():
            return self._record_unlocked(table, key, payload)

    def record_many(self, records: tuple[tuple[str, str, dict[str, Any]], ...]) -> tuple[bool, ...]:
        """Atomically compare-or-insert all records in one transaction."""
        with self._serialized_operation():
            return self._record_many_unlocked(records)

    def _record_unlocked(self, table: str, key: str, payload: dict[str, Any]) -> bool:
        return self._record_many_unlocked(((table, key, payload),))[0]

    def _record_many_unlocked(
        self, records: tuple[tuple[str, str, dict[str, Any]], ...]
    ) -> tuple[bool, ...]:
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
        with self._serialized_operation():
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
        with self._serialized_operation():
            return self._records_with_columns_unlocked(table)

    def _records_with_columns_unlocked(self, table: str) -> list[CatalogRecord]:
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
        with self._serialized_operation():
            return [record.payload for record in self._records_with_columns_unlocked(table)]

    def commit_artifact_receipt(
        self, clip_id: str, sha256: str, size_bytes: int
    ) -> tuple[str, int, bool]:
        """Durably accept a locally verified primary artifact.

        The existing API-owned ``clips`` catalog is the receipt projection:
        content identity remains immutable while the explicit acceptance marker
        is added atomically only after the backend has verified local bytes.
        """
        with self._serialized_operation():
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT payload_json FROM clips WHERE clip_id = ?", (clip_id,)
                ).fetchone()
                if row is None:
                    payload: dict[str, Any] = {
                        "clip_id": clip_id,
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "state": "READY",
                        "backend_receipt_accepted": True,
                    }
                    connection.execute(
                        "INSERT INTO clips (clip_id, state, sha256, size_bytes, payload_json) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            clip_id,
                            "READY",
                            sha256,
                            size_bytes,
                            json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                else:
                    payload = json.loads(str(row[0]))
                    if (
                        not isinstance(payload, dict)
                        or payload.get("sha256") != sha256
                        or payload.get("size_bytes") != size_bytes
                    ):
                        _raise_conflict("clips", clip_id)
                    if payload.get("backend_receipt_accepted") is not True:
                        payload["backend_receipt_accepted"] = True
                        connection.execute(
                            "UPDATE clips SET payload_json = ? WHERE clip_id = ?",
                            (json.dumps(payload, sort_keys=True, separators=(",", ":")), clip_id),
                        )
                connection.execute("COMMIT")
                return sha256, size_bytes, True  # noqa: TRY300
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def artifact_receipt(self, clip_id: str) -> tuple[str, int, bool] | None:
        with self._serialized_operation():
            row = self._connection.execute(
                "SELECT sha256, size_bytes, payload_json FROM clips WHERE clip_id = ?", (clip_id,)
            ).fetchone()
            if row is None:
                return None
            payload = json.loads(str(row[2]))
            if (
                not isinstance(payload, dict)
                or not isinstance(row[0], str)
                or isinstance(row[1], bool)
                or not isinstance(row[1], int)
            ):
                return None
            return row[0], row[1], payload.get("backend_receipt_accepted") is True

    def integrity_check(self) -> str:
        with self._serialized_operation():
            return str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])

    def backfill(
        self,
        clip_store: Any,
        camera_registry: Any | None = None,
    ) -> None:
        with self._serialized_operation():
            for record in strict_manifest_records(clip_store):
                self._record_unlocked("clips", record.manifest.clip_id, record.payload)
            if camera_registry is not None:
                for camera in camera_registry.snapshot().get("cameras", []):
                    if (
                        isinstance(camera, dict)
                        and isinstance(camera.get("id"), str)
                        and camera["id"]
                    ):
                        self._record_unlocked(
                            "cameras", camera["id"], sanitized_camera_payload(camera)
                        )

    @contextmanager
    def _serialized_operation(self) -> Generator[None, None, None]:
        with self._operation_lock:
            with self._state_condition:
                if self._state is not _CatalogStoreLifecycle.OPEN:
                    raise RuntimeError("catalog store is closed")
            yield


__all__ = [
    "CatalogConflictError",
    "CatalogRecord",
    "CatalogSchemaNewerThanSupportedError",
    "CatalogStore",
    "StrictManifest",
    "get_catalog_store",
    "sanitized_camera_payload",
    "strict_camera_snapshot",
    "strict_manifest_records",
]
