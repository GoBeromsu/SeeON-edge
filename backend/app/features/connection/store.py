"""SQLite persistence for deployment-owned URLs and runtime facility enrollment."""
# noqa: SIZE_OK

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Final, TypedDict
from uuid import uuid4

API_CONNECTION_SETTINGS_PATH_ENV: Final = "API_CONNECTION_SETTINGS_PATH"
API_BACKEND_BASE_URL_ENV: Final = "API_BACKEND_BASE_URL"
API_BACKEND_CONFIG_URL_ENV: Final = "API_BACKEND_CONFIG_URL"
API_BACKEND_EVENTS_URL_ENV: Final = "API_BACKEND_EVENTS_URL"
DEFAULT_CONNECTION_SETTINGS_PATH: Final = "/var/lib/ml-api/connection-settings.sqlite3"

logger = logging.getLogger(__name__)

_MIGRATION_VERSION: Final = 1
_SAVE_FIELDS: Final = (
    "events_url",
    "config_url",
    "facility_code",
    "facility_id",
    "facility_token",
    "edge_installation_id",
    "enrollment_generation",
)
_COLUMNS: Final = (
    *_SAVE_FIELDS,
    "enrollment_created_at",
    "enrollment_updated_at",
    "updated_at",
)
_V1_COLUMNS: Final = (
    "facility_code",
    "edge_installation_id",
    "enrollment_generation",
    "enrollment_created_at",
    "enrollment_updated_at",
)
_SCHEMA_SQL: Final = """CREATE TABLE IF NOT EXISTS connection_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    events_url TEXT,
    config_url TEXT,
    facility_id TEXT,
    facility_token TEXT,
    updated_at TEXT,
    facility_code TEXT,
    edge_installation_id TEXT,
    enrollment_generation INTEGER CHECK (enrollment_generation > 0),
    enrollment_created_at TEXT,
    enrollment_updated_at TEXT
) STRICT"""
_MIGRATION_SCHEMA_SQL: Final = """CREATE TABLE IF NOT EXISTS connection_store_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    backup_filename TEXT,
    backup_sha256 TEXT,
    backup_size_bytes INTEGER
) STRICT"""
_ALTER_STATEMENTS: Final = (
    "ALTER TABLE connection_settings ADD COLUMN facility_code TEXT",
    "ALTER TABLE connection_settings ADD COLUMN edge_installation_id TEXT",
    "ALTER TABLE connection_settings ADD COLUMN enrollment_generation INTEGER "
    + "CHECK (enrollment_generation > 0)",
    "ALTER TABLE connection_settings ADD COLUMN enrollment_created_at TEXT",
    "ALTER TABLE connection_settings ADD COLUMN enrollment_updated_at TEXT",
)


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    events_url: str | None
    config_url: str | None
    facility_id: str | None
    facility_token: str | None = field(repr=False)
    updated_at: str | None
    facility_code: str | None = None
    edge_installation_id: str | None = None
    enrollment_generation: int | None = None
    enrollment_created_at: str | None = None
    enrollment_updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class ConnectionStoreBackup:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class InvalidConnectionSettingError(ValueError):
    field_name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.reason}: {self.field_name}"


class MaskedConnectionSettings(TypedDict):
    events_url: str | None
    config_url: str | None
    facility_id: str | None
    facility_token_masked: str | None
    facility_token_set: bool
    updated_at: str | None


def _normalize_api_base(base: str | None) -> str | None:
    if not base:
        return None
    trimmed = base.strip().rstrip("/")
    if not trimmed:
        return None
    return trimmed if trimmed.endswith("/api") else f"{trimmed}/api"


class ConnectionSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self.rollback_directory: Path = self.path.parent / "connection-settings-rollback"
        self._lock: Lock = Lock()

    @classmethod
    def from_env(cls) -> ConnectionSettingsStore:
        return cls(
            os.environ.get(API_CONNECTION_SETTINGS_PATH_ENV, DEFAULT_CONNECTION_SETTINGS_PATH)
        )

    def load(self) -> ConnectionSettings:
        with self._lock:
            saved = self._read_unlocked()
        base = _normalize_api_base(os.environ.get(API_BACKEND_BASE_URL_ENV))
        return ConnectionSettings(
            events_url=(
                _text(saved["events_url"])
                or os.environ.get(API_BACKEND_EVENTS_URL_ENV)
                or (f"{base}/v1/events" if base else None)
            ),
            config_url=(
                _text(saved["config_url"])
                or os.environ.get(API_BACKEND_CONFIG_URL_ENV)
                or (f"{base}/v1/ml-config" if base else None)
            ),
            facility_id=_text(saved["facility_id"]),
            facility_token=_text(saved["facility_token"]),
            updated_at=_text(saved["updated_at"]),
            facility_code=_text(saved["facility_code"]),
            edge_installation_id=_text(saved["edge_installation_id"]),
            enrollment_generation=_positive_int(saved["enrollment_generation"]),
            enrollment_created_at=_text(saved["enrollment_created_at"]),
            enrollment_updated_at=_text(saved["enrollment_updated_at"]),
        )

    def save(self, updates: Mapping[str, str | int | None]) -> ConnectionSettings:
        self._validate_updates(updates)
        with self._lock:
            data = self._read_unlocked()
            data.update(dict(updates))
            timestamp = utc_now_iso()
            data["updated_at"] = timestamp
            if set(updates) & set(_V1_COLUMNS[:3]):
                self._validate_complete_enrollment(data)
                data["enrollment_created_at"] = data["enrollment_created_at"] or timestamp
                data["enrollment_updated_at"] = timestamp
            self._write_unlocked(data)
        return self.load()

    def masked(self) -> MaskedConnectionSettings:
        settings = self.load()
        return {
            "events_url": settings.events_url,
            "config_url": settings.config_url,
            "facility_id": settings.facility_id,
            "facility_token_masked": mask_facility_token(settings.facility_token),
            "facility_token_set": bool(settings.facility_token),
            "updated_at": settings.updated_at,
        }

    def create_pre_v1_backup(self) -> ConnectionStoreBackup:
        with self._lock, closing(self._connect_unlocked(create=False)) as source:
            return self._create_pre_v1_backup_unlocked(source)

    def integrity_check(self, path: str | Path | None = None) -> str:
        database = Path(path) if path is not None else self.path
        with self._lock:
            return self._integrity_check_path(database)

    def restore_pre_v1_backup(self, backup_path: str | Path) -> None:
        backup = Path(backup_path)
        with self._lock:
            if self._integrity_check_path(backup) != "ok":
                raise sqlite3.DatabaseError("connection backup integrity check failed")
            with closing(sqlite3.connect(backup, timeout=5.0)) as source:
                row: tuple[
                    str | None, str | None, str | None, str | None, str | None
                ] | None = source.execute(
                    "SELECT events_url, config_url, facility_id, facility_token, updated_at "
                    + "FROM connection_settings WHERE id = 1"
                ).fetchone()
            with closing(self._connect_unlocked(create=True)) as destination:
                self._ensure_schema_unlocked(destination)
                self._checkpoint_unlocked(destination)
                with destination:
                    if row is None:
                        _ = destination.execute("DELETE FROM connection_settings WHERE id = 1")
                    else:
                        restored: dict[str, str | int | None] = {
                            key: None for key in _COLUMNS
                        }
                        for key, value in zip(
                            (
                                "events_url",
                                "config_url",
                                "facility_id",
                                "facility_token",
                                "updated_at",
                            ),
                            row,
                            strict=True,
                        ):
                            restored[key] = value
                        self._upsert_unlocked(destination, restored)

    def _connect_unlocked(self, *, create: bool) -> sqlite3.Connection:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        os.chmod(self.path, 0o600)
        _ = connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _read_unlocked(self) -> dict[str, str | int | None]:
        empty: dict[str, str | int | None] = {key: None for key in _COLUMNS}
        if not self.path.exists():
            return empty
        try:
            with closing(self._connect_unlocked(create=False)) as connection:
                try:
                    self._ensure_schema_unlocked(connection)
                    return self._select_unlocked(connection)
                except (OSError, sqlite3.DatabaseError) as exc:
                    logger.warning(
                        "connection settings migration unavailable at %s: %r", self.path, exc
                    )
                    return self._select_legacy_unlocked(connection, empty)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("connection settings store unreadable at %s: %r", self.path, exc)
            return empty

    def _write_unlocked(self, data: dict[str, str | int | None]) -> None:
        with closing(self._connect_unlocked(create=True)) as connection:
            self._ensure_schema_unlocked(connection)
            with connection:
                self._upsert_unlocked(connection, data)

    def _ensure_schema_unlocked(self, connection: sqlite3.Connection) -> None:
        table_exists: tuple[int] | None = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'connection_settings'"
        ).fetchone()
        if table_exists is None:
            _ = connection.execute(_SCHEMA_SQL)
            self._record_migration_unlocked(connection, None)
            connection.commit()
            return
        schema_rows: list[tuple[int, str, str, int, str | None, int]] = connection.execute(
            "PRAGMA table_info(connection_settings)"
        ).fetchall()
        columns = {row[1] for row in schema_rows}
        if set(_V1_COLUMNS) <= columns:
            _ = connection.execute(_MIGRATION_SCHEMA_SQL)
            return
        backup = self._create_pre_v1_backup_unlocked(connection)
        with connection:
            for statement in _ALTER_STATEMENTS:
                _ = connection.execute(statement)
            self._record_migration_unlocked(connection, backup)

    def _record_migration_unlocked(
        self, connection: sqlite3.Connection, backup: ConnectionStoreBackup | None
    ) -> None:
        _ = connection.execute(_MIGRATION_SCHEMA_SQL)
        _ = connection.execute(
            "INSERT OR IGNORE INTO connection_store_migrations "
            + "(version, name, applied_at, backup_filename, backup_sha256, backup_size_bytes) "
            + "VALUES (?, ?, ?, ?, ?, ?)",
            (
                _MIGRATION_VERSION,
                "runtime-facility-enrollment",
                utc_now_iso(),
                backup.path.name if backup else None,
                backup.sha256 if backup else None,
                backup.size_bytes if backup else None,
            ),
        )

    def _create_pre_v1_backup_unlocked(
        self, source: sqlite3.Connection
    ) -> ConnectionStoreBackup:
        self.rollback_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.rollback_directory, 0o700)
        self._checkpoint_unlocked(source)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        temporary = self.rollback_directory / f".{self.path.name}.{uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(temporary, timeout=5.0)) as destination:
                type(self)._copy_database_unlocked(source, destination)
                if self._integrity_check_connection(destination) != "ok":
                    raise sqlite3.DatabaseError("connection backup integrity check failed")
            os.chmod(temporary, 0o600)
            with temporary.open("rb") as backup_file:
                os.fsync(backup_file.fileno())
                digest = hashlib.file_digest(backup_file, "sha256").hexdigest()
            final_path = self.rollback_directory / (
                f"{self.path.name}.pre-v1.{timestamp}.{digest}"
            )
            os.replace(temporary, final_path)
            self._fsync_directory(self.rollback_directory)
            return ConnectionStoreBackup(final_path, digest, final_path.stat().st_size)
        except (OSError, sqlite3.Error, KeyboardInterrupt):
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _copy_database_unlocked(
        source: sqlite3.Connection, destination: sqlite3.Connection
    ) -> None:
        source.backup(destination)

    @staticmethod
    def _checkpoint_unlocked(connection: sqlite3.Connection) -> None:
        result: tuple[int, int, int] | None = connection.execute(
            "PRAGMA wal_checkpoint(FULL)"
        ).fetchone()
        if result is not None and int(result[0]) != 0:
            raise sqlite3.OperationalError("connection database WAL checkpoint remained busy")

    @staticmethod
    def _integrity_check_connection(connection: sqlite3.Connection) -> str:
        row: tuple[str] | None = connection.execute("PRAGMA integrity_check").fetchone()
        if row is None:
            raise sqlite3.DatabaseError("connection database integrity check returned no result")
        return row[0]

    def _integrity_check_path(self, path: Path) -> str:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)) as connection:
            return self._integrity_check_connection(connection)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _select_unlocked(connection: sqlite3.Connection) -> dict[str, str | int | None]:
        row: tuple[
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
            int | None,
            str | None,
            str | None,
            str | None,
        ] | None = connection.execute(
            "SELECT events_url, config_url, facility_code, facility_id, facility_token, "
            + "edge_installation_id, enrollment_generation, enrollment_created_at, "
            + "enrollment_updated_at, updated_at FROM connection_settings WHERE id = 1"
        ).fetchone()
        if row is None:
            return {key: None for key in _COLUMNS}
        result: dict[str, str | int | None] = {}
        for key, value in zip(_COLUMNS, row, strict=True):
            result[key] = value if isinstance(value, (str, int)) else None
        return result

    @staticmethod
    def _select_legacy_unlocked(
        connection: sqlite3.Connection, empty: dict[str, str | int | None]
    ) -> dict[str, str | int | None]:
        try:
            row: tuple[
                str | None, str | None, str | None, str | None, str | None
            ] | None = connection.execute(
                "SELECT events_url, config_url, facility_id, facility_token, updated_at "
                + "FROM connection_settings WHERE id = 1"
            ).fetchone()
        except sqlite3.DatabaseError:
            return empty
        if row is not None:
            for key, value in zip(
                ("events_url", "config_url", "facility_id", "facility_token", "updated_at"),
                row,
                strict=True,
            ):
                empty[key] = value
        return empty

    @staticmethod
    def _upsert_unlocked(
        connection: sqlite3.Connection, data: dict[str, str | int | None]
    ) -> None:
        _ = connection.execute(
            "INSERT INTO connection_settings "
            + "(id, events_url, config_url, facility_code, facility_id, facility_token, "
            + "edge_installation_id, enrollment_generation, enrollment_created_at, "
            + "enrollment_updated_at, updated_at) VALUES "
            + "(1, :events_url, :config_url, :facility_code, :facility_id, :facility_token, "
            + ":edge_installation_id, :enrollment_generation, :enrollment_created_at, "
            + ":enrollment_updated_at, :updated_at) ON CONFLICT(id) DO UPDATE SET "
            + "events_url=excluded.events_url, config_url=excluded.config_url, "
            + "facility_code=excluded.facility_code, facility_id=excluded.facility_id, "
            + "facility_token=excluded.facility_token, "
            + "edge_installation_id=excluded.edge_installation_id, "
            + "enrollment_generation=excluded.enrollment_generation, "
            + "enrollment_created_at=excluded.enrollment_created_at, "
            + "enrollment_updated_at=excluded.enrollment_updated_at, "
            + "updated_at=excluded.updated_at",
            data,
        )

    @staticmethod
    def _validate_updates(updates: Mapping[str, str | int | None]) -> None:
        unknown = set(updates) - set(_SAVE_FIELDS)
        if unknown:
            raise InvalidConnectionSettingError(
                field_name=", ".join(sorted(unknown)),
                reason="unknown connection setting field(s)",
            )
        for key in ("facility_code", "facility_id", "facility_token", "edge_installation_id"):
            value = updates.get(key)
            invalid_text = value is not None and (
                not isinstance(value, str) or not value.strip()
            )
            if key in updates and invalid_text:
                raise InvalidConnectionSettingError(
                    field_name=key,
                    reason="invalid connection setting field",
                )
        generation = updates.get("enrollment_generation")
        if generation is not None and (type(generation) is not int or generation < 1):
            raise InvalidConnectionSettingError(
                field_name="enrollment_generation",
                reason="invalid connection setting field",
            )

    @staticmethod
    def _validate_complete_enrollment(data: dict[str, str | int | None]) -> None:
        required = (
            "facility_code",
            "facility_token",
            "facility_id",
            "edge_installation_id",
            "enrollment_generation",
        )
        if any(data[key] is None for key in required):
            raise InvalidConnectionSettingError(
                field_name="runtime_enrollment",
                reason="runtime enrollment fields must be saved atomically",
            )


def _text(value: str | int | None) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int(value: str | int | None) -> int | None:
    return value if type(value) is int and value > 0 else None


def mask_facility_token(token: str | None) -> str | None:
    if not token:
        return None
    return "****" if len(token) <= 4 else f"****{token[-4:]}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "API_BACKEND_BASE_URL_ENV",
    "API_CONNECTION_SETTINGS_PATH_ENV",
    "DEFAULT_CONNECTION_SETTINGS_PATH",
    "ConnectionSettings",
    "ConnectionSettingsStore",
    "ConnectionStoreBackup",
    "InvalidConnectionSettingError",
    "MaskedConnectionSettings",
    "mask_facility_token",
    "utc_now_iso",
]
