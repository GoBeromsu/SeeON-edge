from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeAlias, cast
from uuid import uuid4

from shared.edge_db.connection import RuntimeActor, open_runtime_database

ConnectionValue: TypeAlias = str | int | None
ConnectionData: TypeAlias = dict[str, ConnectionValue]
DatabaseRow: TypeAlias = tuple[ConnectionValue, ...]
SchemaRow: TypeAlias = tuple[int, str, str, int, str | None, int]

SAVE_FIELDS: Final = (
    "events_url", "config_url", "facility_code", "client_installation_ref",
    "facility_id", "facility_token",
    "edge_installation_id", "enrollment_generation",
)
COLUMNS: Final = (*SAVE_FIELDS, "enrollment_created_at", "enrollment_updated_at", "updated_at")
ENROLLMENT_MARKER_FIELDS: Final = (
    "facility_code", "client_installation_ref", "edge_installation_id",
    "enrollment_generation",
)
REQUIRED_ENROLLMENT_FIELDS: Final = (
    "facility_code", "client_installation_ref", "facility_token", "facility_id",
    "edge_installation_id", "enrollment_generation",
)
_SCHEMA_SQL: Final = """CREATE TABLE IF NOT EXISTS connection_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    events_url TEXT, config_url TEXT, facility_id TEXT, facility_token TEXT, updated_at TEXT,
    facility_code TEXT, client_installation_ref TEXT, edge_installation_id TEXT,
    enrollment_generation INTEGER CHECK (enrollment_generation > 0),
    enrollment_created_at TEXT, enrollment_updated_at TEXT
) STRICT"""
_MIGRATION_SCHEMA_SQL: Final = """CREATE TABLE IF NOT EXISTS connection_store_migrations (
    version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL,
    backup_filename TEXT, backup_sha256 TEXT, backup_size_bytes INTEGER
) STRICT"""
_ALTER_STATEMENTS: Final = (
    "ALTER TABLE connection_settings ADD COLUMN facility_code TEXT",
    "ALTER TABLE connection_settings ADD COLUMN edge_installation_id TEXT",
    "ALTER TABLE connection_settings ADD COLUMN enrollment_generation INTEGER "
    + "CHECK (enrollment_generation > 0)",
    "ALTER TABLE connection_settings ADD COLUMN enrollment_created_at TEXT",
    "ALTER TABLE connection_settings ADD COLUMN enrollment_updated_at TEXT",
)
_CLIENT_REF_ALTER: Final = (
    "ALTER TABLE connection_settings ADD COLUMN client_installation_ref TEXT"
)
_LEGACY_COLUMNS: Final = ("events_url", "config_url", "facility_id", "facility_token", "updated_at")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConnectionStoreBackup:
    path: Path
    sha256: str
    size_bytes: int


class ConnectionStoreDatabase:
    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.rollback_directory: Path = path.parent / "connection-settings-rollback"

    def read(self) -> ConnectionData:
        empty = _empty_data()
        if not self.path.exists():
            return empty
        try:
            with closing(self._connect(create=False)) as connection:
                try:
                    self._ensure_schema(connection)
                    return self._select(connection)
                except (OSError, sqlite3.DatabaseError) as exc:
                    logger.warning(
                        "connection settings migration unavailable at %s: %r", self.path, exc
                    )
                    return self._select_legacy(connection, empty)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("connection settings store unreadable at %s: %r", self.path, exc)
            return empty

    def write(self, data: ConnectionData) -> None:
        with closing(self._connect(create=True)) as connection:
            self._ensure_schema(connection)
            with connection:
                self._upsert(connection, data)

    def create_pre_v1_backup(self) -> ConnectionStoreBackup:
        with closing(self._connect(create=False)) as source:
            return self._create_backup(source)

    def integrity_check(self, path: Path) -> str:
        return self._integrity_check_path(path)

    def restore_pre_v1_backup(self, backup: Path) -> None:
        if self._integrity_check_path(backup) != "ok":
            raise sqlite3.DatabaseError("connection backup integrity check failed")
        with closing(sqlite3.connect(backup, timeout=5.0)) as source:
            row = cast(
                DatabaseRow | None,
                source.execute(
                    "SELECT events_url, config_url, facility_id, facility_token, updated_at "
                    + "FROM connection_settings WHERE id = 1"
                ).fetchone(),
            )
        with closing(self._connect(create=True)) as destination:
            self._ensure_schema(destination)
            self._checkpoint(destination)
            with destination:
                if row is None:
                    _ = destination.execute("DELETE FROM connection_settings WHERE id = 1")
                else:
                    restored = _empty_data()
                    restored.update(dict(zip(_LEGACY_COLUMNS, row, strict=True)))
                    self._upsert(destination, restored)

    def _connect(self, *, create: bool) -> sqlite3.Connection:
        if self.path.name == "edge.sqlite3":
            return open_runtime_database(self.path, actor=RuntimeActor.API)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        os.chmod(self.path, 0o600)
        _ = connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        if self.path.name == "edge.sqlite3":
            return
        table_exists = cast(
            tuple[int] | None,
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                + "AND name = 'connection_settings'"
            ).fetchone(),
        )
        if table_exists is None:
            _ = connection.execute(_SCHEMA_SQL)
            self._record_migration(connection, 1, "runtime-facility-enrollment", None)
            connection.commit()
            return
        schema_rows = cast(
            list[SchemaRow],
            connection.execute("PRAGMA table_info(connection_settings)").fetchall(),
        )
        v1_columns = {"facility_code", "edge_installation_id", "enrollment_generation"} | {
            "enrollment_created_at", "enrollment_updated_at",
        }
        existing_columns = {row[1] for row in schema_rows}
        if "client_installation_ref" in existing_columns:
            _ = connection.execute(_MIGRATION_SCHEMA_SQL)
            return
        backup = self._create_backup(connection)
        with connection:
            if not v1_columns <= existing_columns:
                for statement in _ALTER_STATEMENTS:
                    _ = connection.execute(statement)
                self._record_migration(
                    connection, 1, "runtime-facility-enrollment", backup
                )
            _ = connection.execute(_CLIENT_REF_ALTER)
            self._record_migration(
                connection, 2, "enrollment-client-installation-ref", backup
            )

    def _record_migration(
        self,
        connection: sqlite3.Connection,
        version: int,
        name: str,
        backup: ConnectionStoreBackup | None,
    ) -> None:
        _ = connection.execute(_MIGRATION_SCHEMA_SQL)
        _ = connection.execute(
            "INSERT OR IGNORE INTO connection_store_migrations "
            + "(version, name, applied_at, backup_filename, backup_sha256, backup_size_bytes) "
            + "VALUES (?, ?, ?, ?, ?, ?)",
            (
                version,
                name,
                utc_now_iso(),
                backup.path.name if backup else None,
                backup.sha256 if backup else None,
                backup.size_bytes if backup else None,
            ),
        )

    def _create_backup(self, source: sqlite3.Connection) -> ConnectionStoreBackup:
        self.rollback_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.rollback_directory, 0o700)
        self._checkpoint(source)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        temporary = self.rollback_directory / f".{self.path.name}.{uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(temporary, timeout=5.0)) as destination:
                type(self)._copy_database(source, destination)
                if self._integrity_check_connection(destination) != "ok":
                    raise sqlite3.DatabaseError("connection backup integrity check failed")
            os.chmod(temporary, 0o600)
            with temporary.open("rb") as backup_file:
                os.fsync(backup_file.fileno())
                digest = hashlib.file_digest(backup_file, "sha256").hexdigest()
            final_path = self.rollback_directory / f"{self.path.name}.pre-v1.{timestamp}.{digest}"
            os.replace(temporary, final_path)
            _fsync_directory(self.rollback_directory)
            return ConnectionStoreBackup(final_path, digest, final_path.stat().st_size)
        except (OSError, sqlite3.Error, KeyboardInterrupt):
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _copy_database(source: sqlite3.Connection, destination: sqlite3.Connection) -> None:
        source.backup(destination)

    @staticmethod
    def _checkpoint(connection: sqlite3.Connection) -> None:
        result = cast(
            tuple[int, int, int] | None,
            connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone(),
        )
        if result is not None and result[0] != 0:
            raise sqlite3.OperationalError("connection database WAL checkpoint remained busy")

    @staticmethod
    def _integrity_check_connection(connection: sqlite3.Connection) -> str:
        row = cast(tuple[str] | None, connection.execute("PRAGMA integrity_check").fetchone())
        if row is None:
            raise sqlite3.DatabaseError("connection database integrity check returned no result")
        return row[0]

    def _integrity_check_path(self, path: Path) -> str:
        with closing(
            sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        ) as connection:
            return self._integrity_check_connection(connection)

    @staticmethod
    def _select(connection: sqlite3.Connection) -> ConnectionData:
        row = cast(
            DatabaseRow | None,
            connection.execute(
                "SELECT events_url, config_url, facility_code, client_installation_ref, "
                + "facility_id, facility_token, edge_installation_id, enrollment_generation, "
                + "enrollment_created_at, "
                + "enrollment_updated_at, updated_at FROM connection_settings WHERE id = 1"
            ).fetchone(),
        )
        return _empty_data() if row is None else dict(zip(COLUMNS, row, strict=True))

    @staticmethod
    def _select_legacy(
        connection: sqlite3.Connection, empty: ConnectionData
    ) -> ConnectionData:
        try:
            row = cast(
                DatabaseRow | None,
                connection.execute(
                    "SELECT events_url, config_url, facility_id, facility_token, updated_at "
                    + "FROM connection_settings WHERE id = 1"
                ).fetchone(),
            )
        except sqlite3.DatabaseError:
            return empty
        if row is not None:
            empty.update(dict(zip(_LEGACY_COLUMNS, row, strict=True)))
        return empty

    @staticmethod
    def _upsert(connection: sqlite3.Connection, data: ConnectionData) -> None:
        assignments = ", ".join(f"{column}=excluded.{column}" for column in COLUMNS)
        columns = ", ".join(COLUMNS)
        values = ", ".join(f":{column}" for column in COLUMNS)
        _ = connection.execute(
            f"INSERT INTO connection_settings (id, {columns}) VALUES (1, {values}) "
            + f"ON CONFLICT(id) DO UPDATE SET {assignments}",
            data,
        )


def _empty_data() -> ConnectionData:
    return {key: None for key in COLUMNS}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
