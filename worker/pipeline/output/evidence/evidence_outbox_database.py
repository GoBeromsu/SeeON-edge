from __future__ import annotations

import sqlite3
from pathlib import Path

from shared.edge_db.connection import RuntimeActor, open_runtime_database
from worker.pipeline.output.evidence.evidence_outbox_schema import MIGRATIONS, SCHEMA_VERSION
from worker.pipeline.output.evidence.evidence_outbox_types import (
    DatabaseSettings,
    NewerSchemaVersionError,
)


def open_connection(path: Path, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open (creating and migrating if needed) the worker-state database.

    ``busy_timeout_ms`` controls how long a conflicting writer is waited out
    (both the Python-level connect ``timeout`` and ``PRAGMA busy_timeout`` are
    set to the same value). The default (5s) matches normal outbox/config
    write behavior. Callers on a path that must never delay -- e.g. the fault
    handler's hard-exit boundary in ``worker/runtime/faults/record.py`` --
    pass ``busy_timeout_ms=0`` so a lock held by a concurrent writer surfaces
    immediately as ``sqlite3.OperationalError`` instead of being waited out.
    """
    if path.name == "edge.sqlite3":
        connection = open_runtime_database(path, actor=RuntimeActor.WORKER)
        _validate_central_schema(connection)
        return connection
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=busy_timeout_ms / 1000, isolation_level=None)
    version_row = connection.execute("PRAGMA user_version").fetchone()
    version = 0 if version_row is None else int(version_row[0])
    if version > SCHEMA_VERSION:
        connection.close()
        raise NewerSchemaVersionError(found=version, supported=SCHEMA_VERSION)
    try:
        connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        _enable_wal_and_migrate(connection, version)
        connection.execute("PRAGMA synchronous = FULL")
    except (sqlite3.Error, NewerSchemaVersionError):
        connection.close()
        raise
    return connection


def _validate_central_schema(connection: sqlite3.Connection) -> None:
    required = {
        "evidence_events": {"delivery_state"},
        "evidence_clips": {"publish_state", "local_state"},
        "config_current": {"config_version", "payload_json"},
        "config_history": {"config_version", "payload_json"},
        "faults": {"fault_time_iso", "exception_type"},
    }
    for table, columns in required.items():
        found = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if not columns <= found:
            connection.close()
            raise NewerSchemaVersionError(found=0, supported=SCHEMA_VERSION)


def _enable_wal_and_migrate(connection: sqlite3.Connection, version: int) -> None:
    # Acquire the same write lock used by migrations before changing journal
    # mode. PRAGMA journal_mode=WAL can otherwise report SQLITE_BUSY
    # immediately during concurrent first opens, ignoring busy_timeout.
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.commit()
        connection.execute("PRAGMA journal_mode = WAL")
        _migrate(connection, version)
    except (sqlite3.Error, NewerSchemaVersionError):
        connection.rollback()
        raise


def database_settings(connection: sqlite3.Connection) -> DatabaseSettings:
    journal = connection.execute("PRAGMA journal_mode").fetchone()
    synchronous = connection.execute("PRAGMA synchronous").fetchone()
    foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
    return DatabaseSettings(
        journal_mode="" if journal is None else str(journal[0]),
        synchronous=0 if synchronous is None else int(synchronous[0]),
        foreign_keys=0 if foreign_keys is None else int(foreign_keys[0]),
    )


def _migrate(connection: sqlite3.Connection, version: int) -> None:
    if version == SCHEMA_VERSION:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        # Another process may have completed migration while this connection
        # waited for the write lock. Re-read under BEGIN IMMEDIATE before
        # applying any DDL so concurrent first opens cannot replay migrations.
        current_row = connection.execute("PRAGMA user_version").fetchone()
        current = 0 if current_row is None else int(current_row[0])
        if current > SCHEMA_VERSION:
            connection.rollback()
            raise NewerSchemaVersionError(found=current, supported=SCHEMA_VERSION)
        for target_version in range(current + 1, SCHEMA_VERSION + 1):
            for statement in MIGRATIONS[target_version - 1]:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {target_version}")
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise


__all__ = ["database_settings", "open_connection"]
