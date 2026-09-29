"""The old schema-19 SQLite database, as a migration source fixture only.

PostgreSQL is the only durable store. A deployed edge may still hold an
``edge.sqlite3`` written by the retired runtime, and the migration tool reads it
once. This module recreates that file for the migration tests without the
retired bootstrap, and holds the lock the old runtime took while it ran. No
runtime package may import it (import-linter and tests/test_edge_db_ddl_boundary.py).
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from backend.app.edge_db.compact_schema import SCHEMA_19_STATEMENTS
from backend.app.edge_db.compatibility import SCHEMA_19_IDENTITY
from backend.app.edge_db.functions import register_edge_db_functions
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION


def create_schema19_source(path: Path) -> Path:
    """Create a fresh WAL-mode schema-19 database with ledger row 19 only."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    with closing(sqlite3.connect(path, isolation_level=None)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA journal_mode = WAL").fetchone() != ("wal",):
            raise RuntimeError("source database could not enter WAL mode")
        connection.execute("PRAGMA synchronous = FULL")
        register_edge_db_functions(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA_19_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum, applied_at)
                VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                SCHEMA_19_IDENTITY,
            )
            connection.execute(f"PRAGMA user_version = {EDGE_DATABASE_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return path


def open_source_writer(source: Path) -> sqlite3.Connection:
    """A raw writer on the old database; with no auto-checkpoint its commits stay in the WAL."""
    connection = sqlite3.connect(source, isolation_level=None)
    register_edge_db_functions(connection)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA wal_autocheckpoint = 0")
    return connection


@contextmanager
def hold_runtime_lock(source: Path) -> Iterator[None]:
    """Hold the deployment lock shared, as the old runtime did while any connection was open."""
    descriptor = os.open(source.parent / "deployment.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


__all__ = ["create_schema19_source", "hold_runtime_lock", "open_source_writer"]
