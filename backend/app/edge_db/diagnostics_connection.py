"""Connection and create-if-missing bootstrap for the standalone diagnostics database.

Execution-record telemetry (the six schema-19 ``execution_*`` tables) is
written on every worker flush (~250 ms) and pruned toward its retention
budget on that same hot path. Isolating it into its own SQLite file next to
the product database (``EDGE_DATABASE_PATH``) means its writer lock, WAL, and
checkpoint pressure can never be shared with incidents, alerts, or policy
writes in ``edge.sqlite3`` -- SQLite's writer lock and WAL are per-file. The
product database's six ``execution_*`` tables are left in place untouched (no
destructive migration); retiring them (VACUUM/drop) is a later ops step.

This file has exactly one table family and no other feature to protect, so
unlike ``edge_db.connection`` it installs no per-table write authorizer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.connection import NORMAL_BUSY_TIMEOUT_MS, write_transaction
from backend.app.edge_db.execution_records_ddl import EXECUTION_RECORD_CREATE_STATEMENTS
from backend.app.edge_db.paths import prepare_database_path, secure_database_files

DIAGNOSTICS_DATABASE_FILENAME: Final = "edge-diagnostics.sqlite3"

# Any one of the six tables proves the file was already bootstrapped; creation
# below runs inside one write transaction, so partial creation never happens.
_BOOTSTRAP_PROBE_TABLE: Final = "execution_provenance"


def _require_wal(journal_row: tuple[object, ...] | None) -> None:
    if journal_row != ("wal",):
        raise EdgeDatabaseError("diagnostics database could not enter WAL mode")


def _bootstrap_if_empty(connection: sqlite3.Connection) -> None:
    with write_transaction(connection):
        exists = connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?",
            (_BOOTSTRAP_PROBE_TABLE,),
        ).fetchone()
        if exists is None:
            for statement in EXECUTION_RECORD_CREATE_STATEMENTS:
                connection.execute(statement)


def open_diagnostics_database(path: Path) -> sqlite3.Connection:
    """Open the standalone execution-record database, creating it on first use.

    Its own file: an unbounded telemetry writer or a slow retention prune
    here holds only this file's writer lock, never the product database's.
    """
    prepare_database_path(path)
    connection = sqlite3.connect(
        path,
        timeout=NORMAL_BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        connection.execute(f"PRAGMA busy_timeout = {NORMAL_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        _require_wal(connection.execute("PRAGMA journal_mode = WAL").fetchone())
        _bootstrap_if_empty(connection)
        secure_database_files(path)
    except (OSError, sqlite3.Error, EdgeDatabaseError):
        connection.close()
        raise
    return connection


__all__ = ["DIAGNOSTICS_DATABASE_FILENAME", "open_diagnostics_database"]
