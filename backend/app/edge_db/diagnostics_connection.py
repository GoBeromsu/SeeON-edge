"""Connection for the standalone, pre-bootstrapped diagnostics database.

Execution-record telemetry (the six schema-19 ``execution_*`` tables) is
written on every worker flush (~250 ms) and pruned toward its retention
budget on that same hot path. Isolating it into its own SQLite file next to
the product database (``EDGE_DATABASE_PATH``) means its writer lock, WAL, and
checkpoint pressure can never be shared with incidents, alerts, or policy
writes in ``edge.sqlite3`` -- SQLite's writer lock and WAL are per-file. The
product database's six ``execution_*`` tables are left in place untouched (no
destructive migration); retiring them (VACUUM/drop) is a later ops step,
tracked separately in #583.

This file is created and stamped (``PRAGMA user_version``) only by the
one-shot ``python -m backend.app.edge_db`` bootstrap
(``backend/app/edge_db/bootstrap.py:bootstrap_diagnostics_database``), under
the same exclusive ``deployment.lock`` as ``edge.sqlite3``. This module only
opens and verifies it, exactly like ``edge_db.connection.open_runtime_database``
does for the product database -- it never creates the file or its schema
(#579/#580, S1/S2). It has exactly one table family and no other feature to
protect, so unlike ``edge_db.connection`` it installs no per-table write
authorizer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.edge_db.compatibility import EdgeDatabaseError
from backend.app.edge_db.connection import NORMAL_BUSY_TIMEOUT_MS, _require_wal
from backend.app.edge_db.paths import secure_database_files


def open_diagnostics_database(path: Path) -> sqlite3.Connection:
    """Open the already-bootstrapped standalone execution-record database.

    Raises ``EdgeDatabaseError`` if ``path`` does not exist or is not yet in
    WAL mode -- the one-shot bootstrap must run first.
    """
    if not path.is_file():
        raise EdgeDatabaseError(
            f"diagnostics database {path} does not exist; "
            "run the edge_db bootstrap before starting the runtime"
        )
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
        _require_wal(connection.execute("PRAGMA journal_mode").fetchone())
        secure_database_files(path)
    except (OSError, sqlite3.Error, EdgeDatabaseError):
        connection.close()
        raise
    return connection


__all__ = ["open_diagnostics_database"]
