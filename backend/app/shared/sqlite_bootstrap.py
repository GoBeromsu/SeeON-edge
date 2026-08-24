"""Shared connect-and-bootstrap logic for the three ``catalog.sqlite3``-owning
stores (``CameraRegistryStore``, ``DashboardCredentialsStore``).

Each of those stores independently opens the shared catalog database and
idempotently creates its own single table via ``CREATE TABLE IF NOT EXISTS``
(see ``backend/app/features/clips/catalog.py``'s ``_V3_TABLE_STATEMENTS`` for
why: none of them may depend on ``CatalogStore`` without violating the
"backend base (core/shared) does not import upper layers" import-linter
contract in ``pyproject.toml``). This module factors out the identical
connect/pragma/bootstrap/chmod sequence they each repeated; the DDL text
itself stays owned by each caller (and pinned byte-identical to catalog.py's
copy by ``tests/test_clips_catalog.py``), not by this module.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database


def connect_catalog_store(path: Path, create_statements: Iterable[str]) -> sqlite3.Connection:
    """Open ``path``, apply the standard catalog pragmas, and run bootstrap DDL.

    ``create_statements`` are executed in order (typically one or more
    idempotent ``CREATE TABLE IF NOT EXISTS`` statements owned by the caller).
    The database file is made best-effort ``0600`` afterward.
    """
    if path.name == "edge.sqlite3":
        connection = open_runtime_database(path, actor=RuntimeActor.API, check_same_thread=False)
        for statement in create_statements:
            if not statement.lstrip().upper().startswith(("CREATE", "ALTER", "DROP")):
                connection.execute(statement)
        return connection
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0, isolation_level=None, check_same_thread=False)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA foreign_keys = ON")
    for statement in create_statements:
        connection.execute(statement)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return connection


__all__ = ["connect_catalog_store"]
