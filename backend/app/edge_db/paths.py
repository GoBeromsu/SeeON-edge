"""Filesystem contract for the local edge SQLite file and its sidecars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

EDGE_STATE_DIRECTORY: Final = Path("/var/lib/seeon-state")
EDGE_DATABASE_PATH: Final = EDGE_STATE_DIRECTORY / "edge.sqlite3"

# Bare filename only, not a derived absolute path: every consumer resolves it
# against whatever EDGE_DATABASE_PATH currently is (``EDGE_DATABASE_PATH.parent
# / DIAGNOSTICS_DATABASE_FILENAME``) so the test fixture that monkeypatches
# EDGE_DATABASE_PATH per-module also redirects this sibling file, with no
# separate patch target (#579/#580, N2).
DIAGNOSTICS_DATABASE_FILENAME: Final = "edge-diagnostics.sqlite3"


def schema18_backup_path(database: Path) -> Path:
    """Sibling backup taken immediately before a schema-18 extension."""
    return database.with_name(f"{database.name}.schema18-backup.sqlite3")


def prepare_database_path(path: Path) -> None:
    """Create a private local directory and database inode before SQLite opens it."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT, 0o600)
    os.close(descriptor)
    path.chmod(0o600)


def secure_database_files(path: Path) -> None:
    """Best-effort tighten the database, any WAL/SHM sidecars, and the schema-18 backup."""
    path.parent.chmod(0o700)
    for candidate in (
        path,
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        schema18_backup_path(path),
    ):
        try:
            candidate.chmod(0o600)
        except FileNotFoundError:
            continue


__all__ = [
    "DIAGNOSTICS_DATABASE_FILENAME",
    "EDGE_DATABASE_PATH",
    "EDGE_STATE_DIRECTORY",
    "prepare_database_path",
    "schema18_backup_path",
    "secure_database_files",
]
