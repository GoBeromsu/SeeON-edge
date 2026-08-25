"""DDL-free access to schema-18 durable configuration authorities."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from backend.app.edge_db.connection import RuntimeActor, open_runtime_database


def open_configuration_database(path: Path) -> sqlite3.Connection:
    """Open and validate the migrated edge database for API-owned writes."""
    return open_runtime_database(path, actor=RuntimeActor.API, check_same_thread=False)


def ensure_edge_site(connection: sqlite3.Connection) -> None:
    """Materialize the compact singleton without altering sibling fields."""
    connection.execute(
        "INSERT OR IGNORE INTO edge_site (id, updated_at) VALUES (1, ?)",
        (utc_now(),),
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = ["ensure_edge_site", "open_configuration_database", "utc_now"]
