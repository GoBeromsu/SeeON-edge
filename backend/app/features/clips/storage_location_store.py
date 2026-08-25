"""Clip storage selection projected from the schema-18 site singleton."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.configuration import (
    ensure_edge_site,
    open_configuration_database,
    utc_now,
)


class ClipStorageLocationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection = open_configuration_database(self.path)

    @classmethod
    def from_env(cls) -> ClipStorageLocationStore:
        return cls(EDGE_DATABASE_PATH)

    def get(self) -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT clip_store_subdir FROM edge_site WHERE id=1"
            ).fetchone()
        return "" if row is None or row[0] is None else str(row[0])

    def put(
        self,
        selected_path: str,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> str:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                ensure_edge_site(self._connection)
                self._connection.execute(
                    "UPDATE edge_site SET clip_store_subdir=?,updated_at=? WHERE id=1",
                    (selected_path or None, utc_now()),
                )
                if after_write is not None:
                    after_write(self._connection)
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
        return selected_path


__all__ = ["ClipStorageLocationStore"]
