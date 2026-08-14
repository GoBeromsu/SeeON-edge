"""Persisted clip storage location selection for ml-api.

Backs ``PUT /api/v1/clips/storage/location`` (see ``storage_router.py``): a
single relative subdirectory, under the fixed ``CLIP_STORE_DIR`` mount root,
that new clips should be recorded into. ``""`` (empty string) selects the
mount root itself -- the pre-existing, always-valid default.

Single-row table (``id`` CHECK-constrained to 1) mirroring
``CameraRegistryStore``'s single-row-blob shape (see
``backend/app/features/cameras/store.py``): there is exactly one facility-wide
selection at a time, not one per anything else, so a real primary key over
multiple conceptual rows would be the wrong shape here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from shared.edge_db import EDGE_DATABASE_PATH

_CREATE_CLIP_STORAGE_LOCATION_TABLE = (
    "CREATE TABLE IF NOT EXISTS clip_storage_location ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "selected_path TEXT NOT NULL) STRICT"
)


class ClipStorageLocationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    @classmethod
    def from_env(cls) -> ClipStorageLocationStore:
        return cls(EDGE_DATABASE_PATH)

    def get(self) -> str:
        """The selected relative subdirectory; ``""`` (mount root) when
        nothing has ever been explicitly selected."""
        with self._lock:
            connection = self._connect()
            row = connection.execute(
                "SELECT selected_path FROM clip_storage_location WHERE id = 1"
            ).fetchone()
        return "" if row is None else str(row[0])

    def put(self, selected_path: str) -> str:
        with self._lock:
            connection = self._connect()
            connection.execute(
                "INSERT INTO clip_storage_location (id, selected_path) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET selected_path = excluded.selected_path",
                (selected_path,),
            )
        return selected_path

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = connect_catalog_store(
                self.path, (_CREATE_CLIP_STORAGE_LOCATION_TABLE,)
            )
        return self._connection


__all__ = ["ClipStorageLocationStore"]
