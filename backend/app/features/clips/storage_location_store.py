"""Clip storage selection on the API-owned PostgreSQL pool.

Only persistence lives here, not filesystem validation or recording policy.
Bootstrap owns the site singleton; lifespan owns the borrowed pool.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import psycopg

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import PostgresDatabase, PostgresError


class ClipStorageLocationNotInitialized(PostgresError):
    def __init__(self) -> None:
        super().__init__("clip storage location bootstrap row is missing")


class ClipStorageLocationStore:
    """Borrow transactions only; pool lifecycle belongs to the API."""

    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        self.database = database
        self.authority = authority

    def get(self) -> str:
        def read(connection: psycopg.Connection) -> str:
            row = connection.execute(
                "SELECT clip_store_subdir FROM edge_site WHERE id=1"
            ).fetchone()
            if row is None:
                raise ClipStorageLocationNotInitialized()
            return "" if row[0] is None else str(row[0])

        return self.database.read(read)

    def put(
        self,
        selected_path: str,
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> str:
        def persist(connection: psycopg.Connection) -> str:
            require_authority(connection, self.authority)
            row = connection.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE").fetchone()
            if row is None:
                raise ClipStorageLocationNotInitialized()
            now = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            connection.execute(
                "UPDATE edge_site SET clip_store_subdir=%s,updated_at=%s WHERE id=1",
                (selected_path or None, now),
            )
            if after_write is not None:
                after_write(connection)
            return selected_path

        # Never reread after COMMIT or publish a candidate after an unknown outcome.
        return self.database.transact(persist)


__all__ = ["ClipStorageLocationNotInitialized", "ClipStorageLocationStore"]
