"""Privacy-bounded central clip artifact projection for clean and snapshot facts."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from backend.app.edge_db.postgres import PostgresDatabase


@dataclass(frozen=True, slots=True)
class CentralClipArtifacts:
    incident_id: str
    clean_state: str
    snapshot_state: str | None


class CentralClipArtifactQuery:
    """Read stable service facts without exposing worker-owned table rows."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.database = database

    def get(self, clip_id: str) -> CentralClipArtifacts | None:
        def read(connection: psycopg.Connection) -> CentralClipArtifacts | None:
            row = connection.execute(
                "SELECT incident_id,state FROM artifacts WHERE clip_id=%s AND kind='PRIMARY_CLIP'",
                (clip_id,),
            ).fetchone()
            if row is None:
                return None
            snapshot = connection.execute(
                "SELECT state FROM artifacts WHERE incident_id=%s AND kind='SNAPSHOT'",
                (row[0],),
            ).fetchone()
            return CentralClipArtifacts(
                incident_id=str(row[0]),
                clean_state=str(row[1]),
                snapshot_state=None if snapshot is None else str(snapshot[0]),
            )

        return self.database.read_snapshot(read)


__all__ = [
    "CentralClipArtifactQuery",
    "CentralClipArtifacts",
]
