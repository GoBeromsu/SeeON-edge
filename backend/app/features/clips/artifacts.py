"""Privacy-bounded central clip artifact projection for clean and snapshot facts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db import EDGE_DATABASE_PATH
from backend.app.edge_db.connection import RuntimeActor, open_runtime_database


@dataclass(frozen=True, slots=True)
class CentralClipArtifacts:
    incident_id: str
    clean_state: str
    snapshot_state: str | None


class CentralClipArtifactQuery:
    """Read stable service facts without exposing worker-owned table rows."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path: Path = EDGE_DATABASE_PATH if database_path is None else database_path

    def get(self, clip_id: str) -> CentralClipArtifacts | None:
        connection = open_runtime_database(self.database_path, actor=RuntimeActor.API)
        try:
            row = connection.execute(
                "SELECT incident_id,state FROM artifacts "
                "WHERE clip_id=? AND kind='PRIMARY_CLIP'",
                (clip_id,),
            ).fetchone()
            if row is None:
                return None
            snapshot = connection.execute(
                "SELECT state FROM artifacts WHERE incident_id=? AND kind='SNAPSHOT'",
                (row[0],),
            ).fetchone()
        finally:
            connection.close()
        return CentralClipArtifacts(
            incident_id=str(row[0]),
            clean_state=str(row[1]),
            snapshot_state=None if snapshot is None else str(snapshot[0]),
        )


__all__ = [
    "CentralClipArtifactQuery",
    "CentralClipArtifacts",
]
