from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    TopologySuccessEnvelope,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_topology_confirmation_preview (
 id INTEGER PRIMARY KEY CHECK (id = 1), confirmation_id TEXT NOT NULL, digest TEXT NOT NULL,
 expires_at TEXT NOT NULL, snapshot_id TEXT NOT NULL, client_revision INTEGER NOT NULL,
 server_revision INTEGER NOT NULL, registry_version INTEGER NOT NULL,
 edge_installation_id TEXT NOT NULL, enrollment_generation INTEGER NOT NULL,
 cameras INTEGER NOT NULL, rooms INTEGER NOT NULL, floors INTEGER NOT NULL,
 confirmed INTEGER NOT NULL DEFAULT 0
) STRICT
"""


@dataclass(frozen=True, slots=True)
class TopologyConfirmationPreview:
    confirmation_id: str
    digest: str
    expires_at: str
    snapshot_id: str
    client_revision: int
    server_revision: int
    registry_version: int
    principal: MachinePrincipal
    cameras: int
    rooms: int
    floors: int
    confirmed: bool


class TopologyConfirmationStore:
    def __init__(self, path: str | Path) -> None:
        self._connection = connect_catalog_store(Path(path), (_SCHEMA,))

    def save(
        self, response: TopologySuccessEnvelope, principal: MachinePrincipal, registry_version: int
    ) -> None:
        preview = response.omissions
        if preview is None:
            self._connection.execute("DELETE FROM edge_topology_confirmation_preview WHERE id = 1")
            return
        self._connection.execute(
            "INSERT INTO edge_topology_confirmation_preview VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,0) "
            "ON CONFLICT(id) DO UPDATE SET confirmation_id=excluded.confirmation_id, "
            "digest=excluded.digest, expires_at=excluded.expires_at, "
            "snapshot_id=excluded.snapshot_id, client_revision=excluded.client_revision, "
            "server_revision=excluded.server_revision, registry_version=excluded.registry_version, "
            "edge_installation_id=excluded.edge_installation_id, "
            "enrollment_generation=excluded.enrollment_generation, cameras=excluded.cameras, "
            "rooms=excluded.rooms, floors=excluded.floors, confirmed=0",
            (
                preview.confirmation_id,
                preview.digest,
                preview.expires_at,
                response.snapshot_id,
                response.client_revision,
                response.server_revision,
                registry_version,
                principal.edge_installation_id,
                principal.enrollment_generation,
                len(preview.cameras),
                len(preview.rooms),
                len(preview.floors),
            ),
        )

    def load(self) -> TopologyConfirmationPreview | None:
        row = self._connection.execute(
            "SELECT * FROM edge_topology_confirmation_preview WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return TopologyConfirmationPreview(
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            int(row[5]),
            int(row[6]),
            int(row[7]),
            MachinePrincipal(str(row[8]), int(row[9])),
            int(row[10]),
            int(row[11]),
            int(row[12]),
            bool(row[13]),
        )

    def confirm(self, confirmation_id: str) -> None:
        self._connection.execute(
            "UPDATE edge_topology_confirmation_preview SET confirmed = 1 "
            "WHERE id = 1 AND confirmation_id = ?",
            (confirmation_id,),
        )


__all__ = ["TopologyConfirmationPreview", "TopologyConfirmationStore"]
