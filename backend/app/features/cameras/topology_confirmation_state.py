from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.app.edge_db.configuration import open_configuration_database, utc_now
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)


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
    terminal_response: TopologySuccessEnvelope | None

    @property
    def confirmed(self) -> bool:
        return self.terminal_response is not None


class TopologyConfirmationStateConflictError(RuntimeError):
    pass


class TopologyConfirmationStore:
    def __init__(self, path: str | Path) -> None:
        self._connection = open_configuration_database(Path(path))

    def save(
        self,
        response: TopologySuccessEnvelope,
        principal: MachinePrincipal,
        registry_version: int,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        active = self._connection if connection is None else connection
        preview = response.omissions
        if preview is None:
            self._clear(active)
            return
        cursor = active.execute(
            "UPDATE edge_site SET topology_confirmation_id=?,topology_confirmation_digest=?,"
            "topology_confirmation_expires_at=?,topology_confirmation_snapshot_id=?,"
            "topology_confirmation_client_revision=?,topology_confirmation_server_revision=?,"
            "topology_confirmation_registry_version=?,topology_confirmation_cameras=?,"
            "topology_confirmation_rooms=?,topology_confirmation_floors=?,"
            "topology_confirmation_confirmed=0,topology_confirmation_result=NULL,updated_at=? "
            "WHERE id=1 AND edge_installation_id=? AND enrollment_generation=?",
            (
                preview.confirmation_id,
                preview.digest,
                preview.expires_at,
                response.snapshot_id,
                response.client_revision,
                response.server_revision,
                registry_version,
                len(preview.cameras),
                len(preview.rooms),
                len(preview.floors),
                utc_now(),
                principal.edge_installation_id,
                principal.enrollment_generation,
            ),
        )
        if cursor.rowcount != 1:
            raise TopologyConfirmationStateConflictError

    def load(self) -> TopologyConfirmationPreview | None:
        row = self._connection.execute(_SELECT).fetchone()
        if row is None or row[0] is None:
            return None
        principal = MachinePrincipal(str(row[12]), int(row[13]))
        terminal = None
        if row[10] is not None:
            terminal = TopologySuccessEnvelope(
                str(row[3]), int(row[4]), int(row[11]), _decode_result(str(row[10])), None
            )
        return TopologyConfirmationPreview(
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
            int(row[5]),
            int(row[6]),
            principal,
            int(row[7]),
            int(row[8]),
            int(row[9]),
            terminal,
        )

    def complete(
        self,
        preview: TopologyConfirmationPreview,
        response: TopologySuccessEnvelope,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        encoded = _encode_result(response.result)
        self._connection.execute("BEGIN IMMEDIATE")
        completed = False
        try:
            cursor = self._connection.execute(
                "UPDATE edge_site SET topology_confirmation_confirmed=1,"
                "topology_confirmation_result=?,topology_server_revision=?,updated_at=? WHERE id=1 "
                "AND topology_confirmation_id=? AND topology_confirmation_digest=? "
                "AND topology_confirmation_client_revision=? "
                "AND topology_confirmation_server_revision=? "
                "AND topology_confirmation_result IS NULL",
                (
                    encoded,
                    response.server_revision,
                    utc_now(),
                    preview.confirmation_id,
                    preview.digest,
                    preview.client_revision,
                    preview.server_revision,
                ),
            )
            _require_updated(cursor)
            if after_write is not None:
                after_write(self._connection)
            completed = True
        finally:
            self._connection.execute("COMMIT" if completed else "ROLLBACK")

    def _clear(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE edge_site SET topology_confirmation_id=NULL,"
            "topology_confirmation_digest=NULL,topology_confirmation_expires_at=NULL,"
            "topology_confirmation_snapshot_id=NULL,topology_confirmation_client_revision=NULL,"
            "topology_confirmation_server_revision=NULL,topology_confirmation_registry_version=NULL,"
            "topology_confirmation_cameras=NULL,topology_confirmation_rooms=NULL,"
            "topology_confirmation_floors=NULL,topology_confirmation_confirmed=NULL,"
            "topology_confirmation_result=NULL,updated_at=? WHERE id=1",
            (utc_now(),),
        )


def _require_updated(cursor: sqlite3.Cursor) -> None:
    if cursor.rowcount != 1:
        raise TopologyConfirmationStateConflictError


_SELECT = (
    "SELECT topology_confirmation_id,topology_confirmation_digest,"
    "topology_confirmation_expires_at,topology_confirmation_snapshot_id,"
    "topology_confirmation_client_revision,topology_confirmation_server_revision,"
    "topology_confirmation_registry_version,topology_confirmation_cameras,"
    "topology_confirmation_rooms,topology_confirmation_floors,"
    "topology_confirmation_result,topology_server_revision,"
    "edge_installation_id,enrollment_generation "
    "FROM edge_site WHERE id=1"
)


def _encode_result(result: TopologyMutationResult) -> str:
    counts = (result.floors, result.rooms, result.cameras)
    return ";".join(
        ",".join(
            str(value)
            for value in (
                count.created,
                count.updated,
                count.unchanged,
                count.reactivated,
                count.deactivated,
            )
        )
        for count in counts
    )


def _decode_result(encoded: str) -> TopologyMutationResult:
    groups = tuple(
        MutationCounts(*(int(value) for value in group.split(","))) for group in encoded.split(";")
    )
    if len(groups) != 3:
        raise sqlite3.DatabaseError("stored topology confirmation result is malformed")
    return TopologyMutationResult(groups[0], groups[1], groups[2])


__all__ = [
    "TopologyConfirmationPreview",
    "TopologyConfirmationStateConflictError",
    "TopologyConfirmationStore",
]
