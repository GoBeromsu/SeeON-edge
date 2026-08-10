from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    TopologyMutationResult,
    TopologySuccessEnvelope,
    parse_topology_success_envelope,
)

_RESPONSE_ADAPTER = TypeAdapter(dict[str, JsonValue])

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_topology_confirmation_preview (
 id INTEGER PRIMARY KEY CHECK (id = 1), confirmation_id TEXT NOT NULL, digest TEXT NOT NULL,
 expires_at TEXT NOT NULL, snapshot_id TEXT NOT NULL, client_revision INTEGER NOT NULL,
 server_revision INTEGER NOT NULL, registry_version INTEGER NOT NULL,
 edge_installation_id TEXT NOT NULL, enrollment_generation INTEGER NOT NULL,
 cameras INTEGER NOT NULL, rooms INTEGER NOT NULL, floors INTEGER NOT NULL,
 confirmed INTEGER NOT NULL DEFAULT 0,
 terminal_response TEXT
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
    terminal_response: TopologySuccessEnvelope | None

    @property
    def confirmed(self) -> bool:
        return self.terminal_response is not None


class TopologyConfirmationStateConflictError(RuntimeError):
    pass


class TopologyConfirmationStore:
    def __init__(self, path: str | Path) -> None:
        self._connection = connect_catalog_store(Path(path), (_SCHEMA,))
        _ensure_terminal_column(self._connection)

    def save(
        self, response: TopologySuccessEnvelope, principal: MachinePrincipal, registry_version: int
    ) -> None:
        preview = response.omissions
        if preview is None:
            self._connection.execute("DELETE FROM edge_topology_confirmation_preview WHERE id = 1")
            return
        self._connection.execute(
            "INSERT INTO edge_topology_confirmation_preview ("
            "id,confirmation_id,digest,expires_at,snapshot_id,client_revision,"
            "server_revision,registry_version,edge_installation_id,enrollment_generation,"
            "cameras,rooms,floors,confirmed,terminal_response) "
            "VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,0,NULL) "
            "ON CONFLICT(id) DO UPDATE SET confirmation_id=excluded.confirmation_id, "
            "digest=excluded.digest, expires_at=excluded.expires_at, "
            "snapshot_id=excluded.snapshot_id, client_revision=excluded.client_revision, "
            "server_revision=excluded.server_revision, registry_version=excluded.registry_version, "
            "edge_installation_id=excluded.edge_installation_id, "
            "enrollment_generation=excluded.enrollment_generation, cameras=excluded.cameras, "
            "rooms=excluded.rooms, floors=excluded.floors, confirmed=0, "
            "terminal_response=NULL",
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
            _parse_terminal_response(row[14]),
        )

    def complete(
        self,
        preview: TopologyConfirmationPreview,
        response: TopologySuccessEnvelope,
    ) -> None:
        encoded = json.dumps(
            _success_body(response), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        self._connection.execute("BEGIN IMMEDIATE")
        with self._connection:
            preview_update = self._connection.execute(
                "UPDATE edge_topology_confirmation_preview SET confirmed = 1, "
                "terminal_response = ? "
                "WHERE id = 1 AND confirmation_id = ? AND digest = ? "
                "AND client_revision = ? AND server_revision = ? AND terminal_response IS NULL",
                (
                    encoded,
                    preview.confirmation_id,
                    preview.digest,
                    preview.client_revision,
                    preview.server_revision,
                ),
            )
            state_update = self._connection.execute(
                "UPDATE edge_topology_sync_state SET server_revision = ? WHERE id = 1 "
                "AND edge_installation_id = ? AND enrollment_generation = ? "
                "AND last_client_revision = ? AND server_revision = ?",
                (
                    response.server_revision,
                    preview.principal.edge_installation_id,
                    preview.principal.enrollment_generation,
                    preview.client_revision,
                    preview.server_revision,
                ),
            )
            if preview_update.rowcount != 1 or state_update.rowcount != 1:
                raise TopologyConfirmationStateConflictError


def _ensure_terminal_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(edge_topology_confirmation_preview)"
        ).fetchall()
    }
    if "terminal_response" not in columns:
        connection.execute(
            "ALTER TABLE edge_topology_confirmation_preview ADD COLUMN terminal_response TEXT"
        )


def _parse_terminal_response(value: str | None) -> TopologySuccessEnvelope | None:
    if value is None:
        return None
    return parse_topology_success_envelope(_RESPONSE_ADAPTER.validate_json(str(value)))


def _counts_body(counts: MutationCounts) -> dict[str, JsonValue]:
    return {
        "created": counts.created,
        "updated": counts.updated,
        "unchanged": counts.unchanged,
        "reactivated": counts.reactivated,
        "deactivated": counts.deactivated,
    }


def _result_body(result: TopologyMutationResult) -> dict[str, JsonValue]:
    return {
        "floors": _counts_body(result.floors),
        "rooms": _counts_body(result.rooms),
        "cameras": _counts_body(result.cameras),
    }


def _success_body(response: TopologySuccessEnvelope) -> dict[str, JsonValue]:
    return {
        "schemaVersion": 1,
        "snapshotId": response.snapshot_id,
        "clientRevision": response.client_revision,
        "serverRevision": response.server_revision,
        "result": _result_body(response.result),
        "omissions": None,
    }


__all__ = [
    "TopologyConfirmationPreview",
    "TopologyConfirmationStateConflictError",
    "TopologyConfirmationStore",
]
