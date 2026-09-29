"""Confirmation previews and terminal CAS on the API-owned PostgreSQL authority."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import psycopg
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import (
    PostgresDatabase,
    PostgresError,
    PostgresTransactionStateError,
)
from backend.app.features.cameras.camera_repository import lock_registry, utc_now
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

_Result = TypeVar("_Result")


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
    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native confirmation state requires a PostgreSQL owner")
        if not isinstance(authority, AuthorityToken):
            raise TypeError("native confirmation state requires deployment authority")
        self.database, self.authority = database, authority

    def save(
        self,
        response: TopologySuccessEnvelope,
        principal: MachinePrincipal,
        registry_version: int,
        *,
        connection: psycopg.Connection | None = None,
    ) -> None:
        def write(current: psycopg.Connection) -> None:
            preview = response.omissions
            if preview is None:
                self._clear(current, principal)
                return
            cursor = current.execute(
                "UPDATE edge_site SET topology_confirmation_id=%s,topology_confirmation_digest=%s,"
                "topology_confirmation_expires_at=%s,topology_confirmation_snapshot_id=%s,"
                "topology_confirmation_client_revision=%s,topology_confirmation_server_revision=%s,"
                "topology_confirmation_registry_version=%s,topology_confirmation_cameras=%s,"
                "topology_confirmation_rooms=%s,topology_confirmation_floors=%s,"
                "topology_confirmation_confirmed=0,topology_confirmation_result=NULL,updated_at=%s "
                "WHERE id=1 AND edge_installation_id=%s AND enrollment_generation=%s",
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
            _require_updated(cursor)

        self._write(write, connection)

    def load(
        self,
        *,
        connection: psycopg.Connection | None = None,
    ) -> TopologyConfirmationPreview | None:
        if connection is None:
            return self.database.read(self._load)
        self._require_borrowed(connection)
        return self._load(connection)

    @staticmethod
    def _load(connection: psycopg.Connection) -> TopologyConfirmationPreview | None:
        row = connection.execute(_SELECT).fetchone()
        if row is None:
            raise PostgresError("confirmation state bootstrap row is missing")
        if row[0] is None:
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
        after_write: Callable[[psycopg.Connection], None] | None = None,
        connection: psycopg.Connection | None = None,
    ) -> None:
        if (
            response.snapshot_id != preview.snapshot_id
            or response.client_revision != preview.client_revision
        ):
            raise TopologyConfirmationStateConflictError("confirmation response identity changed")
        encoded = _encode_result(response.result)

        def write(current: psycopg.Connection) -> None:
            # Revalidate the pre-network observations under the registry/site
            # lock. An accepted upstream reply does not defeat a later local
            # principal, registry, preview, or state-revision change.
            cursor = current.execute(
                "UPDATE edge_site SET topology_confirmation_confirmed=1,"
                "topology_confirmation_result=%s,topology_server_revision=%s,updated_at=%s "
                "WHERE id=1 "
                "AND topology_confirmation_id=%s AND topology_confirmation_digest=%s "
                "AND topology_confirmation_snapshot_id=%s AND topology_confirmation_expires_at=%s "
                "AND topology_confirmation_client_revision=%s "
                "AND topology_confirmation_server_revision=%s "
                "AND topology_confirmation_registry_version=%s "
                "AND topology_confirmation_cameras=%s AND topology_confirmation_rooms=%s "
                "AND topology_confirmation_floors=%s AND topology_confirmation_confirmed=0 "
                "AND edge_installation_id=%s AND enrollment_generation=%s "
                "AND registry_version=%s AND topology_client_revision=%s "
                "AND topology_server_revision=%s AND topology_confirmation_result IS NULL",
                (
                    encoded,
                    response.server_revision,
                    utc_now(),
                    preview.confirmation_id,
                    preview.digest,
                    preview.snapshot_id,
                    preview.expires_at,
                    preview.client_revision,
                    preview.server_revision,
                    preview.registry_version,
                    preview.cameras,
                    preview.rooms,
                    preview.floors,
                    preview.principal.edge_installation_id,
                    preview.principal.enrollment_generation,
                    preview.registry_version,
                    preview.client_revision,
                    preview.server_revision,
                ),
            )
            _require_updated(cursor)
            if after_write is not None:
                after_write(current)

        self._write(write, connection)

    def _write(
        self,
        callback: Callable[[psycopg.Connection], _Result],
        connection: psycopg.Connection | None,
    ) -> _Result:
        def write(current: psycopg.Connection) -> _Result:
            self._require_borrowed(current)
            require_authority(current, self.authority)
            lock_registry(current)
            return callback(current)

        try:
            return self.database.transact(write) if connection is None else write(connection)
        except (psycopg.IntegrityError, psycopg.DataError):
            # Constraint DETAIL may contain the credential-bearing site row.
            raise PostgresError("confirmation write rejected by a database constraint") from None

    @staticmethod
    def _require_borrowed(connection: psycopg.Connection) -> None:
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise PostgresTransactionStateError("confirmation state requires an active transaction")

    @staticmethod
    def _clear(connection: psycopg.Connection, principal: MachinePrincipal) -> None:
        cursor = connection.execute(
            "UPDATE edge_site SET topology_confirmation_id=NULL,"
            "topology_confirmation_digest=NULL,topology_confirmation_expires_at=NULL,"
            "topology_confirmation_snapshot_id=NULL,topology_confirmation_client_revision=NULL,"
            "topology_confirmation_server_revision=NULL,topology_confirmation_registry_version=NULL,"
            "topology_confirmation_cameras=NULL,topology_confirmation_rooms=NULL,"
            "topology_confirmation_floors=NULL,topology_confirmation_confirmed=NULL,"
            "topology_confirmation_result=NULL,updated_at=%s WHERE id=1 "
            "AND edge_installation_id=%s AND enrollment_generation=%s",
            (utc_now(), principal.edge_installation_id, principal.enrollment_generation),
        )
        _require_updated(cursor)


def _require_updated(cursor: psycopg.Cursor) -> None:
    if cursor.rowcount != 1:
        raise TopologyConfirmationStateConflictError("confirmation state changed")


_SELECT = (
    "SELECT topology_confirmation_id,topology_confirmation_digest,"
    "topology_confirmation_expires_at,topology_confirmation_snapshot_id,"
    "topology_confirmation_client_revision,topology_confirmation_server_revision,"
    "topology_confirmation_registry_version,topology_confirmation_cameras,"
    "topology_confirmation_rooms,topology_confirmation_floors,"
    "topology_confirmation_result,topology_server_revision,"
    "edge_installation_id,enrollment_generation FROM edge_site WHERE id=1"
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
    try:
        values = tuple(
            tuple(int(value) for value in group.split(",")) for group in encoded.split(";")
        )
    except ValueError:
        raise PostgresError("stored topology confirmation result is malformed") from None
    if len(values) != 3 or any(
        len(group) != 5 or any(value < 0 for value in group) for group in values
    ):
        raise PostgresError("stored topology confirmation result is malformed")
    groups = tuple(MutationCounts(*group) for group in values)
    return TopologyMutationResult(groups[0], groups[1], groups[2])


__all__ = [
    "TopologyConfirmationPreview",
    "TopologyConfirmationStateConflictError",
    "TopologyConfirmationStore",
]
