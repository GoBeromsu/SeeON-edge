"""Durable topology retry state on the API-owned native PostgreSQL authority."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final, Protocol, TypeVar

import psycopg
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import AuthorityToken, require_authority
from backend.app.edge_db.postgres import (
    PostgresDatabase,
    PostgresError,
    PostgresTransactionStateError,
)
from backend.app.features.cameras.camera_repository import utc_now
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologySuccessEnvelope

BASE_BACKOFF_SECONDS: Final = 5.0
MAX_BACKOFF_SECONDS: Final = 300.0
_BACKOFF_EXPONENT_LIMIT = math.ceil(math.log2(MAX_BACKOFF_SECONDS / BASE_BACKOFF_SECONDS))
_Result = TypeVar("_Result")


@unique
class TopologyPauseReason(StrEnum):
    AUTH = "auth"
    FORBIDDEN = "forbidden"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class PendingTopologySnapshot:
    snapshot_id: str
    body: bytes
    registry_version: int
    client_revision: int
    expected_server_revision: int
    principal: MachinePrincipal


@dataclass(frozen=True, slots=True)
class EdgeTopologySyncState:
    principal: MachinePrincipal | None
    pending: PendingTopologySnapshot | None
    last_snapshotted_registry_version: int
    last_client_revision: int
    server_revision: int
    consecutive_failures: int
    next_retry_at: float | None
    pause_reason: TopologyPauseReason | None
    last_accepted_at: float | None


class PendingSnapshotBuilder(Protocol):
    @property
    def registry_version(self) -> int: ...
    @property
    def principal(self) -> MachinePrincipal: ...
    @property
    def snapshot_id(self) -> str: ...
    def build(self, client_revision: int, expected_server_revision: int) -> bytes: ...


class TopologySyncStateConflictError(RuntimeError):
    pass


class EdgeTopologySyncStateStore:
    def __init__(self, database: PostgresDatabase, authority: AuthorityToken) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("native topology state requires a PostgreSQL owner")
        if not isinstance(authority, AuthorityToken):
            raise TypeError("native topology state requires deployment authority")
        self.database, self.authority = database, authority

    def load(self, *, connection: psycopg.Connection | None = None) -> EdgeTopologySyncState:
        if connection is None:
            return self.database.read(self._load)
        self._require_borrowed(connection)
        return self._load(connection)

    def operation(
        self,
        callback: Callable[[psycopg.Connection], _Result],
        *,
        after_write: Callable[[psycopg.Connection], None] | None = None,
    ) -> _Result:
        """Own one compound operation; audit once on every normal body return.

        Inner state methods must receive this callback's explicit connection.
        They never open nested transactions, infer ambient ownership, publish an
        audit token, or close the borrowed connection. The caller's result is
        tentative until the complete database owner returns.
        """

        def write(connection: psycopg.Connection) -> _Result:
            result = callback(connection)
            if after_write is not None:
                after_write(connection)
            return result

        return self._write(write)

    def ensure_principal(
        self, principal: MachinePrincipal, *, connection: psycopg.Connection | None = None
    ) -> EdgeTopologySyncState:
        def write(current: psycopg.Connection) -> EdgeTopologySyncState:
            state = self._load(current)
            if state.principal != principal:
                raise TopologySyncStateConflictError("topology principal does not match enrollment")
            return state

        return self._write(write, connection)

    def create_pending(
        self, builder: PendingSnapshotBuilder, *, connection: psycopg.Connection | None = None
    ) -> PendingTopologySnapshot:
        def write(current: psycopg.Connection) -> PendingTopologySnapshot:
            state = self._load(current)
            if state.pending is not None:
                return state.pending
            if state.principal != builder.principal:
                raise TopologySyncStateConflictError("topology principal changed before enqueue")
            client_revision = state.last_client_revision + 1
            body = builder.build(client_revision, state.server_revision)
            current.execute(
                "UPDATE edge_site SET topology_pending_snapshot_id=%s,topology_pending_body=%s,"
                "topology_pending_registry_version=%s,topology_pending_client_revision=%s,"
                "topology_pending_expected_server_revision=%s,topology_consecutive_failures=0,"
                "topology_next_retry_at=NULL,topology_pause_reason=NULL,updated_at=%s WHERE id=1",
                (
                    builder.snapshot_id,
                    body,
                    builder.registry_version,
                    client_revision,
                    state.server_revision,
                    utc_now(),
                ),
            )
            pending = self._load(current).pending
            if pending is None:
                raise TopologySyncStateConflictError("pending topology snapshot was not persisted")
            return pending

        return self._write(write, connection)

    def record_retry(
        self,
        snapshot_id: str,
        *,
        now_epoch: float,
        connection: psycopg.Connection | None = None,
    ) -> EdgeTopologySyncState:
        def write(current: psycopg.Connection) -> EdgeTopologySyncState:
            state = self._require_pending(current, snapshot_id)
            failures = state.consecutive_failures + 1
            exponent = min(failures - 1, _BACKOFF_EXPONENT_LIMIT)
            delay = min(BASE_BACKOFF_SECONDS * (2**exponent), MAX_BACKOFF_SECONDS)
            current.execute(
                "UPDATE edge_site SET topology_consecutive_failures=%s,topology_next_retry_at=%s,"
                "topology_pause_reason=NULL,updated_at=%s WHERE id=1",
                (failures, now_epoch + delay, utc_now()),
            )
            return self._load(current)

        return self._write(write, connection)

    def pause(
        self,
        snapshot_id: str,
        reason: TopologyPauseReason,
        *,
        connection: psycopg.Connection | None = None,
    ) -> EdgeTopologySyncState:
        return self._update_pending(
            snapshot_id,
            "topology_pause_reason=%s,topology_next_retry_at=NULL",
            (reason.value,),
            connection,
        )

    def resume_pending(
        self, snapshot_id: str, *, connection: psycopg.Connection | None = None
    ) -> EdgeTopologySyncState:
        return self._update_pending(snapshot_id, "topology_pause_reason=NULL", (), connection)

    def refresh_conflict(
        self,
        snapshot_id: str,
        server_revision: int,
        *,
        connection: psycopg.Connection | None = None,
    ) -> EdgeTopologySyncState:
        return self._update_pending(
            snapshot_id,
            "topology_server_revision=%s," + _CLEAR_PENDING,
            (server_revision,),
            connection,
        )

    def accept(
        self,
        snapshot_id: str,
        response: TopologySuccessEnvelope,
        *,
        now_epoch: float = 0.0,
        connection: psycopg.Connection | None = None,
    ) -> EdgeTopologySyncState:
        def write(current: psycopg.Connection) -> EdgeTopologySyncState:
            state = self._require_pending(current, snapshot_id)
            pending = state.pending
            if (
                pending is None
                or response.snapshot_id != snapshot_id
                or response.client_revision != pending.client_revision
            ):
                raise TopologySyncStateConflictError("topology acceptance revision mismatch")
            current.execute(
                "UPDATE edge_site SET topology_snapshot_registry_version=%s,"
                "topology_client_revision=%s,topology_server_revision=%s," + _CLEAR_PENDING + ","
                "topology_last_accepted_at=%s,topology_dirty_registry_version="
                "CASE WHEN topology_dirty_registry_version=%s THEN NULL "
                "ELSE topology_dirty_registry_version END,"
                "topology_dirty_created_at=CASE WHEN topology_dirty_registry_version=%s THEN NULL "
                "ELSE topology_dirty_created_at END,updated_at=%s WHERE id=1",
                (
                    pending.registry_version,
                    response.client_revision,
                    response.server_revision,
                    now_epoch,
                    pending.registry_version,
                    pending.registry_version,
                    utc_now(),
                ),
            )
            return self._load(current)

        return self._write(write, connection)

    def _update_pending(
        self,
        snapshot_id: str,
        assignments: str,
        values: tuple[str | int, ...],
        connection: psycopg.Connection | None,
    ) -> EdgeTopologySyncState:
        def write(current: psycopg.Connection) -> EdgeTopologySyncState:
            self._require_pending(current, snapshot_id)
            current.execute(
                f"UPDATE edge_site SET {assignments},updated_at=%s WHERE id=1",
                (*values, utc_now()),
            )
            return self._load(current)

        return self._write(write, connection)

    def _write(
        self,
        callback: Callable[[psycopg.Connection], _Result],
        connection: psycopg.Connection | None = None,
    ) -> _Result:
        def write(current: psycopg.Connection) -> _Result:
            self._require_borrowed(current)
            require_authority(current, self.authority)
            if current.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE").fetchone() is None:
                raise PostgresError("topology state bootstrap row is missing")
            return callback(current)

        return self.database.transact(write) if connection is None else write(connection)

    @staticmethod
    def _require_borrowed(connection: psycopg.Connection) -> None:
        if connection.info.transaction_status is not TransactionStatus.INTRANS:
            raise PostgresTransactionStateError("topology state requires an active transaction")

    def _require_pending(
        self, connection: psycopg.Connection, snapshot_id: str
    ) -> EdgeTopologySyncState:
        state = self._load(connection)
        if state.pending is None or state.pending.snapshot_id != snapshot_id:
            raise TopologySyncStateConflictError("pending topology snapshot changed")
        return state

    @staticmethod
    def _load(connection: psycopg.Connection) -> EdgeTopologySyncState:
        row = connection.execute(_STATE_SELECT).fetchone()
        if row is None:
            raise PostgresError("topology state bootstrap row is missing")
        principal = None if row[0] is None else MachinePrincipal(str(row[0]), int(row[1]))
        pending = None
        if row[5] is not None:
            if principal is None:
                raise TopologySyncStateConflictError("pending snapshot has no principal")
            pending = PendingTopologySnapshot(
                str(row[5]), bytes(row[6]), int(row[7]), int(row[8]), int(row[9]), principal
            )
        pause = None if row[12] is None else TopologyPauseReason(str(row[12]))
        return EdgeTopologySyncState(
            principal,
            pending,
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[10]),
            None if row[11] is None else float(row[11]),
            pause,
            None if row[13] is None else float(row[13]),
        )


_CLEAR_PENDING = (
    "topology_pending_snapshot_id=NULL,topology_pending_body=NULL,"
    "topology_pending_registry_version=NULL,topology_pending_client_revision=NULL,"
    "topology_pending_expected_server_revision=NULL,topology_consecutive_failures=0,"
    "topology_next_retry_at=NULL,topology_pause_reason=NULL"
)
_STATE_SELECT = (
    "SELECT edge_installation_id,enrollment_generation,topology_snapshot_registry_version,"
    "topology_client_revision,topology_server_revision,topology_pending_snapshot_id,"
    "topology_pending_body,topology_pending_registry_version,topology_pending_client_revision,"
    "topology_pending_expected_server_revision,topology_consecutive_failures,"
    "topology_next_retry_at,topology_pause_reason,topology_last_accepted_at "
    "FROM edge_site WHERE id=1"
)

__all__ = [
    "BASE_BACKOFF_SECONDS",
    "EdgeTopologySyncState",
    "EdgeTopologySyncStateStore",
    "PendingSnapshotBuilder",
    "PendingTopologySnapshot",
    "TopologyPauseReason",
    "TopologySyncStateConflictError",
]
