from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from threading import RLock
from typing import Final, Protocol

from backend.app.edge_db.configuration import (
    ensure_edge_site,
    open_configuration_database,
    utc_now,
)
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologySuccessEnvelope

BASE_BACKOFF_SECONDS: Final = 5.0
MAX_BACKOFF_SECONDS: Final = 300.0


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
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._connection = open_configuration_database(self.path)

    def load(self) -> EdgeTopologySyncState:
        with self._lock:
            return self._load(self._connection)

    @contextmanager
    def operation(
        self,
        after_write: Callable[[sqlite3.Connection], None],
    ) -> Generator[sqlite3.Connection]:
        """Own one explicit sync route transaction, including all state stages."""
        with self._lock, self._transaction() as connection:
            yield connection
            after_write(connection)

    def ensure_principal(self, principal: MachinePrincipal) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            ensure_edge_site(connection)
            state = self._load(connection)
            if state.principal != principal:
                raise TopologySyncStateConflictError("topology principal does not match enrollment")
            return state

    def create_pending(self, builder: PendingSnapshotBuilder) -> PendingTopologySnapshot:
        with self._lock, self._transaction() as connection:
            state = self._load(connection)
            if state.pending is not None:
                return state.pending
            if state.principal != builder.principal:
                raise TopologySyncStateConflictError("topology principal changed before enqueue")
            client_revision = state.last_client_revision + 1
            body = builder.build(client_revision, state.server_revision)
            connection.execute(
                "UPDATE edge_site SET topology_pending_snapshot_id=?,topology_pending_body=?,"
                "topology_pending_registry_version=?,topology_pending_client_revision=?,"
                "topology_pending_expected_server_revision=?,topology_consecutive_failures=0,"
                "topology_next_retry_at=NULL,topology_pause_reason=NULL,updated_at=? WHERE id=1",
                (
                    builder.snapshot_id,
                    body,
                    builder.registry_version,
                    client_revision,
                    state.server_revision,
                    utc_now(),
                ),
            )
            pending = self._load(connection).pending
            if pending is None:
                raise TopologySyncStateConflictError("pending topology snapshot was not persisted")
            return pending

    def record_retry(self, snapshot_id: str, *, now_epoch: float) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            state = self._require_pending(connection, snapshot_id)
            failures = state.consecutive_failures + 1
            delay = min(BASE_BACKOFF_SECONDS * (2 ** (failures - 1)), MAX_BACKOFF_SECONDS)
            connection.execute(
                "UPDATE edge_site SET topology_consecutive_failures=?,topology_next_retry_at=?,"
                "topology_pause_reason=NULL,updated_at=? WHERE id=1",
                (failures, now_epoch + delay, utc_now()),
            )
            return self._load(connection)

    def pause(self, snapshot_id: str, reason: TopologyPauseReason) -> EdgeTopologySyncState:
        return self._update_pending(
            snapshot_id,
            "topology_pause_reason=?,topology_next_retry_at=NULL",
            (reason.value,),
        )

    def resume_pending(self, snapshot_id: str) -> EdgeTopologySyncState:
        return self._update_pending(snapshot_id, "topology_pause_reason=NULL", ())

    def refresh_conflict(self, snapshot_id: str, server_revision: int) -> EdgeTopologySyncState:
        return self._update_pending(
            snapshot_id,
            "topology_server_revision=?," + _CLEAR_PENDING,
            (server_revision,),
        )

    def accept(
        self, snapshot_id: str, response: TopologySuccessEnvelope, *, now_epoch: float = 0.0
    ) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            state = self._require_pending(connection, snapshot_id)
            pending = state.pending
            if (
                pending is None
                or response.snapshot_id != snapshot_id
                or response.client_revision != pending.client_revision
            ):
                raise TopologySyncStateConflictError("topology acceptance revision mismatch")
            connection.execute(
                "UPDATE edge_site SET topology_snapshot_registry_version=?,"
                "topology_client_revision=?,topology_server_revision=?," + _CLEAR_PENDING + ","
                "topology_last_accepted_at=?,topology_dirty_registry_version="
                "CASE WHEN topology_dirty_registry_version=? THEN NULL "
                "ELSE topology_dirty_registry_version END,"
                "topology_dirty_created_at=CASE WHEN topology_dirty_registry_version=? THEN NULL "
                "ELSE topology_dirty_created_at END,updated_at=? WHERE id=1",
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
            return self._load(connection)

    def _update_pending(
        self, snapshot_id: str, assignments: str, values: tuple[str | int, ...]
    ) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            self._require_pending(connection, snapshot_id)
            connection.execute(
                f"UPDATE edge_site SET {assignments},updated_at=? WHERE id=1",
                (*values, utc_now()),
            )
            return self._load(connection)

    def _require_pending(
        self, connection: sqlite3.Connection, snapshot_id: str
    ) -> EdgeTopologySyncState:
        state = self._load(connection)
        if state.pending is None or state.pending.snapshot_id != snapshot_id:
            raise TopologySyncStateConflictError("pending topology snapshot changed")
        return state

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        if self._connection.in_transaction:
            yield self._connection
            return
        self._connection.execute("BEGIN IMMEDIATE")
        completed = False
        try:
            yield self._connection
            completed = True
        finally:
            self._connection.execute("COMMIT" if completed else "ROLLBACK")

    @staticmethod
    def _load(connection: sqlite3.Connection) -> EdgeTopologySyncState:
        row = connection.execute(_STATE_SELECT).fetchone()
        if row is None:
            return EdgeTopologySyncState(None, None, 0, 0, 0, 0, None, None, None)
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
