from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from threading import Lock
from typing import Final, Protocol

from backend.app.shared.sqlite_bootstrap import connect_catalog_store
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologySuccessEnvelope

BASE_BACKOFF_SECONDS: Final = 5.0
MAX_BACKOFF_SECONDS: Final = 300.0

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS edge_topology_sync_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  edge_installation_id TEXT,
  enrollment_generation INTEGER CHECK (enrollment_generation IS NULL OR enrollment_generation > 0),
  last_snapshotted_registry_version INTEGER NOT NULL DEFAULT 0
    CHECK (last_snapshotted_registry_version >= 0),
  last_client_revision INTEGER NOT NULL DEFAULT 0 CHECK (last_client_revision >= 0),
  server_revision INTEGER NOT NULL DEFAULT 0 CHECK (server_revision >= 0),
  pending_snapshot_id TEXT,
  pending_body BLOB,
  pending_registry_version INTEGER
    CHECK (pending_registry_version IS NULL OR pending_registry_version >= 0),
  pending_client_revision INTEGER
    CHECK (pending_client_revision IS NULL OR pending_client_revision > 0),
  pending_expected_server_revision INTEGER
    CHECK (pending_expected_server_revision IS NULL OR pending_expected_server_revision >= 0),
  consecutive_failures INTEGER NOT NULL DEFAULT 0 CHECK (consecutive_failures >= 0),
  next_retry_at REAL,
  pause_reason TEXT
    CHECK (pause_reason IS NULL OR pause_reason IN ('auth', 'forbidden', 'conflict')),
  last_accepted_at REAL
) STRICT
"""


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
        self._lock = Lock()
        self._connection = connect_catalog_store(self.path, (_SCHEMA,))
        self._connection.execute("INSERT OR IGNORE INTO edge_topology_sync_state (id) VALUES (1)")

    def load(self) -> EdgeTopologySyncState:
        with self._lock:
            return self._load(self._connection)

    def ensure_principal(self, principal: MachinePrincipal) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            state = self._load(connection)
            if state.principal == principal:
                return state
            connection.execute(
                "UPDATE edge_topology_sync_state SET edge_installation_id = ?, "
                "enrollment_generation = ?, last_snapshotted_registry_version = 0, "
                "last_client_revision = 0, server_revision = 0, pending_snapshot_id = NULL, "
                "pending_body = NULL, pending_registry_version = NULL, "
                "pending_client_revision = NULL, pending_expected_server_revision = NULL, "
                "consecutive_failures = 0, next_retry_at = NULL, pause_reason = NULL, "
                "last_accepted_at = NULL WHERE id = 1",
                (principal.edge_installation_id, principal.enrollment_generation),
            )
            return self._load(connection)

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
                "UPDATE edge_topology_sync_state SET pending_snapshot_id = ?, pending_body = ?, "
                "pending_registry_version = ?, pending_client_revision = ?, "
                "pending_expected_server_revision = ?, consecutive_failures = 0, "
                "next_retry_at = NULL, pause_reason = NULL WHERE id = 1",
                (
                    builder.snapshot_id,
                    body,
                    builder.registry_version,
                    client_revision,
                    state.server_revision,
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
                "UPDATE edge_topology_sync_state SET consecutive_failures = ?, "
                "next_retry_at = ?, pause_reason = NULL WHERE id = 1",
                (failures, now_epoch + delay),
            )
            return self._load(connection)

    def pause(
        self, snapshot_id: str, reason: TopologyPauseReason
    ) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            self._require_pending(connection, snapshot_id)
            connection.execute(
                "UPDATE edge_topology_sync_state SET pause_reason = ?, next_retry_at = NULL "
                "WHERE id = 1",
                (reason.value,),
            )
            return self._load(connection)

    def resume_pending(self, snapshot_id: str) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            self._require_pending(connection, snapshot_id)
            connection.execute(
                "UPDATE edge_topology_sync_state SET pause_reason = NULL WHERE id = 1"
            )
            return self._load(connection)

    def refresh_conflict(self, snapshot_id: str, server_revision: int) -> EdgeTopologySyncState:
        with self._lock, self._transaction() as connection:
            self._require_pending(connection, snapshot_id)
            connection.execute(
                "UPDATE edge_topology_sync_state SET server_revision = ?, "
                "pending_snapshot_id = NULL, pending_body = NULL, "
                "pending_registry_version = NULL, pending_client_revision = NULL, "
                "pending_expected_server_revision = NULL, consecutive_failures = 0, "
                "next_retry_at = NULL, pause_reason = NULL WHERE id = 1",
                (server_revision,),
            )
            return self._load(connection)

    def accept(
        self,
        snapshot_id: str,
        response: TopologySuccessEnvelope,
        *,
        now_epoch: float = 0.0,
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
                "UPDATE edge_topology_sync_state SET last_snapshotted_registry_version = ?, "
                "last_client_revision = ?, server_revision = ?, pending_snapshot_id = NULL, "
                "pending_body = NULL, pending_registry_version = NULL, "
                "pending_client_revision = NULL, pending_expected_server_revision = NULL, "
                "consecutive_failures = 0, next_retry_at = NULL, pause_reason = NULL, "
                "last_accepted_at = ? WHERE id = 1",
                (
                    pending.registry_version,
                    response.client_revision,
                    response.server_revision,
                    now_epoch,
                ),
            )
            connection.execute(
                "DELETE FROM topology_dirty WHERE id = 1 AND registry_version = ?",
                (pending.registry_version,),
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
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    @staticmethod
    def _load(connection: sqlite3.Connection) -> EdgeTopologySyncState:
        row = connection.execute("SELECT * FROM edge_topology_sync_state WHERE id = 1").fetchone()
        if row is None:
            raise TopologySyncStateConflictError("topology sync state row is missing")
        principal = None if row[1] is None else MachinePrincipal(str(row[1]), int(row[2]))
        pending = None
        if row[6] is not None:
            if principal is None:
                raise TopologySyncStateConflictError("pending snapshot has no principal")
            pending = PendingTopologySnapshot(
                str(row[6]), bytes(row[7]), int(row[8]), int(row[9]), int(row[10]), principal
            )
        pause = None if row[13] is None else TopologyPauseReason(str(row[13]))
        return EdgeTopologySyncState(
            principal,
            pending,
            int(row[3]),
            int(row[4]),
            int(row[5]),
            int(row[11]),
            None if row[12] is None else float(row[12]),
            pause,
            None if row[14] is None else float(row[14]),
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
