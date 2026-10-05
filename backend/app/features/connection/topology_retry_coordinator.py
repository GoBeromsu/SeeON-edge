from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections.abc import Callable
from typing import Protocol, assert_never

import psycopg

from backend.app.features.audit.postgres_runtime import AuditMutation
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncState,
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
    TopologyPauseReason,
    TopologySyncStateConflictError,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPaused,
    TopologyPutResult,
    TopologyRetryable,
    TopologySnapshotBuilder,
)
from backend.app.features.cameras.topology_confirmation import (
    TopologyConfirmationCommand,
    TopologyConfirmationResult,
    TopologyConfirmationService,
)
from backend.app.features.cameras.topology_confirmation_state import (
    TopologyConfirmationPreview,
)
from backend.app.features.connection.topology_retry_result import (
    TopologyRetryResult,
    TopologySyncErrorClass,
    TopologySyncStatus,
    current_retry_result,
    retry_result,
    unconfigured_retry_result,
)
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologyConfirmation

_INCOMPLETE = "모든 카메라에 명시적인 층/방/카메라 참조를 배정해야 합니다."


class TopologyClientProtocol(Protocol):
    @property
    def principal(self) -> MachinePrincipal: ...

    def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult: ...

    def refresh_server_revision(self) -> int | None: ...
    def confirm(
        self, snapshot_id: str, confirmation: TopologyConfirmation
    ) -> TopologyPutResult: ...


class TopologyRetryCoordinator:
    def __init__(
        self,
        registry: CameraRegistryStore,
        state_store: EdgeTopologySyncStateStore,
        client_provider: Callable[[], TopologyClientProtocol | None],
    ) -> None:
        self._registry = registry
        self._state_store = state_store
        self._client_provider = client_provider
        self._confirmation = TopologyConfirmationService(registry, state_store, client_provider)
        self._lock = threading.Lock()

    def trigger(
        self,
        *,
        force: bool = False,
        refresh: bool = False,
        now_epoch: float | None = None,
        audit: AuditMutation | None = None,
    ) -> TopologyRetryResult:
        now = time.time() if now_epoch is None else now_epoch
        if not self._lock.acquire(blocking=False):
            return self.current_result(attempted=False)
        try:
            client = self._client_provider()
            if client is None:
                return self._unconfigured_result()
            if audit is None:
                return self._trigger(client, force=force, refresh=refresh, now=now)
            return audit.apply(
                self._state_store,
                lambda append: self._state_store.operation(
                    lambda connection: self._trigger(
                        client,
                        force=force,
                        refresh=refresh,
                        now=now,
                        connection=connection,
                        audit=audit,
                    ),
                    after_write=append,
                ),
            )
        finally:
            self._lock.release()

    def _trigger(
        self,
        client: TopologyClientProtocol,
        *,
        force: bool,
        refresh: bool,
        now: float,
        connection: psycopg.Connection | None = None,
        audit: AuditMutation | None = None,
    ) -> TopologyRetryResult:
        state = self._state_store.ensure_principal(client.principal, connection=connection)
        state = self._resume_if_refreshed(
            client, state, refresh, connection=connection, audit=audit
        )
        if state.pause_reason is not None:
            return self.current_result(attempted=False, connection=connection)
        if (
            state.pending is not None
            and not force
            and state.next_retry_at is not None
            and now < state.next_retry_at
        ):
            return self.current_result(attempted=False, connection=connection)
        pending = state.pending
        if pending is None:
            topology = self._registry.topology_snapshot(connection=connection)
            dirty = topology.dirty
            if dirty is None or dirty.registry_version <= state.last_snapshotted_registry_version:
                return self.current_result(attempted=False, connection=connection)
            if topology.readiness_error is not None:
                return self._result(
                    state, False, "pending", None, _INCOMPLETE, connection=connection
                )
            pending = self._state_store.create_pending(
                TopologySnapshotBuilder(topology, client.principal, _uuid7()),
                connection=connection,
            )
        if audit is not None:
            audit.require_admission(self._state_store)
        outcome = client.put(pending)
        return self._record_outcome(outcome, pending.snapshot_id, now, connection=connection)

    def current_result(
        self, *, attempted: bool = False, connection: psycopg.Connection | None = None
    ) -> TopologyRetryResult:
        return current_retry_result(
            self._registry, self._state_store, attempted=attempted, connection=connection
        )

    def preview(self) -> TopologyConfirmationPreview | None:
        return self._confirmation.preview()

    def confirm(
        self,
        command: TopologyConfirmationCommand,
        *,
        audit: AuditMutation | None = None,
    ) -> TopologyConfirmationResult:
        return self._confirmation.confirm(command, audit=audit)

    def _resume_if_refreshed(
        self,
        client: TopologyClientProtocol,
        state: EdgeTopologySyncState,
        refresh: bool,
        *,
        connection: psycopg.Connection | None = None,
        audit: AuditMutation | None = None,
    ) -> EdgeTopologySyncState:
        pending = state.pending
        if pending is None or state.pause_reason is None or not refresh:
            return state
        if state.pause_reason is TopologyPauseReason.CONFLICT:
            if audit is not None:
                audit.require_admission(self._state_store)
            server_revision = client.refresh_server_revision()
            if server_revision is None:
                return state
            return self._state_store.refresh_conflict(
                pending.snapshot_id, server_revision, connection=connection
            )
        return self._state_store.resume_pending(pending.snapshot_id, connection=connection)

    def _record_outcome(
        self,
        outcome: TopologyPutResult,
        snapshot_id: str,
        now: float,
        *,
        connection: psycopg.Connection | None = None,
    ) -> TopologyRetryResult:
        if connection is None:
            return self._state_store.operation(
                lambda current: self._record_outcome(outcome, snapshot_id, now, connection=current)
            )
        match outcome:
            case TopologyAccepted(response=response):
                pending = self._state_store.load(connection=connection).pending
                if pending is None:
                    raise TopologySyncStateConflictError(
                        "accepted topology has no pending snapshot"
                    )
                self._confirmation.save_preview(
                    response,
                    pending.principal,
                    pending.registry_version,
                    connection=connection,
                )
                state = self._state_store.accept(
                    snapshot_id, response, now_epoch=now, connection=connection
                )
                return self._result(state, True, "synced", None, None, connection=connection)
            case TopologyRetryable(error_class=error_class):
                state = self._state_store.record_retry(
                    snapshot_id, now_epoch=now, connection=connection
                )
                return self._result(state, True, "failed", error_class, None, connection=connection)
            case TopologyPaused(reason=reason):
                self._state_store.pause(snapshot_id, reason, connection=connection)
                return self.current_result(attempted=True, connection=connection)
            case unreachable:
                assert_never(unreachable)

    def _unconfigured_result(self) -> TopologyRetryResult:
        return unconfigured_retry_result(self._registry)

    def _result(
        self,
        state: EdgeTopologySyncState,
        attempted: bool,
        status: TopologySyncStatus,
        error_class: TopologySyncErrorClass | None,
        detail: str | None,
        *,
        connection: psycopg.Connection | None = None,
    ) -> TopologyRetryResult:
        return retry_result(
            self._registry, state, attempted, status, error_class, detail, connection=connection
        )


def _uuid7() -> str:
    timestamp = int(time.time() * 1000) & ((1 << 48) - 1)
    value = (
        (timestamp << 80)
        | (0x7 << 76)
        | (secrets.randbits(12) << 64)
        | (0b10 << 62)
        | secrets.randbits(62)
    )
    return str(uuid.UUID(int=value))


__all__ = [
    "TopologyRetryCoordinator",
    "TopologyRetryResult",
    "TopologySyncErrorClass",
    "TopologySyncStatus",
]
