from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, TypeAlias, assert_never

from fastapi import FastAPI

from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncState,
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
    TopologyPauseReason,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyClient,
    TopologyPaused,
    TopologyPutResult,
    TopologyRetryable,
    TopologySnapshotBuilder,
)
from backend.app.features.cameras.topology_confirmation_state import (
    TopologyConfirmationPreview,
    TopologyConfirmationStore,
)
from backend.app.shared.backend_client_bundle import backend_client_bundle
from contracts.edge_provisioning_v1 import MachinePrincipal, TopologyConfirmation

TopologySyncStatus: TypeAlias = Literal["pending", "synced", "failed", "disabled"]
TopologySyncErrorClass: TypeAlias = Literal[
    "unconfigured", "auth", "timeout", "unreachable", "conflict"
]

_UNCONFIGURED = "백엔드 등록이 완료되지 않아 토폴로지를 동기화할 수 없습니다."
_INCOMPLETE = "모든 카메라에 명시적인 층/방/카메라 참조를 배정해야 합니다."
_PAUSED = "백엔드 상태를 새로 고치기 전까지 토폴로지 동기화를 일시 중지했습니다."


class TopologyClientProtocol(Protocol):
    @property
    def principal(self) -> MachinePrincipal: ...

    def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult: ...

    def refresh_server_revision(self) -> int | None: ...
    def confirm(
        self, snapshot_id: str, confirmation: TopologyConfirmation
    ) -> TopologyPutResult: ...


@dataclass(frozen=True, slots=True)
class TopologyRetryResult:
    attempted: bool
    status: TopologySyncStatus
    error_class: TopologySyncErrorClass | None
    detail: str | None
    last_ok_at: str | None
    next_retry_at: str | None
    camera_count: int


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
        self._preview_store = TopologyConfirmationStore(registry.path)
        self._lock = threading.Lock()

    def trigger(
        self,
        *,
        force: bool = False,
        refresh: bool = False,
        now_epoch: float | None = None,
    ) -> TopologyRetryResult:
        now = time.time() if now_epoch is None else now_epoch
        if not self._lock.acquire(blocking=False):
            return self.current_result(attempted=False)
        try:
            client = self._client_provider()
            if client is None:
                return self._unconfigured_result()
            state = self._state_store.ensure_principal(client.principal)
            state = self._resume_if_refreshed(client, state, refresh)
            if state.pause_reason is not None:
                return self.current_result(attempted=False)
            if (
                state.pending is not None
                and not force
                and state.next_retry_at is not None
                and now < state.next_retry_at
            ):
                return self.current_result(attempted=False)
            pending = state.pending
            if pending is None:
                topology = self._registry.topology_snapshot()
                dirty = topology.dirty
                if (
                    dirty is None
                    or dirty.registry_version <= state.last_snapshotted_registry_version
                ):
                    return self.current_result(attempted=False)
                if topology.readiness_error is not None:
                    return self._result(state, False, "pending", None, _INCOMPLETE)
                pending = self._state_store.create_pending(
                    TopologySnapshotBuilder(topology, client.principal, _uuid7())
                )
            outcome = client.put(pending)
            return self._record_outcome(outcome, pending.snapshot_id, now)
        finally:
            self._lock.release()

    def current_result(self, *, attempted: bool = False) -> TopologyRetryResult:
        state = self._state_store.load()
        if state.pause_reason is not None:
            pause_error: TopologySyncErrorClass = (
                "auth"
                if state.pause_reason in {TopologyPauseReason.AUTH, TopologyPauseReason.FORBIDDEN}
                else "conflict"
            )
            return self._result(state, attempted, "failed", pause_error, _PAUSED)
        if state.pending is not None:
            pending_error: TopologySyncErrorClass | None = (
                "unreachable" if state.consecutive_failures else None
            )
            status: TopologySyncStatus = "failed" if pending_error else "pending"
            return self._result(state, attempted, status, pending_error, None)
        topology = self._registry.topology_snapshot()
        if topology.readiness_error is not None:
            return self._result(state, attempted, "pending", None, _INCOMPLETE)
        status = (
            "synced"
            if topology.dirty is None and state.last_snapshotted_registry_version > 0
            else "pending"
        )
        return self._result(state, attempted, status, None, None)

    def preview(self) -> TopologyConfirmationPreview | None:
        return self._preview_store.load()

    def confirm(
        self, confirmation_id: str, digest: str, client_revision: int, server_revision: int
    ) -> TopologyRetryResult:
        preview = self._preview_store.load()
        client = self._client_provider()
        if preview is None or client is None:
            return self.current_result()
        if preview.confirmed:
            return self.current_result(attempted=False)
        topology = self._registry.topology_snapshot()
        expires_at = datetime.fromisoformat(preview.expires_at.replace("Z", "+00:00"))
        if (
            expires_at <= datetime.now(UTC)
            or preview.principal != client.principal
            or preview.registry_version != topology.registry_version
            or preview.confirmation_id != confirmation_id
            or preview.digest != digest
            or preview.client_revision != client_revision
            or preview.server_revision != server_revision
        ):
            return self.current_result(attempted=False)
        outcome = client.confirm(
            preview.snapshot_id, TopologyConfirmation(confirmation_id, digest, server_revision)
        )
        match outcome:
            case TopologyAccepted():
                self._preview_store.confirm(confirmation_id)
                return self.current_result(attempted=True)
            case TopologyRetryable() | TopologyPaused():
                return self.current_result(attempted=True)
            case unreachable:
                assert_never(unreachable)

    def _resume_if_refreshed(
        self,
        client: TopologyClientProtocol,
        state: EdgeTopologySyncState,
        refresh: bool,
    ) -> EdgeTopologySyncState:
        pending = state.pending
        if pending is None or state.pause_reason is None or not refresh:
            return state
        if state.pause_reason is TopologyPauseReason.CONFLICT:
            server_revision = client.refresh_server_revision()
            if server_revision is None:
                return state
            return self._state_store.refresh_conflict(pending.snapshot_id, server_revision)
        return self._state_store.resume_pending(pending.snapshot_id)

    def _record_outcome(
        self, outcome: TopologyPutResult, snapshot_id: str, now: float
    ) -> TopologyRetryResult:
        match outcome:
            case TopologyAccepted(response=response):
                pending = self._state_store.load().pending
                if pending is None:
                    raise RuntimeError("accepted topology has no pending snapshot")
                self._preview_store.save(response, pending.principal, pending.registry_version)
                state = self._state_store.accept(snapshot_id, response, now_epoch=now)
                return self._result(state, True, "synced", None, None)
            case TopologyRetryable(error_class=error_class):
                state = self._state_store.record_retry(snapshot_id, now_epoch=now)
                return self._result(state, True, "failed", error_class, None)
            case TopologyPaused(reason=reason):
                state = self._state_store.pause(snapshot_id, reason)
                error_class = (
                    "auth"
                    if reason in {TopologyPauseReason.AUTH, TopologyPauseReason.FORBIDDEN}
                    else "conflict"
                )
                return self._result(state, True, "failed", error_class, _PAUSED)
            case unreachable:
                assert_never(unreachable)

    def _unconfigured_result(self) -> TopologyRetryResult:
        topology = self._registry.topology_snapshot()
        status: TopologySyncStatus = "pending" if topology.dirty is not None else "disabled"
        return TopologyRetryResult(
            False,
            status,
            "unconfigured",
            _UNCONFIGURED,
            None,
            None,
            _camera_count(self._registry),
        )

    def _result(
        self,
        state: EdgeTopologySyncState,
        attempted: bool,
        status: TopologySyncStatus,
        error_class: TopologySyncErrorClass | None,
        detail: str | None,
    ) -> TopologyRetryResult:
        return TopologyRetryResult(
            attempted,
            status,
            error_class,
            detail,
            _iso_timestamp(state.last_accepted_at),
            _iso_timestamp(state.next_retry_at),
            _camera_count(self._registry),
        )


def topology_retry_coordinator(app: FastAPI) -> TopologyRetryCoordinator:
    existing = getattr(app.state, "topology_retry_coordinator", None)
    if isinstance(existing, TopologyRetryCoordinator):
        return existing
    registry: CameraRegistryStore
    candidate = getattr(app.state, "camera_registry", None)
    if isinstance(candidate, CameraRegistryStore):
        registry = candidate
    else:
        registry = CameraRegistryStore.from_env()
        app.state.camera_registry = registry

    def client_provider() -> TopologyClient | None:
        bundle = backend_client_bundle(app)
        return None if bundle is None else TopologyClient.from_bundle(bundle)

    coordinator = TopologyRetryCoordinator(
        registry,
        EdgeTopologySyncStateStore(registry.path),
        client_provider,
    )
    app.state.topology_retry_coordinator = coordinator
    return coordinator


def _camera_count(registry: CameraRegistryStore) -> int:
    return len(registry.snapshot()["cameras"])


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
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
    "topology_retry_coordinator",
]
