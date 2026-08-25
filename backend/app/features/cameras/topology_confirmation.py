from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypeAlias, assert_never

from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPaused,
    TopologyPutResult,
    TopologyRetryable,
)
from backend.app.features.cameras.topology_confirmation_state import (
    TopologyConfirmationPreview,
    TopologyConfirmationStateConflictError,
    TopologyConfirmationStore,
)
from contracts.edge_provisioning_v1 import (
    EdgeErrorCode,
    MachinePrincipal,
    TopologyConfirmation,
    TopologySuccessEnvelope,
)


class ConfirmationClientProtocol(Protocol):
    @property
    def principal(self) -> MachinePrincipal: ...

    def confirm(
        self, snapshot_id: str, confirmation: TopologyConfirmation
    ) -> TopologyPutResult: ...


@dataclass(frozen=True, slots=True)
class TopologyConfirmationCommand:
    confirmation_id: str
    digest: str
    client_revision: int
    server_revision: int


@dataclass(frozen=True, slots=True)
class TopologyConfirmationRejected:
    status_code: int
    error_code: EdgeErrorCode


TopologyConfirmationResult: TypeAlias = TopologyPutResult | TopologyConfirmationRejected


class TopologyConfirmationService:
    def __init__(
        self,
        registry: CameraRegistryStore,
        state_store: EdgeTopologySyncStateStore,
        client_provider: Callable[[], ConfirmationClientProtocol | None],
    ) -> None:
        self._registry = registry
        self._state_store = state_store
        self._client_provider = client_provider
        self._store = TopologyConfirmationStore(registry.path)
        self._lock = threading.Lock()

    def preview(self) -> TopologyConfirmationPreview | None:
        return self._store.load()

    def save_preview(
        self,
        response: TopologySuccessEnvelope,
        principal: MachinePrincipal,
        registry_version: int,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._store.save(
            response, principal, registry_version, connection=connection
        )

    def confirm(
        self,
        command: TopologyConfirmationCommand,
        *,
        after_write: Callable[[sqlite3.Connection], None] | None = None,
    ) -> TopologyConfirmationResult:
        with self._lock:
            preview = self._store.load()
            client = self._client_provider()
            if preview is None:
                return TopologyConfirmationRejected(409, EdgeErrorCode.CONFIRMATION_STALE)
            if client is None:
                return TopologyConfirmationRejected(409, EdgeErrorCode.EDGE_AUTH_NOT_CONFIGURED)
            if not _matches_request(preview, client.principal, command):
                return TopologyConfirmationRejected(409, EdgeErrorCode.CONFIRMATION_STALE)
            if preview.terminal_response is not None:
                return TopologyAccepted(preview.terminal_response)
            if self._pending_is_stale(preview):
                return TopologyConfirmationRejected(409, EdgeErrorCode.CONFIRMATION_STALE)
            if _expired(preview.expires_at):
                return TopologyConfirmationRejected(410, EdgeErrorCode.CONFIRMATION_EXPIRED)
            outcome = client.confirm(
                preview.snapshot_id,
                TopologyConfirmation(
                    command.confirmation_id, command.digest, command.server_revision
                ),
            )
            match outcome:
                case TopologyAccepted(response=response):
                    if (
                        response.snapshot_id != preview.snapshot_id
                        or response.client_revision != preview.client_revision
                    ):
                        return TopologyRetryable("unreachable")
                    try:
                        self._store.complete(
                            preview, response, after_write=after_write
                        )
                    except TopologyConfirmationStateConflictError:
                        return TopologyConfirmationRejected(
                            409, EdgeErrorCode.CONFIRMATION_STALE
                        )
                    return outcome
                case TopologyRetryable() | TopologyPaused():
                    return outcome
                case unreachable:
                    assert_never(unreachable)

    def _pending_is_stale(self, preview: TopologyConfirmationPreview) -> bool:
        state = self._state_store.load()
        topology = self._registry.topology_snapshot()
        return (
            preview.registry_version != topology.registry_version
            or state.principal != preview.principal
            or state.last_client_revision != preview.client_revision
            or state.server_revision != preview.server_revision
        )


def _matches_request(
    preview: TopologyConfirmationPreview,
    principal: MachinePrincipal,
    command: TopologyConfirmationCommand,
) -> bool:
    return (
        preview.principal == principal
        and preview.confirmation_id == command.confirmation_id
        and preview.digest == command.digest
        and preview.client_revision == command.client_revision
        and preview.server_revision == command.server_revision
    )


def _expired(expires_at: str) -> bool:
    parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    return parsed <= datetime.now(UTC)


__all__ = [
    "ConfirmationClientProtocol",
    "TopologyConfirmationCommand",
    "TopologyConfirmationRejected",
    "TopologyConfirmationResult",
    "TopologyConfirmationService",
]
