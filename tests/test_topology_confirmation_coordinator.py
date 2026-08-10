from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPutResult,
)
from backend.app.features.cameras.topology_confirmation import (
    TopologyConfirmationCommand,
    TopologyConfirmationRejected,
)
from backend.app.features.connection.topology_retry_coordinator import (
    TopologyRetryCoordinator,
)
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyConfirmation,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 3)
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
DIGEST = "a" * 64


class _Client:
    def __init__(
        self,
        principal: MachinePrincipal,
        put_outcomes: list[Callable[[PendingTopologySnapshot], TopologyPutResult]],
        confirm_outcomes: list[TopologyPutResult],
    ) -> None:
        self.principal = principal
        self.put_outcomes = put_outcomes
        self.confirm_outcomes = confirm_outcomes
        self.sent: list[PendingTopologySnapshot] = []
        self.confirmations: list[tuple[str, TopologyConfirmation]] = []

    def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult:
        self.sent.append(pending)
        return self.put_outcomes.pop(0)(pending)

    def confirm(
        self, snapshot_id: str, confirmation: TopologyConfirmation
    ) -> TopologyPutResult:
        self.confirmations.append((snapshot_id, confirmation))
        return self.confirm_outcomes.pop(0)

    def refresh_server_revision(self) -> int | None:
        return None


def _registry(path: Path) -> CameraRegistryStore:
    store = CameraRegistryStore(path)
    store.create_floor(edge_ref="floor-1", name="First", order_index=1)
    store.create_room(edge_ref="room-101", floor_edge_ref="floor-1", name="101")
    store.create(
        camera_id="local-1",
        label="Lobby",
        rtsp_url="rtsp://private",
        space_id=None,
        status="online",
        edge_ref="camera-1",
        room_edge_ref="room-101",
    )
    return store


def _preview_acceptance(
    pending: PendingTopologySnapshot, *, expires_at: str = "2099-01-01T00:00:00.000Z"
) -> TopologyPutResult:
    unchanged = MutationCounts(0, 0, 1)
    return TopologyAccepted(
        TopologySuccessEnvelope(
            pending.snapshot_id,
            pending.client_revision,
            7,
            TopologyMutationResult(unchanged, unchanged, unchanged),
            OmissionPreview(
                CONFIRMATION_ID,
                DIGEST,
                expires_at,
                ("camera-old",),
                (),
                (),
            ),
        )
    )


def _terminal(pending: PendingTopologySnapshot) -> TopologyAccepted:
    unchanged = MutationCounts(0, 0, 1)
    deactivated = MutationCounts(0, 0, 0, deactivated=1)
    return TopologyAccepted(
        TopologySuccessEnvelope(
            pending.snapshot_id,
            pending.client_revision,
            8,
            TopologyMutationResult(unchanged, unchanged, deactivated),
            None,
        )
    )


def _primed(
    path: Path, *, expires_at: str = "2099-01-01T00:00:00.000Z"
) -> tuple[TopologyRetryCoordinator, CameraRegistryStore, _Client, TopologyAccepted]:
    registry = _registry(path)
    client = _Client(PRINCIPAL, [], [])
    def accept(pending: PendingTopologySnapshot) -> TopologyPutResult:
        client.confirm_outcomes.append(_terminal(pending))
        return _preview_acceptance(pending, expires_at=expires_at)

    client.put_outcomes.append(accept)
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: client
    )
    result = coordinator.trigger(force=True, now_epoch=100.0)
    assert result.status == "synced"
    assert client.confirmations == []
    terminal = client.confirm_outcomes[0]
    assert isinstance(terminal, TopologyAccepted)
    return coordinator, registry, client, terminal


def _confirm(coordinator: TopologyRetryCoordinator):
    return coordinator.confirm(TopologyConfirmationCommand(CONFIRMATION_ID, DIGEST, 1, 7))


def test_manual_confirmation_persists_terminal_result_and_advances_revision(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, _registry_store, client, expected = _primed(path)

    # When
    result = _confirm(coordinator)

    # Then
    assert result == expected
    assert len(client.confirmations) == 1
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is True
    assert EdgeTopologySyncStateStore(path).load().server_revision == 8


def test_exact_confirmation_replay_returns_terminal_result_without_second_upstream_call(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, registry, client, expected = _primed(path)
    first = _confirm(coordinator)
    restarted = TopologyRetryCoordinator(
        CameraRegistryStore(path), EdgeTopologySyncStateStore(path), lambda: client
    )

    # When
    replay = _confirm(restarted)

    # Then
    assert first == replay == expected
    assert len(client.confirmations) == 1
    assert registry.topology_snapshot().registry_version == 3


def test_expired_confirmation_has_zero_upstream_calls_or_local_mutation(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, _registry_store, client, _expected = _primed(
        path, expires_at="2000-01-01T00:00:00.000Z"
    )
    before = EdgeTopologySyncStateStore(path).load()

    # When
    result = _confirm(coordinator)

    # Then
    assert isinstance(result, TopologyConfirmationRejected)
    assert result.status_code == 410
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is False
    assert EdgeTopologySyncStateStore(path).load() == before


def test_changed_registry_rejects_confirmation_without_mutation(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, registry, client, _expected = _primed(path)
    before = EdgeTopologySyncStateStore(path).load()
    registry.update("local-1", {"label": "Changed"})

    # When
    result = _confirm(coordinator)

    # Then
    assert isinstance(result, TopologyConfirmationRejected)
    assert result.status_code == 409
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is False
    assert EdgeTopologySyncStateStore(path).load() == before


@pytest.mark.parametrize(
    ("client_revision", "server_revision"),
    [(2, 7), (1, 8)],
)
def test_stale_request_revision_rejects_without_upstream_call(
    tmp_path: Path, client_revision: int, server_revision: int
) -> None:
    # Given
    path = tmp_path / f"catalog-{client_revision}-{server_revision}.sqlite3"
    coordinator, _registry_store, client, _expected = _primed(path)

    # When
    result = coordinator.confirm(
        TopologyConfirmationCommand(
            CONFIRMATION_ID, DIGEST, client_revision, server_revision
        )
    )

    # Then
    assert isinstance(result, TopologyConfirmationRejected)
    assert result.status_code == 409
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is False


def test_changed_digest_rejects_without_upstream_call_or_local_mutation(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, _registry_store, client, _expected = _primed(path)
    before = EdgeTopologySyncStateStore(path).load()

    # When
    result = coordinator.confirm(
        TopologyConfirmationCommand(CONFIRMATION_ID, "b" * 64, 1, 7)
    )

    # Then
    assert isinstance(result, TopologyConfirmationRejected)
    assert result.status_code == 409
    assert client.confirmations == []
    assert EdgeTopologySyncStateStore(path).load() == before
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is False


def test_changed_generation_rejects_without_upstream_call(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, registry, original_client, _expected = _primed(path)
    changed_client = _Client(
        MachinePrincipal(PRINCIPAL.edge_installation_id, 4), [], []
    )
    changed = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: changed_client
    )

    # When
    result = _confirm(changed)

    # Then
    assert isinstance(result, TopologyConfirmationRejected)
    assert result.status_code == 409
    assert original_client.confirmations == []
    assert changed_client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is False


def test_changed_local_server_revision_fails_confirmation_cas(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    coordinator, _registry_store, client, _expected = _primed(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE edge_topology_sync_state SET server_revision = 9 WHERE id = 1")

    # When
    result = _confirm(coordinator)

    # Then
    assert isinstance(result, TopologyConfirmationRejected)
    assert result.status_code == 409
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None
    assert preview.confirmed is False
    assert EdgeTopologySyncStateStore(path).load().server_revision == 9
