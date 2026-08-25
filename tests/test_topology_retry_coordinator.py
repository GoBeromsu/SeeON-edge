from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
    TopologyPauseReason,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPaused,
    TopologyPutResult,
    TopologyRetryable,
)
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import (
    TopologyRetryCoordinator,
)
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    TopologyConfirmation,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)


@pytest.fixture(autouse=True)
def _enrolled_compact_database(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    migrate_database(path)
    ConnectionSettingsStore(path).save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "token-1",
            "edge_installation_id": PRINCIPAL.edge_installation_id,
            "enrollment_generation": PRINCIPAL.enrollment_generation,
        }
    )


def _ready_registry(path: Path) -> CameraRegistryStore:
    if not path.exists():
        migrate_database(path)
        ConnectionSettingsStore(path).save(
            {
                "facility_code": "NH-1234",
                "client_installation_ref": "install-1",
                "facility_id": "facility-1",
                "facility_token": "token-1",
                "edge_installation_id": PRINCIPAL.edge_installation_id,
                "enrollment_generation": PRINCIPAL.enrollment_generation,
            }
        )
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


def _accepted(pending: PendingTopologySnapshot, server_revision: int = 1):
    counts = MutationCounts(0, 0, 1)
    return TopologyAccepted(
        TopologySuccessEnvelope(
            pending.snapshot_id,
            pending.client_revision,
            server_revision,
            TopologyMutationResult(counts, counts, counts),
            None,
        )
    )


class _Client:
    principal = PRINCIPAL

    def __init__(
        self, outcomes: list[Callable[[PendingTopologySnapshot], TopologyPutResult]]
    ) -> None:
        self.outcomes = outcomes
        self.sent: list[PendingTopologySnapshot] = []
        self.confirmations: list[tuple[str, TopologyConfirmation]] = []
        self.refreshed_revision: int | None = None

    def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult:
        self.sent.append(pending)
        return self.outcomes.pop(0)(pending)

    def refresh_server_revision(self) -> int | None:
        return self.refreshed_revision

    def confirm(self, snapshot_id: str, confirmation: TopologyConfirmation) -> TopologyPutResult:
        self.confirmations.append((snapshot_id, confirmation))
        return TopologyRetryable("unreachable")


def test_timeout_after_commit_restarts_with_byte_identical_pending_snapshot(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = _ready_registry(path)
    first_client = _Client([lambda _pending: TopologyRetryable("timeout")])
    first = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: first_client
    )

    # When
    first_result = first.trigger(force=True, now_epoch=100.0)
    second_client = _Client([_accepted])
    restarted = TopologyRetryCoordinator(
        CameraRegistryStore(path), EdgeTopologySyncStateStore(path), lambda: second_client
    )
    second_result = restarted.trigger(force=True, now_epoch=101.0)

    # Then
    assert first_result.status == "failed"
    assert second_result.status == "synced"
    assert first_client.sent[0] == second_client.sent[0]
    assert first_client.sent[0].body == second_client.sent[0].body


def test_boot_recovery_enqueues_exactly_one_snapshot_for_fresh_dirty_registry(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = _ready_registry(path)
    client = _Client([_accepted])
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: client
    )

    # When
    first = coordinator.trigger(force=True, now_epoch=100.0)
    second = coordinator.trigger(force=True, now_epoch=101.0)

    # Then
    assert first.status == "synced"
    assert second.attempted is False
    assert len(client.sent) == 1


def test_coordinator_crash_leaves_pending_for_exact_restart_resume(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = _ready_registry(path)

    class SimulatedCrash(RuntimeError):
        pass

    def crash(_pending: PendingTopologySnapshot) -> TopologyPutResult:
        raise SimulatedCrash

    crashing_client = _Client([crash])
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: crashing_client
    )

    # When
    with pytest.raises(SimulatedCrash):
        coordinator.trigger(force=True, now_epoch=100.0)
    resumed_client = _Client([_accepted])
    restarted = TopologyRetryCoordinator(
        CameraRegistryStore(path), EdgeTopologySyncStateStore(path), lambda: resumed_client
    )
    resumed = restarted.trigger(force=True, now_epoch=101.0)

    # Then
    assert resumed.status == "synced"
    assert crashing_client.sent[0] == resumed_client.sent[0]


def test_conflict_pauses_until_refresh_then_rebuilds_against_refreshed_revision(
    tmp_path: Path,
) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = _ready_registry(path)
    client = _Client(
        [
            lambda _pending: TopologyPaused(TopologyPauseReason.CONFLICT, 409),
            lambda pending: _accepted(pending, server_revision=5),
        ]
    )
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: client
    )

    # When
    paused = coordinator.trigger(force=True, now_epoch=100.0)
    blind = coordinator.trigger(force=True, now_epoch=101.0)
    client.refreshed_revision = 4
    resumed = coordinator.trigger(force=True, refresh=True, now_epoch=102.0)

    # Then
    assert paused.status == "failed"
    assert blind.attempted is False
    assert resumed.status == "synced"
    assert len(client.sent) == 2
    assert client.sent[0].snapshot_id != client.sent[1].snapshot_id
    assert json_expected_revision(client.sent[1].body) == 4


@pytest.mark.parametrize(
    ("reason", "status"),
    [(TopologyPauseReason.AUTH, 401), (TopologyPauseReason.FORBIDDEN, 403)],
)
def test_auth_pause_resumes_exact_pending_after_connectivity(
    tmp_path: Path, reason: TopologyPauseReason, status: int
) -> None:
    # Given
    path = tmp_path / f"catalog-{status}.sqlite3"
    registry = _ready_registry(path)
    client = _Client([lambda _pending: TopologyPaused(reason, status), _accepted])
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: client
    )

    # When
    coordinator.trigger(force=True, now_epoch=100.0)
    coordinator.trigger(force=True, refresh=True, now_epoch=101.0)

    # Then
    assert client.sent[0] == client.sent[1]


def test_concurrent_trigger_is_single_flight(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = _ready_registry(path)
    entered = threading.Event()
    release = threading.Event()

    def block_then_accept(pending: PendingTopologySnapshot):
        entered.set()
        release.wait(timeout=2)
        return _accepted(pending)

    client = _Client([block_then_accept])
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: client
    )
    first_results: list[object] = []
    thread = threading.Thread(
        target=lambda: first_results.append(coordinator.trigger(force=True, now_epoch=100.0))
    )

    # When
    thread.start()
    assert entered.wait(timeout=1)
    concurrent = coordinator.trigger(force=True, now_epoch=100.0)
    release.set()
    thread.join(timeout=2)

    # Then
    assert concurrent.attempted is False
    assert len(client.sent) == 1


def json_expected_revision(body: bytes) -> int:
    import json

    value = json.loads(body)
    return int(value["expectedServerRevision"])
