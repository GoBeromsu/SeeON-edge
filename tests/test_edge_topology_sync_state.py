from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import ConnectionSettingsStore
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)


@pytest.fixture(autouse=True)
def _enrolled_compact_database(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"
    bootstrap_database(path)
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


@dataclass(frozen=True, slots=True)
class _Builder:
    registry_version: int
    principal: MachinePrincipal
    snapshot_id: str
    marker: bytes

    def build(self, client_revision: int, expected_server_revision: int) -> bytes:
        return self.marker + f":{client_revision}:{expected_server_revision}".encode()


def _accepted(snapshot_id: str, client_revision: int, server_revision: int):
    counts = MutationCounts(0, 0, 1)
    return TopologySuccessEnvelope(
        snapshot_id,
        client_revision,
        server_revision,
        TopologyMutationResult(counts, counts, counts),
        None,
    )


def test_pending_snapshot_and_backoff_survive_restart_byte_identically(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    store = EdgeTopologySyncStateStore(path)
    store.ensure_principal(PRINCIPAL)
    pending = store.create_pending(_Builder(7, PRINCIPAL, "snapshot-a", b"canonical"))

    # When
    store.record_retry(pending.snapshot_id, now_epoch=100.0)
    restarted = EdgeTopologySyncStateStore(path).load()

    # Then
    assert restarted.pending == pending
    assert restarted.pending is not None
    assert restarted.pending.body == b"canonical:1:0"
    assert restarted.consecutive_failures == 1
    assert restarted.next_retry_at == 105.0


def test_accept_clears_only_the_represented_dirty_registry_version(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = CameraRegistryStore(path)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    represented_version = registry.topology_snapshot().registry_version
    state = EdgeTopologySyncStateStore(path)
    state.ensure_principal(PRINCIPAL)
    pending = state.create_pending(_Builder(represented_version, PRINCIPAL, "snapshot-a", b"body"))
    registry.create_floor(edge_ref="floor-2", name="Second", order_index=2)

    # When
    accepted = state.accept(
        pending.snapshot_id,
        _accepted(pending.snapshot_id, pending.client_revision, 1),
    )

    # Then
    assert accepted.pending is None
    assert accepted.last_snapshotted_registry_version == represented_version
    dirty = registry.topology_snapshot().dirty
    assert dirty is not None
    assert dirty.registry_version == represented_version + 1


def test_generation_change_discards_old_pending_but_keeps_registry_dirty(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    registry = CameraRegistryStore(path)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    state = EdgeTopologySyncStateStore(path)
    state.ensure_principal(PRINCIPAL)
    state.create_pending(_Builder(1, PRINCIPAL, "snapshot-a", b"old"))

    # When
    ConnectionSettingsStore(path).save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "token-2",
            "edge_installation_id": PRINCIPAL.edge_installation_id,
            "enrollment_generation": 2,
        }
    )
    changed = state.ensure_principal(MachinePrincipal(PRINCIPAL.edge_installation_id, 2))

    # Then
    assert changed.pending is None
    assert changed.last_client_revision == 0
    assert registry.topology_snapshot().dirty is not None
