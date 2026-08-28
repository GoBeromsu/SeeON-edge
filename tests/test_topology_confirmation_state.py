from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import bootstrap_database
from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.topology_confirmation_state import TopologyConfirmationStore
from backend.app.features.connection.store import ConnectionSettingsStore
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 3)
SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81a"
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
DIGEST = "a" * 64


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


def _result(*, deactivated: int = 0) -> TopologyMutationResult:
    unchanged = MutationCounts(0, 0, 1)
    removed = MutationCounts(0, 0, 0, deactivated=deactivated)
    return TopologyMutationResult(removed, removed, unchanged)


def _preview_response() -> TopologySuccessEnvelope:
    return TopologySuccessEnvelope(
        SNAPSHOT_ID,
        4,
        7,
        _result(),
        OmissionPreview(
            CONFIRMATION_ID,
            DIGEST,
            "2099-01-01T00:00:00.000Z",
            ("camera-old",),
            ("room-old",),
            ("floor-old",),
        ),
    )


def _terminal_response() -> TopologySuccessEnvelope:
    return TopologySuccessEnvelope(SNAPSHOT_ID, 4, 8, _result(deactivated=1), None)


def test_preview_and_terminal_response_survive_store_restart(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    store = TopologyConfirmationStore(path)
    state_store = EdgeTopologySyncStateStore(path)
    _ = state_store.ensure_principal(PRINCIPAL)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE edge_site SET topology_client_revision=4,topology_server_revision=7 WHERE id=1"
        )

    # When
    store.save(_preview_response(), PRINCIPAL, registry_version=12)
    persisted_preview = TopologyConfirmationStore(path).load()
    assert persisted_preview is not None
    store.complete(persisted_preview, _terminal_response())
    terminal_preview = TopologyConfirmationStore(path).load()

    # Then
    assert persisted_preview.confirmed is False
    assert persisted_preview.registry_version == 12
    assert persisted_preview.principal == PRINCIPAL
    assert terminal_preview is not None
    assert terminal_preview.confirmed is True
    assert terminal_preview.terminal_response == _terminal_response()


def test_confirmation_store_uses_no_feature_local_table(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite3"

    assert TopologyConfirmationStore(path).load() is None

    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "edge_topology_confirmation_preview" not in tables
