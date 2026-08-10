from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.topology_confirmation_state import (
    TopologyConfirmationStore,
)
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
            "UPDATE edge_topology_sync_state SET last_client_revision = 4, "
            "server_revision = 7 WHERE id = 1"
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


def test_existing_preview_schema_is_extended_for_terminal_replay(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE edge_topology_confirmation_preview ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), confirmation_id TEXT NOT NULL, "
            "digest TEXT NOT NULL, expires_at TEXT NOT NULL, snapshot_id TEXT NOT NULL, "
            "client_revision INTEGER NOT NULL, server_revision INTEGER NOT NULL, "
            "registry_version INTEGER NOT NULL, edge_installation_id TEXT NOT NULL, "
            "enrollment_generation INTEGER NOT NULL, cameras INTEGER NOT NULL, "
            "rooms INTEGER NOT NULL, floors INTEGER NOT NULL, "
            "confirmed INTEGER NOT NULL DEFAULT 0) STRICT"
        )

    # When
    _ = TopologyConfirmationStore(path)

    # Then
    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(edge_topology_confirmation_preview)"
            ).fetchall()
        }
    assert "terminal_response" in columns
