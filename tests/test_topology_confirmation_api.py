from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import TopologyAccepted, TopologyPutResult
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from backend.app.main import create_app, no_lifespan
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
TOKEN = "server-side-secret-token"


class _Client:
    principal = PRINCIPAL
    bearer_token = TOKEN

    def __init__(
        self, put_outcome: Callable[[PendingTopologySnapshot], TopologyPutResult]
    ) -> None:
        self.put_outcome = put_outcome
        self.confirmation: tuple[str, TopologyConfirmation] | None = None
        self.terminal: TopologyPutResult | None = None

    def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult:
        return self.put_outcome(pending)

    def confirm(
        self, snapshot_id: str, confirmation: TopologyConfirmation
    ) -> TopologyPutResult:
        self.confirmation = (snapshot_id, confirmation)
        assert self.terminal is not None
        return self.terminal

    def refresh_server_revision(self) -> int | None:
        return None


def _app_client(path: Path) -> tuple[TestClient, _Client]:
    registry = CameraRegistryStore(path)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    registry.create_room(edge_ref="room-101", floor_edge_ref="floor-1", name="101")
    registry.create(
        camera_id="local-1",
        label="Lobby",
        rtsp_url="rtsp://private",
        space_id=None,
        status="online",
        edge_ref="camera-1",
        room_edge_ref="room-101",
    )
    unchanged = MutationCounts(0, 0, 1)

    def accepted(pending: PendingTopologySnapshot) -> TopologyPutResult:
        client.terminal = TopologyAccepted(
            TopologySuccessEnvelope(
                pending.snapshot_id,
                pending.client_revision,
                8,
                TopologyMutationResult(unchanged, unchanged, unchanged),
                None,
            )
        )
        return TopologyAccepted(
            TopologySuccessEnvelope(
                pending.snapshot_id,
                pending.client_revision,
                7,
                TopologyMutationResult(unchanged, unchanged, unchanged),
                OmissionPreview(
                    CONFIRMATION_ID,
                    DIGEST,
                    "2099-01-01T00:00:00.000Z",
                    ("camera-old",),
                    (),
                    (),
                ),
            )
        )

    client = _Client(accepted)
    coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(path), lambda: client
    )
    coordinator.trigger(force=True, now_epoch=100.0)
    app = create_app(lifespan=no_lifespan)
    app.state.camera_registry = registry
    app.state.topology_retry_coordinator = coordinator
    return TestClient(app), client


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 204


def test_local_preview_and_confirmation_require_dashboard_auth(tmp_path: Path) -> None:
    # Given
    client, upstream = _app_client(tmp_path / "catalog.sqlite3")

    # When
    preview = client.get("/api/v1/connection/topology-preview")
    confirmation = client.post(
        "/api/v1/connection/topology-preview/confirm",
        json={
            "confirmation_id": CONFIRMATION_ID,
            "digest": DIGEST,
            "client_revision": 1,
            "server_revision": 7,
        },
    )

    # Then
    assert preview.status_code == 401
    assert confirmation.status_code == 401
    assert upstream.confirmation is None


def test_authenticated_local_routes_confirm_with_server_held_token_hidden(
    tmp_path: Path,
) -> None:
    # Given
    client, upstream = _app_client(tmp_path / "catalog.sqlite3")
    _login(client)

    # When
    preview = client.get("/api/v1/connection/topology-preview")
    rejected_injection = client.post(
        "/api/v1/connection/topology-preview/confirm",
        json={
            "confirmation_id": CONFIRMATION_ID,
            "digest": DIGEST,
            "client_revision": 1,
            "server_revision": 7,
            "facility_token": "browser-injected-token",
        },
    )
    confirmed = client.post(
        "/api/v1/connection/topology-preview/confirm",
        json={
            "confirmation_id": CONFIRMATION_ID,
            "digest": DIGEST,
            "client_revision": 1,
            "server_revision": 7,
        },
    )

    # Then
    assert preview.status_code == 200
    assert preview.json()["preview"]["cameras"] == 1
    assert TOKEN not in preview.text
    assert rejected_injection.status_code == 422
    assert upstream.confirmation is not None
    assert confirmed.status_code == 200
    assert confirmed.json()["server_revision"] == 8
    assert TOKEN not in confirmed.text
    assert "browser-injected-token" not in confirmed.text
