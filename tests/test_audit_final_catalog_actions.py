from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import (
    AuditMutation,
    AuditRuntimeUnavailable,
    PostgresAuditRuntime,
)
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import TopologyAccepted, TopologyPutResult
from backend.app.features.cameras.topology_confirmation_state import (
    TopologyConfirmationPreview,
    TopologyConfirmationStore,
)
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from backend.app.features.evidence.event_outbox import EventOutbox, OutboxBudget
from backend.app.features.evidence.postgres_relay_projection import PostgresRelayEvidenceProjection
from backend.app.features.evidence.relay_projection import RelayEvent
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyConfirmation,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)
_SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81a"
_CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
_STAMP = "2026-08-24T00:00:00.000Z"


def _audit(runtime: PostgresAuditRuntime, action: AuditAction, target_id: str) -> AuditMutation:
    return AuditMutation(
        runtime,
        lambda: AuditEvent(
            occurred_at=_STAMP,
            actor_id="admin",
            action=action,
            target_id=target_id,
            detail=empty_detail(action),
        ),
    )


def _reject_audit_inserts(sandbox: ProductSandbox) -> None:
    sandbox.admin.execute(
        "CREATE OR REPLACE FUNCTION reject_audit_test() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
    )
    sandbox.admin.execute(
        "CREATE TRIGGER reject_audit_test BEFORE INSERT ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_audit_test()"
    )


def _restore_audit_inserts(sandbox: ProductSandbox, runtime: PostgresAuditRuntime) -> None:
    # The extra trigger fails the exact trigger contract, so re-verify only after dropping it.
    sandbox.admin.execute("DROP TRIGGER reject_audit_test ON audit_events")
    assert runtime.verify_once()


def _action_count(sandbox: ProductSandbox, action: AuditAction) -> int:
    return sandbox.admin.execute(
        "SELECT COUNT(*) FROM audit_events WHERE action=%s", (action.value,)
    ).fetchone()[0]


def _edge_site(sandbox: ProductSandbox) -> tuple[object, ...]:
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def _enroll(sandbox: ProductSandbox) -> None:
    ConnectionSettingsStore(sandbox.database, sandbox.authority).save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "token-1",
            "edge_installation_id": _PRINCIPAL.edge_installation_id,
            "enrollment_generation": _PRINCIPAL.enrollment_generation,
        }
    )


def _sync_fixture(
    sandbox: ProductSandbox,
) -> tuple[TopologyRetryCoordinator, EdgeTopologySyncStateStore]:
    _enroll(sandbox)
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    registry.create_room(edge_ref="room-1", floor_edge_ref="floor-1", name="101")
    registry.create(
        camera_id="local-1",
        label="A",
        rtsp_url="rtsp://private",
        space_id=None,
        status="online",
        edge_ref="camera-1",
        room_edge_ref="room-1",
    )
    unchanged = MutationCounts(0, 0, 1)

    class Client:
        principal = _PRINCIPAL

        def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult:
            return TopologyAccepted(
                TopologySuccessEnvelope(
                    pending.snapshot_id,
                    pending.client_revision,
                    1,
                    TopologyMutationResult(unchanged, unchanged, unchanged),
                    None,
                )
            )

        def refresh_server_revision(self) -> int | None:
            return None

        def confirm(
            self, _snapshot_id: str, _confirmation: TopologyConfirmation
        ) -> TopologyPutResult:
            raise AssertionError("not used")

    state = EdgeTopologySyncStateStore(sandbox.database, sandbox.authority)
    client = Client()
    return TopologyRetryCoordinator(registry, state, lambda: client), state


def test_connection_sync_route_commits_canonical_action_and_detail(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: an enrolled roster whose sync coordinator shares the product database.
    sandbox = postgres_product_sandbox
    coordinator, _state = _sync_fixture(sandbox)
    app = postgres_api_app(sandbox, postgres_audit_runtime)
    app.state.topology_retry_coordinator = coordinator
    with TestClient(app) as client:
        login = client.post("/api/v1/auth/session", json={"username": "admin", "password": "admin"})
        assert login.status_code == 204

        # When: the dashboard requests an explicit roster sync.
        response = client.post("/api/v1/connection/sync-cameras")

    # Then: the route commits exactly the canonical sync action and detail.
    assert response.status_code == 200
    rows = sandbox.admin.execute(
        "SELECT action,target_id,actor_type,auth_mechanism,detail_json FROM audit_events "
        "WHERE action NOT LIKE 'audit.%' AND action NOT LIKE 'auth.%' ORDER BY audit_id"
    ).fetchall()
    assert rows == [
        ("connection.sync", "camera-roster", "user", "dashboard_session", '{"version":1}')
    ]


def test_connection_sync_audit_is_one_atomic_operation(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: an enrolled roster and a real PostgreSQL rejection at audit INSERT.
    sandbox = postgres_product_sandbox
    runtime = postgres_audit_runtime
    coordinator, state = _sync_fixture(sandbox)
    before = _edge_site(sandbox)
    _reject_audit_inserts(sandbox)

    # When: the sync commits its acceptance and audit in one transaction.
    with pytest.raises(AuditRuntimeUnavailable):
        coordinator.trigger(
            force=True,
            now_epoch=1.0,
            audit=_audit(runtime, AuditAction.CONNECTION_SYNC, "camera-roster"),
        )

    # Then: the failed audit leaves the sync state untouched.
    assert _edge_site(sandbox) == before
    assert _action_count(sandbox, AuditAction.CONNECTION_SYNC) == 0

    # When: audit inserts are accepted again and the sync repeats.
    _restore_audit_inserts(sandbox, runtime)
    result = coordinator.trigger(
        force=True,
        now_epoch=1.0,
        audit=_audit(runtime, AuditAction.CONNECTION_SYNC, "camera-roster"),
    )

    # Then: acceptance and its one audit row commit together.
    assert result.status == "synced"
    assert _action_count(sandbox, AuditAction.CONNECTION_SYNC) == 1
    assert state.load().last_client_revision == 1


def _confirmation_fixture(
    sandbox: ProductSandbox,
) -> tuple[TopologyConfirmationStore, TopologyConfirmationPreview, TopologySuccessEnvelope]:
    _enroll(sandbox)
    # The terminal CAS revalidates these accepted-snapshot versions against the preview.
    sandbox.admin.execute(
        "UPDATE edge_site SET registry_version=12,topology_client_revision=1,"
        "topology_server_revision=7 WHERE id=1"
    )
    counts = MutationCounts(0, 0, 1)
    result = TopologyMutationResult(counts, counts, counts)
    preview_response = TopologySuccessEnvelope(
        _SNAPSHOT_ID,
        1,
        7,
        result,
        OmissionPreview(_CONFIRMATION_ID, "a" * 64, "2099-01-01T00:00:00.000Z", (), (), ()),
    )
    terminal = TopologySuccessEnvelope(_SNAPSHOT_ID, 1, 8, result, None)
    store = TopologyConfirmationStore(sandbox.database, sandbox.authority)
    store.save(preview_response, _PRINCIPAL, registry_version=12)
    preview = store.load()
    assert preview is not None
    return store, preview, terminal


def test_topology_confirmation_audit_rolls_back_terminal_state(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: a saved omission preview and a real PostgreSQL rejection at audit INSERT.
    sandbox = postgres_product_sandbox
    runtime = postgres_audit_runtime
    store, preview, terminal = _confirmation_fixture(sandbox)
    _reject_audit_inserts(sandbox)

    # When: the terminal confirmation and its audit share one transaction.
    with pytest.raises(AuditRuntimeUnavailable):
        _audit(runtime, AuditAction.TOPOLOGY_CONFIRM, _SNAPSHOT_ID).apply(
            store, lambda append: store.complete(preview, terminal, after_write=append)
        )

    # Then: the preview stays unconfirmed and no audit row escaped.
    loaded = store.load()
    assert loaded is not None and loaded.confirmed is False
    assert _action_count(sandbox, AuditAction.TOPOLOGY_CONFIRM) == 0

    # When: audit inserts are accepted again and the confirmation repeats.
    _restore_audit_inserts(sandbox, runtime)
    _audit(runtime, AuditAction.TOPOLOGY_CONFIRM, _SNAPSHOT_ID).apply(
        store, lambda append: store.complete(preview, terminal, after_write=append)
    )

    # Then: the terminal state and its one audit row commit together.
    loaded = store.load()
    assert loaded is not None and loaded.confirmed is True
    assert _action_count(sandbox, AuditAction.TOPOLOGY_CONFIRM) == 1


def _event(edge_event_id: str) -> RelayEvent:
    return RelayEvent(
        edge_event_id,
        "fall",
        0.9,
        "2026-08-24T00:00:00Z",
        "camera-1",
        "facility-1",
        None,
        None,
        None,
    )


def _snapshot_artifacts(sandbox: ProductSandbox) -> list[tuple[object, ...]]:
    return sandbox.admin.execute(
        "SELECT incidents.edge_event_id,artifacts.artifact_id,artifacts.state FROM artifacts "
        "JOIN incidents USING (incident_id) WHERE artifacts.kind='SNAPSHOT' "
        "ORDER BY incidents.edge_event_id"
    ).fetchall()


def test_snapshot_actions_share_projection_transactions(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: three accepted events and the native late-snapshot owner.
    sandbox = postgres_product_sandbox
    runtime = postgres_audit_runtime
    outbox = EventOutbox(
        sandbox.database, sandbox.authority, OutboxBudget(10, 1_048_576), audit_runtime=runtime
    )
    for edge_event_id in ("event-attach", "event-disposition", "event-fault"):
        outbox.accept(_event(edge_event_id), backend_camera_id=None, forward=False)
    projection = PostgresRelayEvidenceProjection(sandbox.database, sandbox.authority)

    # When: an attachment and a disposition each carry their audit into the projection.
    _audit(runtime, AuditAction.RELAY_SNAPSHOT_ATTACHMENT, "snapshot-1").apply(
        projection,
        lambda append: projection.attach_snapshot(
            edge_event_id="event-attach",
            snapshot_id="snapshot-1",
            sha256="a" * 64,
            media_reference="clips/snapshot.jpg",
            size_bytes=10,
            mime_type="image/jpeg",
            after_write=append,
        ),
    )
    _audit(runtime, AuditAction.RELAY_SNAPSHOT_DISPOSITION, "snapshot-2").apply(
        projection,
        lambda append: projection.record_snapshot_disposition(
            edge_event_id="event-disposition",
            snapshot_id="snapshot-2",
            disposition="unavailable",
            reason="capture_failed",
            after_write=append,
        ),
    )

    # Then: each fact commits with exactly one audit row.
    committed = [
        ("event-attach", "snapshot-1", "AVAILABLE"),
        ("event-disposition", None, "UNAVAILABLE"),
    ]
    assert _snapshot_artifacts(sandbox) == committed
    assert _action_count(sandbox, AuditAction.RELAY_SNAPSHOT_ATTACHMENT) == 1
    assert _action_count(sandbox, AuditAction.RELAY_SNAPSHOT_DISPOSITION) == 1

    # When: PostgreSQL rejects the audit INSERT of a further attachment.
    _reject_audit_inserts(sandbox)
    with pytest.raises(AuditRuntimeUnavailable):
        _audit(runtime, AuditAction.RELAY_SNAPSHOT_ATTACHMENT, "snapshot-3").apply(
            projection,
            lambda append: projection.attach_snapshot(
                edge_event_id="event-fault",
                snapshot_id="snapshot-3",
                sha256="b" * 64,
                media_reference="clips/fault.jpg",
                size_bytes=10,
                mime_type="image/jpeg",
                after_write=append,
            ),
        )

    # Then: the attachment rolled back with its audit row.
    assert _snapshot_artifacts(sandbox) == committed
    assert _action_count(sandbox, AuditAction.RELAY_SNAPSHOT_ATTACHMENT) == 1
