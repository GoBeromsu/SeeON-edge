from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.migrator import migrate_database
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.store import AuditEvent, AuditStore, utc_now
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import TopologyAccepted, TopologyPutResult
from backend.app.features.cameras.topology_confirmation_state import TopologyConfirmationStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from backend.app.features.evidence.compact_receipts import CompactArtifactReceiptStore
from backend.app.features.evidence.receipt_store import ArtifactReceipt, verified_artifact
from backend.app.features.evidence.relay_projection import RelayEvent, RelayEvidenceProjection
from backend.app.main import create_app, no_lifespan
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyConfirmation,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)
from tests_support.compact_authority_db import seed_enrollment

_PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)


def _hook(path: Path, action: AuditAction, *, deny: bool = False):
    store = AuditStore(path)
    event = AuditEvent(utc_now(), "test", action, action.value, empty_detail(action))

    def append(connection: sqlite3.Connection) -> None:
        if deny:
            connection.set_authorizer(
                lambda code, table, *_: sqlite3.SQLITE_DENY
                if code == sqlite3.SQLITE_INSERT and table == "audit_events"
                else sqlite3.SQLITE_OK
            )
        store.append(event, connection=connection)

    return append


def _count(path: Path, action: AuditAction) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE action=?", (action.value,)
        ).fetchone()[0]


def _sync_fixture(path: Path) -> tuple[TopologyRetryCoordinator, EdgeTopologySyncStateStore]:
    migrate_database(path)
    seed_enrollment(
        path,
        edge_installation_id=_PRINCIPAL.edge_installation_id,
        enrollment_generation=_PRINCIPAL.enrollment_generation,
    )
    registry = CameraRegistryStore(path)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    registry.create_room(edge_ref="room-1", floor_edge_ref="floor-1", name="101")
    registry.create(
        camera_id="local-1", label="A", rtsp_url="rtsp://private", space_id=None,
        status="online", edge_ref="camera-1", room_edge_ref="room-1",
    )
    unchanged = MutationCounts(0, 0, 1)

    class Client:
        principal = _PRINCIPAL

        def put(self, pending: PendingTopologySnapshot) -> TopologyPutResult:
            return TopologyAccepted(
                TopologySuccessEnvelope(
                    pending.snapshot_id, pending.client_revision, 1,
                    TopologyMutationResult(unchanged, unchanged, unchanged), None,
                )
            )

        def refresh_server_revision(self) -> int | None:
            return None

        def confirm(
            self, _snapshot_id: str, _confirmation: TopologyConfirmation
        ) -> TopologyPutResult:
            raise AssertionError("not used")

    state = EdgeTopologySyncStateStore(path)
    client = Client()
    return TopologyRetryCoordinator(registry, state, lambda: client), state


def test_connection_sync_route_commits_canonical_action_and_detail(tmp_path: Path) -> None:
    path = tmp_path / "sync-route" / "edge.sqlite3"
    coordinator, _state = _sync_fixture(path)
    app = create_app(lifespan=no_lifespan)
    app.state.topology_retry_coordinator = coordinator
    client = TestClient(app)
    assert client.post(
        "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
    ).status_code == 204

    response = client.post("/api/v1/connection/sync-cameras")

    assert response.status_code == 200
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT action,target_id,actor_type,auth_mechanism,detail_json "
            "FROM audit_events WHERE action NOT LIKE 'audit.%'"
        ).fetchall()
    assert rows == [
        ("connection.sync", "camera-roster", "user", "dashboard_session", '{"version":1}')
    ]


def test_connection_sync_audit_is_one_atomic_operation(tmp_path: Path) -> None:
    healthy_path = tmp_path / "sync-healthy" / "edge.sqlite3"
    healthy, healthy_state = _sync_fixture(healthy_path)
    result = healthy.trigger(
        force=True, now_epoch=1.0,
        after_write=_hook(healthy_path, AuditAction.CONNECTION_SYNC),
    )
    assert result.status == "synced"
    assert _count(healthy_path, AuditAction.CONNECTION_SYNC) == 1
    assert healthy_state.load().last_client_revision == 1

    fault_path = tmp_path / "sync-fault" / "edge.sqlite3"
    fault, fault_state = _sync_fixture(fault_path)
    before = fault_state.load()
    with pytest.raises(sqlite3.DatabaseError):
        fault.trigger(
            force=True, now_epoch=1.0,
            after_write=_hook(fault_path, AuditAction.CONNECTION_SYNC, deny=True),
        )
    assert fault_state.load() == before
    assert _count(fault_path, AuditAction.CONNECTION_SYNC) == 0


def _confirmation_fixture(path: Path):
    migrate_database(path)
    seed_enrollment(
        path,
        edge_installation_id=_PRINCIPAL.edge_installation_id,
        enrollment_generation=_PRINCIPAL.enrollment_generation,
    )
    counts = MutationCounts(0, 0, 1)
    result = TopologyMutationResult(counts, counts, counts)
    preview_response = TopologySuccessEnvelope(
        "snapshot-1", 1, 7, result,
        OmissionPreview("confirmation-1", "a" * 64, "2099-01-01T00:00:00Z", (), (), ()),
    )
    terminal = TopologySuccessEnvelope("snapshot-1", 1, 8, result, None)
    store = TopologyConfirmationStore(path)
    store.save(preview_response, _PRINCIPAL, 0)
    preview = store.load()
    assert preview is not None
    return store, preview, terminal


def test_topology_confirmation_audit_rolls_back_terminal_state(tmp_path: Path) -> None:
    healthy_path = tmp_path / "confirm-healthy" / "edge.sqlite3"
    store, preview, terminal = _confirmation_fixture(healthy_path)
    store.complete(
        preview, terminal,
        after_write=_hook(healthy_path, AuditAction.TOPOLOGY_CONFIRM),
    )
    loaded = store.load()
    assert loaded is not None and loaded.confirmed is True
    assert _count(healthy_path, AuditAction.TOPOLOGY_CONFIRM) == 1

    fault_path = tmp_path / "confirm-fault" / "edge.sqlite3"
    fault_store, fault_preview, fault_terminal = _confirmation_fixture(fault_path)
    with pytest.raises(sqlite3.DatabaseError):
        fault_store.complete(
            fault_preview, fault_terminal,
            after_write=_hook(fault_path, AuditAction.TOPOLOGY_CONFIRM, deny=True),
        )
    loaded = fault_store.load()
    assert loaded is not None and loaded.confirmed is False
    assert _count(fault_path, AuditAction.TOPOLOGY_CONFIRM) == 0


def _event(edge_event_id: str) -> RelayEvent:
    return RelayEvent(
        edge_event_id, "fall", 0.9, "2026-08-24T00:00:00Z",
        "camera-1", "facility-1", None, None, None,
    )


def test_snapshot_actions_share_projection_transactions(tmp_path: Path) -> None:
    path = tmp_path / "relay.sqlite3"
    migrate_database(path)
    projection = RelayEvidenceProjection(path)
    projection.project_event(_event("event-attach"))
    projection.attach_snapshot(
        edge_event_id="event-attach", snapshot_id="snapshot-1", sha256="a" * 64,
        media_reference="clips/snapshot.jpg", size_bytes=10, mime_type="image/jpeg",
        after_write=_hook(path, AuditAction.RELAY_SNAPSHOT_ATTACHMENT),
    )
    projection.project_event(_event("event-disposition"))
    projection.record_snapshot_disposition(
        edge_event_id="event-disposition", snapshot_id="snapshot-2",
        disposition="unavailable", reason="capture_failed",
        after_write=_hook(path, AuditAction.RELAY_SNAPSHOT_DISPOSITION),
    )
    assert _count(path, AuditAction.RELAY_SNAPSHOT_ATTACHMENT) == 1
    assert _count(path, AuditAction.RELAY_SNAPSHOT_DISPOSITION) == 1

    projection.project_event(_event("event-fault"))
    with pytest.raises(sqlite3.DatabaseError):
        projection.attach_snapshot(
            edge_event_id="event-fault", snapshot_id="snapshot-3", sha256="b" * 64,
            media_reference="clips/fault.jpg", size_bytes=10, mime_type="image/jpeg",
            after_write=_hook(path, AuditAction.RELAY_SNAPSHOT_ATTACHMENT, deny=True),
        )
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE artifact_id='snapshot-3'"
        ).fetchone()[0]
    assert stored == 0


def _media(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "clip-store" / "clips" / "clip-1" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "clip_id": "clip-1", "camera_id": "camera-1", "event_ref": "event-1",
                "event_type": "fall", "started_at": "2026-08-24T00:00:00Z",
                "duration_s": 1.0, "codec": "h264", "path": "clips/clip-1",
                "video_available": True, "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_evidence_receipt_audit_rolls_back_compact_facts(tmp_path: Path) -> None:
    path = tmp_path / "receipt.sqlite3"
    migrate_database(path)
    data = b"verified video"
    media_path = _media(tmp_path, data)
    receipt = ArtifactReceipt("clip-1", hashlib.sha256(data).hexdigest(), len(data))
    store = CompactArtifactReceiptStore(path, tmp_path / "clip-store")
    with media_path.open("rb") as handle:
        store.commit_verified(
            receipt, verified_artifact(handle),
            after_write=_hook(path, AuditAction.EVIDENCE_RECEIPT),
        )
    assert store.get("clip-1") == receipt
    assert _count(path, AuditAction.EVIDENCE_RECEIPT) == 1

    fault_path = tmp_path / "receipt-fault.sqlite3"
    migrate_database(fault_path)
    fault_store = CompactArtifactReceiptStore(fault_path, tmp_path / "clip-store")
    with media_path.open("rb") as handle, pytest.raises(sqlite3.DatabaseError):
        fault_store.commit_verified(
            receipt, verified_artifact(handle),
            after_write=_hook(fault_path, AuditAction.EVIDENCE_RECEIPT, deny=True),
        )
    assert fault_store.get("clip-1") is None
    assert _count(fault_path, AuditAction.EVIDENCE_RECEIPT) == 0
