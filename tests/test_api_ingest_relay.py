from __future__ import annotations

import base64
import hashlib
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.evidence.relay_projection import RelayEvidenceProjection
from backend.app.features.relay.router import (
    MAX_CATALOG_PAYLOAD_BYTES,
    MAX_CATALOG_PAYLOAD_DEPTH,
    MAX_INLINE_SNAPSHOT_BASE64_CHARS,
    MAX_INLINE_SNAPSHOT_BYTES,
)
from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from backend.app.main import create_app, no_lifespan
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from tests_support.compact_authority_db import prepare_compact_database


class FakeBackendIngestClient:
    def __init__(self, *, alert_ok: bool = True, heartbeat_ok: bool = True) -> None:
        self.alert_ok = alert_ok
        self.heartbeat_ok = heartbeat_ok
        self.alerts: list[dict] = []
        self.heartbeats = 0
        self.egress_camera_ids: list[str] = []

    def for_camera(self, camera_id: str) -> FakeBackendIngestClient:
        self.egress_camera_ids.append(camera_id)
        return self

    def send_alert(self, **kwargs) -> bool:
        self.alerts.append(kwargs)
        return self.alert_ok

    def send_heartbeat(self) -> bool:
        self.heartbeats += 1
        return self.heartbeat_ok

    def send_alert_receipt(self, **kwargs) -> EventReceipt:
        self.alerts.append(kwargs)
        return EventReceipt("accepted", kwargs["edge_event_id"], "event-1")


def _client(
    fake: FakeBackendIngestClient | None = None,
    *,
    catalog_path: Path | None = None,
    registry_dir: Path | None = None,
) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    registry_root = registry_dir if registry_dir is not None else Path(tempfile.mkdtemp())
    registry_path = catalog_path or registry_root / "edge.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    store.create(
        camera_id="camera-1",
        label="camera-1",
        rtsp_url="rtsp://example/camera-1",
        space_id=None,
        status="online",
        backend_camera_id="camera-1",
    )
    app.state.camera_registry = store
    app.state.backend_ingest_client = fake or FakeBackendIngestClient()
    app.state.relay_evidence_projection = RelayEvidenceProjection(registry_path)
    app.state.edge_database_path = registry_path
    return TestClient(app)


def _compact_incident(database: Path, edge_event_id: str) -> tuple[object, ...] | None:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT edge_event_id,camera_id,facility_id,event_type,probability,detected_at "
            "FROM incidents WHERE edge_event_id=?",
            (edge_event_id,),
        ).fetchone()


def _alert_payload(**overrides) -> dict:
    payload = {
        "event_type": "bed-exit",
        "probability": 0.87,
        "detected_at": "2026-06-25T12:00:00.000Z",
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "evidence": {"domain": "night-bed-exit", "clip_id": "clip-123"},
    }
    payload.update(overrides)
    return payload


def _snapshot_metadata(content: bytes, **overrides: object) -> dict[str, object]:
    snapshot = {
        "snapshot_id": "snapshot-1",
        "path": "snapshots/camera-1/event-1.jpg",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "mime_type": "image/jpeg",
        "captured_at": "2026-06-25T12:00:00.000Z",
        "camera_id": "camera-1",
        "edge_event_id": "00000000-0000-4000-8000-000000000020",
    }
    snapshot.update(overrides)
    return snapshot


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/relay/system-tests",
        "/api/v1/relay/system-tests/auth-check",
    ),
)
def test_removed_system_test_relay_routes_are_not_registered(path: str) -> None:
    response = _client().post(
        path,
        json={},
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 404


def test_system_test_variant_cannot_cross_the_ordinary_alert_contract() -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json={
            "edge_event_id": "00000000-0000-4000-8000-000000000099",
            "type": "SYSTEM_TEST",
            "source": "SYSTEM_TEST",
            "test_mode": "SYSTEM_TEST",
            "detected_at": "2026-08-12T00:00:00Z",
        },
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 422
    assert fake.alerts == []


def test_relay_alert_rejects_missing_token() -> None:
    response = _client().post("/api/v1/relay/alerts", json=_alert_payload())

    assert response.status_code == 401


def test_unenrolled_runtime_accepts_alert_locally_without_cloud_egress() -> None:
    """An edge that hasn't completed backend enrollment yet (no
    ``backend_ingest_client`` published -- see ``apply_connection_settings``)
    must still accept and locally record alerts for a camera the registry
    already knows about; cloud egress is attempted only once a backend
    client exists (see relay/router.py's ``relay_alert``, "Registry-bound
    local accept; cloud only when store built a client", #183/#202). This
    replaces a prior expectation of a 503 "backend enrollment is required"
    refusal, which described pre-store-only-mapping behavior no longer
    present in the route.
    """
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    # Camera binding is registry-only now (no camera_inventory fallback --
    # see _camera_binding_from_registry in relay/router.py); the camera must
    # resolve here so the request reaches the backend-enrollment branch this
    # test actually exercises, instead of 403ing earlier as an unknown camera.
    registry_path = Path(tempfile.mkdtemp()) / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    store.create(
        camera_id="camera-1",
        label="camera-1",
        rtsp_url="rtsp://example/camera-1",
        space_id="facility-1",
        status="online",
    )
    app.state.camera_registry = store
    # No app.state.backend_ingest_client: this IS the unenrolled condition.

    response = TestClient(app).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_relay_alert_rejects_wrong_token() -> None:
    response = _client().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(),
        headers={"X-Edge-Relay-Token": "wrong"},
    )

    assert response.status_code == 403


def test_relay_alert_rejects_unknown_camera() -> None:
    response = _client().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(camera_id="camera-unknown"),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 403
    assert "unknown camera" in response.json()["detail"]


def test_relay_alert_accepts_any_wire_facility_when_registry_has_camera() -> None:
    """Worker wire facility_id is not compared to env; registry camera_id binds."""
    response = _client().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(facility_id="facility-2"),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_relay_alert_records_catalog_even_when_camera_unresolved(tmp_path) -> None:
    """The local catalog is edge-local audit trail, not backend egress -- an
    unresolved camera (which still 403s to the worker, unchanged) must not
    leave zero local trace of the attempt (see #183, #202)."""
    payload = _alert_payload(
        camera_id="camera-unknown",
        edge_event_id="00000000-0000-4000-8000-000000000030",
    )

    with _client(catalog_path=tmp_path / "catalog.sqlite3") as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json=payload,
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

        assert response.status_code == 403
        assert _compact_incident(tmp_path / "catalog.sqlite3", str(payload["edge_event_id"])) == (
            payload["edge_event_id"],
            "camera-unknown",
            "facility-1",
            "bed-exit",
            0.87,
            "2026-06-25T12:00:00.000Z",
        )


def test_relay_alert_forwards_valid_event_to_backend_ingest_client() -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert fake.alerts == [
        {
            "event_type": "bed-exit",
            "detected_at": "2026-06-25T12:00:00.000Z",
            "probability": 0.87,
            "clip_id": "clip-123",
        }
    ]


def test_relay_alert_compact_projection_preserves_incident_identity(tmp_path) -> None:
    fake = FakeBackendIngestClient()
    evidence = {"domain": "night-bed-exit", "window": {"start": 1, "end": 2}}
    payload = _alert_payload(
        edge_event_id="00000000-0000-4000-8000-000000000010", evidence=evidence
    )

    with _client(fake, catalog_path=tmp_path / "catalog.sqlite3") as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json=payload,
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert _compact_incident(tmp_path / "catalog.sqlite3", str(payload["edge_event_id"])) == (
        payload["edge_event_id"],
        "camera-1",
        "facility-1",
        "bed-exit",
        0.87,
        "2026-06-25T12:00:00.000Z",
    )


def test_relay_alert_projects_identity_without_persisting_oversized_metadata() -> None:
    # Compact incidents retain bounded identity, not arbitrary evidence metadata.
    fake = FakeBackendIngestClient()
    client = _client(fake)
    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000011",
            evidence={"detail": "x" * MAX_CATALOG_PAYLOAD_BYTES},
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert len(fake.alerts) == 1
    assert (
        _compact_incident(
            Path(client.app.state.edge_database_path),
            "00000000-0000-4000-8000-000000000011",
        )
        is not None
    )


def test_relay_alert_projects_identity_without_persisting_overdeep_metadata() -> None:
    """Deep metadata is omitted while compact incident identity remains durable."""
    fake = FakeBackendIngestClient()
    evidence: dict[str, object] = {}
    current = evidence
    for _ in range(MAX_CATALOG_PAYLOAD_DEPTH):
        child: dict[str, object] = {}
        current["child"] = child
        current = child

    client = _client(fake)
    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000012", evidence=evidence
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert len(fake.alerts) == 1
    assert (
        _compact_incident(
            Path(client.app.state.edge_database_path),
            "00000000-0000-4000-8000-000000000012",
        )
        is not None
    )


def test_relay_alert_fails_before_egress_when_compact_projection_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeBackendIngestClient()

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(RelayEvidenceProjection, "project_event", fail_projection)
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(edge_event_id="00000000-0000-4000-8000-000000000013"),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "central evidence projection unavailable"}
    assert fake.alerts == []


def test_relay_alert_omits_missing_clip_id_for_backward_compatibility() -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(evidence={"domain": "night-bed-exit"}),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert fake.alerts == [
        {
            "event_type": "bed-exit",
            "detected_at": "2026-06-25T12:00:00.000Z",
            "probability": 0.87,
        }
    ]


def test_relay_heartbeat_forwards_valid_camera_to_backend_ingest_client() -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "camera-1", "facility_id": "facility-1"},
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    assert fake.heartbeats == 1


def test_relay_heartbeat_records_local_liveness_even_when_camera_unresolved() -> None:
    """Local liveness reflects edge-local truth and must not depend on
    camera_inventory/registry binding: a camera absent from both still 403s
    to the worker (backend egress is legitimately unresolvable), but ml-api's
    own /status view must already know this camera beat in (see #183, #202)."""
    client = _client()

    response = client.post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "camera-unknown", "facility_id": "facility-1"},
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 403
    snapshot = get_heartbeat_store(client.app).snapshot()
    assert snapshot["cameras"]["camera-unknown"]["status"] == ONLINE


def test_relay_accepts_canonical_camera_id_from_registry_when_inventory_missing(tmp_path) -> None:
    fake = FakeBackendIngestClient()
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    store.create(
        camera_id="provisional-camera",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        backend_camera_id="backend-camera-1",
    )
    app.state.camera_registry = store
    app.state.backend_ingest_client = fake

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "backend-camera-1", "facility_id": "local"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert fake.heartbeats == 1


def _registry_app(fake: FakeBackendIngestClient, tmp_path, *, backend_camera_id: str | None):
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    registry_path = tmp_path / "catalog.sqlite3"
    prepare_compact_database(registry_path)
    store = CameraRegistryStore(registry_path)
    store.create(
        camera_id="local-uuid-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        backend_camera_id=backend_camera_id,
    )
    app.state.camera_registry = store
    app.state.backend_ingest_client = fake
    return app


def test_relay_heartbeat_egresses_canonical_backend_id_for_mapped_local_camera(tmp_path) -> None:
    """The worker sends its local registry id; backend egress must use the
    explicit backend mapping — the backend only knows its own camera ids."""
    fake = FakeBackendIngestClient()
    app = _registry_app(fake, tmp_path, backend_camera_id="backend-camera-1")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert fake.heartbeats == 1
    assert fake.egress_camera_ids == ["backend-camera-1"]


def test_relay_alert_egresses_canonical_backend_id_for_mapped_local_camera(tmp_path) -> None:
    fake = FakeBackendIngestClient()
    app = _registry_app(fake, tmp_path, backend_camera_id="backend-camera-1")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json=_alert_payload(camera_id="local-uuid-1", facility_id="local-facility"),
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert len(fake.alerts) == 1
    assert fake.egress_camera_ids == ["backend-camera-1"]


def test_relay_heartbeat_never_egresses_local_id_when_camera_is_unmapped(tmp_path) -> None:
    """An unmapped camera does NOT forward its local id to the Hub.

    This test previously asserted the opposite, on the reasoning that forwarding
    the local id would draw a loud rejection from the authoritative backend and
    was preferable to a silently re-attributed identity. Issue #308 disproved the
    premise in production: the Hub rejects the unissued id with
    FACILITY_BINDING_MISMATCH, but that reaches the edge as an opaque relay 502
    and was repeatedly misdiagnosed as an authentication failure. The failure was
    not loud, it was misleading -- and every heartbeat for that camera 502'd.

    The anti-pattern the original docstring guarded against is still prevented:
    no identity is re-attributed. The id is simply not sent, and the reason is
    named in a local warning. This also makes the one-shot route consistent with
    the periodic tick in backend_heartbeat_relay, which already refuses to send
    under an unmapped id rather than emit a guaranteed reject. Local liveness,
    policy acknowledgement, and never_connected bookkeeping all still run.
    """
    fake = FakeBackendIngestClient()
    app = _registry_app(fake, tmp_path, backend_camera_id=None)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    # No Hub egress at all, and specifically never under the local id.
    assert fake.egress_camera_ids == []
    assert fake.heartbeats == 0


def test_relay_heartbeat_clears_never_connected_on_first_heartbeat(tmp_path) -> None:
    """A registry record's never_connected flips False on its first heartbeat,
    even before any successful probe -- a live worker beating in is itself
    evidence the camera has connected at least once."""
    fake = FakeBackendIngestClient()
    app = _registry_app(fake, tmp_path, backend_camera_id=None)
    store: CameraRegistryStore = app.state.camera_registry
    assert store.get("local-uuid-1")["never_connected"] is True

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert store.get("local-uuid-1")["never_connected"] is False


def test_relay_heartbeat_never_connected_flip_is_a_single_write(tmp_path) -> None:
    """Once never_connected is cleared it must stay cleared without an extra
    store.update() on every subsequent heartbeat (avoids write amplification
    on the file-backed, lock-guarded registry for a value that never reverts)."""
    fake = FakeBackendIngestClient()
    app = _registry_app(fake, tmp_path, backend_camera_id=None)
    store: CameraRegistryStore = app.state.camera_registry
    original_update = store.update
    call_count = {"count": 0}

    def counting_update(camera_id: str, updates: dict[str, object]):
        call_count["count"] += 1
        return original_update(camera_id, updates)

    store.update = counting_update  # type: ignore[method-assign]

    with TestClient(app) as client:
        for _ in range(3):
            response = client.post(
                "/api/v1/relay/heartbeat",
                json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
                headers={"X-Edge-Relay-Token": "relay-token"},
            )
            assert response.status_code == 202

    assert call_count["count"] == 1
    assert store.get("local-uuid-1")["never_connected"] is False


def test_relay_alert_rejects_raw_frame_payloads() -> None:
    payload = _alert_payload(frame=[0, 1, 2])

    response = _client().post(
        "/api/v1/relay/alerts",
        json=payload,
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 422


def test_relay_alert_forwards_audit_and_snapshot_when_present() -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            audit={
                "config_version": 7,
                "model_version": "rf-2026",
                "clock_source": "edge_wall_clock",
            },
            snapshot_jpeg_base64=base64.b64encode(b"jpeg-bytes").decode("ascii"),
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert len(fake.alerts) == 1
    forwarded = fake.alerts[0]
    assert forwarded["event_type"] == "bed-exit"
    assert forwarded["audit"] == {
        "config_version": 7,
        "model_version": "rf-2026",
        "clock_source": "edge_wall_clock",
    }
    assert forwarded["snapshot_bytes"] == b"jpeg-bytes"


def test_relay_alert_accepts_inline_snapshot_at_decoded_size_limit(tmp_path) -> None:
    content = b"x" * MAX_INLINE_SNAPSHOT_BYTES
    fake = FakeBackendIngestClient()
    response = _client(fake, catalog_path=tmp_path / "catalog.sqlite3").post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000020",
            snapshot_jpeg_base64=base64.b64encode(content).decode("ascii"),
            snapshot=_snapshot_metadata(content),
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert fake.alerts[0]["snapshot_bytes"] == content


@pytest.mark.parametrize(
    "snapshot_jpeg_base64",
    [
        base64.b64encode(b"x" * (MAX_INLINE_SNAPSHOT_BYTES + 1)).decode("ascii"),
        "A" * (MAX_INLINE_SNAPSHOT_BASE64_CHARS + 1),
        "not-valid-base64!",
    ],
    ids=["decoded-too-large", "encoded-too-large", "malformed"],
)
def test_relay_alert_rejects_invalid_inline_snapshot(
    snapshot_jpeg_base64: str,
) -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(snapshot_jpeg_base64=snapshot_jpeg_base64),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 422
    assert fake.alerts == []


@pytest.mark.parametrize(
    "snapshot_override",
    [
        {"mime_type": "image/png"},
        {"size_bytes": 1},
        {"sha256": "0" * 64},
        {"camera_id": "camera-2"},
        {"edge_event_id": "00000000-0000-4000-8000-000000000021"},
    ],
    ids=["mime", "size", "sha256", "camera", "event"],
)
def test_relay_alert_rejects_inline_snapshot_metadata_mismatch(
    snapshot_override: dict[str, object],
) -> None:
    content = b"jpeg-bytes"
    fake = FakeBackendIngestClient()
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000020",
            snapshot_jpeg_base64=base64.b64encode(content).decode("ascii"),
            snapshot=_snapshot_metadata(content, **snapshot_override),
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 422
    assert fake.alerts == []


def test_relay_alert_keeps_metadata_only_snapshot_compatible(tmp_path) -> None:
    fake = FakeBackendIngestClient()
    response = _client(fake, catalog_path=tmp_path / "catalog.sqlite3").post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000020",
            snapshot=_snapshot_metadata(
                b"",
                mime_type="application/octet-stream",
                sha256="legacy",
                size_bytes=0,
            ),
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert "snapshot_bytes" not in fake.alerts[0]
    assert (
        _compact_incident(tmp_path / "catalog.sqlite3", "00000000-0000-4000-8000-000000000020")
        is not None
    )
    with sqlite3.connect(tmp_path / "catalog.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


class ReceiptBackendIngestClient(FakeBackendIngestClient):
    def __init__(
        self,
        *,
        accepted_at: float | None = None,
        failure: DeliveryFailure | None = None,
    ) -> None:
        super().__init__()
        self.accepted_at = accepted_at
        self.failure = failure

    def send_alert_receipt(self, **kwargs) -> EventReceipt | DeliveryFailure:
        if self.failure is not None:
            return self.failure
        callback = kwargs["on_accepted"]
        assert callable(callback)
        assert self.accepted_at is not None
        callback(self.accepted_at)
        return EventReceipt("accepted", kwargs["edge_event_id"], "event-1")


def _receipt_client(fake: ReceiptBackendIngestClient, tmp_path) -> TestClient:
    client = _client(fake)
    client.app.state.runtime_status_store = RuntimeStatusStore(
        latency_state_path=tmp_path / "catalog.sqlite3"
    )
    return client


def test_relay_latency_uses_remote_acceptance_time_for_first_attempt(tmp_path) -> None:
    fake = ReceiptBackendIngestClient(accepted_at=105.0)
    client = _receipt_client(fake, tmp_path)

    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000001",
            detected_at="1970-01-01T00:01:40Z",
            attempt_ordinal=1,
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert client.app.state.runtime_status_store._latency_for_facility("facility-1") == {
        "first_attempt_samples": 1,
        "max_sec": 5.0,
        "since_sec": 105.0,
    }


def test_relay_latency_excludes_failed_and_retried_delivery(tmp_path) -> None:
    failure = DeliveryFailure(DeliveryDisposition.RETRY, "TIMEOUT")
    failed_client = _receipt_client(ReceiptBackendIngestClient(failure=failure), tmp_path)
    retry_client = _receipt_client(ReceiptBackendIngestClient(accepted_at=105.0), tmp_path)

    failed = failed_client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000001",
            attempt_ordinal=1,
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )
    retried = retry_client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000002",
            attempt_ordinal=2,
        ),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert failed.status_code == 503
    assert retry_client.app.state.runtime_status_store._latency_for_facility("facility-1") is None
    assert retried.status_code == 202
    assert retry_client.app.state.runtime_status_store._latency_for_facility("facility-1") is None
