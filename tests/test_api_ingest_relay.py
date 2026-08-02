from __future__ import annotations

import base64
import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.clips.catalog import CatalogStore
from backend.app.features.relay.router import (
    MAX_CATALOG_PAYLOAD_BYTES,
    MAX_CATALOG_PAYLOAD_DEPTH,
    MAX_INLINE_SNAPSHOT_BASE64_CHARS,
    MAX_INLINE_SNAPSHOT_BYTES,
)
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from backend.app.main import create_app, no_lifespan
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)


@pytest.fixture(autouse=True)
def default_catalog_path_to_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Redirect the catalog resolver so no test in this file can reach the
    real ``~/.local/state/ml-api/catalog.sqlite3``, even if a call site
    forgets to pass ``catalog_path`` into ``_client()``.

    Mirrors ``tests/conftest.py``'s
    ``default_dashboard_credentials_store_to_tmp_path`` fixture: relay POSTs
    that never explicitly inject a catalog path fall through to
    ``get_catalog_store()`` -> ``_catalog_path()``, which otherwise resolves
    the real home-directory state dir (no env override exists for this path
    anymore). That ambient-filesystem write shouldn't happen from the suite
    at all, so ``_catalog_path()`` itself is patched to build the path from
    a per-test tmp path instead -- a belt-and-suspenders guard on top of the
    explicit ``catalog_path=`` call sites below.
    """

    monkeypatch.setattr(
        "backend.app.features.clips.catalog._catalog_path",
        lambda: tmp_path / "catalog.sqlite3",
    )
    yield


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
    fake: FakeBackendIngestClient | None = None, *, catalog_path: Path | None = None
) -> TestClient:
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    app.state.camera_inventory = {
        "camera-1": {
            "camera_id": "camera-1",
            "facility_id": "facility-1",
            "resident_id": "resident-1",
        }
    }
    app.state.backend_ingest_client = fake or FakeBackendIngestClient()
    if catalog_path is not None:
        # Pre-set app.state.catalog_store so get_catalog_store() (called
        # lazily on first relay use) short-circuits on the isinstance check
        # and never resolves a path itself -- constructor injection instead
        # of env-var injection now that no state-path env override exists.
        app.state.catalog_store = CatalogStore.open(catalog_path)
    return TestClient(app)


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




def test_relay_alert_rejects_missing_token() -> None:
    response = _client().post("/api/v1/relay/alerts", json=_alert_payload())

    assert response.status_code == 401


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


def test_relay_alert_rejects_facility_mismatch() -> None:
    response = _client().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(facility_id="facility-2"),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 403
    assert "facility" in response.json()["detail"]


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


def test_relay_alert_catalog_preserves_normal_evidence_losslessly(tmp_path) -> None:
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
    assert client.app.state.catalog_store.records("events") == [{**payload}]


def test_relay_alert_skips_oversized_catalog_record_but_delivers_alert() -> None:
    # No catalog is ever opened on this path: the size guard rejects the
    # payload before the router reaches ``get_catalog_store`` (see
    # ``backend/app/features/relay/router.py``), so ``catalog_store`` must
    # stay unset -- matching the assertion below.
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
    assert response.json()["catalog"] == "not_recorded"
    assert "maximum size" in response.json()["catalog_reason"]
    assert len(fake.alerts) == 1
    assert not hasattr(client.app.state, "catalog_store")


def test_relay_alert_skips_overdeep_catalog_record_but_delivers_alert() -> None:
    """Catalog limits never block safety-alert egress because it is only an index."""
    # No catalog is ever opened on this path either -- see the sibling
    # oversized-record test above.
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
    assert response.json()["catalog"] == "not_recorded"
    assert "maximum depth" in response.json()["catalog_reason"]
    assert len(fake.alerts) == 1
    assert not hasattr(client.app.state, "catalog_store")
class FailingCatalogStore:
    def record_many(self, records: tuple[tuple[str, str, dict], ...]) -> None:
        raise sqlite3.OperationalError("database is locked")


def test_relay_alert_delivers_before_sqlite_catalog_failure(monkeypatch) -> None:
    fake = FakeBackendIngestClient()
    monkeypatch.setattr(
        "backend.app.features.relay.router.get_catalog_store",
        lambda app: FailingCatalogStore(),
    )
    response = _client(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(edge_event_id="00000000-0000-4000-8000-000000000013"),
        headers={"X-Edge-Relay-Token": "relay-token"},
    )

    assert response.status_code == 202
    assert response.json()["catalog"] == "not_recorded"
    assert "operational failure" in response.json()["catalog_reason"]
    assert len(fake.alerts) == 1




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


def test_relay_accepts_canonical_camera_id_from_registry_when_inventory_missing(tmp_path) -> None:
    fake = FakeBackendIngestClient()
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="provisional-camera",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        backend_camera_id="backend-camera-1",
    )
    app.state.camera_registry = store
    app.state.camera_inventory = {}
    app.state.backend_ingest_client = fake

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "backend-camera-1", "facility_id": "local-facility"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert fake.heartbeats == 1


def _registry_app(fake: FakeBackendIngestClient, tmp_path, *, backend_camera_id: str | None):
    app = create_app(lifespan=no_lifespan)
    app.state.edge_relay_token = "relay-token"
    store = CameraRegistryStore(tmp_path / "cameras.json")
    store.create(
        camera_id="local-uuid-1",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        status="online",
        backend_camera_id=backend_camera_id,
    )
    app.state.camera_registry = store
    app.state.camera_inventory = {}
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


def test_relay_heartbeat_egresses_local_id_when_camera_is_unmapped(tmp_path) -> None:
    """Unmapped cameras forward their local id; the authoritative backend may
    reject it (loud failure), never a silently re-attributed identity."""
    fake = FakeBackendIngestClient()
    app = _registry_app(fake, tmp_path, backend_camera_id=None)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
            headers={"X-Edge-Relay-Token": "relay-token"},
        )

    assert response.status_code == 202
    assert fake.egress_camera_ids == ["local-uuid-1"]


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
        latency_state_path=tmp_path / "runtime-latency.json"
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
