from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.relay.router import (
    MAX_INLINE_SNAPSHOT_BASE64_CHARS,
    MAX_INLINE_SNAPSHOT_BYTES,
)
from backend.app.features.status.heartbeat_store import ONLINE, get_heartbeat_store
from backend.app.features.status.runtime_status_store import RuntimeStatusStore
from shared.events.evidence_export_contract import (
    DeliveryDisposition,
    DeliveryFailure,
    EventReceipt,
)
from shared.events.evidence_http_transport import parse_event_result
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import (
    RELAY_HEADERS,
    artifact_count,
    camera_revision,
    fail_inserts,
    incident_rows,
    outbox_rows,
    relay_postgres_app,
    row_counts,
)

pytest_plugins = ("tests_support.postgres_sandbox",)


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


BuildRelay = Callable[..., TestClient]


@pytest.fixture
def build_relay(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> BuildRelay:
    """Relay on the real PostgreSQL root; ``enrolled=False`` publishes no Hub client."""

    def build(
        fake: FakeBackendIngestClient | None = None, *, enrolled: bool = True, **options: Any
    ) -> TestClient:
        client = (fake or FakeBackendIngestClient()) if enrolled else None
        app = relay_postgres_app(
            postgres_product_sandbox, postgres_audit_runtime, client=client, **options
        )
        return TestClient(app)

    return build


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


def _registry_camera(**options: Any) -> dict[str, Any]:
    return {
        "camera_id": "local-uuid-1",
        "label": "Lobby",
        "rtsp_url": "rtsp://camera/stream",
        "space_id": "space-1",
        **options,
    }


@pytest.mark.parametrize(
    "path",
    (
        "/api/v1/relay/system-tests",
        "/api/v1/relay/system-tests/auth-check",
    ),
)
def test_removed_system_test_relay_routes_are_not_registered(
    build_relay: BuildRelay, path: str
) -> None:
    response = build_relay().post(path, json={}, headers=RELAY_HEADERS)

    assert response.status_code == 404


def test_system_test_variant_cannot_cross_the_ordinary_alert_contract(
    build_relay: BuildRelay,
) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json={
            "edge_event_id": "00000000-0000-4000-8000-000000000099",
            "type": "SYSTEM_TEST",
            "source": "SYSTEM_TEST",
            "test_mode": "SYSTEM_TEST",
            "detected_at": "2026-08-12T00:00:00Z",
        },
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 422
    assert fake.alerts == []


def test_relay_alert_rejects_missing_token(build_relay: BuildRelay) -> None:
    response = build_relay().post("/api/v1/relay/alerts", json=_alert_payload())

    assert response.status_code == 401


def test_unenrolled_runtime_accepts_alert_locally_without_cloud_egress(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
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
    client = build_relay(enrolled=False, space_id="facility-1", backend_camera_id=None)

    response = client.post("/api/v1/relay/alerts", json=_alert_payload(), headers=RELAY_HEADERS)

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert row_counts(postgres_product_sandbox) == (1, 1)
    assert [row[1] for row in outbox_rows(postgres_product_sandbox)] == ["LOCAL_ONLY"]


def test_relay_alert_rejects_wrong_token(build_relay: BuildRelay) -> None:
    response = build_relay().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(),
        headers={"X-Edge-Relay-Token": "wrong"},
    )

    assert response.status_code == 403


def test_relay_alert_rejects_unknown_camera(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    response = build_relay().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(camera_id="camera-unknown"),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 403
    assert "unknown camera" in response.json()["detail"]
    assert row_counts(postgres_product_sandbox) == (0, 0)


def test_relay_alert_accepts_any_wire_facility_when_registry_has_camera(
    build_relay: BuildRelay,
) -> None:
    """Worker wire facility_id is not compared to env; registry camera_id binds."""
    response = build_relay().post(
        "/api/v1/relay/alerts",
        json=_alert_payload(facility_id="facility-2"),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_relay_alert_for_unresolved_camera_commits_no_incident(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """An unresolved camera 403s before anything is committed.

    The SQLite catalog used to keep an edge-local trace of the refused attempt
    (#183, #202). The incident projection and its event_outbox obligation are
    now one PostgreSQL transaction bound to a registry camera, so a refused
    alert leaves no incident that could later be mistaken for an accepted one.
    """
    payload = _alert_payload(
        camera_id="camera-unknown",
        edge_event_id="00000000-0000-4000-8000-000000000030",
    )

    response = build_relay().post("/api/v1/relay/alerts", json=payload, headers=RELAY_HEADERS)

    assert response.status_code == 403
    assert row_counts(postgres_product_sandbox) == (0, 0)


def test_relay_alert_forwards_valid_event_to_backend_ingest_client(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts", json=_alert_payload(), headers=RELAY_HEADERS
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert len(fake.alerts) == 1
    forwarded = dict(fake.alerts[0])
    forwarded.pop("on_accepted")
    edge_event_id = forwarded.pop("edge_event_id")
    assert forwarded == {
        "event_type": "bed-exit",
        "detected_at": "2026-06-25T12:00:00.000Z",
        "probability": 0.87,
        "clip_id": "clip-123",
    }
    # The Hub idempotency key is the one the committed incident carries.
    assert [row[0] for row in incident_rows(postgres_product_sandbox)] == [edge_event_id]


def test_relay_alert_compact_projection_preserves_incident_identity(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    fake = FakeBackendIngestClient()
    evidence = {"domain": "night-bed-exit", "window": {"start": 1, "end": 2}}
    payload = _alert_payload(
        edge_event_id="00000000-0000-4000-8000-000000000010", evidence=evidence
    )

    response = build_relay(fake).post("/api/v1/relay/alerts", json=payload, headers=RELAY_HEADERS)

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert incident_rows(postgres_product_sandbox) == [
        (
            payload["edge_event_id"],
            "camera-1",
            "facility-1",
            "bed-exit",
            0.87,
            "2026-06-25T12:00:00.000Z",
        )
    ]


def test_relay_alert_projects_identity_with_large_evidence_metadata(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """Evidence metadata is carried by the outbox envelope, not the incident row."""
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000011",
            evidence={"detail": "x" * 16 * 1024},
        ),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert len(fake.alerts) == 1
    assert [row[0] for row in incident_rows(postgres_product_sandbox)] == [
        "00000000-0000-4000-8000-000000000011"
    ]


def test_relay_alert_projects_identity_with_deep_evidence_metadata(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """Deep metadata does not block the compact incident identity."""
    fake = FakeBackendIngestClient()
    evidence: dict[str, object] = {}
    current = evidence
    for _ in range(8):
        child: dict[str, object] = {}
        current["child"] = child
        current = child

    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000012", evidence=evidence
        ),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"
    assert len(fake.alerts) == 1
    assert [row[0] for row in incident_rows(postgres_product_sandbox)] == [
        "00000000-0000-4000-8000-000000000012"
    ]


def test_relay_alert_fails_before_egress_when_compact_projection_is_unavailable(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    fake = FakeBackendIngestClient()
    client = build_relay(fake)
    fail_inserts(postgres_product_sandbox, "incidents")

    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(edge_event_id="00000000-0000-4000-8000-000000000013"),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 503
    assert row_counts(postgres_product_sandbox) == (0, 0)
    assert fake.alerts == []


def test_relay_alert_omits_missing_clip_id_for_backward_compatibility(
    build_relay: BuildRelay,
) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(evidence={"domain": "night-bed-exit"}),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert len(fake.alerts) == 1
    forwarded = dict(fake.alerts[0])
    forwarded.pop("on_accepted")
    forwarded.pop("edge_event_id")
    assert forwarded == {
        "event_type": "bed-exit",
        "detected_at": "2026-06-25T12:00:00.000Z",
        "probability": 0.87,
    }


def test_relay_heartbeat_forwards_valid_camera_to_backend_ingest_client(
    build_relay: BuildRelay,
) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "camera-1", "facility_id": "facility-1"},
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    assert fake.heartbeats == 1


def test_relay_heartbeat_records_local_liveness_even_when_camera_unresolved(
    build_relay: BuildRelay,
) -> None:
    """Local liveness reflects edge-local truth and must not depend on
    camera_inventory/registry binding: a camera absent from both still 403s
    to the worker (backend egress is legitimately unresolvable), but ml-api's
    own /status view must already know this camera beat in (see #183, #202)."""
    client = build_relay()

    response = client.post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "camera-unknown", "facility_id": "facility-1"},
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 403
    snapshot = get_heartbeat_store(client.app).snapshot()
    assert snapshot["cameras"]["camera-unknown"]["status"] == ONLINE


def test_relay_accepts_canonical_camera_id_from_registry_when_inventory_missing(
    build_relay: BuildRelay,
) -> None:
    fake = FakeBackendIngestClient()
    client = build_relay(
        fake,
        camera_id="provisional-camera",
        label="Lobby",
        rtsp_url="rtsp://camera/stream",
        space_id="space-1",
        backend_camera_id="backend-camera-1",
    )

    response = client.post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "backend-camera-1", "facility_id": "local"},
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert fake.heartbeats == 1


def test_relay_heartbeat_egresses_canonical_backend_id_for_mapped_local_camera(
    build_relay: BuildRelay,
) -> None:
    """The worker sends its local registry id; backend egress must use the
    explicit backend mapping — the backend only knows its own camera ids."""
    fake = FakeBackendIngestClient()
    client = build_relay(fake, **_registry_camera(backend_camera_id="backend-camera-1"))

    response = client.post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert fake.heartbeats == 1
    assert fake.egress_camera_ids == ["backend-camera-1"]


def test_relay_alert_egresses_canonical_backend_id_for_mapped_local_camera(
    build_relay: BuildRelay,
) -> None:
    fake = FakeBackendIngestClient()
    client = build_relay(fake, **_registry_camera(backend_camera_id="backend-camera-1"))

    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(camera_id="local-uuid-1", facility_id="local-facility"),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert len(fake.alerts) == 1
    assert fake.egress_camera_ids == ["backend-camera-1"]


def test_relay_heartbeat_never_egresses_local_id_when_camera_is_unmapped(
    build_relay: BuildRelay,
) -> None:
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
    client = build_relay(fake, **_registry_camera(backend_camera_id=None))

    response = client.post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    # No Hub egress at all, and specifically never under the local id.
    assert fake.egress_camera_ids == []
    assert fake.heartbeats == 0


def test_relay_heartbeat_clears_never_connected_on_first_heartbeat(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """A registry record's never_connected flips False on its first heartbeat,
    even before any successful probe -- a live worker beating in is itself
    evidence the camera has connected at least once."""
    client = build_relay(**_registry_camera(backend_camera_id=None))
    assert camera_revision(postgres_product_sandbox, "local-uuid-1") == (1, 1)

    response = client.post(
        "/api/v1/relay/heartbeat",
        json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert camera_revision(postgres_product_sandbox, "local-uuid-1")[1] == 0


def test_relay_heartbeat_never_connected_flip_is_a_single_write(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """Once never_connected is cleared it must stay cleared without an extra
    registry update on every subsequent heartbeat (avoids write amplification
    for a value that never reverts). Every update bumps the row revision."""
    client = build_relay(**_registry_camera(backend_camera_id=None))

    for _ in range(3):
        response = client.post(
            "/api/v1/relay/heartbeat",
            json={"camera_id": "local-uuid-1", "facility_id": "local-facility"},
            headers=RELAY_HEADERS,
        )
        assert response.status_code == 202

    assert camera_revision(postgres_product_sandbox, "local-uuid-1") == (2, 0)


def test_relay_alert_rejects_raw_frame_payloads(build_relay: BuildRelay) -> None:
    payload = _alert_payload(frame=[0, 1, 2])

    response = build_relay().post("/api/v1/relay/alerts", json=payload, headers=RELAY_HEADERS)

    assert response.status_code == 422


def test_relay_alert_forwards_audit_and_snapshot_when_present(build_relay: BuildRelay) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000020",
            audit={
                "config_version": 7,
                "model_version": "rf-2026",
                "clock_source": "edge_wall_clock",
            },
            snapshot_jpeg_base64=base64.b64encode(b"jpeg-bytes").decode("ascii"),
            snapshot=_snapshot_metadata(b"jpeg-bytes"),
        ),
        headers=RELAY_HEADERS,
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


def test_relay_alert_accepts_inline_snapshot_at_decoded_size_limit(
    build_relay: BuildRelay,
) -> None:
    content = b"x" * MAX_INLINE_SNAPSHOT_BYTES
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000020",
            snapshot_jpeg_base64=base64.b64encode(content).decode("ascii"),
            snapshot=_snapshot_metadata(content),
        ),
        headers=RELAY_HEADERS,
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
    build_relay: BuildRelay,
    postgres_product_sandbox: ProductSandbox,
    snapshot_jpeg_base64: str,
) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(snapshot_jpeg_base64=snapshot_jpeg_base64),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 422
    assert fake.alerts == []
    assert row_counts(postgres_product_sandbox) == (0, 0)


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
    build_relay: BuildRelay,
    snapshot_override: dict[str, object],
) -> None:
    content = b"jpeg-bytes"
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000020",
            snapshot_jpeg_base64=base64.b64encode(content).decode("ascii"),
            snapshot=_snapshot_metadata(content, **snapshot_override),
        ),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 422
    assert fake.alerts == []


def test_relay_alert_keeps_metadata_only_snapshot_compatible(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    fake = FakeBackendIngestClient()
    response = build_relay(fake).post(
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
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert "snapshot_bytes" not in fake.alerts[0]
    assert [row[0] for row in incident_rows(postgres_product_sandbox)] == [
        "00000000-0000-4000-8000-000000000020"
    ]
    assert artifact_count(postgres_product_sandbox) == 0


def _status_store() -> RuntimeStatusStore:
    store = RuntimeStatusStore(clock=lambda: 200.0)
    store.record({"facility_id": "facility-1", "seq": 1, "cameras": [], "clip_recorder": {}})
    return store


def test_relay_latency_uses_remote_acceptance_time_for_first_attempt(
    build_relay: BuildRelay,
) -> None:
    client = build_relay(ReceiptBackendIngestClient(accepted_at=105.0))
    store = _status_store()
    client.app.state.runtime_status_store = store

    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000001",
            detected_at="1970-01-01T00:01:40Z",
            attempt_ordinal=1,
        ),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 202
    assert store.snapshot()["facilities"]["facility-1"]["latency"] == {
        "first_attempt_samples": 1,
        "max_sec": 5.0,
        "since_sec": 105.0,
    }


def test_relay_latency_excludes_failed_and_retried_delivery(build_relay: BuildRelay) -> None:
    failure = DeliveryFailure(DeliveryDisposition.RETRY, "TIMEOUT")
    client = build_relay(ReceiptBackendIngestClient(failure=failure))
    store = _status_store()
    client.app.state.runtime_status_store = store

    failed = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000001",
            attempt_ordinal=1,
        ),
        headers=RELAY_HEADERS,
    )
    assert failed.status_code == 503
    assert store.snapshot()["facilities"]["facility-1"]["latency"] is None

    client.app.state.backend_ingest_client = ReceiptBackendIngestClient(accepted_at=105.0)
    retried = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000002",
            attempt_ordinal=2,
        ),
        headers=RELAY_HEADERS,
    )

    assert retried.status_code == 202
    assert store.snapshot()["facilities"]["facility-1"]["latency"] is None


def test_local_accept_with_no_persistence_anywhere_is_a_retryable_refusal(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """A terminal receipt deletes the worker's only other copy, so it may only be
    issued once something local holds the alert. When PostgreSQL cannot commit
    the incident, the relay must refuse with a 503 the worker classifies as RETRY."""
    client = build_relay(backend_camera_id=None)
    fail_inserts(postgres_product_sandbox, "incidents")

    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(edge_event_id="00000000-0000-4000-8000-000000000040"),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 503
    classified = parse_event_result(
        (response.status_code, {}, response.content),
        "00000000-0000-4000-8000-000000000040",
    )
    assert isinstance(classified, DeliveryFailure)
    assert classified.disposition is DeliveryDisposition.RETRY
    assert row_counts(postgres_product_sandbox) == (0, 0)


def test_local_accept_with_a_payload_postgres_can_never_hold_is_permanent(
    build_relay: BuildRelay, postgres_product_sandbox: ProductSandbox
) -> None:
    """Review of the 503 guard: a stored-fact constraint rejection is identical on
    every retry. Answering 503 there would recreate #431 as a permanent poison
    entry, so the refusal must be a 4xx the worker classifies as PERMANENT."""
    client = build_relay(backend_camera_id=None)

    response = client.post(
        "/api/v1/relay/alerts",
        json=_alert_payload(
            edge_event_id="00000000-0000-4000-8000-000000000041",
            detected_at="2026-06-25T12:00:00+00:00",
        ),
        headers=RELAY_HEADERS,
    )

    assert response.status_code == 422
    classified = parse_event_result(
        (response.status_code, {}, response.content),
        "00000000-0000-4000-8000-000000000041",
    )
    assert isinstance(classified, DeliveryFailure)
    assert classified.disposition is DeliveryDisposition.PERMANENT
    assert row_counts(postgres_product_sandbox) == (0, 0)
