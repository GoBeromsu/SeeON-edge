from __future__ import annotations

import base64
import hashlib

import psycopg
import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.evidence.event_outbox import (
    EventIdentityConflict,
    EventOutbox,
    OutboxBudget,
)
from backend.app.features.evidence.record_store import CentralEvidenceQuery
from backend.app.features.evidence.relay_projection import RelayEvent, RelaySnapshot
from tests_support.postgres_sandbox import ProductSandbox
from tests_support.relay_postgres_runtime import (
    RELAY_HEADERS,
    artifact_count,
    incident_rows,
    relay_postgres_app,
    row_counts,
)

pytest_plugins = ("tests_support.postgres_sandbox",)

EVENT_ID = "00000000-0000-4000-8000-000000000088"
TS = "2026-08-24T01:02:03Z"
SNAPSHOT_BYTES = b"jpeg-snapshot"
COMMITTED = [(EVENT_ID, "camera-1", "facility-1", "fall", 0.8, TS)]


def _outbox(sandbox: ProductSandbox, runtime: PostgresAuditRuntime) -> EventOutbox:
    return EventOutbox(
        sandbox.database, sandbox.authority, OutboxBudget(10, 1_048_576), audit_runtime=runtime
    )


def _event(*, probability: float = 0.8) -> RelayEvent:
    return RelayEvent(
        edge_event_id=EVENT_ID,
        event_type="fall",
        probability=probability,
        detected_at=TS,
        camera_id="camera-1",
        facility_id="facility-1",
        resident_id=None,
        evidence=None,
        audit=None,
    )


def _snapshot() -> RelaySnapshot:
    return RelaySnapshot(
        snapshot_id="snapshot-1",
        path="snapshots/camera-1/snapshot-1.jpg",
        sha256=hashlib.sha256(SNAPSHOT_BYTES).hexdigest(),
        size_bytes=len(SNAPSHOT_BYTES),
        mime_type="image/jpeg",
        captured_at=TS,
    )


def test_alert_and_inline_snapshot_commit_atomically_and_replay_idempotently(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: the PostgreSQL authority and one alert with inline snapshot bytes.
    sandbox = postgres_product_sandbox
    outbox = _outbox(sandbox, postgres_audit_runtime)

    # When: the exact alert is delivered twice.
    accepted = tuple(
        outbox.accept(
            _event(),
            backend_camera_id=None,
            forward=False,
            snapshot=_snapshot(),
            snapshot_bytes=SNAPSHOT_BYTES,
        )
        for _ in range(2)
    )

    # Then: one incident and one available snapshot are committed and the replay is a duplicate.
    assert [receipt.duplicate for receipt in accepted] == [False, True]
    assert sandbox.admin.execute("SELECT count(*) FROM incidents").fetchone() == (1,)
    assert sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
    summary = CentralEvidenceQuery(sandbox.database).get(EVENT_ID)
    assert summary is not None
    assert summary.snapshot_artifact_state == "AVAILABLE"


@pytest.mark.parametrize("committed_by", ["outbox", "migration"])
def test_edge_event_id_replay_rejects_changed_identity(
    postgres_product_sandbox: ProductSandbox,
    postgres_audit_runtime: PostgresAuditRuntime,
    committed_by: str,
) -> None:
    # Given: an already committed edge_event_id, either accepted through the outbox
    # or imported as a bare incident with no outbox row (the migration shape).
    sandbox = postgres_product_sandbox
    outbox = _outbox(sandbox, postgres_audit_runtime)
    if committed_by == "outbox":
        outbox.accept(_event(), backend_camera_id=None, forward=False)
    else:
        sandbox.admin.execute(
            "INSERT INTO incidents (incident_id,edge_event_id,facility_id,camera_id,event_type,"
            "probability,detected_at,lifecycle_state,provenance_state,provenance_missing_reason,"
            "review_version,revision,created_at,updated_at) "
            "VALUES (%s,%s,'facility-1','camera-1','fall',0.8,%s,'OPEN','MISSING',"
            "'NOT_RECORDED',0,1,%s,%s)",
            (f"incident:{EVENT_ID}", EVENT_ID, TS, TS, TS),
        )
    before = row_counts(sandbox)

    # When/Then: a changed immutable probability cannot reuse that key.
    with pytest.raises(EventIdentityConflict):
        outbox.accept(_event(probability=0.7), backend_camera_id=None, forward=False)

    # Then: the committed fact is unchanged and no delivery obligation was added.
    assert incident_rows(sandbox) == COMMITTED
    assert row_counts(sandbox) == before


def test_snapshot_database_failure_rolls_back_incident(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: an artifact tuple whose MIME exceeds the locked schema boundary.
    sandbox = postgres_product_sandbox
    outbox = _outbox(sandbox, postgres_audit_runtime)
    invalid = RelaySnapshot(
        snapshot_id="snapshot-1",
        path="snapshots/camera-1/snapshot-1.jpg",
        sha256=hashlib.sha256(SNAPSHOT_BYTES).hexdigest(),
        size_bytes=len(SNAPSHOT_BYTES),
        mime_type="x" * 129,
        captured_at=TS,
    )

    # When: the invalid artifact fails after incident insertion in the real transaction.
    with pytest.raises(psycopg.errors.CheckViolation):
        outbox.accept(_event(), backend_camera_id=None, forward=False, snapshot=invalid)

    # Then: neither the incident, its outbox row nor the artifact is partially committed.
    assert row_counts(sandbox) == (0, 0)
    assert artifact_count(sandbox) == 0

    # And: the same key is still new, so a corrected alert is accepted, not a duplicate.
    accepted = outbox.accept(_event(), backend_camera_id=None, forward=False, snapshot=_snapshot())
    assert not accepted.duplicate
    assert incident_rows(sandbox) == COMMITTED
    assert (row_counts(sandbox), artifact_count(sandbox)) == ((1, 1), 1)


def test_unmapped_relay_alert_is_locally_accepted_on_real_http_surface(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: a registered camera without a Hub mapping on the PostgreSQL authority.
    sandbox = postgres_product_sandbox
    app = relay_postgres_app(sandbox, postgres_audit_runtime, backend_camera_id=None)
    snapshot = _snapshot()
    payload = {
        "edge_event_id": EVENT_ID,
        "event_type": "fall",
        "probability": 0.8,
        "detected_at": TS,
        "camera_id": "camera-1",
        "facility_id": "facility-1",
        "snapshot_jpeg_base64": base64.b64encode(SNAPSHOT_BYTES).decode(),
        "snapshot": {
            "snapshot_id": snapshot.snapshot_id,
            "path": snapshot.path,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "mime_type": snapshot.mime_type,
            "captured_at": snapshot.captured_at,
            "camera_id": "camera-1",
            "edge_event_id": EVENT_ID,
        },
    }

    # When: the worker POSTs through the authenticated relay route.
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/relay/alerts",
            json=payload,
            headers=RELAY_HEADERS,
        )

    # Then: local atomic acceptance does not require an upstream camera id, and
    # it says so by name. The bare {"status": "accepted"} this used to pin was
    # unactionable for the sender: it needs a receipt echoing its edge_event_id,
    # an absent one is indistinguishable from a mangled response, so it retried
    # this event forever and every newer event queued behind it never left the
    # edge (#431). The backend is the only party that knows the push was
    # deliberately skipped, so the backend is the party that must state it.
    assert response.status_code == 202
    assert response.json() == {"status": "accepted_local", "edge_event_id": EVENT_ID}
    assert CentralEvidenceQuery(sandbox.database).get(EVENT_ID) is not None
    assert row_counts(sandbox) == (1, 1)
