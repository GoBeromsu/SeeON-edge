from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.evidence.event_outbox import EventOutbox, OutboxBudget
from backend.app.features.evidence.record_store import (
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
    EvidenceReviewConflictError,
    ReviewDisposition,
)
from backend.app.features.evidence.relay_projection import RelayEvent
from backend.app.main import create_app, no_lifespan
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

pytest_plugins = ("tests_support.postgres_sandbox",)
EVENT_ID = "event-review"
INCIDENT_ID = "incident:" + EVENT_ID
CLIP_ID = "clip-review"
REVIEWED_AT = "2026-08-13T12:00:00Z"
HASH = "ab" * 32


@pytest.fixture
def setup(postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=lambda: 0.0,
    )
    assert runtime.verify_once() and runtime.start_session_once()
    outbox = EventOutbox(
        sandbox.database, sandbox.authority, OutboxBudget(10, 1_048_576), audit_runtime=runtime
    )
    _accept(outbox, EVENT_ID)
    return sandbox, runtime, outbox


def _accept(outbox, event_id):
    outbox.accept(
        RelayEvent(event_id, "fall", 0.8, REVIEWED_AT, "camera-1", "facility-1", None, None, None),
        backend_camera_id=None,
        forward=False,
    )


def _primary(setup, state):
    sandbox = setup[0]
    with sandbox.admin.transaction():
        sandbox.admin.execute(
            "INSERT INTO clips (clip_id,camera_id,event_facet,started_at,manifest_relpath,"
            "media_relpath,manifest_sha256,media_sha256,manifest_size_bytes,media_size_bytes,"
            "local_state,publish_state,retention_state,revision,created_at,updated_at) "
            "VALUES (%s,'camera-1','fall',%s,'clips/clip-review/manifest.json',"
            "'clips/clip-review/clip.mp4',%s,%s,10,10,'AVAILABLE','WAITING','RETAINED',1,%s,%s)",
            (CLIP_ID, REVIEWED_AT, HASH, HASH, REVIEWED_AT, REVIEWED_AT),
        )
        if state == "AVAILABLE":
            sandbox.admin.execute(
                "INSERT INTO artifacts "
                "(incident_id,kind,artifact_id,clip_id,state,contained_relpath,"
                "content_sha256,size_bytes,mime_type,codec,revision,created_at,updated_at) "
                "VALUES (%s,'PRIMARY_CLIP','artifact-review',%s,'AVAILABLE',"
                "'clips/clip-review/clip.mp4',%s,10,'video/mp4','h264',1,%s,%s)",
                (INCIDENT_ID, CLIP_ID, HASH, REVIEWED_AT, REVIEWED_AT),
            )
        else:
            assert state == "PURGED"
            sandbox.admin.execute(
                "INSERT INTO artifacts (incident_id,kind,artifact_id,clip_id,state,reason,"
                "revision,created_at,updated_at) "
                "VALUES (%s,'PRIMARY_CLIP','artifact-review',%s,"
                "'PURGED','OPERATOR_DELETE',1,%s,%s)",
                (INCIDENT_ID, CLIP_ID, REVIEWED_AT, REVIEWED_AT),
            )


def _store(setup):
    return CentralEvidenceReviewStore(setup[0].database, setup[0].authority)


def _review(store, **changes):
    values = {
        "incident_id": INCIDENT_ID,
        "expected_version": 0,
        "actor_id": "operator-1",
        "reviewed_at": REVIEWED_AT,
        "disposition": ReviewDisposition.TRUE_POSITIVE,
        "notes": None,
    }
    values.update(changes)
    return store.update(**values)


def test_review_is_incident_local_when_primary_clip_is_missing(setup):
    review = _review(_store(setup))
    assert review.version == 1 and review.clip_id is None


def test_review_updates_are_optimistically_concurrent(setup):
    _primary(setup, "AVAILABLE")
    store = _store(setup)
    _review(store, actor_id="operator-first", notes="first")
    barrier = Barrier(2, timeout=2)

    def revise(actor_id):
        barrier.wait()
        try:
            return _review(
                store,
                expected_version=1,
                actor_id=actor_id,
                reviewed_at="2026-08-13T12:01:00Z",
                disposition=ReviewDisposition.FALSE_POSITIVE,
            )
        except EvidenceReviewConflictError as error:
            return error

    executor = ThreadPoolExecutor(max_workers=2)
    futures = [executor.submit(revise, actor) for actor in ("operator-a", "operator-b")]
    try:
        outcomes = [future.result(timeout=5) for future in futures]
        assert sum(not isinstance(value, Exception) for value in outcomes) == 1
        assert sum(isinstance(value, EvidenceReviewConflictError) for value in outcomes) == 1
    finally:
        barrier.abort()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "review operations did not drain"
    assert setup[0].admin.execute(
        "SELECT review_version,review_disposition,revision FROM incidents"
    ).fetchone() == (2, "FP", 3)


def test_review_contract_bounds_actor_time_and_notes(setup):
    store = _store(setup)
    for actor, reviewed_at, notes in (
        ("x" * 129, REVIEWED_AT, None),
        ("operator", "not-a-time", None),
        ("operator", REVIEWED_AT, "x" * 1001),
    ):
        with pytest.raises(ValueError):
            _review(store, actor_id=actor, reviewed_at=reviewed_at, notes=notes)


def test_primary_clip_is_projected_from_purged_artifact(setup):
    _primary(setup, "PURGED")
    summary = CentralEvidenceQuery(setup[0].database).get(INCIDENT_ID)
    assert summary is not None and summary.primary_clip_id == CLIP_ID
    assert summary.primary_artifact_state == "PURGED"
    review = _review(
        _store(setup), actor_id="operator", disposition=ReviewDisposition.FALSE_POSITIVE
    )
    assert review.version == 1


def test_operator_incident_api_reviews_without_available_primary_clip(setup):
    app = create_app(lifespan=no_lifespan)
    app.state.central_evidence_query = CentralEvidenceQuery(setup[0].database)
    app.state.central_evidence_review_store = _store(setup)
    app.state.audit_runtime = setup[1]
    app.state.dashboard_credentials_store = PostgresDashboardCredentialsStore(
        setup[0].database, setup[0].authority
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            ).status_code
            == 204
        )
        listed = client.get("/api/v1/incidents", params={"limit": 10})
        reviewed = client.put(
            f"/api/v1/incident-reviews/{INCIDENT_ID}",
            json={"expected_version": 0, "disposition": "TRUE_POSITIVE", "notes": None},
        )
        conflict = client.put(
            f"/api/v1/incident-reviews/{INCIDENT_ID}",
            json={"expected_version": 0, "disposition": "FALSE_POSITIVE", "notes": None},
        )
        malformed = client.get("/api/v1/incidents", params={"cursor": "not-a-cursor"})
    assert listed.status_code == 200 and listed.json()["incidents"][0]["review"] is None
    assert reviewed.status_code == 200 and reviewed.json()["review"]["version"] == 1
    assert conflict.status_code == 409 and malformed.status_code == 400


def test_incident_keyset_does_not_skip_equal_timestamps(setup):
    for suffix in ("a", "b"):
        _accept(setup[2], f"event-{suffix}")
    query = CentralEvidenceQuery(setup[0].database)
    seen, cursor = [], None
    while True:
        page, cursor = query.list(limit=1, cursor=cursor)
        seen.extend(item.incident_id for item in page)
        if cursor is None:
            break
    assert seen == ["incident:event-review", "incident:event-b", "incident:event-a"]
