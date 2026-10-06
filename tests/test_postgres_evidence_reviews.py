"""Native incident queries and governed review transactions, not HTTP activation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import Event
from time import monotonic

import psycopg
import pytest

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.edge_db.reviews import ReviewDisposition
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.store import AuditEvent
from backend.app.features.evidence.event_outbox import EventOutbox, OutboxBudget
from backend.app.features.evidence.record_store import (
    LEGACY_DELIVERY_STATE,
    CentralEvidenceQuery,
    CentralEvidenceReviewStore,
    EvidenceProjectionUnavailable,
    EvidenceReviewConflictError,
)
from backend.app.features.evidence.relay_projection import RelayEvent

pytest_plugins = ("tests_support.postgres_sandbox",)
_TIME = "2026-09-28T03:00:00.000Z"


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
    for event_id in ("a", "b", "c"):
        outbox.accept(
            RelayEvent(event_id, "fall", 0.8, _TIME, "camera-1", "facility-1", None, None, None),
            backend_camera_id="hub-1" if event_id == "b" else None,
            forward=event_id == "b",
        )
    return (
        sandbox,
        runtime,
        CentralEvidenceReviewStore(sandbox.database, sandbox.authority),
        CentralEvidenceQuery(sandbox.database),
    )


def _update(store, **changes):
    values = {
        "incident_id": "incident:a",
        "expected_version": 0,
        "actor_id": "operator",
        "reviewed_at": _TIME,
        "disposition": ReviewDisposition.TRUE_POSITIVE,
        "notes": None,
    }
    values.update(changes)
    return store.update(**values)


def _rows(sandbox):
    return {
        name: sandbox.admin.execute("SELECT * FROM " + name + " ORDER BY 1").fetchall()
        for name in ("incidents", "artifacts", "event_outbox", "audit_events")
    }


def _append(runtime, tokens):
    def write(connection):
        token = runtime.append_borrowed(
            connection,
            AuditEvent(
                occurred_at=_TIME,
                actor_id="operator",
                action=AuditAction.INCIDENT_REVIEW,
                target_id="incident:a",
                detail=empty_detail(AuditAction.INCIDENT_REVIEW),
            ),
        )
        runtime.validate_publication(token)
        tokens.append(token)

    return write


def test_review_without_clip_remains_tentative_until_owner_exit(setup, monkeypatch):
    sandbox, runtime, store, query = setup
    before, tokens, trace = _rows(sandbox), [], []
    pool_connection = sandbox.database._pool.connection
    append = _append(runtime, tokens)

    @contextmanager
    def observed_exit(*args, **kwargs):
        with pool_connection(*args, **kwargs) as connection:
            yield connection
        trace.append("pool-exit")

    monkeypatch.setattr(sandbox.database._pool, "connection", observed_exit)

    def callback(connection):
        assert connection.execute(
            "SELECT review_version FROM incidents WHERE incident_id='incident:a'"
        ).fetchone() == (1,)
        append(connection)
        assert _rows(sandbox) == before
        trace.append("callback")

    review = _update(store, after_write=callback)
    trace.append("returned")
    assert trace == ["callback", "pool-exit", "returned"]
    assert review.version == 1 and review.clip_id is None
    assert len(tokens) == 1
    runtime.publish_committed(tokens.pop())
    assert query.get("a").review == review
    before = _rows(sandbox)
    calls = []
    with pytest.raises(EvidenceReviewConflictError):
        _update(store, after_write=lambda c: calls.append(True))
    assert calls == [] and _rows(sandbox) == before


@pytest.mark.parametrize("cancelled", [False, True])
def test_callback_failure_preserves_error_and_rolls_back_borrowed_audit(setup, cancelled):
    sandbox, runtime, store, _ = setup
    error = BaseException("cancelled") if cancelled else ValueError("refused")
    before, tokens = _rows(sandbox), []
    append = _append(runtime, tokens)

    def callback(connection):
        append(connection)
        raise error

    with pytest.raises(type(error)) as caught:
        _update(store, after_write=callback)
    assert caught.value is error and _rows(sandbox) == before and len(tokens) == 1
    runtime.publish_failed(tokens.pop(), error)


def test_real_commit_rejection_cannot_publish_review(setup):
    sandbox, runtime, store, _ = setup
    before, tokens = _rows(sandbox), []
    sandbox.admin.execute(
        "CREATE FUNCTION reject_review_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'review commit rejection' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_review_commit AFTER UPDATE ON incidents "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_review_commit()"
    )
    with pytest.raises(psycopg.errors.CheckViolation) as caught:
        _update(store, after_write=_append(runtime, tokens))
    assert len(tokens) == 1 and _rows(sandbox) == before
    runtime.publish_failed(tokens.pop(), caught.value)


def test_authority_and_conflict_do_not_call_audit_hook(setup):
    sandbox, _, store, _ = setup
    calls = []
    with pytest.raises(EvidenceReviewConflictError):
        _update(store, incident_id="missing", after_write=lambda c: calls.append(True))
    freeze_authority(sandbox.database, sandbox.authority)
    before = _rows(sandbox)
    with pytest.raises(AuthorityFenced):
        _update(store, after_write=lambda c: calls.append(True))
    assert not calls and _rows(sandbox) == before


def test_query_preserves_keyset_order_and_real_delivery_states(setup):
    _, _, _, query = setup
    assert query.get("a").event_delivery_state == "LOCAL_ONLY"
    assert query.get("b").event_delivery_state == "PENDING"
    assert query.get("missing") is None
    seen, cursor = [], None
    while True:
        page, cursor = query.list(limit=1, cursor=cursor)
        seen.extend(value.incident_id for value in page)
        if cursor is None:
            break
    assert seen == ["incident:c", "incident:b", "incident:a"]
    assert query.get("incident:a") == query.get("a")
    with pytest.raises(ValueError, match="cursor"):
        query.list(cursor="not-base64!")
    with pytest.raises(ValueError, match="limit"):
        query.list(limit=0)


def test_missing_delivery_obligation_is_not_reported_as_acknowledged(setup):
    sandbox, _, _, query = setup
    sandbox.admin.execute(
        "INSERT INTO incidents (incident_id,edge_event_id,facility_id,camera_id,event_type,"
        "detected_at,lifecycle_state,provenance_state,provenance_missing_reason,"
        "review_version,revision,created_at,updated_at) "
        "VALUES ('unbound','unbound','facility-1','camera-1','fall',%s,'OPEN','MISSING',"
        "'NOT_RECORDED',0,1,%s,%s)",
        (_TIME, _TIME, _TIME),
    )
    with pytest.raises(EvidenceProjectionUnavailable, match="obligation is missing"):
        query.get("unbound")
    with pytest.raises(EvidenceProjectionUnavailable, match="obligation is missing"):
        query.list()


def test_incident_imported_from_sqlite_lists_without_a_delivery_obligation(setup):
    sandbox, _, _, query = setup
    sandbox.admin.execute(
        "INSERT INTO incidents (incident_id,edge_event_id,facility_id,camera_id,event_type,"
        "detected_at,lifecycle_state,provenance_state,provenance_missing_reason,"
        "review_version,revision,created_at,updated_at) "
        "VALUES ('legacy','legacy','facility-1','camera-1','fall',%s,'OPEN','MISSING',"
        "'NOT_RECORDED',0,1,%s,%s)",
        (_TIME, _TIME, _TIME),
    )
    # Given: the ledger carries the stamp the SQLite import leaves behind.
    sandbox.admin.execute(
        "INSERT INTO schema_migrations (version,name,checksum,applied_at,"
        "source_schema_version,source_db_sha256,reconciliation_sha256) "
        "VALUES (9999,'sqlite-import',repeat('c',64),%s,19,repeat('a',64),repeat('b',64))",
        (_TIME,),
    )

    legacy = query.get("legacy")
    listed, _ = query.list()

    assert legacy is not None and legacy.event_delivery_state == LEGACY_DELIVERY_STATE
    assert {item.incident_id: item.event_delivery_state for item in listed}["legacy"] == (
        LEGACY_DELIVERY_STATE
    )
    # Incidents accepted through the outbox keep their real delivery state.
    assert query.get("a").event_delivery_state != LEGACY_DELIVERY_STATE


def test_purged_primary_identity_remains_available_to_incident_review(setup):
    sandbox, _, store, query = setup
    sandbox.admin.execute(
        "INSERT INTO clips (clip_id,camera_id,event_facet,started_at,manifest_relpath,"
        "media_relpath,manifest_sha256,media_sha256,manifest_size_bytes,media_size_bytes,"
        "local_state,publish_state,retention_state,revision,created_at,updated_at) "
        "VALUES ('clip-a','camera-1','fall',%s,'clips/clip-a/manifest.json',"
        "'clips/clip-a/clip.mp4',%s,%s,10,10,'AVAILABLE','WAITING','RETAINED',1,%s,%s)",
        (_TIME, "a" * 64, "b" * 64, _TIME, _TIME),
    )
    sandbox.admin.execute(
        "INSERT INTO artifacts "
        "(incident_id,kind,artifact_id,clip_id,state,reason,revision,created_at,updated_at) "
        "VALUES ('incident:a','PRIMARY_CLIP','purged-primary','clip-a',"
        "'PURGED','RETENTION',1,%s,%s)",
        (_TIME, _TIME),
    )
    review = _update(store)
    summary = query.get("a")
    assert review.version == 1 and summary.review == review
    assert summary.primary_artifact_state == "PURGED"
    assert review.clip_id == summary.primary_clip_id == "clip-a"
    assert summary.clip_publish_state == "WAITING" and summary.retention_state == "RETAINED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor_id", ""),
        ("notes", ""),
        ("notes", "x" * 1001),
        ("expected_version", -1),
        ("reviewed_at", "2026-01-01"),
    ],
)
def test_invalid_review_is_rejected_before_mutation(setup, field, value):
    sandbox, _, store, _ = setup
    before, callbacks = _rows(sandbox), []
    with pytest.raises(ValueError):
        _update(store, **{field: value}, after_write=lambda c: callbacks.append(True))
    assert not callbacks and _rows(sandbox) == before


def test_post_commit_error_is_not_replayed(setup, monkeypatch):
    sandbox, _, store, _ = setup
    original, error, calls = sandbox.database.transact, CommitOutcomeUnknown(), []

    def uncertain(callback):
        original(callback)
        calls.append(True)
        raise error

    monkeypatch.setattr(sandbox.database, "transact", uncertain)
    with pytest.raises(CommitOutcomeUnknown) as caught:
        _update(store)
    assert caught.value is error and calls == [True]
    assert sandbox.admin.execute(
        "SELECT review_version FROM incidents WHERE incident_id='incident:a'"
    ).fetchone() == (1,)


def test_concurrent_compare_and_swap_commits_one_review(setup):
    sandbox, _, store, _ = setup
    entered, release = Event(), Event()
    executor = ThreadPoolExecutor(max_workers=2)
    futures = []
    pids, loser_calls = [], []

    def hold(connection):
        pids.append(connection.info.backend_pid)
        entered.set()
        assert release.wait(2), "first reviewer not released"

    try:
        futures.append(executor.submit(_update, store, after_write=hold))
        assert entered.wait(1)
        futures.append(
            executor.submit(_update, store, after_write=lambda c: loser_calls.append(True))
        )
        deadline, pacing = monotonic() + 1.5, Event()
        while not sandbox.admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)))",
            (pids[0],),
        ).fetchone()[0]:
            assert monotonic() < deadline, "second reviewer did not wait on the incident lock"
            pacing.wait(0.01)
        assert not any(future.done() for future in futures)
        release.set()
        assert futures[0].result(timeout=2).version == 1
        with pytest.raises(EvidenceReviewConflictError):
            futures[1].result(timeout=2)
        assert loser_calls == []
        assert sandbox.admin.execute(
            "SELECT review_version,revision FROM incidents WHERE incident_id='incident:a'"
        ).fetchone() == (1, 2)
    finally:
        release.set()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "review operations did not drain"
