"""Real isolated PostgreSQL event acceptance, fencing, and retry conservation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

import psycopg
import pytest

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown, PoolBudget, PostgresDatabase
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.postgres_runtime import (
    AuditRuntimeUnavailable,
    PendingAuditPublication,
    PostgresAuditRuntime,
)
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.sessions import AuditSession
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.evidence.event_outbox import (
    EventIdentityConflict,
    EventOutbox,
    OutboxBudget,
    OutboxCapacityExceeded,
)
from backend.app.features.evidence.outbox_delivery import (
    DeliveryBudget,
    DeliveryOutcome,
    DeliveryResponseConflict,
    OutboxDelivery,
)
from backend.app.features.evidence.relay_projection import RelayEvent, RelaySnapshot

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.000Z"


class _AuditClock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def audit_clock() -> _AuditClock:
    return _AuditClock()


@pytest.fixture
def audit_runtime(
    postgres_product_sandbox: ProductSandbox, audit_clock: _AuditClock
) -> PostgresAuditRuntime:
    sandbox = postgres_product_sandbox
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=audit_clock,
    )
    # Exercise real verification and session establishment, not a seeded session.
    # This component fixture is not evidence of HTTP/lifespan activation.
    assert runtime.verify_once()
    assert runtime.start_session_once()
    assert runtime.snapshot().ready
    return runtime


@pytest.fixture
def store(
    postgres_product_sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime
) -> tuple[EventOutbox, psycopg.Connection]:
    sandbox = postgres_product_sandbox
    return (
        EventOutbox(
            sandbox.database,
            sandbox.authority,
            OutboxBudget(100, 1_048_576),
            audit_runtime=audit_runtime,
        ),
        sandbox.admin,
    )


def _event():
    return RelayEvent(
        str(uuid4()),
        "FALL",
        0.8,
        _TIME,
        "camera-1",
        "facility-1",
        None,
        {"clip_id": "clip-1"},
        None,
    )


def _delivery(store, attempts=2):
    return OutboxDelivery(
        store.database, store.authority, DeliveryBudget(attempts, 30.0, 10.0, 1.0)
    )


def _counts(connection):
    return tuple(
        connection.execute("SELECT count(*) FROM " + name).fetchone()[0]
        for name in ("incidents", "event_outbox", "audit_events")
    )


def _delivery_history(connection):
    return {
        name: connection.execute("SELECT * FROM " + name + " ORDER BY 1").fetchall()
        for name in (
            "event_outbox",
            "event_delivery_attempts",
            "event_delivery_results",
            "event_delivery_observations",
        )
    }


def _publications(runtime, monkeypatch):
    observations = []
    committed, failed = runtime.publish_committed, runtime.publish_failed

    def commit(token):
        observations.append(("committed", None))
        return committed(token)

    def fail(token, error):
        observations.append(("failed", error))
        return failed(token, error)

    monkeypatch.setattr(runtime, "publish_committed", commit)
    monkeypatch.setattr(runtime, "publish_failed", fail)
    return observations


def _pending_publication(runtime, target):
    event = AuditEvent(
        occurred_at=utc_now(),
        actor_id="worker-relay",
        action=AuditAction.RELAY_ALERT,
        target_id=target,
        detail=empty_detail(AuditAction.RELAY_ALERT),
        actor_type=AuditActorType.SERVICE,
        auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
    )
    return runtime.database.transact(lambda connection: runtime.append_borrowed(connection, event))


def test_commit_persists_incident_obligation_audit_before_receipt_and_dedupes(store):
    outbox, admin = store
    event = _event()
    receipt = outbox.accept(event, backend_camera_id="hub-camera", forward=True)
    assert receipt.edge_event_id == event.edge_event_id and not receipt.duplicate
    assert receipt.delivery_state == "PENDING" and _counts(admin) == (1, 1, 2)
    duplicate = outbox.accept(event, backend_camera_id="changed-mapping", forward=False)
    assert duplicate.duplicate and duplicate.delivery_state == "PENDING"
    assert _counts(admin) == (1, 1, 2)
    assert admin.execute("SELECT backend_camera_id FROM event_outbox").fetchone() == ("hub-camera",)
    with pytest.raises(EventIdentityConflict):
        outbox.accept(replace(event, probability=0.9), backend_camera_id="hub-camera", forward=True)
    assert _counts(admin) == (1, 1, 2)


def test_failing_audit_rolls_back_incident_outbox_and_snapshot(store):
    outbox, admin = store
    admin.execute(
        "CREATE FUNCTION reject_audit_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
    )
    admin.execute(
        "CREATE TRIGGER reject_audit_test BEFORE INSERT ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_audit_test()"
    )
    snapshot = RelaySnapshot("snap-1", "snapshots/snap-1.jpg", "a" * 64, 20, "image/jpeg", _TIME)
    with pytest.raises(AuditRuntimeUnavailable, match="^audit runtime unavailable$"):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True, snapshot=snapshot)
    assert _counts(admin) == (0, 0, 1)
    assert admin.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)


def test_concurrent_duplicate_acceptance_has_one_durable_identity(store):
    outbox, admin = store
    event = _event()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(outbox.accept, event, backend_camera_id="hub-camera", forward=True)
            for _ in range(3)
        ]
        receipts = [future.result(timeout=5) for future in futures]
    assert sum(not receipt.duplicate for receipt in receipts) == 1
    assert _counts(admin) == (1, 1, 2)


def test_capacity_refusal_cannot_ack_or_remove_previous_acceptance(store):
    outbox, admin = store
    limited = EventOutbox(
        outbox.database,
        outbox.authority,
        OutboxBudget(1, 1_048_576),
        audit_runtime=outbox.audit_runtime,
    )
    event = _event()
    limited.accept(event, backend_camera_id=None, forward=False)
    with pytest.raises(OutboxCapacityExceeded):
        limited.accept(_event(), backend_camera_id=None, forward=False)
    assert limited.accept(event, backend_camera_id=None, forward=False).duplicate
    assert _counts(admin) == (1, 1, 2)
    assert _delivery(outbox).claim() is None


def test_sender_fence_blocks_new_admission_claims_and_stale_generation(store):
    outbox, admin = store
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert freeze_authority(outbox.database, outbox.authority) == 1
    with pytest.raises(AuthorityFenced):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    with pytest.raises(AuthorityFenced):
        _delivery(outbox).claim()
    admin.execute(
        "UPDATE deployment_authority SET generation=2,writer_token=%s,"
        "accepting=true,egress_enabled=true",
        (uuid4(),),
    )
    with pytest.raises(AuthorityFenced):
        outbox.accept(_event(), backend_camera_id=None, forward=False)
    assert _counts(admin) == (1, 1, 2)


def test_exclusive_fence_waits_for_prior_acceptance_transaction(store):
    from backend.app.edge_db.authority import require_authority

    outbox, admin = store
    admitted, release = Event(), Event()

    def in_flight(connection):
        require_authority(connection, outbox.authority)
        admitted.set()
        assert release.wait(2)
        return "committed"

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(outbox.database.transact, in_flight)
        assert admitted.wait(2)
        fencer = pool.submit(freeze_authority, outbox.database, outbox.authority)
        query = (
            "SELECT generation, writer_token FROM deployment_authority "
            "WHERE singleton = 1 FOR UPDATE"
        )
        deadline = monotonic() + 1.5
        pacing = Event()
        while not admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
            "WHERE application_name='seeon-edge' AND query=%s AND wait_event_type='Lock')",
            (query,),
        ).fetchone()[0]:
            assert monotonic() < deadline, "fencer never waited for the real writer lock"
            pacing.wait(0.01)
        assert not fencer.done()
        assert admin.execute("SELECT accepting FROM deployment_authority").fetchone() == (True,)
        release.set()
        assert writer.result(timeout=3) == "committed"
        assert fencer.result(timeout=3) == 1
    assert admin.execute(
        "SELECT accepting,egress_enabled FROM deployment_authority"
    ).fetchone() == (False, False)


def test_deferred_commit_rejection_cannot_release_event_receipt(store, monkeypatch):
    outbox, admin = store
    publications = _publications(outbox.audit_runtime, monkeypatch)
    admin.execute(
        "CREATE FUNCTION reject_commit_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected deferred failure' USING ERRCODE='23514'; END $$"
    )
    admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_commit_test AFTER INSERT ON event_outbox "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_commit_test()"
    )
    with pytest.raises(psycopg.errors.CheckViolation) as raised:
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert _counts(admin) == (0, 0, 1)
    assert publications == [("failed", raised.value)]
    assert not outbox.audit_runtime.snapshot().ready
    # The synchronous owner has completed rollback/exit; no tentative token
    # remains to block closure. This is not an HTTP/lifespan drain proof.
    outbox.audit_runtime.stop()
    assert outbox.audit_runtime.close_session_once()


def test_unknown_then_retry_keeps_same_event_and_append_only_history(store):
    outbox, admin = store
    event = _event()
    outbox.accept(event, backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox)
    first = sender.claim()
    assert first and first.edge_event_id == event.edge_event_id and first.ordinal == 1
    assert sender.claim() is None
    assert sender.finish(first, DeliveryOutcome.UNKNOWN, reason="NETWORK_TIMEOUT")
    assert sender.claim() is None
    admin.execute("UPDATE event_outbox SET retry_at=clock_timestamp()-interval '1 second'")
    second = sender.claim()
    assert second and second.edge_event_id == first.edge_event_id and second.ordinal == 2
    assert second.attempt_id != first.attempt_id and second.envelope == first.envelope
    assert sender.finish(
        second,
        DeliveryOutcome.SENT,
        reason="ACCEPTED",
        http_status=202,
        backend_event_id="central-1",
    )
    assert not sender.finish(first, DeliveryOutcome.UNKNOWN, reason="NETWORK_TIMEOUT")
    with pytest.raises(DeliveryResponseConflict):
        sender.finish(
            first,
            DeliveryOutcome.SENT,
            reason="LATE",
            http_status=202,
            backend_event_id="central-1",
        )
    assert sender.claim() is None
    assert admin.execute("SELECT state,attempt_count FROM event_outbox").fetchone() == ("SENT", 2)
    assert admin.execute(
        "SELECT outcome FROM event_delivery_results ORDER BY finished_at"
    ).fetchall() == [("UNKNOWN",), ("SENT",)]
    assert admin.execute(
        "SELECT outcome,reason,backend_event_id FROM event_delivery_observations ORDER BY ordinal"
    ).fetchall() == [("UNKNOWN", "NETWORK_TIMEOUT", None), ("SENT", "ACCEPTED", "central-1")]
    for table in (
        "event_outbox",
        "event_delivery_attempts",
        "event_delivery_results",
        "event_delivery_observations",
    ):
        with pytest.raises(psycopg.errors.CheckViolation), admin.transaction():
            admin.execute("DELETE FROM " + table)


def test_expired_lease_restarts_with_durable_unknown_and_old_lease_cannot_finish(store):
    outbox, admin = store
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox)
    first = sender.claim()
    admin.execute("UPDATE event_outbox SET lease_until=clock_timestamp()-interval '1 second'")
    second = sender.claim()
    assert first and second and second.ordinal == 2
    assert not sender.finish(first, DeliveryOutcome.REJECTED, reason="OLD_LEASE", http_status=422)
    assert sender.finish(
        second, DeliveryOutcome.RETRY, reason="CENTRAL_UNAVAILABLE", http_status=503
    )
    assert admin.execute("SELECT state FROM event_outbox").fetchone() == ("EXHAUSTED",)
    assert admin.execute("SELECT count(*) FROM event_delivery_results").fetchone() == (2,)
    assert _counts(admin) == (1, 1, 2) and sender.claim() is None


@pytest.mark.parametrize("attempts", [1, 2])
def test_late_sent_observation_preserves_expiry_and_current_claim(store, attempts):
    outbox, admin = store
    event = _event()
    outbox.accept(event, backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox, attempts)
    first = sender.claim()
    assert first
    admin.execute("UPDATE event_outbox SET lease_until=clock_timestamp()-interval '1 second'")
    replacement = sender.claim()
    if attempts == 1:
        assert replacement is None
        expected = ("EXHAUSTED", 1, None)
    else:
        assert replacement and replacement.ordinal == 2
        assert replacement.attempt_id != first.attempt_id
        expected = ("IN_FLIGHT", 2, replacement.attempt_id)
    assert (
        admin.execute("SELECT state,attempt_count,active_attempt FROM event_outbox").fetchone()
        == expected
    )
    assert admin.execute(
        "SELECT attempt_id,outcome,reason,http_status,backend_event_id FROM event_delivery_results"
    ).fetchall() == [(first.attempt_id, "UNKNOWN", "LEASE_EXPIRED", None, None)]
    before = _delivery_history(admin)
    response = {"reason": "ACCEPTED", "http_status": 202, "backend_event_id": "central-late"}
    assert not sender.finish(first, DeliveryOutcome.SENT, **response)
    observed = _delivery_history(admin)
    for table in ("event_outbox", "event_delivery_attempts", "event_delivery_results"):
        assert observed[table] == before[table]
    assert admin.execute(
        "SELECT attempt_id,edge_event_id,ordinal,outcome,reason,http_status,backend_event_id "
        "FROM event_delivery_observations"
    ).fetchall() == [
        (first.attempt_id, event.edge_event_id, 1, "SENT", "ACCEPTED", 202, "central-late")
    ]
    assert not sender.finish(first, DeliveryOutcome.SENT, **response)
    assert _delivery_history(admin) == observed
    for outcome, changes in (
        (DeliveryOutcome.SENT, {"reason": "DIFFERENT"}),
        (DeliveryOutcome.SENT, {"http_status": 200}),
        (DeliveryOutcome.SENT, {"backend_event_id": "central-other"}),
        (DeliveryOutcome.UNKNOWN, {"backend_event_id": None}),
    ):
        with pytest.raises(DeliveryResponseConflict):
            sender.finish(first, outcome, **(response | changes))
        assert _delivery_history(admin) == observed
    assert sender.claim() is None and _counts(admin) == (1, 1, 2)
    assert _delivery_history(admin) == observed


def test_concurrent_duplicate_finish_records_one_response_and_one_result(store):
    outbox, admin = store
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox)
    claim = sender.claim()
    assert claim
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                sender.finish,
                claim,
                DeliveryOutcome.SENT,
                reason="ACCEPTED",
                http_status=202,
                backend_event_id="central-1",
            )
            for _ in range(3)
        ]
        finished = [future.result(timeout=5) for future in futures]
    assert finished.count(True) == 1 and finished.count(False) == 2
    assert all(len(rows) == 1 for rows in _delivery_history(admin).values())
    assert admin.execute("SELECT state,attempt_count FROM event_outbox").fetchone() == ("SENT", 1)
    for table in ("event_delivery_results", "event_delivery_observations"):
        assert admin.execute(
            "SELECT outcome,reason,http_status,backend_event_id FROM " + table
        ).fetchone() == ("SENT", "ACCEPTED", 202, "central-1")


@pytest.mark.parametrize("invalid", ["missing_attempt", "missing_event", "cross_event", "ordinal"])
def test_finish_rejects_invalid_attempt_association_before_writing(store, invalid):
    outbox, admin = store
    sender = _delivery(outbox)
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    first = sender.claim()
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    second = sender.claim()
    assert first and second
    if invalid == "missing_attempt":
        claim = replace(first, attempt_id=uuid4())
    elif invalid == "missing_event":
        claim = replace(first, edge_event_id=str(uuid4()))
    elif invalid == "cross_event":
        claim = replace(first, edge_event_id=second.edge_event_id)
    else:
        claim = replace(first, ordinal=first.ordinal + 1)
    before = _delivery_history(admin)
    with pytest.raises(ValueError, match="does not match its persisted attempt"):
        sender.finish(
            claim,
            DeliveryOutcome.SENT,
            reason="ACCEPTED",
            http_status=202,
            backend_event_id="central-1",
        )
    assert _delivery_history(admin) == before
    assert _counts(admin) == (2, 2, 3)


@pytest.mark.parametrize("invalid", ["missing_attempt", "cross_event", "ordinal"])
def test_active_attempt_fk_rejects_invalid_owner_and_preserves_expiry_claim(store, invalid):
    outbox, admin = store
    sender = _delivery(outbox)
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    first = sender.claim()
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    second = sender.claim()
    assert first and second
    before = _delivery_history(admin)
    with pytest.raises(psycopg.errors.ForeignKeyViolation), admin.transaction():
        if invalid == "ordinal":
            admin.execute(
                "UPDATE event_outbox SET attempt_count=attempt_count+1 WHERE edge_event_id=%s",
                (first.edge_event_id,),
            )
        else:
            admin.execute(
                "UPDATE event_outbox SET active_attempt=%s WHERE edge_event_id=%s",
                (
                    uuid4() if invalid == "missing_attempt" else second.attempt_id,
                    first.edge_event_id,
                ),
            )
    assert _delivery_history(admin) == before
    other = admin.execute(
        "SELECT * FROM event_outbox WHERE edge_event_id=%s", (second.edge_event_id,)
    ).fetchone()
    admin.execute(
        "UPDATE event_outbox SET lease_until=clock_timestamp()-interval '1 second' "
        "WHERE edge_event_id=%s",
        (first.edge_event_id,),
    )
    replacement = sender.claim()
    assert replacement and replacement.edge_event_id == first.edge_event_id
    assert replacement.ordinal == 2 and replacement.attempt_id != first.attempt_id
    assert admin.execute(
        "SELECT state,attempt_count,active_attempt FROM event_outbox WHERE edge_event_id=%s",
        (first.edge_event_id,),
    ).fetchone() == ("IN_FLIGHT", 2, replacement.attempt_id)
    assert (
        admin.execute(
            "SELECT * FROM event_outbox WHERE edge_event_id=%s", (second.edge_event_id,)
        ).fetchone()
        == other
    )
    assert admin.execute("SELECT count(*) FROM event_delivery_attempts").fetchone() == (3,)
    assert admin.execute(
        "SELECT attempt_id,outcome,reason FROM event_delivery_results"
    ).fetchall() == [(first.attempt_id, "UNKNOWN", "LEASE_EXPIRED")]
    assert admin.execute("SELECT count(*) FROM event_delivery_observations").fetchone() == (0,)
    assert _counts(admin) == (2, 2, 3)


@pytest.mark.parametrize("disposition", ["active", "exhausted", "reclaimed", "finished"])
def test_authority_freeze_blocks_normal_late_and_duplicate_response_writes(store, disposition):
    outbox, admin = store
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox, attempts=1 if disposition == "exhausted" else 2)
    claim = sender.claim()
    assert claim
    response = {"reason": "ACCEPTED", "http_status": 202, "backend_event_id": "central-1"}
    if disposition in ("exhausted", "reclaimed"):
        admin.execute("UPDATE event_outbox SET lease_until=clock_timestamp()-interval '1 second'")
        replacement = sender.claim()
        assert (replacement is None) == (disposition == "exhausted")
    elif disposition == "finished":
        assert sender.finish(claim, DeliveryOutcome.SENT, **response)
    before = _delivery_history(admin)
    assert freeze_authority(outbox.database, outbox.authority) == 1
    with pytest.raises(AuthorityFenced):
        sender.finish(claim, DeliveryOutcome.SENT, **response)
    assert _delivery_history(admin) == before


@pytest.mark.parametrize("expired", [False, True])
def test_observation_commit_failure_cannot_return_owned_or_stale_finish(store, expired):
    outbox, admin = store
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox, attempts=1)
    claim = sender.claim()
    assert claim
    if expired:
        admin.execute("UPDATE event_outbox SET lease_until=clock_timestamp()-interval '1 second'")
        assert sender.claim() is None
    before = _delivery_history(admin)
    admin.execute(
        "CREATE FUNCTION reject_response_commit_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected response commit failure' "
        "USING ERRCODE='23514'; END $$"
    )
    admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_response_commit_test "
        "AFTER INSERT ON event_delivery_observations "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        "EXECUTE FUNCTION reject_response_commit_test()"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        sender.finish(
            claim,
            DeliveryOutcome.SENT,
            reason="ACCEPTED",
            http_status=202,
            backend_event_id="central-1",
        )
    assert _delivery_history(admin) == before


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"outcome": "SENT"}, TypeError),
        ({"reason": "raw response text"}, ValueError),
        ({"reason": "A" * 65}, ValueError),
        ({"http_status": True}, ValueError),
        ({"http_status": 99}, ValueError),
        ({"http_status": 600}, ValueError),
        ({"backend_event_id": None}, ValueError),
        ({"backend_event_id": ""}, ValueError),
        ({"backend_event_id": "x" * 129}, ValueError),
        ({"outcome": DeliveryOutcome.UNKNOWN}, ValueError),
    ],
)
def test_late_response_validation_cannot_persist_unclassified_or_unbounded_fields(
    store, changes, error
):
    outbox, admin = store
    outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    sender = _delivery(outbox, attempts=1)
    claim = sender.claim()
    assert claim
    admin.execute("UPDATE event_outbox SET lease_until=clock_timestamp()-interval '1 second'")
    assert sender.claim() is None
    before = _delivery_history(admin)
    response = {
        "outcome": DeliveryOutcome.SENT,
        "reason": "ACCEPTED",
        "http_status": 202,
        "backend_event_id": "central-1",
    } | changes
    with pytest.raises(error):
        sender.finish(claim, **response)
    assert _delivery_history(admin) == before


def test_admission_rejects_missing_or_incoherent_audit_ownership(
    postgres_product_sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime
):
    sandbox = postgres_product_sandbox
    budget = OutboxBudget(10, 1024)
    with pytest.raises(TypeError, match="requires the native audit runtime"):
        EventOutbox(sandbox.database, sandbox.authority, budget, audit_runtime=None)
    wrong_authority = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, AuthorityToken(sandbox.authority.generation, uuid4())),
        maximum_snapshot_age_sec=10,
    )
    with pytest.raises(ValueError, match="must share database and authority"):
        EventOutbox(sandbox.database, sandbox.authority, budget, audit_runtime=wrong_authority)
    other = PostgresDatabase(
        sandbox.dsn,
        sandbox.schema,
        PoolBudget(
            max_connections=1,
            max_waiting=1,
            acquire_timeout_sec=1,
            statement_timeout_ms=5000,
            lock_timeout_ms=3000,
            startup_timeout_sec=5,
        ),
    )
    try:
        with pytest.raises(ValueError, match="must share database and authority"):
            EventOutbox(other, sandbox.authority, budget, audit_runtime=audit_runtime)
    finally:
        other.close(timeout_sec=3.0)


@pytest.mark.parametrize("stage", ["unverified", "session_missing", "expired", "stopped"])
def test_new_acceptance_cannot_bypass_native_audit_admission(
    postgres_product_sandbox: ProductSandbox, audit_clock: _AuditClock, stage: str
):
    sandbox = postgres_product_sandbox
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=audit_clock,
    )
    if stage != "unverified":
        assert runtime.verify_once()
    if stage not in ("unverified", "session_missing"):
        assert runtime.start_session_once()
    if stage == "expired":
        audit_clock.value = 10.0
    if stage == "stopped":
        runtime.stop()
    outbox = EventOutbox(
        sandbox.database,
        sandbox.authority,
        OutboxBudget(10, 1_048_576),
        audit_runtime=runtime,
    )
    before = _counts(sandbox.admin)
    snapshot = RelaySnapshot(
        "snap-audit", "snapshots/snap-audit.jpg", "a" * 64, 20, "image/jpeg", _TIME
    )
    with pytest.raises(AuditRuntimeUnavailable):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True, snapshot=snapshot)
    assert _counts(sandbox.admin) == before
    assert sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (0,)
    assert not runtime.snapshot().ready


def test_recovery_publication_follows_complete_owned_return(store, monkeypatch):
    outbox, admin = store
    runtime = outbox.audit_runtime
    runtime.record_failure(OSError("injected prior failure"))
    assert runtime.verify_once()
    assert runtime.snapshot().eligible_to_attempt and not runtime.snapshot().ready
    publications = _publications(runtime, monkeypatch)
    transact = outbox.database.transact
    owned_returns = []

    def observe_owned_return(operation):
        result = transact(operation)
        owned_returns.append(result)
        assert publications == []
        assert not runtime.snapshot().ready
        assert _counts(admin) == (1, 1, 3)
        return result

    monkeypatch.setattr(outbox.database, "transact", observe_owned_return)
    accepted = outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert owned_returns == [accepted]
    assert publications == [("committed", None)]
    assert runtime.snapshot().ready
    assert admin.execute("SELECT action FROM audit_events ORDER BY audit_id").fetchall() == [
        (AuditAction.AUDIT_SESSION_START.value,),
        (AuditAction.RECOVERY_FENCE.value,),
        (AuditAction.RELAY_ALERT.value,),
    ]


@pytest.mark.parametrize("kind", ["unknown", "cancel", "exit"])
def test_injected_loss_after_owned_commit_never_publishes_success_or_replays(
    store, monkeypatch, kind
):
    """Inject after a real full owned return; not a real wire/pool-exit fault."""

    class Cancelled(BaseException):
        pass

    outbox, admin = store
    runtime = outbox.audit_runtime
    publications = _publications(runtime, monkeypatch)
    failure = {
        "unknown": CommitOutcomeUnknown(),
        "cancel": Cancelled("injected cancellation"),
        "exit": OSError("injected lost owned result"),
    }[kind]
    event = _event()
    transact = outbox.database.transact
    attempts = []

    def lose_owned_result(operation):
        attempts.append(True)
        transact(operation)
        raise failure

    monkeypatch.setattr(outbox.database, "transact", lose_owned_result)
    with pytest.raises(type(failure)) as raised:
        outbox.accept(event, backend_camera_id="hub-camera", forward=True)
    assert raised.value is failure
    assert attempts == [True] and publications == [("failed", failure)]
    assert _counts(admin) == (1, 1, 2)
    assert not runtime.snapshot().ready
    assert runtime.snapshot().indeterminate == (kind == "unknown")
    monkeypatch.setattr(outbox.database, "transact", transact)
    # A producer's explicit replay can observe the committed identity without
    # inventing another audit append or healing indeterminate runtime state.
    duplicate = outbox.accept(event, backend_camera_id=None, forward=False)
    assert duplicate.duplicate and duplicate.delivery_state == "PENDING"
    assert _counts(admin) == (1, 1, 2)
    assert publications == [("failed", failure)] and not runtime.snapshot().ready


def test_stale_verification_allows_only_explicit_duplicate_without_publication(
    store, audit_clock: _AuditClock, monkeypatch
):
    outbox, admin = store
    publications = _publications(outbox.audit_runtime, monkeypatch)
    event = _event()
    outbox.accept(event, backend_camera_id="hub-camera", forward=True)
    audit_clock.value = 10.0
    assert not outbox.audit_runtime.snapshot().ready
    duplicate = outbox.accept(event, backend_camera_id=None, forward=False)
    assert duplicate.duplicate and duplicate.delivery_state == "PENDING"
    assert publications == [("committed", None)]
    assert not outbox.audit_runtime.snapshot().ready
    with pytest.raises(AuditRuntimeUnavailable):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert _counts(admin) == (1, 1, 2)
    assert publications == [("committed", None)]


@pytest.mark.parametrize("invalid", [None, False, "not-a-publication"])
def test_missing_audit_receipt_aborts_before_commit(store, monkeypatch, invalid):
    outbox, admin = store
    runtime = outbox.audit_runtime
    publications = _publications(runtime, monkeypatch)
    # Deliberately broken collaborator: never an implementation or gate fallback.
    monkeypatch.setattr(runtime, "append_borrowed", lambda connection, event: invalid)
    with pytest.raises(AuditRuntimeUnavailable, match="event audit publication is invalid"):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert _counts(admin) == (0, 0, 1) and publications == []
    assert not runtime.snapshot().ready


@pytest.mark.parametrize("when", ["before_validation", "after_owned_return"])
def test_newer_failure_wins_after_known_event_commit(store, monkeypatch, when):
    outbox, admin = store
    runtime = outbox.audit_runtime
    publications = _publications(runtime, monkeypatch)
    transact = outbox.database.transact

    def fail_after_owned_return(operation):
        result = transact(operation)
        runtime.record_failure(OSError("newer failure"))
        return result

    if when == "before_validation":
        append = runtime.append_borrowed

        def invalidate_before_validation(connection, event):
            token = append(connection, event)
            runtime.record_failure(OSError("newer failure"))
            return token

        monkeypatch.setattr(runtime, "append_borrowed", invalidate_before_validation)
    else:
        monkeypatch.setattr(outbox.database, "transact", fail_after_owned_return)
    accepted = outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert not accepted.duplicate and accepted.delivery_state == "PENDING"
    assert _counts(admin) == (1, 1, 2)
    assert publications == [("committed", None)]
    assert not runtime.snapshot().ready
    assert runtime.snapshot().failure_code == "database_unavailable"


@pytest.mark.parametrize("kind", ["foreign_owner", "foreign_session", "consumed", "unregistered"])
def test_semantic_invalid_receipt_rolls_back_without_consuming_unrelated_work(
    store, audit_clock: _AuditClock, monkeypatch, kind
):
    outbox, admin = store
    runtime = outbox.audit_runtime
    unrelated = _pending_publication(runtime, "unrelated")
    other = None
    original_session = None
    if kind == "foreign_owner":
        # Two real runtimes are an adversarial ownership fixture, not an
        # application assembly or preseeded-session qualification shortcut.
        other = PostgresAuditRuntime(
            PostgresAuditStore(outbox.database, outbox.authority),
            maximum_snapshot_age_sec=10,
            clock=audit_clock,
        )
        assert other.verify_once() and other.start_session_once()
        candidate = _pending_publication(other, "foreign-owner")
    elif kind == "foreign_session":
        candidate = _pending_publication(runtime, "wrong-session")
        original_session = candidate._session
        candidate._session = AuditSession(uuid4().hex)
    elif kind == "consumed":
        candidate = _pending_publication(runtime, "consumed")
        assert runtime.publish_committed(candidate)
    else:
        # Matching fields and another live operation's pending count are not
        # proof that this receipt was ever minted by append_borrowed.
        candidate = PendingAuditPublication(runtime, unrelated._session, unrelated._revision)
    before = _counts(admin)
    publications = _publications(runtime, monkeypatch)
    monkeypatch.setattr(runtime, "append_borrowed", lambda connection, event: candidate)
    try:
        with pytest.raises(AuditRuntimeUnavailable, match="event audit publication is invalid"):
            outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
        assert _counts(admin) == before and publications == []
        assert not runtime.snapshot().ready
        runtime.validate_publication(unrelated)
        if other is not None:
            other.validate_publication(candidate)
            assert other.publish_committed(candidate)
            other.stop()
            assert other.close_session_once()
        elif original_session is not None:
            candidate._session = original_session
            runtime.validate_publication(candidate)
            assert not runtime.publish_committed(candidate)
        assert not runtime.publish_committed(unrelated)
        assert not runtime._pending
    finally:
        if original_session is not None:
            candidate._session = original_session


@pytest.mark.parametrize("kind", ["ordinary", "cancel", "unknown"])
def test_secondary_publication_rejection_preserves_owned_failure_and_other_work(
    store, monkeypatch, caplog, kind
):
    """Post-return injection tests precedence, not actual COMMIT/pool loss."""

    class Cancelled(BaseException):
        pass

    outbox, admin = store
    runtime = outbox.audit_runtime
    unrelated = _pending_publication(runtime, "unrelated")
    failure = {
        "ordinary": OSError("private-owned-error"),
        "cancel": Cancelled("private-owned-cancellation"),
        "unknown": CommitOutcomeUnknown(),
    }[kind]
    append = runtime.append_borrowed
    consume = runtime.publish_failed
    transact = outbox.database.transact
    publications = _publications(runtime, monkeypatch)
    captured, attempts = [], []

    def capture(connection, event):
        token = append(connection, event)
        captured.append(token)
        return token

    def invalidate_then_fail(operation):
        attempts.append(True)
        transact(operation)
        consume(captured[0], OSError("controlled prior consumption"))
        raise failure

    monkeypatch.setattr(runtime, "append_borrowed", capture)
    monkeypatch.setattr(outbox.database, "transact", invalidate_then_fail)
    with pytest.raises(type(failure)) as raised:
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert raised.value is failure
    assert attempts == [True] and publications == [("failed", failure)]
    assert _counts(admin) == (1, 1, 3)
    assert not runtime.snapshot().ready
    assert runtime.snapshot().indeterminate == (kind == "unknown")
    if kind == "unknown":
        assert runtime.snapshot().failure_code == "commit_outcome_unknown"
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.app.features.evidence.event_outbox"
    ]
    assert messages == ["event audit publication accounting failed after owned failure"]
    assert "private-owned" not in caplog.text
    runtime.validate_publication(unrelated)
    runtime.publish_failed(unrelated, failure)
    assert not runtime._pending
