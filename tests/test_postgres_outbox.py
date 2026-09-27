"""Real isolated PostgreSQL event acceptance, fencing, and retry conservation."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase
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

_DDL = Path(__file__).resolve().parents[1] / "backend/app/edge_db"
_TIME = "2026-09-27T04:00:00.000Z"


@pytest.fixture
def store():
    dsn = os.environ.get("SEEON_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("requires an isolated SEEON_TEST_POSTGRES_DSN; not admission evidence")
    schema = "seeon_outbox_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        database = None
        try:
            admin.execute(
                sql.SQL("SET search_path TO {},pg_catalog").format(sql.Identifier(schema))
            )
            with admin.transaction():
                admin.execute((_DDL / "postgres_product.sql").read_text(), prepare=False)
                admin.execute((_DDL / "postgres_delivery.sql").read_text(), prepare=False)
                token = AuthorityToken(1, uuid4())
                admin.execute(
                    "INSERT INTO deployment_authority VALUES (1,%s,%s,true,true)",
                    (token.generation, token.writer_token),
                )
            database = PostgresDatabase(dsn, schema, PoolBudget(3, 3, 2.0, 4000, 2000, 3.0))
            database.start()
            yield EventOutbox(database, token, OutboxBudget(100, 1_048_576)), admin
        finally:
            if database is not None:
                database.close()
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


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


def test_commit_persists_incident_obligation_audit_before_receipt_and_dedupes(store):
    outbox, admin = store
    event = _event()
    receipt = outbox.accept(event, backend_camera_id="hub-camera", forward=True)
    assert receipt.edge_event_id == event.edge_event_id and not receipt.duplicate
    assert receipt.delivery_state == "PENDING" and _counts(admin) == (1, 1, 1)
    duplicate = outbox.accept(event, backend_camera_id="changed-mapping", forward=False)
    assert duplicate.duplicate and duplicate.delivery_state == "PENDING"
    assert _counts(admin) == (1, 1, 1)
    assert admin.execute("SELECT backend_camera_id FROM event_outbox").fetchone() == ("hub-camera",)
    with pytest.raises(EventIdentityConflict):
        outbox.accept(replace(event, probability=0.9), backend_camera_id="hub-camera", forward=True)
    assert _counts(admin) == (1, 1, 1)


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
    with pytest.raises(psycopg.Error):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True, snapshot=snapshot)
    assert _counts(admin) == (0, 0, 0)
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
    assert _counts(admin) == (1, 1, 1)


def test_capacity_refusal_cannot_ack_or_remove_previous_acceptance(store):
    outbox, admin = store
    limited = EventOutbox(outbox.database, outbox.authority, OutboxBudget(1, 1_048_576))
    event = _event()
    limited.accept(event, backend_camera_id=None, forward=False)
    with pytest.raises(OutboxCapacityExceeded):
        limited.accept(_event(), backend_camera_id=None, forward=False)
    assert limited.accept(event, backend_camera_id=None, forward=False).duplicate
    assert _counts(admin) == (1, 1, 1)
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
    assert _counts(admin) == (1, 1, 1)


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


def test_deferred_commit_rejection_cannot_release_event_receipt(store):
    outbox, admin = store
    admin.execute(
        "CREATE FUNCTION reject_commit_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected deferred failure' USING ERRCODE='23514'; END $$"
    )
    admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_commit_test AFTER INSERT ON event_outbox "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_commit_test()"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        outbox.accept(_event(), backend_camera_id="hub-camera", forward=True)
    assert _counts(admin) == (0, 0, 0)


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
    assert _counts(admin) == (1, 1, 1) and sender.claim() is None


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
    assert sender.claim() is None and _counts(admin) == (1, 1, 1)
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
    assert _counts(admin) == (2, 2, 2)


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
    assert _counts(admin) == (2, 2, 2)


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
