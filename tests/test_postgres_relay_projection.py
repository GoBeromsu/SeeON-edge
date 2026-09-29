"""Native snapshot-owner contracts; not HTTP/lifespan or transport qualification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import Event
from time import monotonic
from uuid import uuid4

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import (
    AuthorityFenced,
    AuthorityToken,
    freeze_authority,
    require_authority,
)
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    empty_detail,
)
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.store import AuditEvent, utc_now
from backend.app.features.evidence.event_outbox import (
    EventIdentityConflict,
    EventOutbox,
    OutboxBudget,
)
from backend.app.features.evidence.postgres_relay_projection import PostgresRelayEvidenceProjection
from backend.app.features.evidence.relay_projection import (
    RelayEvent,
    RelayEvidenceProjectionConflict,
    RelayEvidenceProjectionError,
    RelayEvidenceProjectionMissingEvent,
    RelaySnapshot,
)

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.000Z"
_EPOCH = "1970-01-01T00:00:00Z"


class _Call:
    def __init__(self, operation):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(operation)

    def result(self):
        return self.future.result(timeout=2)

    def close(self):
        _, pending = wait((self.future,), timeout=2)
        self.executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "snapshot operation did not terminate"


@pytest.fixture
def audit_runtime(postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=lambda: 0.0,
    )
    assert runtime.verify_once() and runtime.start_session_once()
    return runtime


@pytest.fixture
def outbox(postgres_product_sandbox, audit_runtime):
    sandbox = postgres_product_sandbox
    return EventOutbox(
        sandbox.database,
        sandbox.authority,
        OutboxBudget(10, 1_048_576),
        audit_runtime=audit_runtime,
    )


def _event(event_id="evt-1"):
    return RelayEvent(event_id, "FALL", 0.8, _TIME, "camera-1", "facility-1", None, None, None)


@pytest.fixture
def projection(postgres_product_sandbox, outbox):
    sandbox = postgres_product_sandbox
    outbox.accept(_event(), backend_camera_id=None, forward=False)
    return PostgresRelayEvidenceProjection(sandbox.database, sandbox.authority)


def _attach(projection, *, after_write=None, **changes):
    values = {
        "edge_event_id": "evt-1",
        "snapshot_id": "snapshot-1",
        "sha256": "a" * 64,
        "media_reference": "snapshots/snapshot-1.jpg",
        "size_bytes": 12,
        "mime_type": "image/jpeg",
    }
    values.update(changes)
    return projection.attach_snapshot(**values, after_write=after_write)


def _dispose(projection, *, after_write=None, **changes):
    values = {
        "edge_event_id": "evt-1",
        "snapshot_id": "snapshot-1",
        "disposition": "UNAVAILABLE",
        "reason": "not-captured",
    }
    values.update(changes)
    return projection.record_snapshot_disposition(**values, after_write=after_write)


def _history(connection):
    return {
        table: connection.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall()
        for table in ("incidents", "artifacts", "event_outbox", "audit_events")
    }


def _audit_callback(runtime, kind, tokens):
    action = (
        AuditAction.RELAY_SNAPSHOT_ATTACHMENT
        if kind == "attach"
        else AuditAction.RELAY_SNAPSHOT_DISPOSITION
    )

    def append(connection):
        token = runtime.append_borrowed(
            connection,
            AuditEvent(
                occurred_at=utc_now(),
                actor_id="worker-relay",
                action=action,
                target_id="snapshot-1",
                detail=empty_detail(action),
                actor_type=AuditActorType.SERVICE,
                auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
            ),
        )
        tokens.append(token)
        runtime.validate_publication(token)

    return append


@pytest.mark.parametrize("kind", ["attach", "dispose"])
def test_callback_is_atomic_once_even_for_matching_facts_and_returns_after_pool_exit(
    projection, postgres_product_sandbox, audit_runtime, monkeypatch, kind
):
    sandbox = postgres_product_sandbox
    operation = _attach if kind == "attach" else _dispose
    before = _history(sandbox.admin)
    trace, tokens = [], []
    append = _audit_callback(audit_runtime, kind, tokens)
    pool_connection = sandbox.database._pool.connection

    @contextmanager
    def observed_exit(*args, **kwargs):
        with pool_connection(*args, **kwargs) as connection:
            yield connection
        trace.append("pool-exit")

    monkeypatch.setattr(sandbox.database._pool, "connection", observed_exit)
    for existing in (0, 1):

        def callback(connection, existing=existing):
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
            assert sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (existing,)
            append(connection)
            assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
                2 + existing,
            )
            trace.append("callback")

        assert operation(projection, after_write=callback) is None
        trace.append("returned")
        assert trace == ["callback", "pool-exit", "returned"] * (existing + 1)
        assert len(tokens) == existing + 1
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (
            3 + existing,
        )
        assert audit_runtime.publish_committed(tokens[-1])
        if existing == 0:
            first_artifact = _history(sandbox.admin)["artifacts"]
    after = _history(sandbox.admin)
    assert after["artifacts"] == first_artifact
    assert after["incidents"] == before["incidents"]
    assert after["event_outbox"] == before["event_outbox"]
    assert sandbox.admin.execute("SELECT captured_at FROM artifacts").fetchone() == (_EPOCH,)


@pytest.mark.parametrize("kind", ["attach", "dispose"])
def test_missing_incident_does_not_call_hook_or_persist(projection, postgres_product_sandbox, kind):
    before = _history(postgres_product_sandbox.admin)
    calls = []
    operation = _attach if kind == "attach" else _dispose
    with pytest.raises(RelayEvidenceProjectionMissingEvent):
        operation(
            projection, edge_event_id="absent", after_write=lambda connection: calls.append(1)
        )
    assert calls == [] and _history(postgres_product_sandbox.admin) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("snapshot_id", "another"),
        ("sha256", "b" * 64),
        ("media_reference", "snapshots/another.jpg"),
        ("size_bytes", 13),
        ("mime_type", "image/png"),
    ],
)
def test_attachment_identity_conflicts_do_not_call_hook(
    projection, postgres_product_sandbox, field, value
):
    _attach(projection)
    before = _history(postgres_product_sandbox.admin)
    calls = []
    with pytest.raises(RelayEvidenceProjectionConflict):
        _attach(projection, **{field: value}, after_write=lambda connection: calls.append(1))
    assert calls == [] and _history(postgres_product_sandbox.admin) == before


@pytest.mark.parametrize("first", ["attach", "dispose"])
def test_terminal_available_and_unavailable_facts_cannot_replace_each_other(
    projection, postgres_product_sandbox, first
):
    initial, conflicting = (_attach, _dispose) if first == "attach" else (_dispose, _attach)
    initial(projection)
    before = _history(postgres_product_sandbox.admin)
    with pytest.raises(RelayEvidenceProjectionConflict):
        conflicting(projection)
    assert _history(postgres_product_sandbox.admin) == before


def test_disposition_identity_uses_bounded_reason_not_unstored_snapshot_id(
    projection, postgres_product_sandbox
):
    reason = "원인" * 40
    _dispose(projection, reason=reason)
    before = _history(postgres_product_sandbox.admin)
    calls = []
    _dispose(
        projection,
        reason=reason,
        snapshot_id="other",
        after_write=lambda connection: calls.append(1),
    )
    assert calls == [1] and _history(postgres_product_sandbox.admin) == before
    assert postgres_product_sandbox.admin.execute("SELECT reason FROM artifacts").fetchone() == (
        ("UNAVAILABLE:" + reason)[:64],
    )
    with pytest.raises(RelayEvidenceProjectionConflict):
        _dispose(projection, reason="different")
    assert _history(postgres_product_sandbox.admin) == before


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"media_reference": "../outside.jpg"}, RelayEvidenceProjectionError),
        ({"size_bytes": 0}, RelayEvidenceProjectionError),
        ({"sha256": "not-a-hash"}, psycopg.errors.CheckViolation),
    ],
)
def test_invalid_snapshot_rolls_back_without_callback(
    projection, postgres_product_sandbox, changes, error
):
    before = _history(postgres_product_sandbox.admin)
    calls = []
    with pytest.raises(error):
        _attach(projection, **changes, after_write=lambda connection: calls.append(1))
    assert calls == [] and _history(postgres_product_sandbox.admin) == before


@pytest.mark.parametrize("kind", ["attach", "dispose"])
@pytest.mark.parametrize("fence", ["frozen", "generation", "writer"])
def test_all_snapshot_admission_requires_explicit_current_authority(
    projection, postgres_product_sandbox, kind, fence
):
    sandbox = postgres_product_sandbox
    authority = sandbox.authority
    if fence == "frozen":
        freeze_authority(sandbox.database, authority)
    else:
        authority = AuthorityToken(
            authority.generation + (fence == "generation"),
            uuid4() if fence == "writer" else authority.writer_token,
        )
        projection = PostgresRelayEvidenceProjection(sandbox.database, authority)
    before = _history(sandbox.admin)
    calls = []
    operation = _attach if kind == "attach" else _dispose
    with pytest.raises(AuthorityFenced):
        operation(projection, after_write=lambda connection: calls.append(1))
    assert calls == [] and _history(sandbox.admin) == before


@pytest.mark.parametrize("kind", ["attach", "dispose"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_callback_failure_rolls_back_artifact_and_borrowed_audit(
    projection, postgres_product_sandbox, audit_runtime, kind, cancelled
):
    class Cancelled(BaseException):
        pass

    failure = Cancelled() if cancelled else OSError("callback failed")
    before = _history(postgres_product_sandbox.admin)
    tokens = []
    append = _audit_callback(audit_runtime, kind, tokens)

    def fail(connection):
        append(connection)
        raise failure

    operation = _attach if kind == "attach" else _dispose
    with pytest.raises(type(failure)) as raised:
        operation(projection, after_write=fail)
    assert raised.value is failure and len(tokens) == 1
    assert _history(postgres_product_sandbox.admin) == before
    audit_runtime.publish_failed(tokens[0], failure)
    assert not audit_runtime.snapshot().ready


@pytest.mark.parametrize("kind", ["attach", "dispose"])
def test_deferred_commit_rejection_rolls_back_callback_and_artifact(
    projection, postgres_product_sandbox, audit_runtime, kind
):
    admin = postgres_product_sandbox.admin
    before = _history(admin)
    admin.execute(
        "CREATE FUNCTION reject_snapshot_commit_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected snapshot commit failure' "
        "USING ERRCODE='23514'; END $$"
    )
    admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_snapshot_commit_test AFTER INSERT ON artifacts "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        "EXECUTE FUNCTION reject_snapshot_commit_test()"
    )
    tokens = []
    operation = _attach if kind == "attach" else _dispose
    with pytest.raises(psycopg.errors.CheckViolation) as raised:
        operation(projection, after_write=_audit_callback(audit_runtime, kind, tokens))
    assert len(tokens) == 1 and _history(admin) == before
    audit_runtime.publish_failed(tokens[0], raised.value)
    assert not audit_runtime.snapshot().ready


@pytest.mark.parametrize("kind", ["attach", "dispose"])
@pytest.mark.parametrize("outcome", ["ordinary", "cancel", "unknown"])
def test_owner_return_loss_preserves_failure_without_replay(
    projection, postgres_product_sandbox, monkeypatch, kind, outcome
):
    """Inject after real COMMIT/exit; this does not simulate wire/pool loss."""

    class Cancelled(BaseException):
        pass

    failure = {
        "ordinary": OSError("owned return failed"),
        "cancel": Cancelled(),
        "unknown": CommitOutcomeUnknown(),
    }[outcome]
    transact = projection.database.transact
    attempts, calls = [], []

    def lose(operation):
        attempts.append(1)
        transact(operation)
        raise failure

    monkeypatch.setattr(projection.database, "transact", lose)
    operation = _attach if kind == "attach" else _dispose
    with pytest.raises(type(failure)) as raised:
        operation(projection, after_write=lambda connection: calls.append(1))
    assert raised.value is failure and attempts == [1] and calls == [1]
    assert postgres_product_sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone() == (
        1,
    )


@pytest.mark.parametrize("second", ["matching", "conflicting", "disposition"])
def test_concurrent_decisions_serialize_on_the_real_incident_lock(
    projection, postgres_product_sandbox, second
):
    entered, release = Event(), Event()
    pids, calls = [], []

    def hold(connection):
        pids.append(connection.info.backend_pid)
        entered.set()
        assert release.wait(2), "writer was not released"

    first = _Call(lambda: _attach(projection, after_write=hold))
    other = None
    try:
        assert entered.wait(1)

        def callback(connection):
            calls.append(1)

        def operation():
            if second == "disposition":
                return _dispose(projection, after_write=callback)
            return _attach(
                projection,
                sha256=("b" if second == "conflicting" else "a") * 64,
                after_write=callback,
            )

        other = _Call(operation)
        deadline, pacing = monotonic() + 1.5, Event()
        while not postgres_product_sandbox.admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)))",
            (pids[0],),
        ).fetchone()[0]:
            assert monotonic() < deadline, "companion never waited for the real incident lock"
            pacing.wait(0.01)
        assert calls == [] and not first.future.done() and not other.future.done()
        release.set()
        assert first.result() is None
        if second == "matching":
            assert other.result() is None and calls == [1]
        else:
            with pytest.raises(RelayEvidenceProjectionConflict):
                other.result()
            assert calls == []
        assert postgres_product_sandbox.admin.execute(
            "SELECT content_sha256,state FROM artifacts"
        ).fetchall() == [("a" * 64, "AVAILABLE")]
    finally:
        release.set()
        try:
            first.close()
        finally:
            if other is not None:
                other.close()


def test_initial_outbox_and_late_attachment_share_snapshot_identity(
    postgres_product_sandbox, outbox
):
    snapshot = RelaySnapshot(
        "snapshot-1", "snapshots/snapshot-1.jpg", "a" * 64, 12, "image/jpeg", _EPOCH
    )
    outbox.accept(_event(), backend_camera_id=None, forward=False, snapshot=snapshot)
    sandbox = postgres_product_sandbox
    projection = PostgresRelayEvidenceProjection(sandbox.database, sandbox.authority)
    before = _history(sandbox.admin)
    calls = []
    _attach(projection, after_write=lambda connection: calls.append(1))
    assert calls == [1] and _history(sandbox.admin) == before
    with pytest.raises(RelayEvidenceProjectionConflict):
        _attach(projection, sha256="b" * 64)
    with pytest.raises(EventIdentityConflict):
        outbox.accept(
            _event(),
            backend_camera_id=None,
            forward=False,
            snapshot=RelaySnapshot("snapshot-1", snapshot.path, "b" * 64, 12, "image/jpeg", _EPOCH),
        )
    assert _history(sandbox.admin) == before


def test_outbox_preserves_conflict_boundary_for_a_preprojected_incident(
    postgres_product_sandbox, outbox
):
    sandbox = postgres_product_sandbox

    def seed_existing_fact(connection):
        # A preprojected incident has no new outbox envelope to short-circuit
        # comparison. This real row exercises the extracted SQL's conflict path.
        require_authority(connection, sandbox.authority)
        connection.execute(
            "INSERT INTO incidents (incident_id,edge_event_id,facility_id,camera_id,"
            "event_type,probability,detected_at,lifecycle_state,provenance_state,"
            "provenance_missing_reason,review_version,revision,created_at,updated_at) "
            "VALUES ('incident:evt-1','evt-1','facility-1','camera-1','FALL',0.8,%s,"
            "'OPEN','MISSING','NOT_RECORDED',0,1,%s,%s)",
            (_TIME, _TIME, _TIME),
        )

    sandbox.database.transact(seed_existing_fact)
    projection = PostgresRelayEvidenceProjection(sandbox.database, sandbox.authority)
    _attach(projection)
    before = _history(sandbox.admin)
    with pytest.raises(
        EventIdentityConflict, match="snapshot identity conflicts with accepted content"
    ):
        outbox.accept(
            _event(),
            backend_camera_id=None,
            forward=False,
            snapshot=RelaySnapshot(
                "snapshot-1", "snapshots/snapshot-1.jpg", "b" * 64, 12, "image/jpeg", _EPOCH
            ),
        )
    assert _history(sandbox.admin) == before
