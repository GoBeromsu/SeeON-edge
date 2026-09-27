"""Real PostgreSQL audit append/lifecycle components, not API or verifier qualification.

The shared sandbox requires explicit SEEON_TEST_POSTGRES_DSN and owns its isolated
namespace and pool. An absent DSN skip is not qualification evidence.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row, tuple_row

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.functions import audit_record_hash
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit import postgres_sessions
from backend.app.features.audit.catalog import (
    AuditAction,
    AuditActorType,
    AuditAuthMechanism,
    camera_probe_detail,
    empty_detail,
    session_detail,
)
from backend.app.features.audit.postgres_sessions import (
    append_with_recovery,
    close_session,
    start_session,
)
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.sessions import AuditSession
from backend.app.features.audit.store import AuditEvent, AuditRecord
from backend.app.features.audit.verification import GENESIS_HASH, AuditVerificationError

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:00:00.123Z"
_OPERATIONS = ("append", "batch", "empty_batch", "start", "close", "recovery")
_ADDED_ROWS = {"append": 1, "batch": 2, "empty_batch": 0, "start": 2, "close": 1, "recovery": 2}


@pytest.fixture
def audit_store(postgres_product_sandbox: ProductSandbox) -> PostgresAuditStore:
    return PostgresAuditStore(postgres_product_sandbox.database, postgres_product_sandbox.authority)


def _event(target_id: str = "protected") -> AuditEvent:
    return AuditEvent(
        occurred_at=_TIME,
        actor_id="test-operator",
        action=AuditAction.AUDIT_LIST,
        target_id=target_id,
        detail=empty_detail(AuditAction.AUDIT_LIST),
    )


def _session_event(action: AuditAction, session: AuditSession) -> AuditEvent:
    return AuditEvent(
        occurred_at=_TIME,
        actor_id="audit-readiness",
        action=action,
        target_id=session.session_id,
        detail=session_detail(action),
        actor_type=AuditActorType.SYSTEM,
        auth_mechanism=AuditAuthMechanism.INTERNAL,
    )


def _history(sandbox: ProductSandbox) -> list[dict]:
    with sandbox.admin.cursor(row_factory=dict_row) as cursor:
        return cursor.execute("SELECT * FROM audit_events ORDER BY audit_id").fetchall()


def _identities(rows: list[dict]) -> list[tuple[str, str]]:
    return [(row["action"], row["target_id"]) for row in rows]


def _assert_chain(rows: list[dict]) -> None:
    previous = GENESIS_HASH
    previous_id = 0
    for row in rows:
        assert row["audit_id"] > previous_id
        assert row["previous_hash"] == previous
        payload = {
            key: value for key, value in row.items() if key not in {"audit_id", "record_hash"}
        }
        assert row["record_hash"] == audit_record_hash(
            previous, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        previous = row["record_hash"]
        previous_id = row["audit_id"]


def _owned_operation(store: PostgresAuditStore, operation: str, session: AuditSession) -> object:
    if operation == "append":
        return store.append(_event())
    if operation == "batch":
        return store.append_batch((_event("batch-first"), _event("batch-second")))
    if operation == "empty_batch":
        return store.append_batch(())
    if operation == "start":
        return start_session(store)
    if operation == "close":
        return close_session(store, session)
    if operation == "recovery":
        return append_with_recovery(store, _event(), session, "database_unavailable")
    raise AssertionError("unknown test operation")


def _caller_operation(
    store: PostgresAuditStore,
    operation: str,
    session: AuditSession,
    connection: psycopg.Connection,
) -> object:
    if operation == "append":
        return store.append(_event(), connection=connection)
    if operation == "start":
        return start_session(store, connection=connection)
    if operation == "recovery":
        return append_with_recovery(
            store, _event(), session, "database_unavailable", connection=connection
        )
    raise AssertionError("unknown caller-owned test operation")


def _wait_for_lock(sandbox: ProductSandbox, pid: int, *, advisory: bool = False) -> None:
    deadline = monotonic() + 1.5
    pacing = Event()
    while True:
        row = sandbox.database.read(
            lambda connection: connection.execute(
                "SELECT wait_event FROM pg_stat_activity WHERE pid=%s AND wait_event_type='Lock'",
                (pid,),
            ).fetchone()
        )
        if row is not None and (not advisory or row == ("advisory",)):
            return
        assert monotonic() < deadline, "caller never waited on the real transaction lock"
        pacing.wait(0.005)


def test_linear_append_returns_committed_canonical_records(
    postgres_product_sandbox: ProductSandbox, audit_store: PostgresAuditStore
) -> None:
    events = (_event("first"), _event("second"), _event("third"))
    records = [audit_store.append(event) for event in events]
    rows = _history(postgres_product_sandbox)
    assert len(rows) == len(records) == 3
    for event, record, row in zip(events, records, rows, strict=True):
        assert isinstance(record, AuditRecord)
        assert record.audit_id == row["audit_id"]
        assert record.occurred_at == row["occurred_at"] == event.occurred_at
        assert record.recorded_at == row["recorded_at"]
        assert record.actor_id == row["actor_id"] == event.actor_id
        assert record.action == row["action"] == event.action
        assert record.target_id == row["target_id"] == event.target_id
        assert record.target_type == row["target_type"] == "audit"
        assert record.detail == event.detail
        assert row["detail_json"] == event.detail.json
        assert record.previous_hash == row["previous_hash"]
        assert record.record_hash == row["record_hash"]
    _assert_chain(rows)


def test_unicode_nulls_and_detail_text_keep_the_canonical_envelope(
    postgres_product_sandbox: ProductSandbox, audit_store: PostgresAuditStore
) -> None:
    events = [
        replace(
            _event('대상 "침대"\\' + suffix),
            actor_id="운영자 " + suffix,
            action=AuditAction.CAMERA_PROBE,
            detail=camera_probe_detail(True, None),
        )
        for suffix in ("é", "é")
    ]
    records = audit_store.append_batch(events)
    rows = _history(postgres_product_sandbox)
    assert isinstance(records, tuple) and len(records) == 2
    for event, record, row in zip(events, records, rows, strict=True):
        assert row["actor_id"] == event.actor_id
        assert row["target_id"] == record.target_id == event.target_id
        assert row["target_type"] == record.target_type == "camera"
        assert row["detail_json"] == '{"error_class":null,"ok":true,"version":1}'
        assert all(
            row[key] is None for key in ("reason", "request_id", "interaction_id", "hold_reference")
        )
        assert row["record_hash"] == record.record_hash
    assert rows[0]["actor_id"] != rows[1]["actor_id"]
    _assert_chain(rows)


@pytest.mark.parametrize("caller_owned", [False, True])
def test_action_detail_mismatch_never_inserts(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    caller_owned: bool,
) -> None:
    sandbox = postgres_product_sandbox
    event = replace(_event(), detail=empty_detail(AuditAction.CONNECTION_UPDATE))
    if caller_owned:
        with sandbox.admin.transaction():
            with pytest.raises(AuditVerificationError, match="action/detail variants"):
                audit_store.append(event, connection=sandbox.admin)
            assert sandbox.admin.info.transaction_status is TransactionStatus.INTRANS
    else:
        with pytest.raises(AuditVerificationError, match="action/detail variants"):
            audit_store.append(event)
    assert _history(sandbox) == []


def test_concurrent_appends_form_one_chain_on_distinct_borrowed_connections(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = postgres_product_sandbox
    transact = sandbox.database.transact
    ready = Barrier(3)
    pids: Queue[int] = Queue()

    def synchronized(callback):
        def run(connection):
            pids.put(connection.info.backend_pid)
            ready.wait(timeout=3)
            return callback(connection)

        return transact(run)

    monkeypatch.setattr(sandbox.database, "transact", synchronized)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(audit_store.append, _event(f"append-{index}")) for index in range(3)]
        records = [future.result(timeout=5) for future in futures]
    assert len({pids.get_nowait() for _ in range(3)}) == 3
    rows = _history(sandbox)
    assert len(rows) == 3
    assert {record.audit_id for record in records} == {row["audit_id"] for row in rows}
    assert {row["target_id"] for row in rows} == {f"append-{index}" for index in range(3)}
    _assert_chain(rows)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_owned_mutations_use_one_callback_and_release_only_after_commit(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    transact = sandbox.database.transact
    calls = []
    candidates = []

    def observed(callback):
        calls.append("transact")

        def run(connection):
            calls.append("callback")
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert connection.row_factory is tuple_row
            assert connection.execute("SHOW transaction_isolation").fetchone() == (
                "read committed",
            )
            candidate = callback(connection)
            candidates.append(candidate)
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert _history(sandbox) == before
            return candidate

        return transact(run)

    def no_reload(_callback):
        pytest.fail("mutation must not open a second reload")

    monkeypatch.setattr(sandbox.database, "transact", observed)
    monkeypatch.setattr(sandbox.database, "read", no_reload)
    result = _owned_operation(audit_store, operation, session)
    assert calls == ["transact", "callback"]
    assert candidates == [result]
    rows = _history(sandbox)
    assert rows[: len(before)] == before
    assert len(rows) == len(before) + _ADDED_ROWS[operation]
    if operation == "batch":
        assert isinstance(result, tuple)
        assert [record.target_id for record in result] == ["batch-first", "batch-second"]
        assert [record.audit_id for record in result] == [row["audit_id"] for row in rows[1:]]
    elif operation == "empty_batch":
        assert result == ()
    _assert_chain(rows)


@pytest.mark.parametrize("operation", ["append", "start", "recovery"])
@pytest.mark.parametrize("committed", [False, True], ids=["rollback", "commit"])
def test_caller_owned_results_are_tentative_and_atomic_with_product_writes(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    committed: bool,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    transact = sandbox.database.transact
    tentative = []

    def no_owned_transaction(_callback):
        pytest.fail("caller-owned audit must not open another database transaction")

    monkeypatch.setattr(sandbox.database, "transact", no_owned_transaction)
    monkeypatch.setattr(sandbox.database, "read", no_owned_transaction)

    def product_write(connection):
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        candidate = _caller_operation(audit_store, operation, session, connection)
        tentative.append(candidate)
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert _history(sandbox) == before
        assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (0,)
        if not committed:
            raise RuntimeError("injected product failure")
        return candidate

    if committed:
        assert transact(product_write) == tentative[0]
    else:
        with pytest.raises(RuntimeError, match="injected product failure"):
            transact(product_write)
    assert len(tentative) == 1
    rows = _history(sandbox)
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (
        int(committed),
    )
    if committed:
        assert len(rows) == len(before) + _ADDED_ROWS[operation]
        candidate = tentative[0]
        if isinstance(candidate, AuditRecord):
            assert rows[-1]["audit_id"] == candidate.audit_id
            assert rows[-1]["record_hash"] == candidate.record_hash
        else:
            assert isinstance(candidate, AuditSession)
            assert rows[-1]["target_id"] == candidate.session_id
    else:
        assert rows == before
    _assert_chain(rows)


@pytest.mark.parametrize("operation", ["append", "start", "recovery"])
@pytest.mark.parametrize("aborted", [False, True], ids=["idle", "aborted"])
def test_caller_owned_operations_require_an_active_successful_transaction(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    operation: str,
    aborted: bool,
) -> None:
    sandbox = postgres_product_sandbox
    session = AuditSession(uuid4().hex)
    if aborted:
        with sandbox.admin.transaction():
            with pytest.raises(psycopg.errors.DivisionByZero):
                sandbox.admin.execute("SELECT 1/0")
            assert sandbox.admin.info.transaction_status is TransactionStatus.INERROR
            with pytest.raises(AuditVerificationError, match="active transaction"):
                _caller_operation(audit_store, operation, session, sandbox.admin)
    else:
        assert sandbox.admin.info.transaction_status is TransactionStatus.IDLE
        with pytest.raises(AuditVerificationError, match="active transaction"):
            _caller_operation(audit_store, operation, session, sandbox.admin)
    assert _history(sandbox) == []


@pytest.mark.parametrize("failure", ["mismatch", "constraint"])
def test_batch_failure_rolls_back_the_whole_group_and_preserves_the_tail(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    failure: str,
) -> None:
    sandbox = postgres_product_sandbox
    original = audit_store.append(_event("existing"))
    before = _history(sandbox)
    if failure == "mismatch":
        invalid = replace(_event("bad"), detail=empty_detail(AuditAction.CONNECTION_UPDATE))
        error = AuditVerificationError
    else:
        invalid = replace(_event("bad"), actor_id="")
        error = psycopg.errors.CheckViolation
    with pytest.raises(error):
        audit_store.append_batch((_event("rolled-back"), invalid, _event("unreached")))
    assert _history(sandbox) == before
    successor = audit_store.append(_event("after-rollback"))
    assert successor.previous_hash == original.record_hash
    assert [row["target_id"] for row in _history(sandbox)] == ["existing", "after-rollback"]
    _assert_chain(_history(sandbox))


@pytest.mark.parametrize(
    ("operation", "caller_owned"),
    [(operation, False) for operation in _OPERATIONS]
    + [(operation, True) for operation in ("append", "start", "recovery")],
)
def test_frozen_authority_precedes_lifecycle_reads_and_even_idempotent_bodies(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    caller_owned: bool,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    append_with_recovery(audit_store, _event("seed"), session, "database_unavailable")
    close_session(audit_store, session)
    before = _history(sandbox)
    freeze_authority(sandbox.database, sandbox.authority)
    execute = psycopg.Connection.execute

    def refuse_body(connection, query, *args, **kwargs):
        if isinstance(query, str) and "audit_events" in query:
            pytest.fail("audit body ran before authority admission")
        return execute(connection, query, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "execute", refuse_body)
        if caller_owned:
            with sandbox.admin.transaction():
                with pytest.raises(AuthorityFenced):
                    _caller_operation(audit_store, operation, session, sandbox.admin)
        else:
            with pytest.raises(AuthorityFenced):
                _owned_operation(audit_store, operation, session)
    assert _history(sandbox) == before


@pytest.mark.parametrize("operation", ["append", "batch", "recovery"])
def test_frozen_authority_precedes_event_validation(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    invalid = replace(_event(), detail=empty_detail(AuditAction.CONNECTION_UPDATE))
    freeze_authority(sandbox.database, sandbox.authority)
    with pytest.raises(AuthorityFenced):
        if operation == "append":
            audit_store.append(invalid)
        elif operation == "batch":
            audit_store.append_batch((_event("first"), invalid))
        else:
            append_with_recovery(audit_store, invalid, session, "database_unavailable")
    assert _history(sandbox) == before


@pytest.mark.parametrize("wrong_identity", ["generation", "writer"])
def test_stale_writer_cannot_append_or_publish_a_batch(
    postgres_product_sandbox: ProductSandbox, wrong_identity: str
) -> None:
    sandbox = postgres_product_sandbox
    authority = AuthorityToken(
        sandbox.authority.generation + int(wrong_identity == "generation"),
        uuid4() if wrong_identity == "writer" else sandbox.authority.writer_token,
    )
    stale = PostgresAuditStore(sandbox.database, authority)
    with pytest.raises(AuthorityFenced):
        stale.append(_event())
    with pytest.raises(AuthorityFenced):
        stale.append_batch((_event("first"), _event("second")))
    assert _history(sandbox) == []


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_authority_lock_is_held_until_the_actual_owned_commit(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    transact = sandbox.database.transact
    admitted, release = Event(), Event()
    pids: Queue[int] = Queue()

    def observed(callback):
        def run(connection):
            pids.put(connection.info.backend_pid)
            candidate = callback(connection)
            if not admitted.is_set():
                admitted.set()
                assert release.wait(3), "test never released the admitted writer"
            return candidate

        return transact(run)

    monkeypatch.setattr(sandbox.database, "transact", observed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(_owned_operation, audit_store, operation, session)
        try:
            assert admitted.wait(2)
            pids.get(timeout=1)
            fencer = pool.submit(freeze_authority, sandbox.database, sandbox.authority)
            fence_pid = pids.get(timeout=1)
            _wait_for_lock(sandbox, fence_pid)
            assert not fencer.done()
            assert _history(sandbox) == before
            assert sandbox.admin.execute(
                "SELECT accepting FROM deployment_authority"
            ).fetchone() == (True,)
        finally:
            release.set()
        writer.result(timeout=5)
        assert fencer.result(timeout=5) == sandbox.authority.generation
    assert len(_history(sandbox)) == len(before) + _ADDED_ROWS[operation]
    assert sandbox.admin.execute(
        "SELECT accepting,egress_enabled FROM deployment_authority"
    ).fetchone() == (False, False)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("committed", [False, True], ids=["rolled-back", "committed"])
def test_unknown_commit_never_replays_reloads_or_publishes_success(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    committed: bool,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    transact = sandbox.database.transact
    commit = psycopg.Connection.commit
    calls = []
    callback_pids = []
    commit_pids = []
    published = []

    def observed(callback):
        calls.append("transact")

        def run(connection):
            callback_pids.append(connection.info.backend_pid)
            return callback(connection)

        return transact(run)

    def lose_receipt(connection):
        pid = connection.info.backend_pid
        if pid in callback_pids:
            commit_pids.append(pid)
            if committed:
                commit(connection)
            else:
                connection.rollback()
            raise psycopg.OperationalError("injected COMMIT receipt loss")
        commit(connection)

    def no_reload(_callback):
        pytest.fail("unknown COMMIT must not be resolved by a second reload")

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "transact", observed)
        patch.setattr(sandbox.database, "read", no_reload)
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(_owned_operation(audit_store, operation, session))
    assert not published
    assert calls == ["transact"]
    assert len(callback_pids) == len(commit_pids) == 1
    assert callback_pids == commit_pids
    rows = _history(sandbox)
    if committed:
        assert rows[: len(before)] == before
        assert len(rows) == len(before) + _ADDED_ROWS[operation]
    else:
        assert rows == before
    _assert_chain(rows)


@pytest.mark.parametrize("operation", ["append", "batch", "start", "close", "recovery"])
def test_deferred_commit_rejection_never_releases_a_result(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    sandbox.admin.execute(
        "CREATE FUNCTION reject_audit_commit_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RAISE EXCEPTION 'injected deferred audit failure' "
        "USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_audit_commit_test AFTER INSERT ON audit_events "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        "EXECUTE FUNCTION reject_audit_commit_test()"
    )
    published = []
    with pytest.raises(psycopg.errors.CheckViolation):
        published.append(_owned_operation(audit_store, operation, session))
    assert not published
    assert _history(sandbox) == before


def test_first_and_clean_startup_keep_shared_opaque_sessions_without_fences(
    postgres_product_sandbox: ProductSandbox, audit_store: PostgresAuditStore
) -> None:
    first = start_session(audit_store)
    close_session(audit_store, first)
    close_session(audit_store, first)
    second = start_session(audit_store)
    close_session(audit_store, second)
    close_session(audit_store, second)
    assert type(first) is type(second) is AuditSession
    assert UUID(hex=first.session_id).version == UUID(hex=second.session_id).version == 4
    assert len(first.session_id) == len(second.session_id) == 32
    assert first.session_id != second.session_id
    rows = _history(postgres_product_sandbox)
    assert _identities(rows) == [
        (AuditAction.AUDIT_SESSION_START, first.session_id),
        (AuditAction.AUDIT_SESSION_CLOSE, first.session_id),
        (AuditAction.AUDIT_SESSION_START, second.session_id),
        (AuditAction.AUDIT_SESSION_CLOSE, second.session_id),
    ]
    _assert_chain(rows)


def test_unclean_startup_orders_one_recovery_before_the_new_start(
    postgres_product_sandbox: ProductSandbox, audit_store: PostgresAuditStore
) -> None:
    abandoned = start_session(audit_store)
    second = start_session(audit_store)
    close_session(audit_store, second)
    third = start_session(audit_store)
    rows = _history(postgres_product_sandbox)
    assert _identities(rows) == [
        (AuditAction.AUDIT_SESSION_START, abandoned.session_id),
        (AuditAction.RECOVERY_FENCE, abandoned.session_id),
        (AuditAction.AUDIT_SESSION_START, second.session_id),
        (AuditAction.AUDIT_SESSION_CLOSE, second.session_id),
        (AuditAction.AUDIT_SESSION_START, third.session_id),
    ]
    assert json.loads(rows[1]["detail_json"])["failure_code"] == "unclean_restart"
    _assert_chain(rows)


def test_startup_inspects_only_the_latest_start_not_every_unclosed_session(
    postgres_product_sandbox: ProductSandbox, audit_store: PostgresAuditStore
) -> None:
    older, latest = AuditSession(uuid4().hex), AuditSession(uuid4().hex)
    audit_store.append_batch(
        (
            _session_event(AuditAction.AUDIT_SESSION_START, older),
            _session_event(AuditAction.AUDIT_SESSION_START, latest),
            _session_event(AuditAction.AUDIT_SESSION_CLOSE, latest),
        )
    )
    opened = start_session(audit_store)
    rows = _history(postgres_product_sandbox)
    assert _identities(rows) == [
        (AuditAction.AUDIT_SESSION_START, older.session_id),
        (AuditAction.AUDIT_SESSION_START, latest.session_id),
        (AuditAction.AUDIT_SESSION_CLOSE, latest.session_id),
        (AuditAction.AUDIT_SESSION_START, opened.session_id),
    ]
    _assert_chain(rows)


def test_recovery_identity_ignores_failure_code_and_startup_does_not_refence(
    postgres_product_sandbox: ProductSandbox, audit_store: PostgresAuditStore
) -> None:
    session = start_session(audit_store)
    first = append_with_recovery(audit_store, _event("first"), session, "database_unavailable")
    second = append_with_recovery(audit_store, _event("second"), session, "capacity_refused")
    opened = start_session(audit_store)
    rows = _history(postgres_product_sandbox)
    assert _identities(rows) == [
        (AuditAction.AUDIT_SESSION_START, session.session_id),
        (AuditAction.RECOVERY_FENCE, session.session_id),
        (AuditAction.AUDIT_LIST, "first"),
        (AuditAction.AUDIT_LIST, "second"),
        (AuditAction.AUDIT_SESSION_START, opened.session_id),
    ]
    assert json.loads(rows[1]["detail_json"])["failure_code"] == "database_unavailable"
    assert (rows[2]["audit_id"], rows[3]["audit_id"]) == (first.audit_id, second.audit_id)
    _assert_chain(rows)


def test_lifecycle_attribution_and_timestamp_call_order_match_the_reference(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = tuple(f"2026-09-27T04:01:0{index}.000Z" for index in range(4))
    clock = iter(times)
    monkeypatch.setattr(postgres_sessions, "utc_now", lambda: next(clock))
    protected = replace(
        _event(),
        actor_id="edge-service",
        actor_type=AuditActorType.SERVICE,
        auth_mechanism=AuditAuthMechanism.RELAY_TOKEN,
    )
    session = start_session(audit_store)
    record = append_with_recovery(audit_store, protected, session, "database_unavailable")
    close_session(audit_store, session)
    rows = _history(postgres_product_sandbox)
    assert [row["occurred_at"] for row in rows] == [times[0], times[1], _TIME, times[3]]
    for row in (rows[0], rows[1], rows[3]):
        assert (row["actor_id"], row["actor_type"], row["auth_mechanism"]) == (
            "audit-readiness",
            "system",
            "internal",
        )
        assert (row["target_type"], row["target_id"]) == ("audit", session.session_id)
        assert row["request_id"] is row["interaction_id"] is None
    assert (
        json.loads(rows[0]["detail_json"]) == json.loads(rows[3]["detail_json"]) == {"version": 1}
    )
    assert json.loads(rows[1]["detail_json"]) == {
        "version": 1,
        "failure_code": "database_unavailable",
        "ended_at": times[2],
    }
    assert (rows[2]["actor_id"], rows[2]["actor_type"], rows[2]["auth_mechanism"]) == (
        "edge-service",
        "service",
        "relay_token",
    )
    assert rows[2]["audit_id"] == record.audit_id
    assert record.detail == protected.detail
    _assert_chain(rows)


@pytest.mark.parametrize("operation", ["close", "recovery"])
def test_concurrent_closes_and_different_failure_codes_are_session_idempotent(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    transact = sandbox.database.transact
    ready = Barrier(3)
    pids: Queue[int] = Queue()
    codes = ("database_unavailable", "capacity_refused", "temporary_failure")

    def synchronized(callback):
        def run(connection):
            pids.put(connection.info.backend_pid)
            ready.wait(timeout=3)
            return callback(connection)

        return transact(run)

    def mutate(index):
        if operation == "close":
            return close_session(audit_store, session)
        return append_with_recovery(audit_store, _event(f"event-{index}"), session, codes[index])

    monkeypatch.setattr(sandbox.database, "transact", synchronized)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(mutate, index) for index in range(3)]
        results = [future.result(timeout=5) for future in futures]
    assert len({pids.get_nowait() for _ in range(3)}) == 3
    rows = _history(sandbox)
    if operation == "close":
        assert results == [None] * 3
        assert _identities(rows) == [
            (AuditAction.AUDIT_SESSION_START, session.session_id),
            (AuditAction.AUDIT_SESSION_CLOSE, session.session_id),
        ]
    else:
        assert _identities(rows[:2]) == [
            (AuditAction.AUDIT_SESSION_START, session.session_id),
            (AuditAction.RECOVERY_FENCE, session.session_id),
        ]
        assert len(rows) == 5
        assert {row["target_id"] for row in rows[2:]} == {f"event-{index}" for index in range(3)}
        winner = int(rows[2]["target_id"].removeprefix("event-"))
        assert json.loads(rows[1]["detail_json"])["failure_code"] == codes[winner]
        assert {result.audit_id for result in results} == {row["audit_id"] for row in rows[2:]}
    _assert_chain(rows)


@pytest.mark.parametrize("operation", ["start", "close", "recovery"])
def test_lifecycle_lock_precedes_reads_and_observes_the_committing_predecessor(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    transact = sandbox.database.transact
    pids: Queue[int] = Queue()

    def observed(callback):
        def run(connection):
            pids.put(connection.info.backend_pid)
            return callback(connection)

        return transact(run)

    monkeypatch.setattr(sandbox.database, "transact", observed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with sandbox.admin.transaction():
            if operation == "start":
                predecessor = start_session(audit_store, connection=sandbox.admin)
            elif operation == "close":
                audit_store.append(
                    _session_event(AuditAction.AUDIT_SESSION_CLOSE, session),
                    connection=sandbox.admin,
                )
            else:
                append_with_recovery(
                    audit_store,
                    _event("predecessor"),
                    session,
                    "capacity_refused",
                    connection=sandbox.admin,
                )
            future = pool.submit(_owned_operation, audit_store, operation, session)
            _wait_for_lock(sandbox, pids.get(timeout=2), advisory=True)
            assert not future.done()
        result = future.result(timeout=5)
    rows = _history(sandbox)
    if operation == "start":
        assert isinstance(result, AuditSession)
        assert _identities(rows) == [
            (AuditAction.AUDIT_SESSION_START, session.session_id),
            (AuditAction.RECOVERY_FENCE, session.session_id),
            (AuditAction.AUDIT_SESSION_START, predecessor.session_id),
            (AuditAction.RECOVERY_FENCE, predecessor.session_id),
            (AuditAction.AUDIT_SESSION_START, result.session_id),
        ]
    elif operation == "close":
        assert result is None
        assert _identities(rows) == [
            (AuditAction.AUDIT_SESSION_START, session.session_id),
            (AuditAction.AUDIT_SESSION_CLOSE, session.session_id),
        ]
    else:
        assert isinstance(result, AuditRecord)
        assert _identities(rows) == [
            (AuditAction.AUDIT_SESSION_START, session.session_id),
            (AuditAction.RECOVERY_FENCE, session.session_id),
            (AuditAction.AUDIT_LIST, "predecessor"),
            (AuditAction.AUDIT_LIST, "protected"),
        ]
        assert json.loads(rows[1]["detail_json"])["failure_code"] == "capacity_refused"
        assert rows[-1]["audit_id"] == result.audit_id
    _assert_chain(rows)


@pytest.mark.parametrize(
    ("operation", "rejected_action"),
    [
        ("start", AuditAction.RECOVERY_FENCE),
        ("start", AuditAction.AUDIT_SESSION_START),
        ("close", AuditAction.AUDIT_SESSION_CLOSE),
        ("recovery", AuditAction.RECOVERY_FENCE),
        ("recovery", AuditAction.AUDIT_LIST),
    ],
)
def test_lifecycle_failure_rolls_back_fences_and_protected_actions_together(
    postgres_product_sandbox: ProductSandbox,
    audit_store: PostgresAuditStore,
    operation: str,
    rejected_action: AuditAction,
) -> None:
    sandbox = postgres_product_sandbox
    session = start_session(audit_store)
    before = _history(sandbox)
    sandbox.admin.execute(
        "CREATE FUNCTION reject_audit_action_test() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN IF NEW.action = TG_ARGV[0] THEN "
        "RAISE EXCEPTION 'injected audit failure' USING ERRCODE='23514'; "
        "END IF; RETURN NEW; END $$"
    )
    sandbox.admin.execute(
        sql.SQL(
            "CREATE TRIGGER reject_audit_action_test BEFORE INSERT ON audit_events "
            "FOR EACH ROW EXECUTE FUNCTION reject_audit_action_test({})"
        ).format(sql.Literal(rejected_action.value))
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        _owned_operation(audit_store, operation, session)
    assert _history(sandbox) == before
    sandbox.admin.execute("DROP TRIGGER reject_audit_action_test ON audit_events")
    _owned_operation(audit_store, operation, session)
    rows = _history(sandbox)
    assert len(rows) == len(before) + _ADDED_ROWS[operation]
    assert rows[: len(before)] == before
    _assert_chain(rows)
