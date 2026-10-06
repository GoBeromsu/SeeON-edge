"""Real PostgreSQL publication evidence, not HTTP/lifespan or deployment qualification.

A missing or invalid SEEON_TEST_POSTGRES_DSN fails in postgres_sandbox; it never
skips. Fault hooks wrap actual native transactions and session helpers.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, replace
from queue import Queue
from threading import Barrier, BrokenBarrierError, Event, Lock, Thread
from typing import TYPE_CHECKING

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PostgresDatabase,
    PostgresError,
    PostgresPoolBusy,
    PostgresStartupError,
    PostgresTransactionStateError,
    PostgresUnavailable,
)
from backend.app.features.audit import postgres_sessions
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import (
    AuditRuntimeUnavailable,
    InvalidAuditPublication,
    PendingAuditPublication,
    PostgresAuditRuntime,
)
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.postgres_verification import _verify_snapshot
from backend.app.features.audit.store import AuditEvent
from backend.app.features.audit.verification import AuditVerificationError
from backend.app.features.runtime_settings.store import (
    RuntimeSettingsStore,
    RuntimeSettingsVersionConflict,
)

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_SECRET = "postgresql://private-user:private-password@private-host/private-db SELECT secret"
_STAGES = ("before_commit", "pool_exit", "unknown_rollback", "unknown_commit")
# Families the worker may surface. Anything else is recorded and re-raised in the
# thread so cancellation and surprises cannot disappear before result() joins.
_THREAD_EXCEPTIONS = (
    AssertionError,
    AuditRuntimeUnavailable,
    AuditVerificationError,
    AuthorityFenced,
    BrokenBarrierError,
    CommitOutcomeUnknown,
    OSError,
    PostgresError,
    ValueError,
    psycopg.Error,
)


class Clock:
    def __init__(self) -> None:
        self._value = 100.0
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class Call:
    """Bounded thread result, including assertion failures inside fault hooks."""

    def __init__(self, callback: Callable[[], object]) -> None:
        self._results: Queue[tuple[bool, object]] = Queue()

        def run() -> None:
            try:
                self._results.put((True, callback()))
            except _THREAD_EXCEPTIONS as error:
                self._results.put((False, error))
            except BaseException as error:
                self._results.put((False, error))
                raise

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def result(self, timeout: float = 2.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "runtime call did not finish within its bound"
        success, result = self._results.get_nowait()
        if not success:
            raise result
        return result


@pytest.fixture
def audit_store(postgres_product_sandbox: ProductSandbox) -> PostgresAuditStore:
    return PostgresAuditStore(postgres_product_sandbox.database, postgres_product_sandbox.authority)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def runtime(audit_store: PostgresAuditStore, clock: Clock) -> PostgresAuditRuntime:
    return PostgresAuditRuntime(audit_store, maximum_snapshot_age_sec=10, clock=clock)


def _event(target: str = "protected") -> AuditEvent:
    return AuditEvent(
        occurred_at="2026-09-27T00:00:00Z",
        actor_id="test-operator",
        action=AuditAction.AUDIT_LIST,
        target_id=target,
        detail=empty_detail(AuditAction.AUDIT_LIST),
    )


def _history(sandbox: ProductSandbox) -> list[dict]:
    with sandbox.admin.cursor(row_factory=dict_row) as cursor:
        return cursor.execute("SELECT * FROM audit_events ORDER BY audit_id").fetchall()


def _actions(sandbox: ProductSandbox) -> list[str]:
    return [row["action"] for row in _history(sandbox)]


def _ready(runtime: PostgresAuditRuntime) -> None:
    assert runtime.verify_once()
    assert runtime.start_session_once()
    assert runtime.snapshot().ready


def _private(failure, runtime, caplog) -> None:
    expected = (
        "PostgreSQL commit outcome is unknown; automatic retry is forbidden"
        if isinstance(failure.value, CommitOutcomeUnknown)
        else "audit runtime unavailable"
    )
    assert str(failure.value) == expected
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__
    assert _SECRET not in "".join(traceback.format_exception(failure.value))
    assert _SECRET not in caplog.text + repr(runtime.snapshot())


@contextmanager
def _paused_owner(sandbox, monkeypatch, method="transact", *, after_commit=False):
    owner = getattr(sandbox.database, method)
    entered, release = Event(), Event()
    calls = []

    def pause():
        entered.set()
        assert release.wait(3), "test did not release the database owner"

    def observed(callback):
        def run(connection):
            calls.append(connection.info.backend_pid)
            result = callback(connection)
            if not after_commit:
                pause()
            return result

        result = owner(run)
        if after_commit:
            pause()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, method, observed)
        try:
            yield entered, release, calls
        finally:
            release.set()


@contextmanager
def _fault(sandbox, monkeypatch, stage, *, method="transact") -> Iterator[list[int]]:
    owner = getattr(sandbox.database, method)
    commit = psycopg.Connection.commit
    acquire = sandbox.database._pool.connection
    calls = []

    def observed(callback):
        def run(connection):
            calls.append(connection.info.backend_pid)
            result = callback(connection)
            if stage == "before_commit":
                connection.execute("SELECT %s::bigint", (_SECRET,))
            return result

        return owner(run)

    def lose_receipt(connection):
        if connection.info.backend_pid in calls:
            if stage == "unknown_commit":
                commit(connection)
            else:
                connection.rollback()
            raise psycopg.OperationalError(_SECRET)
        return commit(connection)

    @contextmanager
    def failed_exit(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            yield connection
        raise psycopg.OperationalError(_SECRET)

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, method, observed)
        if stage.startswith("unknown_"):
            patch.setattr(psycopg.Connection, "commit", lose_receipt)
        if stage == "pool_exit":
            patch.setattr(sandbox.database._pool, "connection", failed_exit)
        yield calls


@pytest.mark.parametrize(
    "age", [0, -1, float("inf"), -float("inf"), float("nan"), True, "10", None]
)
def test_age_is_required_finite_positive_policy(audit_store, age) -> None:
    with pytest.raises(ValueError, match="^maximum_snapshot_age_sec must be finite and positive$"):
        PostgresAuditRuntime(audit_store, maximum_snapshot_age_sec=age)
    with pytest.raises(TypeError):
        PostgresAuditRuntime(audit_store)


@pytest.mark.parametrize("after_commit", [False, True])
def test_start_order_once_and_publication_only_after_owned_return(
    postgres_product_sandbox, runtime, monkeypatch, after_commit
) -> None:
    sandbox = postgres_product_sandbox
    assert not runtime.start_session_once()
    with pytest.raises(AuditRuntimeUnavailable, match="^audit runtime unavailable$"):
        runtime.append_owned(_event())
    assert _history(sandbox) == []
    assert runtime.verify_once()
    with _paused_owner(sandbox, monkeypatch, after_commit=after_commit) as (
        entered,
        release,
        calls,
    ):
        start = Call(runtime.start_session_once)
        assert entered.wait(2)
        status = Call(runtime.snapshot).result()
        assert not status.ready and status.verification_current and not status.session_established
        assert _actions(sandbox) == ([AuditAction.AUDIT_SESSION_START] if after_commit else [])
        assert Call(runtime.start_session_once).result() is False
        release.set()
        assert start.result() is True
        assert len(calls) == 1
    rows = _history(sandbox)
    assert len(rows) == 1 and rows[0]["action"] == AuditAction.AUDIT_SESSION_START
    assert runtime.snapshot().ready
    assert not runtime.start_session_once()
    assert not runtime.close_session_once()  # No stopping admission yet.
    runtime.stop()
    assert runtime.close_session_once()
    rows = _history(sandbox)
    assert [row["action"] for row in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]
    assert rows[0]["target_id"] == rows[1]["target_id"]
    assert sandbox.database.read(lambda connection: connection.execute("SELECT 1").fetchone()) == (
        1,
    )
    # Only the caller closes the shared pool, after the close commit.
    sandbox.database.close(timeout_sec=3.0)
    assert not runtime.close_session_once()


@pytest.mark.parametrize("stage", _STAGES)
def test_failed_establishment_never_publishes_retries_or_reconstructs_session(
    postgres_product_sandbox, runtime, monkeypatch, caplog, stage
) -> None:
    sandbox = postgres_product_sandbox
    assert runtime.verify_once()
    error_type = CommitOutcomeUnknown if stage.startswith("unknown_") else AuditRuntimeUnavailable
    with _fault(sandbox, monkeypatch, stage) as calls:
        with pytest.raises(error_type) as failure:
            runtime.start_session_once()
    _private(failure, runtime, caplog)
    assert len(calls) == 1
    persisted = stage in {"pool_exit", "unknown_commit"}
    assert _actions(sandbox) == ([AuditAction.AUDIT_SESSION_START] if persisted else [])
    status = runtime.snapshot()
    assert not status.session_established and not status.eligible_to_attempt and not status.ready
    assert status.indeterminate == stage.startswith("unknown_")
    assert runtime.verify_once() == (not status.indeterminate)
    assert not runtime.start_session_once()
    with pytest.raises(AuditRuntimeUnavailable):
        runtime.append_owned(_event())
    runtime.stop()
    assert not runtime.close_session_once()
    assert _actions(sandbox) == ([AuditAction.AUDIT_SESSION_START] if persisted else [])


def test_owned_recovery_requires_new_verification_and_committed_fence_once_per_session(
    postgres_product_sandbox, runtime, audit_store
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)
    runtime.record_failure(OSError(_SECRET))
    assert _history(sandbox) == before
    assert not runtime.snapshot().verification_current
    with pytest.raises(AuditRuntimeUnavailable):
        runtime.append_owned(_event("refused"))
    assert runtime.verify_once()
    status = runtime.snapshot()
    assert status.eligible_to_attempt and not status.ready
    assert status.failure_code == "database_unavailable"
    assert _history(sandbox) == before  # Verification did not commit a fence.
    record = runtime.append_owned(_event("first"))
    rows = _history(sandbox)
    assert [row["action"] for row in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.RECOVERY_FENCE,
        AuditAction.AUDIT_LIST,
    ]
    assert rows[0]["target_id"] == rows[1]["target_id"]
    assert json.loads(rows[1]["detail_json"])["failure_code"] == "database_unavailable"
    assert rows[2]["audit_id"] == record.audit_id and rows[2]["target_id"] == "first"
    assert runtime.snapshot().ready
    runtime.record_failure(PostgresPoolBusy(_SECRET))
    assert runtime.verify_once()
    runtime.append_owned(_event("second"))
    rows = _history(sandbox)
    assert [row["target_id"] for row in rows[2:]] == ["first", "second"]
    assert _actions(sandbox).count(AuditAction.RECOVERY_FENCE) == 1
    assert audit_store.verify().row_count == 4


@pytest.mark.parametrize("borrowed", [False, True])
def test_recovery_event_failure_rolls_back_the_fence_and_product_write(
    postgres_product_sandbox, runtime, audit_store, borrowed, caplog
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)
    runtime.record_failure(PostgresUnavailable(_SECRET))
    assert runtime.verify_once()
    invalid = replace(_event(), detail=empty_detail(AuditAction.CONNECTION_UPDATE))

    def product_write(connection):
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        return runtime.append_borrowed(connection, invalid)

    with pytest.raises(AuditRuntimeUnavailable) as failure:
        if borrowed:
            sandbox.database.transact(product_write)
        else:
            runtime.append_owned(invalid)
    _private(failure, runtime, caplog)
    assert _history(sandbox) == before
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (0,)
    assert runtime.snapshot().failure_code == "verification_failed"
    assert not runtime.snapshot().verification_current
    assert runtime.verify_once()
    runtime.append_owned(_event("retry-with-valid-event"))
    assert _actions(sandbox) == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.RECOVERY_FENCE,
        AuditAction.AUDIT_LIST,
    ]
    assert audit_store.verify().row_count == 3
    runtime.stop()
    assert runtime.close_session_once()  # Failed borrowed calls did not leak pending counts.


def test_borrowed_commit_is_tentative_without_checkout_verify_or_session_creation(
    postgres_product_sandbox, runtime, audit_store, monkeypatch
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    runtime.record_failure(OSError(_SECRET))
    assert runtime.verify_once()
    transact = sandbox.database.transact
    before = _history(sandbox)

    def forbidden(*args, **kwargs):
        pytest.fail("borrowed append performed extra owner work")

    def product_write(connection):
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        token = runtime.append_borrowed(connection, _event())
        assert isinstance(token, PendingAuditPublication)
        assert not runtime.snapshot().ready
        assert _history(sandbox) == before
        assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (0,)
        assert connection.execute(
            "SELECT action FROM audit_events ORDER BY audit_id"
        ).fetchall() == [
            (AuditAction.AUDIT_SESSION_START,),
            (AuditAction.RECOVERY_FENCE,),
            (AuditAction.AUDIT_LIST,),
        ]
        return token

    with monkeypatch.context() as patch:
        for method in ("transact", "read", "read_snapshot"):
            patch.setattr(sandbox.database, method, forbidden)
        patch.setattr(audit_store, "verify", forbidden)
        patch.setattr(postgres_sessions, "start_session", forbidden)
        token = transact(product_write)
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (1,)
    assert len(_history(sandbox)) == 3
    assert not runtime.snapshot().ready
    assert runtime.publish_committed(token)
    assert runtime.snapshot().ready
    assert audit_store.verify().row_count == 3


def test_borrowed_outer_rollback_consumes_failure_without_publishing_tentative_fence(
    postgres_product_sandbox, runtime, audit_store
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    runtime.record_failure(OSError(_SECRET))
    assert runtime.verify_once()
    before, tokens = _history(sandbox), []

    def product_write(connection):
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        tokens.append(runtime.append_borrowed(connection, _event("rolled-back")))
        raise OSError(_SECRET)

    with pytest.raises(OSError) as failure:
        sandbox.database.transact(product_write)
    runtime.publish_failed(tokens[0], failure.value)
    assert _history(sandbox) == before
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (0,)
    assert not runtime.snapshot().eligible_to_attempt
    with pytest.raises(ValueError, match="^invalid audit publication token$"):
        runtime.publish_committed(tokens[0])
    assert runtime.verify_once()
    token = sandbox.database.transact(
        lambda connection: runtime.append_borrowed(connection, _event())
    )
    assert runtime.publish_committed(token)
    rows = _history(sandbox)
    assert rows[-1]["target_id"] == "protected" and len(rows) == 3
    assert _actions(sandbox).count(AuditAction.RECOVERY_FENCE) == 1
    assert audit_store.verify().row_count == 3


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("stage", _STAGES)
@pytest.mark.parametrize("recovery", [False, True])
def test_append_outcome_is_not_inferred_from_errors_or_replayed(
    postgres_product_sandbox, runtime, monkeypatch, caplog, stage, borrowed, recovery
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    if recovery:
        runtime.record_failure(OSError(_SECRET))
        assert runtime.verify_once()
    tokens = []

    def product_write(connection):
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        tokens.append(runtime.append_borrowed(connection, _event("uncertain")))

    unknown = stage.startswith("unknown_")
    expected = (
        CommitOutcomeUnknown
        if unknown
        else ((psycopg.Error, PostgresUnavailable) if borrowed else AuditRuntimeUnavailable)
    )
    with _fault(sandbox, monkeypatch, stage) as calls:
        with pytest.raises(expected) as failure:
            if borrowed:
                sandbox.database.transact(product_write)
            else:
                runtime.append_owned(_event("uncertain"))
    if borrowed:
        assert len(tokens) == 1
        runtime.publish_failed(tokens[0], failure.value)
    else:
        _private(failure, runtime, caplog)
    assert len(calls) == 1
    persisted = stage in {"pool_exit", "unknown_commit"}
    rows = _history(sandbox)
    expected_actions = [AuditAction.AUDIT_SESSION_START]
    if persisted:
        if recovery:
            expected_actions.append(AuditAction.RECOVERY_FENCE)
        expected_actions.append(AuditAction.AUDIT_LIST)
        assert rows[-1]["target_id"] == "uncertain"
    assert [row["action"] for row in rows] == expected_actions
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (
        int(borrowed and persisted),
    )
    assert runtime.snapshot().indeterminate == unknown
    assert not runtime.snapshot().ready and not runtime.snapshot().eligible_to_attempt
    if unknown:
        runtime.record_failure(OSError(_SECRET))
        assert runtime.snapshot().failure_code == "commit_outcome_unknown"
        assert not runtime.verify_once()
        assert not runtime.start_session_once()
        with pytest.raises(AuditRuntimeUnavailable):
            runtime.append_owned(_event("forbidden-retry"))
        runtime.stop()
        assert not runtime.close_session_once()
        assert _history(sandbox) == rows
    else:
        assert runtime.verify_once()
        assert not runtime.snapshot().ready
        runtime.append_owned(_event("distinct-later-event"))
        assert [
            row["target_id"] for row in _history(sandbox) if row["action"] == AuditAction.AUDIT_LIST
        ] == (["uncertain", "distinct-later-event"] if persisted else ["distinct-later-event"])
        assert _actions(sandbox).count(AuditAction.RECOVERY_FENCE) == 1


@pytest.mark.parametrize("unknown", [False, True])
def test_older_owned_success_cannot_erase_new_failure_or_unknown_latch(
    postgres_product_sandbox, runtime, monkeypatch, unknown
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    with _paused_owner(sandbox, monkeypatch, after_commit=True) as (entered, release, calls):
        append = Call(lambda: runtime.append_owned(_event("older")))
        assert entered.wait(2)
        assert _history(sandbox)[-1]["target_id"] == "older"
        error = CommitOutcomeUnknown() if unknown else PostgresPoolBusy(_SECRET)
        Call(lambda: runtime.record_failure(error)).result()
        assert runtime.verify_once() == (not unknown)
        release.set()
        assert append.result().target_id == "older"
        assert len(calls) == 1
    status = runtime.snapshot()
    assert not status.ready and status.eligible_to_attempt == (not unknown)
    assert status.failure_code == ("commit_outcome_unknown" if unknown else "database_busy")
    if not unknown:
        runtime.append_owned(_event("newer-recovery"))
        assert _actions(sandbox) == [
            AuditAction.AUDIT_SESSION_START,
            AuditAction.AUDIT_LIST,
            AuditAction.RECOVERY_FENCE,
            AuditAction.AUDIT_LIST,
        ]
        assert runtime.snapshot().ready


def test_old_foreign_and_consumed_tokens_never_promote_or_drain_other_work(
    postgres_product_sandbox, runtime, audit_store
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    tokens = sandbox.database.transact(
        lambda connection: tuple(
            runtime.append_borrowed(connection, _event(str(index))) for index in range(2)
        )
    )
    aliases = (copy(tokens[0]), deepcopy(tokens[0]))
    other = PostgresAuditRuntime(audit_store, maximum_snapshot_age_sec=10)
    with pytest.raises(ValueError, match="^invalid audit publication token$"):
        other.publish_committed(tokens[0])
    with pytest.raises(ValueError, match="^invalid audit publication token$"):
        runtime.publish_committed(object())
    runtime.record_failure(OSError(_SECRET))
    assert runtime.verify_once()
    assert not runtime.publish_committed(tokens[0])  # Old revision, but still consumes its count.
    assert not runtime.snapshot().ready
    with pytest.raises(ValueError, match="^invalid audit publication token$"):
        runtime.publish_failed(tokens[0], CommitOutcomeUnknown())
    for alias in aliases:
        with pytest.raises(ValueError, match="^invalid audit publication token$"):
            runtime.publish_committed(alias)
    assert not runtime.snapshot().indeterminate
    runtime.stop()
    assert not runtime.close_session_once()  # The foreign-token rejection did not consume token 1.
    assert not runtime.publish_committed(tokens[1])
    assert runtime.close_session_once()
    rows = _history(sandbox)
    assert [row["action"] for row in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_LIST,
        AuditAction.AUDIT_LIST,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]
    for token in tokens:
        assert rows[0]["target_id"] not in repr(token)
    assert rows[0]["target_id"] == rows[-1]["target_id"]


def test_verification_preserves_anchor_on_failure_and_stale_success(
    postgres_product_sandbox, runtime, audit_store, monkeypatch, caplog
) -> None:
    sandbox = postgres_product_sandbox
    verify = audit_store.verify
    inputs, outputs = [], []

    def observed(checkpoint=None):
        inputs.append(checkpoint)
        result = verify(checkpoint)
        outputs.append(result)
        return result

    monkeypatch.setattr(audit_store, "verify", observed)
    _ready(runtime)
    original = outputs[-1]
    sandbox.admin.execute("ALTER TABLE audit_events DISABLE TRIGGER audit_events_immutable_update")
    with pytest.raises(AuditRuntimeUnavailable) as failure:
        runtime.verify_once()
    _private(failure, runtime, caplog)
    assert inputs[-1] is original
    sandbox.admin.execute("ALTER TABLE audit_events ENABLE TRIGGER audit_events_immutable_update")
    with _paused_owner(sandbox, monkeypatch, "read_snapshot", after_commit=True) as (
        entered,
        release,
        calls,
    ):
        scan = Call(runtime.verify_once)
        assert entered.wait(2)
        Call(lambda: runtime.record_failure(PostgresPoolBusy(_SECRET))).result()
        release.set()
        assert scan.result() is False
        assert len(calls) == 1
    assert inputs[-1] is original and outputs[-1].row_count == 1
    assert runtime.verify_once()
    assert inputs[-1] is original
    assert outputs[-1].row_count == 1
    assert not runtime.snapshot().ready and runtime.snapshot().eligible_to_attempt
    runtime.append_owned(_event())
    assert _actions(sandbox) == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.RECOVERY_FENCE,
        AuditAction.AUDIT_LIST,
    ]


@pytest.mark.parametrize("stage", _STAGES)
def test_verifier_owner_failure_never_replaces_previous_checkpoint(
    postgres_product_sandbox, runtime, audit_store, monkeypatch, caplog, stage
) -> None:
    sandbox = postgres_product_sandbox
    verify, inputs = audit_store.verify, []

    def observed(checkpoint=None):
        inputs.append(checkpoint)
        return verify(checkpoint)

    monkeypatch.setattr(audit_store, "verify", observed)
    _ready(runtime)
    runtime.append_owned(_event())
    with _fault(sandbox, monkeypatch, stage, method="read_snapshot") as calls:
        expected = CommitOutcomeUnknown if stage.startswith("unknown_") else AuditRuntimeUnavailable
        with pytest.raises(expected) as failure:
            runtime.verify_once()
    _private(failure, runtime, caplog)
    assert len(calls) == 1
    previous = inputs[-1]
    assert previous.row_count == 0
    if stage.startswith("unknown_"):
        assert not runtime.verify_once()
    else:
        assert runtime.verify_once()
        assert inputs[-1] is previous
    assert not runtime.snapshot().ready
    assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START, AuditAction.AUDIT_LIST]


@pytest.mark.parametrize("borrowed", [False, True])
def test_serialized_verifier_has_no_queue_and_expiry_is_measured_from_scan_start(
    postgres_product_sandbox, runtime, clock, monkeypatch, borrowed
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    clock.advance(4)
    with _paused_owner(sandbox, monkeypatch, "read_snapshot") as (entered, release, calls):
        scan = Call(runtime.verify_once)
        assert entered.wait(2)
        barrier = Barrier(3)

        def contend():
            barrier.wait(timeout=2)
            return runtime.verify_once()

        contenders = [Call(contend), Call(contend)]
        barrier.wait(timeout=2)
        assert [contender.result() for contender in contenders] == [False, False]
        clock.advance(6)  # Original observation at 100 expires while scan at 104 is blocked.
        assert not Call(runtime.snapshot).result().verification_current
        with pytest.raises(AuditRuntimeUnavailable):
            runtime.append_owned(_event("expired"))
        assert len(calls) == 1
        release.set()
        assert scan.result() is True
    assert runtime.snapshot().ready  # New observation is six seconds old, not zero.
    clock.advance(3)
    if borrowed:
        token = sandbox.database.transact(
            lambda connection: runtime.append_borrowed(connection, _event("does-not-renew"))
        )
    else:
        runtime.append_owned(_event("does-not-renew"))
    clock.advance(1)
    if borrowed:
        assert runtime.publish_committed(token)
    assert not runtime.snapshot().verification_current  # START 104 + policy 10, not end 110 + 10.
    with pytest.raises(AuditRuntimeUnavailable):
        runtime.append_owned(_event("still-expired"))
    assert [row["target_id"] for row in _history(sandbox)[1:]] == ["does-not-renew"]


def test_slow_scan_does_not_create_fresh_evidence_or_establish_session(
    postgres_product_sandbox, runtime, clock, monkeypatch
) -> None:
    sandbox = postgres_product_sandbox
    with _paused_owner(sandbox, monkeypatch, "read_snapshot") as (entered, release, calls):
        scan = Call(runtime.verify_once)
        assert entered.wait(2)
        clock.advance(10)
        release.set()
        assert scan.result() is True
        assert len(calls) == 1
    assert not runtime.snapshot().verification_current
    assert not runtime.start_session_once()
    assert _history(sandbox) == []
    assert runtime.verify_once()
    assert runtime.start_session_once()  # Refused admission did not consume the one attempt.
    assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START]


@pytest.mark.parametrize("transition", ["failure", "stop"])
def test_known_committed_start_is_retained_despite_concurrent_invalidation(
    postgres_product_sandbox, runtime, monkeypatch, transition
) -> None:
    sandbox = postgres_product_sandbox
    assert runtime.verify_once()
    with _paused_owner(sandbox, monkeypatch, after_commit=True) as (entered, release, calls):
        start = Call(runtime.start_session_once)
        assert entered.wait(2)
        if transition == "stop":
            Call(runtime.stop).result()
        else:
            Call(lambda: runtime.record_failure(OSError(_SECRET))).result()
        assert not runtime.snapshot().ready
        assert not Call(runtime.close_session_once).result()
        release.set()
        assert start.result() is True
        assert len(calls) == 1
    assert runtime.snapshot().session_established and not runtime.snapshot().ready
    assert not runtime.start_session_once()
    runtime.stop()
    assert runtime.close_session_once()
    rows = _history(sandbox)
    assert [row["action"] for row in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]
    assert rows[0]["target_id"] == rows[1]["target_id"]


def test_stop_during_scan_invalidates_publication_and_requires_scan_drain(
    postgres_product_sandbox, runtime, monkeypatch
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    with _paused_owner(sandbox, monkeypatch, "read_snapshot") as (entered, release, calls):
        scan = Call(runtime.verify_once)
        assert entered.wait(2)
        Call(runtime.stop).result()
        assert not Call(runtime.close_session_once).result()
        assert not runtime.verify_once() and not runtime.start_session_once()
        with pytest.raises(AuditRuntimeUnavailable):
            runtime.append_owned(_event())
        release.set()
        assert scan.result() is False
        assert len(calls) == 1
    assert not runtime.snapshot().ready and runtime.snapshot().stopping
    assert runtime.close_session_once()
    assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START, AuditAction.AUDIT_SESSION_CLOSE]


@pytest.mark.parametrize("borrowed", [False, True])
def test_stop_during_append_waits_for_owned_return_and_pending_publication(
    postgres_product_sandbox, runtime, monkeypatch, borrowed
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)

    def append():
        if borrowed:
            return sandbox.database.transact(
                lambda connection: runtime.append_borrowed(connection, _event())
            )
        return runtime.append_owned(_event())

    with _paused_owner(sandbox, monkeypatch) as (entered, release, calls):
        operation = Call(append)
        assert entered.wait(2)
        assert _history(sandbox) == before
        Call(runtime.stop).result()
        assert not Call(runtime.close_session_once).result()
        release.set()
        result = operation.result()
        assert len(calls) == 1
    assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START, AuditAction.AUDIT_LIST]
    if borrowed:
        assert not runtime.close_session_once()  # DB return alone does not drain publication.
        assert not runtime.publish_committed(result)
    assert not runtime.snapshot().ready
    assert runtime.close_session_once()
    assert _actions(sandbox) == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_LIST,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]


@pytest.mark.parametrize("stage", _STAGES)
def test_failed_or_unknown_close_is_never_replayed_or_used_to_close_pool(
    postgres_product_sandbox, runtime, monkeypatch, caplog, stage
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    runtime.stop()
    with _fault(sandbox, monkeypatch, stage) as calls:
        expected = CommitOutcomeUnknown if stage.startswith("unknown_") else AuditRuntimeUnavailable
        with pytest.raises(expected) as failure:
            runtime.close_session_once()
    _private(failure, runtime, caplog)
    assert len(calls) == 1
    assert not runtime.close_session_once()
    assert runtime.snapshot().stopping and not runtime.snapshot().ready
    persisted = stage in {"pool_exit", "unknown_commit"}
    assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START] + (
        [AuditAction.AUDIT_SESSION_CLOSE] if persisted else []
    )
    assert sandbox.database.read(lambda connection: connection.execute("SELECT 1").fetchone()) == (
        1,
    )


def test_close_in_flight_is_once_only_and_does_not_hold_the_status_lock(
    postgres_product_sandbox, runtime, monkeypatch
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    runtime.stop()
    with _paused_owner(sandbox, monkeypatch) as (entered, release, calls):
        close = Call(runtime.close_session_once)
        assert entered.wait(2)
        assert Call(runtime.snapshot).result().stopping
        assert not Call(runtime.close_session_once).result()
        assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START]
        release.set()
        assert close.result() is True
        assert len(calls) == 1
    assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START, AuditAction.AUDIT_SESSION_CLOSE]


@pytest.mark.parametrize("borrowed", [False, True])
def test_native_authority_fence_escapes_and_rolls_back_product_work(
    postgres_product_sandbox, runtime, borrowed, caplog
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)
    freeze_authority(sandbox.database, sandbox.authority)

    def product_write(connection):
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        return runtime.append_borrowed(connection, _event())

    with pytest.raises(AuditRuntimeUnavailable) as failure:
        if borrowed:
            sandbox.database.transact(product_write)
        else:
            runtime.append_owned(_event())
    _private(failure, runtime, caplog)
    assert runtime.snapshot().failure_code == "authority_fenced"
    assert not runtime.snapshot().eligible_to_attempt
    assert _history(sandbox) == before
    assert sandbox.admin.execute("SELECT clip_export_enabled FROM edge_site").fetchone() == (0,)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (AuthorityFenced(_SECRET), "authority_fenced"),
        (PostgresPoolBusy(_SECRET), "database_busy"),
        (PostgresStartupError(_SECRET), "database_startup_failed"),
        (PostgresTransactionStateError(_SECRET), "transaction_state"),
        (PostgresUnavailable(_SECRET), "database_unavailable"),
        (PostgresError(_SECRET), "database_error"),
        (AuditVerificationError(_SECRET), "verification_failed"),
        (psycopg.errors.CheckViolation(_SECRET), "database_error"),
        (OSError(_SECRET), "database_unavailable"),
        (RuntimeError(_SECRET), "operation_failed"),
        (CommitOutcomeUnknown(), "commit_outcome_unknown"),
    ],
    ids=[
        "authority",
        "busy",
        "startup",
        "state",
        "unavailable",
        "owner",
        "verify",
        "sql",
        "io",
        "other",
        "unknown",
    ],
)
def test_fail_open_recording_and_immutable_status_are_static_private_and_io_free(
    postgres_product_sandbox, runtime, audit_store, monkeypatch, caplog, error, code
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)

    def forbidden(*args, **kwargs):
        pytest.fail("status/fail-open recording attempted database work")

    with monkeypatch.context() as patch:
        for method in ("read", "read_snapshot", "transact"):
            patch.setattr(sandbox.database, method, forbidden)
        patch.setattr(audit_store, "verify", forbidden)
        ready = runtime.snapshot()
        runtime.record_failure(error)
        status = runtime.snapshot()
    assert ready.ready and not status.ready and not status.eligible_to_attempt
    assert status.failure_code == code and status.session_established
    assert status.indeterminate == isinstance(error, CommitOutcomeUnknown)
    with pytest.raises(FrozenInstanceError):
        status.ready = True
    assert _history(sandbox) == before
    text = repr(status) + caplog.text
    assert _SECRET not in text and before[0]["target_id"] not in text
    assert sandbox.dsn not in repr(status)
    assert set(status.__dataclass_fields__) == {
        "ready",
        "verification_current",
        "eligible_to_attempt",
        "session_established",
        "failure_code",
        "indeterminate",
        "stopping",
    }


class _Cancelled(BaseException):
    """Non-Exception cancellation; admission cleanup must still run."""


@pytest.mark.parametrize("operation", ["verify", "start", "append", "close"])
def test_cancellation_cleans_admission_and_propagates(
    postgres_product_sandbox, runtime, audit_store, monkeypatch, operation
) -> None:
    sandbox = postgres_product_sandbox
    if operation == "start":
        assert runtime.verify_once()
    else:
        _ready(runtime)

    def cancel(*args, **kwargs):
        raise _Cancelled("cancelled")

    if operation == "verify":
        monkeypatch.setattr(audit_store, "verify", cancel)
    elif operation == "start":
        monkeypatch.setattr(postgres_sessions, "start_session", cancel)
    elif operation == "append":
        monkeypatch.setattr(audit_store, "append", cancel)
    else:
        runtime.stop()
        monkeypatch.setattr(postgres_sessions, "close_session", cancel)

    with pytest.raises(_Cancelled, match="^cancelled$") as failure:
        if operation == "verify":
            runtime.verify_once()
        elif operation == "start":
            runtime.start_session_once()
        elif operation == "append":
            runtime.append_owned(_event("cancelled"))
        else:
            runtime.close_session_once()
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__
    assert _SECRET not in "".join(traceback.format_exception(failure.value))
    status = runtime.snapshot()
    assert status.failure_code == "operation_failed"
    assert not status.indeterminate and not status.ready
    runtime.stop()
    if operation == "start":
        assert not status.session_established
        assert not runtime.close_session_once()
        assert _actions(sandbox) == []
    elif operation == "close":
        assert not runtime.close_session_once()
        assert _actions(sandbox) == [AuditAction.AUDIT_SESSION_START]
    else:
        assert runtime.close_session_once()
        assert _actions(sandbox) == [
            AuditAction.AUDIT_SESSION_START,
            AuditAction.AUDIT_SESSION_CLOSE,
        ]


@pytest.mark.parametrize("cancelled", [False, True], ids=["failure", "cancellation"])
def test_failed_borrowed_append_local_drain_does_not_certify_outer_rollback(
    postgres_product_sandbox, runtime, audit_store, monkeypatch, cancelled
) -> None:
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)
    append, rollback = audit_store.append, psycopg.Connection.rollback
    entered, release, rolled_back = Event(), Event(), Event()
    at_exit, release_exit, draining = Event(), Event(), Event()
    acquire, close_pool = sandbox.database._pool.connection, sandbox.database._pool.close
    wait = sandbox.database._condition.wait
    calls, written, finalized, pool_closes = [], [], [], []
    closing = None
    event = _event("rolled-back-after-append")
    append_error = _Cancelled("cancelled") if cancelled else OSError("append failed")
    expected = _Cancelled if cancelled else AuditRuntimeUnavailable

    def fail_after_write(event, *, connection=None):
        assert connection is not None
        written.append(append(event, connection=connection).audit_id)
        raise append_error

    def product_write(connection):
        calls.append(connection.info.backend_pid)
        return runtime.append_borrowed(connection, event)

    def pause_rollback(connection):
        if connection.info.backend_pid not in calls or entered.is_set():
            return rollback(connection)
        try:
            assert len(written) == 1
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert connection.execute(
                "SELECT action,target_id FROM audit_events WHERE audit_id=%s", (written[0],)
            ).fetchone() == (AuditAction.AUDIT_LIST, event.target_id)
            entered.set()
            assert release.wait(3), "test did not release the borrowed transaction rollback"
        finally:
            rollback(connection)
            rolled_back.set()

    def outer_operation():
        with pytest.raises(expected) as failure:
            sandbox.database.transact(product_write)
        assert str(failure.value) == ("cancelled" if cancelled else "audit runtime unavailable")
        if cancelled:
            assert failure.value is append_error

    @contextmanager
    def paused_pool_exit(*, timeout):
        first = not at_exit.is_set()
        try:
            with acquire(timeout=timeout) as connection:
                yield connection
        finally:
            if first:
                assert rolled_back.is_set()
                at_exit.set()
                assert release_exit.wait(3), "test did not release outer pool exit"

    def observed_wait(timeout=None):
        draining.set()
        return wait(timeout)

    def checked_finalizer():
        assert rolled_back.is_set() and release_exit.is_set()
        assert runtime.close_session_once() is True
        finalized.append(1)

    def observed_pool_close(*, timeout):
        assert finalized == [1]
        pool_closes.append(1)
        close_pool(timeout=timeout)

    with monkeypatch.context() as patch:
        patch.setattr(audit_store, "append", fail_after_write)
        patch.setattr(psycopg.Connection, "rollback", pause_rollback)
        patch.setattr(sandbox.database._pool, "connection", paused_pool_exit)
        patch.setattr(sandbox.database._pool, "close", observed_pool_close)
        patch.setattr(sandbox.database._condition, "wait", observed_wait)
        operation = Call(outer_operation)
        try:
            assert entered.wait(2)
            assert len(calls) == 1
            assert operation.thread.is_alive()
            assert not rolled_back.is_set()
            # The failed callback consumed its token, not its owner's transaction.
            assert not runtime._pending
            assert sandbox.admin.execute(
                "SELECT state FROM pg_stat_activity WHERE pid=%s", (calls[0],)
            ).fetchone() == ("idle in transaction",)
            assert _history(sandbox) == before
            Call(runtime.stop).result()
            assert runtime.snapshot().stopping
            closing = Call(
                lambda: sandbox.database.close(timeout_sec=3.0, finalizer=checked_finalizer)
            )
            assert draining.wait(2)
            assert closing.thread.is_alive()
            assert not finalized and not pool_closes
            assert not sandbox.database._pool.closed
            release.set()
            assert at_exit.wait(2)
            assert rolled_back.is_set() and not runtime._pending
            assert operation.thread.is_alive() and closing.thread.is_alive()
            assert _history(sandbox) == before
            assert not finalized and not pool_closes
            assert not sandbox.database._pool.closed
        finally:
            release.set()
            release_exit.set()
            try:
                operation.result()
            finally:
                if closing is not None:
                    assert closing.result() is None

    assert rolled_back.is_set()
    assert finalized == pool_closes == [1]
    assert sandbox.database._pool.closed
    rows = _history(sandbox)
    assert rows[:-1] == before
    assert [row["action"] for row in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]
    assert rows[0]["target_id"] == rows[1]["target_id"]
    assert not runtime.close_session_once()
    assert _history(sandbox) == rows
    # Inspect the closed owner's chain through the existing independent test
    # connection, in a real committed RR/RO snapshot; never reopen admission.
    with sandbox.admin.transaction():
        sandbox.admin.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        checkpoint = _verify_snapshot(sandbox.admin, sandbox.schema, None)
    assert checkpoint.row_count == 2


@pytest.fixture
def mutation_owner(postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    return RuntimeSettingsStore(sandbox.database, sandbox.authority)


def _setting_event():
    return AuditEvent(
        occurred_at="2026-09-27T00:00:00Z",
        actor_id="test-operator",
        action=AuditAction.RUNTIME_SETTINGS_UPDATE,
        target_id="runtime-settings",
        detail=empty_detail(AuditAction.RUNTIME_SETTINGS_UPDATE),
    )


def _apply_setting(runtime, owner, *, enabled=True, expected_version=None):
    return runtime.apply_mutation(
        owner,
        _setting_event,
        lambda append: owner.set_clip_export_enabled(
            enabled, expected_version=expected_version, after_write=append
        ),
    )


def test_mutation_publishes_after_complete_owned_return_and_before_caller_work(
    runtime, mutation_owner, postgres_product_sandbox, monkeypatch
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    trace = []
    transact, publish = sandbox.database.transact, runtime.publish_committed

    def owned(callback):
        def within(connection):
            value = callback(connection)
            assert trace == []
            assert sandbox.admin.execute(
                "SELECT clip_export_enabled FROM edge_site WHERE id=1"
            ).fetchone() == (0,)
            return value

        value = transact(within)
        trace.append("complete-owned-return")
        return value

    def committed(token):
        assert trace == ["complete-owned-return"]
        assert sandbox.admin.execute(
            "SELECT clip_export_enabled FROM edge_site WHERE id=1"
        ).fetchone() == (1,)
        trace.append("published")
        return publish(token)

    monkeypatch.setattr(sandbox.database, "transact", owned)
    monkeypatch.setattr(runtime, "publish_committed", committed)
    result = _apply_setting(runtime, mutation_owner)
    trace.append("later-caller-work")
    assert result.clip_export_enabled
    assert trace == ["complete-owned-return", "published", "later-caller-work"]
    assert _actions(sandbox) == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.RUNTIME_SETTINGS_UPDATE,
    ]


def test_matching_mutation_audits_once_and_conflicting_mutation_audits_never(
    runtime, mutation_owner, postgres_product_sandbox
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    first = _apply_setting(runtime, mutation_owner)
    assert _apply_setting(runtime, mutation_owner, expected_version=first.version) == first
    assert _actions(sandbox).count(AuditAction.RUNTIME_SETTINGS_UPDATE) == 2
    before = _history(sandbox)
    with pytest.raises(RuntimeSettingsVersionConflict):
        _apply_setting(runtime, mutation_owner, expected_version=first.version - 1)
    assert _history(sandbox) == before and runtime.snapshot().ready
    assert not runtime._pending


def test_mutation_recovery_fence_and_newer_failure_keep_revision_order(
    runtime, mutation_owner, postgres_product_sandbox, monkeypatch
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    runtime.record_failure(OSError("earlier failure"))
    assert runtime.verify_once() and not runtime.snapshot().ready
    _apply_setting(runtime, mutation_owner)
    assert _actions(sandbox) == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.RECOVERY_FENCE,
        AuditAction.RUNTIME_SETTINGS_UPDATE,
    ]
    assert runtime.snapshot().ready
    transact = sandbox.database.transact

    def newer_failure_after_commit(callback):
        result = transact(callback)
        runtime.record_failure(OSError("newer failure"))
        return result

    monkeypatch.setattr(sandbox.database, "transact", newer_failure_after_commit)
    assert _apply_setting(runtime, mutation_owner).clip_export_enabled
    assert not runtime.snapshot().ready and not runtime._pending
    assert _actions(sandbox)[-1] == AuditAction.RUNTIME_SETTINGS_UPDATE


@pytest.mark.parametrize("state", ["unverified", "session_missing", "expired", "stopped"])
def test_mutation_requires_admission_before_entering_feature_owner(
    runtime, mutation_owner, clock, state
):
    if state == "session_missing":
        assert runtime.verify_once()
    elif state in ("expired", "stopped"):
        _ready(runtime)
        clock.advance(10) if state == "expired" else runtime.stop()
    entered = []
    with pytest.raises(AuditRuntimeUnavailable):
        runtime.apply_mutation(mutation_owner, _setting_event, lambda append: entered.append(True))
    assert not entered


@pytest.mark.parametrize("mismatch", ["database", "authority"])
def test_mutation_rejects_incoherent_feature_owner_before_invocation(
    runtime, postgres_product_sandbox, mismatch
):
    sandbox = postgres_product_sandbox
    other = PostgresDatabase(sandbox.dsn, sandbox.schema, sandbox.database._budget)
    owner = RuntimeSettingsStore(
        other if mismatch == "database" else sandbox.database,
        replace(sandbox.authority, generation=sandbox.authority.generation + 1)
        if mismatch == "authority"
        else sandbox.authority,
    )
    _ready(runtime)
    entered = []
    try:
        with pytest.raises(ValueError, match="share database and authority"):
            runtime.apply_mutation(owner, _setting_event, lambda append: entered.append(True))
        assert not entered and runtime.snapshot().ready
    finally:
        other.close(timeout_sec=3.0)


@pytest.mark.parametrize("candidate", [None, False, "not-a-token"])
def test_mutation_invalid_publication_aborts_before_commit(
    runtime, mutation_owner, postgres_product_sandbox, monkeypatch, candidate
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)
    monkeypatch.setattr(runtime, "append_borrowed", lambda connection, event: candidate)
    with pytest.raises(AuditRuntimeUnavailable, match="publication is invalid"):
        _apply_setting(runtime, mutation_owner)
    assert _history(sandbox) == before and not runtime.snapshot().ready
    assert sandbox.admin.execute(
        "SELECT clip_export_enabled FROM edge_site WHERE id=1"
    ).fetchone() == (0,)


def test_mutation_rejects_consumed_token_without_consuming_unrelated_work(
    runtime, mutation_owner, postgres_product_sandbox, monkeypatch
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    consumed = sandbox.database.transact(lambda c: runtime.append_borrowed(c, _event("consumed")))
    runtime.publish_committed(consumed)
    unrelated = sandbox.database.transact(lambda c: runtime.append_borrowed(c, _event("unrelated")))
    before = _history(sandbox)
    monkeypatch.setattr(runtime, "append_borrowed", lambda connection, event: consumed)
    with pytest.raises(AuditRuntimeUnavailable, match="publication is invalid"):
        _apply_setting(runtime, mutation_owner)
    assert _history(sandbox) == before
    runtime.validate_publication(unrelated)
    assert runtime.publish_committed(unrelated) is False
    assert not runtime._pending


def test_mutation_duplicate_callback_rolls_back_the_first_append(
    runtime, mutation_owner, postgres_product_sandbox
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)

    def write(append):
        def twice(connection):
            append(connection)
            append(connection)

        return mutation_owner.set_clip_export_enabled(True, after_write=twice)

    with pytest.raises(AuditRuntimeUnavailable, match="was repeated"):
        runtime.apply_mutation(mutation_owner, _setting_event, write)
    assert _history(sandbox) == before and not runtime._pending
    assert not mutation_owner.get().clip_export_enabled and not runtime.snapshot().ready


def test_missing_mutation_callback_refuses_success_without_claiming_rollback(
    runtime, mutation_owner, postgres_product_sandbox
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before = _history(sandbox)
    with pytest.raises(AuditRuntimeUnavailable, match="was not called"):
        # Deliberate feature-owner contract violation: a wrapper cannot undo
        # an already-committed owner that ignored its required callback.
        runtime.apply_mutation(
            mutation_owner,
            _setting_event,
            lambda append: mutation_owner.set_clip_export_enabled(True),
        )
    assert mutation_owner.get().clip_export_enabled
    assert _history(sandbox) == before and not runtime.snapshot().ready


def test_mutation_deferred_commit_rejection_never_publishes_success(
    runtime, mutation_owner, postgres_product_sandbox, monkeypatch
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    before, successful, failed = _history(sandbox), [], []
    publish_failed = runtime.publish_failed
    sandbox.admin.execute(
        "CREATE FUNCTION reject_setting() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'setting rejected' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_setting AFTER UPDATE ON edge_site "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_setting()"
    )

    def failure(token, error):
        assert _history(sandbox) == before
        failed.append(error)
        return publish_failed(token, error)

    monkeypatch.setattr(runtime, "publish_failed", failure)
    monkeypatch.setattr(runtime, "publish_committed", lambda token: successful.append(True))
    with pytest.raises(psycopg.errors.CheckViolation) as caught:
        _apply_setting(runtime, mutation_owner)
    assert failed == [caught.value] and not successful and not runtime._pending
    assert not mutation_owner.get().clip_export_enabled


@pytest.mark.parametrize("failure_kind", ["ordinary", "cancelled", "unknown"])
@pytest.mark.parametrize("secondary_invalid", [False, True])
def test_mutation_post_owned_failure_is_published_without_masking_or_replay(
    runtime,
    mutation_owner,
    postgres_product_sandbox,
    monkeypatch,
    caplog,
    failure_kind,
    secondary_invalid,
):
    sandbox = postgres_product_sandbox
    _ready(runtime)
    errors = {
        "ordinary": OSError("owned return " + _SECRET),
        "cancelled": BaseException("owned cancellation " + _SECRET),
        "unknown": CommitOutcomeUnknown(),
    }
    error = errors[failure_kind]
    original, publish = sandbox.database.transact, runtime.publish_failed
    writes, failures, successful = [], [], []

    def after_owned(callback):
        original(callback)
        writes.append(True)
        raise error

    def failure(token, original_error):
        failures.append(original_error)
        publish(token, original_error)
        if secondary_invalid:
            raise InvalidAuditPublication(_SECRET)

    monkeypatch.setattr(sandbox.database, "transact", after_owned)
    monkeypatch.setattr(runtime, "publish_failed", failure)
    monkeypatch.setattr(runtime, "publish_committed", lambda token: successful.append(True))
    with pytest.raises(type(error)) as caught:
        _apply_setting(runtime, mutation_owner)
    assert caught.value is error and writes == [True] and failures == [error]
    assert not successful and not runtime._pending
    assert not runtime.snapshot().ready
    assert runtime.snapshot().indeterminate is (failure_kind == "unknown")
    assert sandbox.admin.execute(
        "SELECT clip_export_enabled FROM edge_site WHERE id=1"
    ).fetchone() == (1,)
    assert _SECRET not in caplog.text
    if secondary_invalid:
        record = caplog.records[-1]
        assert (
            record.getMessage()
            == "mutation audit publication accounting failed after owned failure"
        )
        assert record.exc_info is None
