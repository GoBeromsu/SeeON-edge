"""Real PostgreSQL checks. A missing SEEON_TEST_POSTGRES_DSN fails; it never skips."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from queue import Queue
from threading import Event, Thread, get_ident
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.pq import TransactionStatus
from psycopg_pool import PoolTimeout
from psycopg_pool.pool import ReturnConnection, StopWorker, WaitingClient

from backend.app.edge_db import postgres
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PoolBudget,
    PostgresDatabase,
    PostgresError,
    PostgresPoolBusy,
    PostgresShutdownError,
    PostgresShutdownTimeout,
    PostgresStartupError,
    PostgresTransactionStateError,
    PostgresUnavailable,
)

pytest_plugins = ("tests_support.postgres_sandbox",)

_OPERATIONS = ("read", "read_snapshot", "transact")


class _Cancelled(BaseException):
    """Exercise cancellation without turning it into an ordinary exception."""


class Call:
    """Join every test worker with a bound and re-raise its actual outcome."""

    def __init__(self, callback):
        self.results = Queue()

        def run():
            try:
                self.results.put((True, callback()))
            except (
                AssertionError,
                PostgresError,
                ValueError,
                OSError,
                psycopg.Error,
                _Cancelled,
            ) as error:
                self.results.put((False, error))
            except BaseException as error:
                self.results.put((False, error))
                raise

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(3)
        assert not self.thread.is_alive(), "PostgreSQL test worker did not unwind"

    def result(self):
        self.join()
        success, result = self.results.get_nowait()
        if not success:
            raise result
        return result


def _observe_shutdown(owner, monkeypatch):
    draining, closes = Event(), []
    wait, close = owner._condition.wait, owner._pool.close

    def observe_wait(timeout=None):
        draining.set()
        return wait(timeout)

    def observe_close(*, timeout):
        closes.append(timeout)
        return close(timeout=timeout)

    monkeypatch.setattr(owner._condition, "wait", observe_wait)
    monkeypatch.setattr(owner._pool, "close", observe_close)
    return draining, closes


def _assert_stopped(owner):
    before = owner.stats().get("requests_num", 0)

    def forbidden(connection):
        pytest.fail("stopped owner entered a late callback")

    for operation in _OPERATIONS:
        with pytest.raises(PostgresUnavailable, match="not running"):
            getattr(owner, operation)(forbidden)
    assert owner.stats().get("requests_num", 0) == before


@dataclass
class _Sandbox:
    dsn: str = field(repr=False)
    connection: psycopg.Connection = field(repr=False)
    schema: str

    def owner(self, budget: PoolBudget | None = None) -> PostgresDatabase:
        return PostgresDatabase(self.dsn, self.schema, budget or _budget())


def _budget() -> PoolBudget:
    return PoolBudget(
        max_connections=1,
        max_waiting=1,
        acquire_timeout_sec=0.5,
        statement_timeout_ms=5_000,
        lock_timeout_ms=3_000,
        startup_timeout_sec=5.0,
    )


@pytest.fixture
def postgres_sandbox():
    dsn = os.environ.get("SEEON_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.fail(
            "SEEON_TEST_POSTGRES_DSN is required; point it at an isolated test database",
            pytrace=False,
        )
    if not dsn.strip() or "\x00" in dsn:
        pytest.fail("SEEON_TEST_POSTGRES_DSN must be nonblank without NUL bytes", pytrace=False)
    try:
        connection = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except (psycopg.Error, OSError, ValueError, TypeError):
        connection = None
    if connection is None:
        pytest.fail("the configured PostgreSQL test service is unavailable", pytrace=False)
    # An embedded quote proves schema selection uses Identifier, not interpolation.
    schema = 'seeon_test_"' + uuid4().hex
    try:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        connection.execute(
            sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(sql.Identifier(schema))
        )
        connection.execute("SET statement_timeout = '8s'")
        connection.execute("SET lock_timeout = '5s'")
        connection.execute(
            "CREATE TABLE committed_values (id bigint PRIMARY KEY, value text NOT NULL)"
        )
        connection.execute("CREATE TABLE parents (id bigint PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE children (id bigint PRIMARY KEY, parent_id bigint NOT NULL "
            "REFERENCES parents(id) DEFERRABLE INITIALLY DEFERRED)"
        )
        yield _Sandbox(dsn, connection, schema)
    finally:
        connection.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
        )
        connection.close()


@pytest.fixture
def database(postgres_sandbox):
    owner = postgres_sandbox.owner()
    try:
        owner.start()
        yield owner
    finally:
        owner.close(timeout_sec=3.0)


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    pause = Event()
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail("bounded PostgreSQL observation did not arrive")
        pause.wait(min(0.005, remaining))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_connections", 0),
        ("max_connections", True),
        ("max_connections", 1.5),
        ("max_waiting", 0),
        ("max_waiting", -1),
        ("statement_timeout_ms", 0),
        ("statement_timeout_ms", 2**31),
        ("lock_timeout_ms", -1),
        ("lock_timeout_ms", float("inf")),
        ("acquire_timeout_sec", 0),
        ("acquire_timeout_sec", float("nan")),
        ("acquire_timeout_sec", float("inf")),
        ("startup_timeout_sec", -1),
        ("startup_timeout_sec", True),
        ("startup_timeout_sec", float("inf")),
    ],
)
def test_pool_budget_rejects_unbounded_or_nonpositive_values(name, value):
    with pytest.raises(ValueError, match=name):
        replace(_budget(), **{name: value})


@pytest.mark.parametrize(
    ("conninfo", "error_type"),
    [
        (None, TypeError),
        (False, TypeError),
        (42, TypeError),
        (b"host=isolated.invalid", TypeError),
        ({}, TypeError),
        ("", ValueError),
        (" \t\r\n", ValueError),
        ("\x00", ValueError),
        ("password=test-invalid-input-sentinel\x00", ValueError),
    ],
)
def test_invalid_conninfo_is_rejected_before_pool_or_connection_creation(
    monkeypatch, caplog, conninfo, error_type
):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(1)
        pytest.fail("invalid connection information reached connection creation")

    monkeypatch.setattr(postgres, "ConnectionPool", forbidden)
    monkeypatch.setattr(psycopg, "connect", forbidden)
    with pytest.raises(error_type) as failure:
        PostgresDatabase(conninfo, "isolated_test", _budget())
    assert str(failure.value) == (
        "PostgreSQL connection information must be text"
        if error_type is TypeError
        else "PostgreSQL connection information must be nonblank without NUL bytes"
    )
    assert not attempts
    assert "test-invalid-input-sentinel" not in caplog.text


@pytest.mark.parametrize("sandbox_kind", ["owner", "product"])
@pytest.mark.parametrize("dsn", [None, "", " \t\r\n", "\x00", "password=test-sentinel\x00"])
def test_sandbox_rejects_missing_or_invalid_dsn_without_connecting(monkeypatch, sandbox_kind, dsn):
    from tests_support.postgres_sandbox import postgres_product_sandbox

    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(1)
        pytest.fail("sandbox attempted an ambient connection")

    monkeypatch.setattr(os, "environ", {} if dsn is None else {"SEEON_TEST_POSTGRES_DSN": dsn})
    monkeypatch.setattr(psycopg, "connect", forbidden)
    fixture_factory = postgres_sandbox if sandbox_kind == "owner" else postgres_product_sandbox
    fixture = fixture_factory.__wrapped__()
    try:
        # Skipped is caught too, so a regression to skipping fails here instead of hiding.
        with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as failure:
            next(fixture)
        assert failure.type is pytest.fail.Exception
        if dsn is None:
            assert "SEEON_TEST_POSTGRES_DSN is required" in str(failure.value)
        else:
            assert str(failure.value) == (
                "SEEON_TEST_POSTGRES_DSN must be nonblank without NUL bytes"
            )
        assert not attempts
    finally:
        fixture.close()


def test_schema_property_exposes_only_the_configured_namespace(database, postgres_sandbox):
    assert database.schema == postgres_sandbox.schema
    with pytest.raises(AttributeError):
        database.schema = "replacement"
    assert database.schema == postgres_sandbox.schema
    assert repr(database) == "<PostgresDatabase redacted>"


def test_commit_is_visible_on_an_independent_connection_before_return(database, postgres_sandbox):
    receipt = object()
    calls = 0

    def write(connection):
        nonlocal calls
        calls += 1
        connection.execute("INSERT INTO committed_values VALUES (1, 'committed')")
        return receipt

    assert database.transact(write) is receipt
    assert calls == 1
    assert postgres_sandbox.connection.execute("SELECT * FROM committed_values").fetchall() == [
        (1, "committed")
    ]
    assert database.stats()["requests_num"] >= 1


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_callback_exception_rolls_back_without_success(database, postgres_sandbox, operation):
    calls = 0
    key = uuid4().int % (2**63 - 1)

    def write(connection):
        nonlocal calls
        calls += 1
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'rolled back')")
        raise ValueError("callback aborted")

    with pytest.raises(ValueError, match="callback aborted"):
        getattr(database, operation)(write)
    assert calls == 1
    assert postgres_sandbox.connection.execute(
        "SELECT pg_try_advisory_xact_lock(%s)", (key,)
    ).fetchone() == (True,)
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (0,)
    assert database.read(lambda connection: connection.execute("SELECT 42").fetchone()) == (42,)
    assert getattr(database, operation)(
        lambda connection: connection.execute("SELECT 42").fetchone()
    ) == (42,)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_swallowed_statement_error_cannot_turn_rollback_into_success(
    database, postgres_sandbox, operation
):
    calls = 0

    def write(connection):
        nonlocal calls
        calls += 1
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'must roll back')")
        try:
            connection.execute("SELECT 1 / 0")
        except psycopg.errors.DivisionByZero:
            pass
        return "not a success"

    with pytest.raises(PostgresTransactionStateError):
        getattr(database, operation)(write)
    assert calls == 1
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (0,)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("managed_end", ["commit", "rollback"])
def test_callback_cannot_end_its_borrowed_transaction(database, operation, managed_end):
    calls = 0

    def end_transaction(connection):
        nonlocal calls
        calls += 1
        connection.execute("SELECT 42")
        getattr(connection, managed_end)()
        return "must not escape"

    with pytest.raises(PostgresTransactionStateError):
        getattr(database, operation)(end_transaction)
    assert calls == 1
    assert getattr(database, operation)(
        lambda connection: connection.execute("SELECT 42").fetchone()
    ) == (42,)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_failed_rollback_preserves_callback_error_and_discards_connection(
    database, postgres_sandbox, monkeypatch, operation
):
    callback_pids = []
    rollback_pids = []
    rollback = psycopg.Connection.rollback

    def fail_rollback(connection):
        pid = connection.info.backend_pid
        if pid in callback_pids:
            rollback_pids.append(pid)
            raise psycopg.OperationalError("injected rollback failure")
        rollback(connection)

    def abort(connection):
        callback_pids.append(connection.info.backend_pid)
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'must roll back')")
        else:
            connection.execute("SELECT * FROM committed_values")
        raise ValueError("original callback failure")

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "rollback", fail_rollback)
        with pytest.raises(ValueError, match="^original callback failure$"):
            getattr(database, operation)(abort)
    assert len(callback_pids) == 1
    assert rollback_pids == callback_pids
    _wait_for(
        lambda: (
            postgres_sandbox.connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (callback_pids[0],)
            ).fetchone()
            == (0,)
        )
    )
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (0,)
    assert database.read_snapshot(
        lambda connection: connection.execute("SELECT 42").fetchone()
    ) == (42,)


def test_deferred_constraint_rejection_is_known_commit_failure(database, postgres_sandbox):
    calls = 0

    def write(connection):
        nonlocal calls
        calls += 1
        connection.execute("INSERT INTO children VALUES (1, 999)")
        return "must not escape"

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        database.transact(write)
    assert calls == 1
    assert postgres_sandbox.connection.execute("SELECT count(*) FROM children").fetchone() == (0,)


@pytest.mark.parametrize("operation", ["read", "read_snapshot"])
def test_read_transaction_prohibits_writes(database, postgres_sandbox, operation):
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        getattr(database, operation)(
            lambda connection: connection.execute(
                "INSERT INTO committed_values VALUES (1, 'forbidden')"
            )
        )
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (0,)


@pytest.mark.parametrize("isolation", ["repeatable read", "serializable"])
@pytest.mark.parametrize("default_readonly", ["on", "off"])
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_transactions_override_nondefault_isolation(
    postgres_sandbox, isolation, default_readonly, operation
):
    options = (
        "-c default_transaction_isolation="
        + isolation.replace(" ", "\\ ")
        + f" -c default_transaction_read_only={default_readonly}"
    )
    owner = PostgresDatabase(
        make_conninfo(postgres_sandbox.dsn, options=options),
        postgres_sandbox.schema,
        _budget(),
    )
    try:
        owner.start()
        settings = getattr(owner, operation)(
            lambda connection: connection.execute(
                "SELECT current_setting('default_transaction_isolation'), "
                "current_setting('default_transaction_read_only'), "
                "current_setting('transaction_isolation'), "
                "current_setting('transaction_read_only')"
            ).fetchone()
        )
        assert settings == (
            isolation,
            default_readonly,
            "repeatable read" if operation == "read_snapshot" else "read committed",
            "off" if operation == "transact" else "on",
        )
        assert owner.read(
            lambda connection: connection.execute(
                "SELECT current_setting('default_transaction_isolation'), "
                "current_setting('default_transaction_read_only'), "
                "current_setting('transaction_isolation')"
            ).fetchone()
        ) == (isolation, default_readonly, "read committed")
    finally:
        owner.close(timeout_sec=3.0)


def test_snapshot_keeps_one_prefix_without_blocking_an_independent_owner_append(
    database, postgres_sandbox
):
    database.transact(
        lambda connection: connection.execute("INSERT INTO committed_values VALUES (1, 'first')")
    )
    writer = postgres_sandbox.owner()
    observed = Event()
    release = Event()
    reader_pids = []
    writer_pids = []
    receipt = object()

    def inspect_prefix(connection):
        reader_pids.append(connection.info.backend_pid)
        before = connection.execute("SELECT * FROM committed_values ORDER BY id").fetchall()
        observed.set()
        assert release.wait(4), "test failed to release its snapshot reader"
        after = connection.execute("SELECT * FROM committed_values ORDER BY id").fetchall()
        return before, after

    def append(connection):
        writer_pids.append(connection.info.backend_pid)
        connection.execute("INSERT INTO committed_values VALUES (2, 'append')")
        return receipt

    try:
        writer.start()
        with ThreadPoolExecutor(max_workers=2) as executor:
            reader = executor.submit(database.read_snapshot, inspect_prefix)
            try:
                assert observed.wait(2)
                pending_write = executor.submit(writer.transact, append)
                assert pending_write.result(timeout=2) is receipt
                assert not reader.done()
                assert postgres_sandbox.connection.execute(
                    "SELECT state, xact_start IS NOT NULL FROM pg_stat_activity WHERE pid = %s",
                    (reader_pids[0],),
                ).fetchone() == ("idle in transaction", True)
                assert postgres_sandbox.connection.execute(
                    "SELECT * FROM committed_values ORDER BY id"
                ).fetchall() == [(1, "first"), (2, "append")]
            finally:
                release.set()
            assert reader.result(timeout=2) == ([(1, "first")], [(1, "first")])
        assert len(reader_pids) == len(writer_pids) == 1
        assert reader_pids[0] != writer_pids[0]
        for operation in ("read", "read_snapshot"):
            assert getattr(database, operation)(
                lambda connection: connection.execute(
                    "SELECT * FROM committed_values ORDER BY id"
                ).fetchall()
            ) == [(1, "first"), (2, "append")]
    finally:
        release.set()
        writer.close(timeout_sec=3.0)


def test_advisory_lock_wait_refreshes_prior_snapshot(postgres_sandbox):
    owner = PostgresDatabase(
        make_conninfo(
            postgres_sandbox.dsn,
            options="-c default_transaction_isolation=repeatable\\ read",
        ),
        postgres_sandbox.schema,
        _budget(),
    )
    admin = postgres_sandbox.connection
    before_lock = Event()

    def inspect_after_lock(connection):
        before = connection.execute("SELECT count(*) FROM committed_values").fetchone()
        before_lock.set()
        connection.execute("SELECT pg_advisory_xact_lock(421773)")
        after = connection.execute("SELECT count(*) FROM committed_values").fetchone()
        return before, after

    try:
        owner.start()
        admin.execute("BEGIN")
        admin.execute("SELECT pg_advisory_xact_lock(421773)")
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(owner.transact, inspect_after_lock)
            assert before_lock.wait(2)
            admin.execute("INSERT INTO committed_values VALUES (1, 'concurrent admission')")
            admin.commit()
            assert pending.result(timeout=5) == ((0,), (1,))
    finally:
        admin.rollback()
        owner.close(timeout_sec=3.0)


def test_connections_have_durable_bounded_namespace_configuration(database, postgres_sandbox):
    row = database.read(
        lambda connection: connection.execute(
            "SELECT current_schema(), current_setting('TimeZone'), "
            "current_setting('synchronous_commit'), current_setting('fsync'), "
            "current_setting('full_page_writes'), "
            "(SELECT setting::bigint FROM pg_settings WHERE name = 'statement_timeout'), "
            "(SELECT setting::bigint FROM pg_settings WHERE name = 'lock_timeout'), "
            "(SELECT setting::bigint FROM pg_settings "
            "WHERE name = 'idle_in_transaction_session_timeout'), "
            "current_setting('transaction_read_only'), "
            "current_setting('session_replication_role')"
        ).fetchone()
    )
    assert row == (
        postgres_sandbox.schema,
        "UTC",
        "on",
        "on",
        "on",
        5_000,
        3_000,
        5_000,
        "on",
        "origin",
    )


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_explicit_temp_last_path_prevents_shadowing_real_relations(
    database, postgres_sandbox, operation
):
    def create_shadow(connection):
        connection.execute("INSERT INTO committed_values VALUES (1, 'real schema')")
        connection.execute("CREATE TEMP TABLE committed_values (id bigint, value text)")
        connection.execute("INSERT INTO pg_temp.committed_values VALUES (2, 'temporary shadow')")
        return connection.info.backend_pid

    pid = database.transact(create_shadow)

    def inspect(connection):
        assert connection.info.backend_pid == pid
        return (
            connection.execute("SHOW search_path").fetchone(),
            connection.execute("SELECT * FROM committed_values").fetchall(),
            connection.execute("SELECT * FROM pg_temp.committed_values").fetchall(),
        )

    assert getattr(database, operation)(inspect) == (
        (
            sql.SQL("{}, pg_catalog, pg_temp")
            .format(sql.Identifier(postgres_sandbox.schema))
            .as_string(postgres_sandbox.connection),
        ),
        [(1, "real schema")],
        [(2, "temporary shadow")],
    )


def test_product_bootstrap_keeps_temp_last_in_captured_audit_search_path(
    postgres_product_sandbox,
):
    sandbox = postgres_product_sandbox
    admin = sandbox.admin
    admin.execute("CREATE TEMP TABLE edge_site (id bigint)")
    admin.execute("INSERT INTO pg_temp.edge_site VALUES (99)")
    assert admin.execute("SELECT id FROM edge_site").fetchall() == [(1,)]
    assert admin.execute("SELECT id FROM pg_temp.edge_site").fetchall() == [(99,)]
    temp_schema = admin.execute(
        "SELECT nspname FROM pg_namespace WHERE oid = pg_my_temp_schema()"
    ).fetchone()[0]
    assert admin.execute("SELECT current_schemas(false)").fetchone() == (
        [sandbox.schema, "pg_catalog", temp_schema],
    )
    search_path = admin.execute("SHOW search_path").fetchone()[0]
    assert search_path.endswith(", pg_catalog, pg_temp")
    assert admin.execute(
        "SELECT p.proconfig FROM pg_proc AS p "
        "JOIN pg_namespace AS n ON n.oid = p.pronamespace "
        "WHERE n.nspname = %s AND p.proname = 'seeon_audit_insert'",
        (sandbox.schema,),
    ).fetchone() == ([f"search_path={search_path}"],)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_statement_timeout_rolls_back_earlier_callback_writes(postgres_sandbox, operation):
    owner = postgres_sandbox.owner(replace(_budget(), statement_timeout_ms=100))
    calls = []

    def write(connection):
        calls.append(1)
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'must roll back')")
        connection.execute("SELECT pg_sleep(1)")
        return "must not escape"

    try:
        owner.start()
        with pytest.raises(PostgresUnavailable):
            getattr(owner, operation)(write)
    finally:
        owner.close(timeout_sec=3.0)
    assert calls == [1]
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (0,)


def test_lock_timeout_bounds_contention_without_replay(postgres_sandbox):
    admin = postgres_sandbox.connection
    admin.execute("INSERT INTO committed_values VALUES (1, 'original')")
    owner = postgres_sandbox.owner(replace(_budget(), lock_timeout_ms=100))
    calls = []

    def write(connection):
        calls.append(1)
        connection.execute("UPDATE committed_values SET value = 'must roll back' WHERE id = 1")

    try:
        owner.start()
        with admin.transaction():
            admin.execute("SELECT * FROM committed_values WHERE id = 1 FOR UPDATE")
            with pytest.raises(PostgresUnavailable):
                owner.transact(write)
    finally:
        owner.close(timeout_sec=3.0)
    assert calls == [1]
    assert admin.execute("SELECT value FROM committed_values").fetchone() == ("original",)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_pool_bounds_waiters_and_acquisition_without_replaying(postgres_sandbox, operation):
    owner = postgres_sandbox.owner(replace(_budget(), acquire_timeout_sec=0.25))
    entered = Event()
    release = Event()
    queued_calls = []

    def hold(connection):
        connection.execute("INSERT INTO committed_values VALUES (1, 'held')")
        entered.set()
        assert release.wait(5), "test failed to release its own transaction"
        return "committed once"

    def queued(connection):
        queued_calls.append(connection)
        return "unexpected admission"

    try:
        owner.start()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(owner.transact, hold)
            try:
                assert entered.wait(2)
                second = executor.submit(getattr(owner, operation), queued)
                _wait_for(lambda: owner.stats().get("requests_waiting") == 1)
                with pytest.raises(PostgresPoolBusy):
                    getattr(owner, operation)(queued)
                with pytest.raises(PostgresPoolBusy):
                    second.result(timeout=2)
                assert not queued_calls
                assert owner.stats()["pool_size"] == 1
            finally:
                release.set()
            assert first.result(timeout=2) == "committed once"
    finally:
        release.set()
        owner.close(timeout_sec=3.0)
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (1,)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("fsync", "off"),
        ("full_page_writes", "off"),
        ("session_replication_role", "replica"),
        ("session_replication_role", "local"),
    ],
)
def test_startup_rejects_unsafe_server_options_and_closes(postgres_sandbox, setting, value, caplog):
    # PostgreSQL itself rejects startup changes to fsync/full_page_writes.
    # Session replication modes reach the owner's configuration check instead.
    # No ALTER SYSTEM or global configuration is touched.
    sentinel = "test-unsafe-settings-sentinel"
    unsafe = make_conninfo(
        postgres_sandbox.dsn,
        options=f"-c {setting}={value} -c application_name={sentinel}",
    )
    owner = PostgresDatabase(
        unsafe, postgres_sandbox.schema, replace(_budget(), startup_timeout_sec=0.2)
    )
    started = time.monotonic()
    try:
        with pytest.raises(PostgresStartupError) as failure:
            owner.start()
        assert str(failure.value) == "PostgreSQL bounded startup failed"
        assert time.monotonic() - started < 3
        assert owner._pool.closed
        for operation in _OPERATIONS:
            with pytest.raises(PostgresUnavailable):
                getattr(owner, operation)(lambda connection: connection.execute("SELECT 1"))
        assert unsafe not in caplog.text
        assert sentinel not in caplog.text
        if setting == "session_replication_role":
            assert value not in caplog.text
    finally:
        owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("role", ["replica", "local"])
def test_configuration_rejects_unsafe_replication_role_without_silently_changing_it(
    postgres_sandbox, role, caplog
):
    owner = postgres_sandbox.owner()
    admin = postgres_sandbox.connection
    original = admin.execute("SHOW session_replication_role").fetchone()[0]
    try:
        admin.execute(sql.SQL("SET session_replication_role TO {}").format(sql.Literal(role)))
        with pytest.raises(PostgresStartupError) as failure:
            owner._configure(admin)
        assert str(failure.value) == ("PostgreSQL durability or namespace configuration is unsafe")
        assert admin.execute("SHOW session_replication_role").fetchone() == (role,)
        assert role not in str(failure.value)
        assert role not in caplog.text
        assert postgres_sandbox.dsn not in caplog.text
    finally:
        admin.execute(sql.SQL("SET session_replication_role TO {}").format(sql.Literal(original)))
        owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("role", ["replica", "local"])
def test_owned_return_discards_unsafe_replication_role_instead_of_repairing_it(
    database, postgres_sandbox, role, caplog
):
    poisoned_pids = []

    def poison(connection):
        # Deliberate callback-contract violation: owned return must refuse it.
        poisoned_pids.append(connection.info.backend_pid)
        connection.execute(sql.SQL("SET session_replication_role TO {}").format(sql.Literal(role)))
        assert connection.execute("SHOW session_replication_role").fetchone() == (role,)

    with pytest.raises(PostgresStartupError) as failure:
        database.transact(poison)
    assert str(failure.value) == "PostgreSQL durability or namespace configuration is unsafe"
    assert len(poisoned_pids) == 1
    poisoned_pid = poisoned_pids[0]
    _wait_for(
        lambda: (
            postgres_sandbox.connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (poisoned_pid,)
            ).fetchone()
            == (0,)
        )
    )
    pid, actual_role = database.read_snapshot(
        lambda connection: connection.execute(
            "SELECT pg_backend_pid(), current_setting('session_replication_role')"
        ).fetchone()
    )
    assert pid != poisoned_pid
    assert actual_role == "origin"
    assert role not in caplog.text
    assert postgres_sandbox.dsn not in caplog.text


def test_missing_namespace_fails_configuration_and_closes_pool(postgres_sandbox):
    owner = PostgresDatabase(
        postgres_sandbox.dsn,
        postgres_sandbox.schema + "_missing",
        replace(_budget(), startup_timeout_sec=0.2),
    )
    try:
        with pytest.raises(PostgresStartupError, match="bounded startup failed"):
            owner.start()
        assert owner._pool.closed
    finally:
        owner.close(timeout_sec=3.0)


def test_connection_errors_and_representations_redact_conninfo(postgres_sandbox, caplog):
    sentinel = "test-connection-privacy-sentinel"
    invalid = make_conninfo(
        postgres_sandbox.dsn, options=f"-c nonexistent_seeon_setting={sentinel}"
    )
    owner = PostgresDatabase(
        invalid, postgres_sandbox.schema, replace(_budget(), startup_timeout_sec=0.2)
    )
    try:
        with pytest.raises(PostgresStartupError) as failure:
            owner.start()
        assert sentinel not in str(failure.value)
        assert sentinel not in repr(owner)
        assert sentinel not in caplog.text
        assert str(failure.value) == "PostgreSQL bounded startup failed"
    finally:
        owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_close_is_terminal_and_releases_test_owned_backend(postgres_sandbox, operation):
    owner = postgres_sandbox.owner()
    owner.start()
    pid = owner.read(lambda connection: connection.info.backend_pid)
    owner.close(timeout_sec=3.0)
    owner.close(timeout_sec=3.0)
    with pytest.raises(PostgresUnavailable):
        getattr(owner, operation)(lambda connection: connection.execute("SELECT 1"))
    with pytest.raises(PostgresStartupError, match="closed"):
        owner.start()
    _wait_for(
        lambda: (
            postgres_sandbox.connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (pid,)
            ).fetchone()
            == (0,)
        )
    )


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_termination_immediately_before_commit_is_unknown_not_retried(
    database, postgres_sandbox, operation
):
    calls = 0

    def write(connection):
        nonlocal calls
        calls += 1
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'unacknowledged')")
        else:
            connection.execute("SELECT * FROM committed_values")
        pid = connection.info.backend_pid
        assert pid != postgres_sandbox.connection.info.backend_pid
        assert postgres_sandbox.connection.execute(
            "SELECT pg_terminate_backend(%s, 1000)", (pid,)
        ).fetchone() == (True,)
        return "must not escape"

    with pytest.raises(CommitOutcomeUnknown, match="automatic retry is forbidden"):
        getattr(database, operation)(write)
    assert calls == 1
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (0,)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("committed", [False, True], ids=["rolled-back", "committed"])
@pytest.mark.parametrize("error_type", [psycopg.OperationalError, psycopg.InterfaceError])
def test_lost_commit_receipt_never_replays_or_returns_callback_result(
    database, postgres_sandbox, monkeypatch, operation, committed, error_type
):
    callback_pids = []
    commit_pids = []
    published = []
    commit = psycopg.Connection.commit

    def callback(connection):
        callback_pids.append(connection.info.backend_pid)
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'receipt lost')")
        else:
            connection.execute("SELECT * FROM committed_values").fetchall()
        return "must not escape"

    def lose_receipt(connection):
        pid = connection.info.backend_pid
        if pid in callback_pids:
            commit_pids.append(pid)
            if committed:
                commit(connection)
            else:
                connection.rollback()
            raise error_type("injected COMMIT receipt loss")
        commit(connection)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown) as failure:
            published.append(getattr(database, operation)(callback))
    assert str(failure.value) == (
        "PostgreSQL commit outcome is unknown; automatic retry is forbidden"
    )
    assert not published
    assert len(callback_pids) == 1
    assert commit_pids == callback_pids
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (int(committed and operation == "transact"),)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_pool_exit_failure_after_real_commit_does_not_release_a_result(
    database, postgres_sandbox, monkeypatch, operation
):
    acquire = database._pool.connection
    calls = []
    exits = []
    published = []

    @contextmanager
    def fail_release(*, timeout):
        with acquire(timeout=timeout) as connection:
            yield connection
            assert connection.info.transaction_status is TransactionStatus.IDLE
        exits.append(1)
        raise psycopg.OperationalError("injected pool release failure")

    def callback(connection):
        calls.append(1)
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'really committed')")
        else:
            connection.execute("SELECT * FROM committed_values").fetchall()
        return "must not escape"

    with monkeypatch.context() as patch:
        patch.setattr(database._pool, "connection", fail_release)
        with pytest.raises(PostgresUnavailable) as failure:
            published.append(getattr(database, operation)(callback))
    assert str(failure.value) == "PostgreSQL connection unavailable"
    assert calls == exits == [1]
    assert not published
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (int(operation == "transact"),)
    assert database.read_snapshot(
        lambda connection: connection.execute("SELECT 42").fetchone()
    ) == (42,)


def test_termination_while_server_is_executing_commit_is_unknown(database, postgres_sandbox):
    admin = postgres_sandbox.connection
    key = uuid4().int % (2**63 - 1)
    admin.execute(
        sql.SQL(
            "CREATE FUNCTION pause_test_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN PERFORM pg_advisory_xact_lock({}); RETURN NEW; END; $$"
        ).format(sql.Literal(key))
    )
    admin.execute(
        "CREATE CONSTRAINT TRIGGER pause_test_commit AFTER INSERT ON committed_values "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION pause_test_commit()"
    )
    admin.execute("SELECT pg_advisory_lock(%s)", (key,))
    pids = Queue()
    calls = []

    def write(connection):
        calls.append(1)
        connection.execute("INSERT INTO committed_values VALUES (1, 'commit in flight')")
        pids.put(connection.info.backend_pid)
        return "must not escape"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(database.transact, write)
        try:
            pid = pids.get(timeout=2)
            assert pid != admin.info.backend_pid
            _wait_for(
                lambda: (
                    admin.execute(
                        "SELECT wait_event = 'advisory' AND query = 'COMMIT' "
                        "FROM pg_stat_activity WHERE pid = %s",
                        (pid,),
                    ).fetchone()
                    == (True,)
                )
            )
            assert admin.execute("SELECT pg_terminate_backend(%s, 1000)", (pid,)).fetchone() == (
                True,
            )
            with pytest.raises(CommitOutcomeUnknown) as failure:
                future.result(timeout=2)
            assert (
                str(failure.value)
                == "PostgreSQL commit outcome is unknown; automatic retry is forbidden"
            )
        finally:
            admin.execute("SELECT pg_advisory_unlock(%s)", (key,))
    assert calls == [1]
    assert admin.execute("SELECT count(*) FROM committed_values").fetchone() == (0,)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("cancelled", [False, True])
def test_shutdown_reserves_work_before_checkout_and_releases_acquisition_cancellation(
    database, monkeypatch, operation, cancelled
):
    acquire = database._pool.connection
    entered, release = Event(), Event()
    draining, closes = _observe_shutdown(database, monkeypatch)
    callbacks, finalized = [], []
    cancellation = _Cancelled("acquisition cancelled")
    closing = None

    @contextmanager
    def paused_acquisition(*, timeout):
        entered.set()
        assert release.wait(3), "test did not release acquisition"
        if cancelled:
            raise cancellation
        with acquire(timeout=timeout) as connection:
            yield connection

    def callback(connection):
        callbacks.append(1)
        return connection.execute("SELECT 42").fetchone()

    monkeypatch.setattr(database._pool, "connection", paused_acquisition)
    operation_call = Call(lambda: getattr(database, operation)(callback))
    try:
        assert entered.wait(2)
        closing = Call(
            lambda: database.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        )
        assert draining.wait(2)
        _assert_stopped(database)
        assert Call(database.stop_admission).result() is None
        assert Call(database.stats).result()["pool_size"] == 1
        assert not callbacks and not finalized and not closes
        assert closing.thread.is_alive()
    finally:
        release.set()
        operation_call.join()
        if closing is not None:
            closing.join()
    if cancelled:
        with pytest.raises(_Cancelled) as failure:
            operation_call.result()
        assert failure.value is cancellation
        assert not callbacks
    else:
        assert operation_call.result() == (42,)
        assert callbacks == [1]
    assert closing.result() is None
    assert finalized == [1] and len(closes) == 1


def test_finalizer_has_only_one_same_thread_same_pool_write(database, monkeypatch):
    pool = database._pool
    pid = database.read(lambda connection: connection.info.backend_pid)
    checkouts, finalizations = [], []
    acquire = pool.connection
    closer_thread = get_ident()

    @contextmanager
    def observed_checkout(*, timeout):
        with acquire(timeout=timeout) as connection:
            checkouts.append(connection.info.backend_pid)
            yield connection

    def forbidden(connection):
        pytest.fail("finalizer privilege escaped its single write scope")

    def write(connection):
        assert get_ident() == closer_thread
        assert database._pool is pool
        assert connection.info.backend_pid == pid
        assert connection.execute(
            "SELECT current_setting('transaction_isolation'), "
            "current_setting('transaction_read_only')"
        ).fetchone() == ("read committed", "off")
        _assert_stopped(database)  # Nested operations, including transact, are forbidden.
        with pytest.raises(PostgresShutdownError, match="own scope"):
            database.close(timeout_sec=3.0)
        connection.execute("INSERT INTO committed_values VALUES (1, 'finalized')")
        return "finalizer committed"

    def finalize():
        finalizations.append(1)
        assert get_ident() == closer_thread
        with pytest.raises(PostgresShutdownError, match="own scope"):
            database.close(timeout_sec=3.0)
        for method in ("read", "read_snapshot"):
            with pytest.raises(PostgresUnavailable):
                getattr(database, method)(forbidden)
        assert Call(lambda: _assert_stopped(database)).result() is None
        assert database.transact(write) == "finalizer committed"
        _assert_stopped(database)  # The second write cannot reuse the privilege.

    monkeypatch.setattr(pool, "connection", observed_checkout)
    database.close(timeout_sec=3.0, finalizer=finalize)
    database.close(timeout_sec=3.0, finalizer=finalize)
    _assert_stopped(database)
    assert finalizations == [1] and checkouts == [pid]
    assert pool.closed


@pytest.mark.parametrize(
    "outcome", ["failure", "cancellation", "unknown_rollback", "unknown_commit", "exit_error"]
)
@pytest.mark.parametrize("swallowed", [False, True])
def test_finalizer_transaction_failure_is_latched_even_when_callback_swallows_it(
    postgres_sandbox, monkeypatch, caplog, outcome, swallowed
):
    # Deliberately poisoned finalizers never belong to the shared owner fixture.
    owner = postgres_sandbox.owner()
    finalizations, callbacks, faults, closes = [], [], [], []
    cancellation = _Cancelled("finalizer cancelled")
    secret = "private-finalizer-error password=must-not-escape SQL"
    expected = (
        CommitOutcomeUnknown
        if outcome.startswith("unknown_")
        else _Cancelled
        if outcome == "cancellation" and not swallowed
        else PostgresShutdownError
    )
    try:
        owner.start()
        acquire, commit, close = (
            owner._pool.connection,
            psycopg.Connection.commit,
            owner._pool.close,
        )

        @contextmanager
        def failing_exit(*, timeout):
            with acquire(timeout=timeout) as connection:
                yield connection
            if outcome == "exit_error":
                raise psycopg.OperationalError(secret)

        def commit_outcome(connection):
            if outcome.startswith("unknown_") and callbacks and not faults:
                faults.append(1)
                if outcome == "unknown_commit":
                    commit(connection)
                else:
                    connection.rollback()
                raise psycopg.OperationalError(secret)
            return commit(connection)

        def observe_close(*, timeout):
            closes.append(timeout)
            return close(timeout=timeout)

        def write(connection):
            callbacks.append(1)
            connection.execute("INSERT INTO committed_values VALUES (1, 'finalizer')")
            if outcome == "failure":
                raise ValueError(secret)
            if outcome == "cancellation":
                raise cancellation

        def finalize():
            finalizations.append(1)
            try:
                owner.transact(write)
            except BaseException:
                if not swallowed:
                    raise

        monkeypatch.setattr(owner._pool, "connection", failing_exit)
        monkeypatch.setattr(owner._pool, "close", observe_close)
        monkeypatch.setattr(psycopg.Connection, "commit", commit_outcome)
        with pytest.raises(expected) as failure:
            owner.close(timeout_sec=3.0, finalizer=finalize)
        if expected is _Cancelled:
            assert failure.value is cancellation
        assert secret not in str(failure.value) + caplog.text
        assert failure.value.__cause__ is None
        assert failure.value.__suppress_context__
        assert owner._pool.closed
        later_error = (
            CommitOutcomeUnknown if outcome.startswith("unknown_") else PostgresShutdownError
        )
        for finalizer in (None, finalize):
            with pytest.raises(later_error):
                owner.close(timeout_sec=3.0, finalizer=finalizer)
        _assert_stopped(owner)
        assert finalizations == callbacks == [1]
        assert len(closes) == 1
        assert postgres_sandbox.connection.execute(
            "SELECT count(*) FROM committed_values"
        ).fetchone() == (int(outcome in ("unknown_commit", "exit_error")),)
    finally:
        # Cleanup cannot erase the deliberately latched finalizer failure.
        with suppress(PostgresShutdownError, CommitOutcomeUnknown):
            owner.close(timeout_sec=3.0)


@pytest.mark.parametrize(
    "timeout", [None, True, False, 0, -1, "3", float("inf"), -float("inf"), float("nan"), 10**1000]
)
def test_shutdown_timeout_is_required_and_validated_before_stopping(database, timeout):
    before = database.stats()["requests_num"]
    with pytest.raises(ValueError, match="^timeout_sec must be finite and positive$"):
        database.close(timeout_sec=timeout)
    with pytest.raises(TypeError):
        database.close()
    assert database.stats()["requests_num"] == before
    assert database.read(lambda connection: connection.execute("SELECT 42").fetchone()) == (42,)


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("cancelled", [False, True])
def test_interrupted_drain_stops_admission_without_disposal_and_can_resume(
    database, monkeypatch, operation, cancelled
):
    entered, release = Event(), Event()
    draining, closes = _observe_shutdown(database, monkeypatch)
    finalized = []
    closing = None
    cancellation = _Cancelled("drain cancelled")

    def callback(connection):
        connection.execute("SELECT 42")
        entered.set()
        assert release.wait(3), "test did not release admitted callback"
        return "completed original work"

    def cancel_wait(timeout=None):
        raise cancellation

    work = Call(lambda: getattr(database, operation)(callback))
    try:
        assert entered.wait(2)
        with monkeypatch.context() as patch:
            if cancelled:
                patch.setattr(database._condition, "wait", cancel_wait)
            with pytest.raises(_Cancelled if cancelled else PostgresShutdownTimeout) as failure:
                database.close(timeout_sec=0.02, finalizer=lambda: finalized.append(1))
            if cancelled:
                assert failure.value is cancellation
        assert not finalized and not closes and not database._pool.closed
        _assert_stopped(database)
        assert Call(database.stop_admission).result() is None
        draining.clear()
        closing = Call(
            lambda: database.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        )
        assert draining.wait(2)
        with pytest.raises(PostgresShutdownError, match="already in progress"):
            database.close(timeout_sec=3.0)
        assert not finalized and not closes
    finally:
        release.set()
        work.join()
        if closing is not None:
            closing.join()
    assert work.result() == "completed original work"
    assert closing.result() is None
    assert finalized == [1] and len(closes) == 1


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("stage", ["acquisition", "callback", "pool_exit"])
def test_reentrant_close_refuses_through_entire_operation_without_stopping_owner(
    database, monkeypatch, operation, stage
):
    acquire = database._pool.connection
    refusals = []

    def refuse():
        with pytest.raises(PostgresShutdownError, match="own scope"):
            database.close(timeout_sec=3.0)
        refusals.append(1)

    @contextmanager
    def observed_scope(*, timeout):
        if stage == "acquisition":
            refuse()
        with acquire(timeout=timeout) as connection:
            yield connection
        if stage == "pool_exit":
            refuse()

    def callback(connection):
        if stage == "callback":
            refuse()
        return connection.execute("SELECT 42").fetchone()

    with monkeypatch.context() as patch:
        patch.setattr(database._pool, "connection", observed_scope)
        assert getattr(database, operation)(callback) == (42,)
    assert refusals == [1]
    assert database.read_snapshot(
        lambda connection: connection.execute("SELECT 43").fetchone()
    ) == (43,)


@pytest.mark.parametrize("stop_first", [False, True])
def test_shutdown_before_start_needs_no_finalizer_and_is_terminal(
    postgres_sandbox, monkeypatch, stop_first
):
    owner = postgres_sandbox.owner()
    finalized = []
    _, closes = _observe_shutdown(owner, monkeypatch)
    if stop_first:
        owner.stop_admission()
        owner.stop_admission()
    owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
    owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
    assert len(closes) == 1 and not finalized
    _assert_stopped(owner)
    with pytest.raises(PostgresStartupError, match="closed"):
        owner.start()


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("queue_outcome", ["success", "timeout", "cancellation"])
def test_shutdown_drains_sole_connection_queue_after_exhaustion_or_cancellation(
    postgres_sandbox, monkeypatch, operation, queue_outcome
):
    owner = postgres_sandbox.owner(
        replace(_budget(), acquire_timeout_sec=1.0 if queue_outcome == "success" else 0.25)
    )
    entered, release = Event(), Event()
    queued_entered, release_queued = Event(), Event()
    callbacks, finalized = [], []
    first = queued = closing = None
    try:
        owner.start()
        draining, closes = _observe_shutdown(owner, monkeypatch)
        acquire = owner._pool.connection

        @contextmanager
        def cancellable_acquisition(*, timeout):
            try:
                with acquire(timeout=timeout) as connection:
                    yield connection
            except PoolTimeout:
                if queue_outcome == "cancellation":
                    raise _Cancelled("queued acquisition cancelled") from None
                raise

        monkeypatch.setattr(owner._pool, "connection", cancellable_acquisition)

        def hold(connection):
            connection.execute("INSERT INTO committed_values VALUES (1, 'held')")
            entered.set()
            assert release.wait(3), "test did not release held transaction"
            return "committed once"

        def late(connection):
            pytest.fail("exhausted queue entered a late callback")

        def queued_callback(connection):
            callbacks.append(1)
            if operation == "transact":
                connection.execute("INSERT INTO committed_values VALUES (2, 'queued')")
            else:
                assert connection.execute("SELECT count(*) FROM committed_values").fetchone() == (
                    1,
                )
            queued_entered.set()
            assert release_queued.wait(3), "test did not release the admitted queue waiter"
            return "queued once"

        first = Call(lambda: owner.transact(hold))
        assert entered.wait(2)
        queued = Call(lambda: getattr(owner, operation)(queued_callback))
        _wait_for(lambda: owner.stats().get("requests_waiting") == 1)
        with pytest.raises(PostgresPoolBusy):
            getattr(owner, operation)(late)
        closing = Call(lambda: owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1)))
        assert draining.wait(2)
        _assert_stopped(owner)
        if queue_outcome == "success":
            release.set()
            assert queued_entered.wait(2)
            assert callbacks == [1] and queued.thread.is_alive()
        else:
            with pytest.raises(_Cancelled if queue_outcome == "cancellation" else PostgresPoolBusy):
                queued.result()
            assert not callbacks and first.thread.is_alive()
        assert not finalized and not closes and closing.thread.is_alive()
    finally:
        release.set()
        release_queued.set()
        for call in (first, queued, closing):
            if call is not None:
                call.join()
        owner.close(timeout_sec=3.0)
    assert first.result() == "committed once"
    if queue_outcome == "success":
        assert queued.result() == "queued once"
    assert closing.result() is None
    assert finalized == [1] and len(closes) == 1
    expected = [(1, "held")]
    if operation == "transact" and queue_outcome == "success":
        expected.append((2, "queued"))
    assert (
        postgres_sandbox.connection.execute("SELECT * FROM committed_values ORDER BY id").fetchall()
        == expected
    )


def test_shutdown_finalizer_cannot_overtake_server_blocked_commit(
    database, postgres_sandbox, monkeypatch
):
    admin = postgres_sandbox.connection
    key = uuid4().int % (2**63 - 1)
    admin.execute(
        sql.SQL(
            "CREATE FUNCTION pause_shutdown_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN PERFORM pg_advisory_xact_lock({}); RETURN NEW; END; $$"
        ).format(sql.Literal(key))
    )
    admin.execute(
        "CREATE CONSTRAINT TRIGGER pause_shutdown_commit AFTER INSERT ON committed_values "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION pause_shutdown_commit()"
    )
    admin.execute("SELECT pg_advisory_lock(%s)", (key,))
    pids, finalized = Queue(), []
    draining, closes = _observe_shutdown(database, monkeypatch)
    writing = closing = None

    def write(connection):
        connection.execute("INSERT INTO committed_values VALUES (1, 'ordinary')")
        pids.put(connection.info.backend_pid)
        return "known commit"

    def finalize():
        assert admin.execute("SELECT * FROM committed_values").fetchall() == [(1, "ordinary")]
        database.transact(
            lambda connection: connection.execute(
                "INSERT INTO committed_values VALUES (2, 'finalizer')"
            )
        )
        finalized.append(1)

    try:
        writing = Call(lambda: database.transact(write))
        pid = pids.get(timeout=2)
        _wait_for(
            lambda: (
                admin.execute(
                    "SELECT wait_event = 'advisory' AND query = 'COMMIT' "
                    "FROM pg_stat_activity WHERE pid = %s",
                    (pid,),
                ).fetchone()
                == (True,)
            )
        )
        closing = Call(lambda: database.close(timeout_sec=3.0, finalizer=finalize))
        assert draining.wait(2)
        _assert_stopped(database)
        assert writing.thread.is_alive() and closing.thread.is_alive()
        assert not finalized and not closes
        assert admin.execute("SELECT count(*) FROM committed_values").fetchone() == (0,)
    finally:
        admin.execute("SELECT pg_advisory_unlock(%s)", (key,))
        for call in (writing, closing):
            if call is not None:
                call.join()
    assert writing.result() == "known commit"
    assert closing.result() is None
    assert finalized == [1] and len(closes) == 1
    assert admin.execute("SELECT * FROM committed_values ORDER BY id").fetchall() == [
        (1, "ordinary"),
        (2, "finalizer"),
    ]


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("exit_error", ["none", "ordinary", "cancellation"])
@pytest.mark.parametrize(
    "outcome",
    [
        "known",
        "unknown_rollback",
        "unknown_commit",
        "rollback_failure",
        "begin_error",
        "begin_cancellation",
        "callback_cancellation",
    ],
)
def test_shutdown_keeps_lease_through_real_cleanup_and_paused_pool_exit(
    database, postgres_sandbox, monkeypatch, operation, outcome, exit_error
):
    acquire = database._pool.connection
    commit, rollback, execute = (
        psycopg.Connection.commit,
        psycopg.Connection.rollback,
        psycopg.Connection.execute,
    )
    at_exit, release = Event(), Event()
    draining, closes = _observe_shutdown(database, monkeypatch)
    calls, published, finalized, faulted = [], [], [], []
    closing = None
    receipt = object()
    cancellation = _Cancelled("transaction cancelled")
    cleanup_cancellation = _Cancelled("pool exit cancelled")

    @contextmanager
    def held_exit(*, timeout):
        first = not at_exit.is_set()
        try:
            with acquire(timeout=timeout) as connection:
                yield connection
        finally:
            if first:
                at_exit.set()
                assert release.wait(3), "test did not release pool exit"
                if exit_error == "ordinary":
                    raise psycopg.OperationalError("test pool exit failure")
                if exit_error == "cancellation":
                    raise cleanup_cancellation

    def fail_begin(connection, query, *args, **kwargs):
        if outcome.startswith("begin_") and isinstance(query, str) and query.startswith("BEGIN "):
            if not faulted:
                faulted.append(1)
                if outcome == "begin_cancellation":
                    execute(connection, query, *args, **kwargs)
                    raise cancellation
                # A real server error before the callback, not a fake transaction.
                return execute(connection, "SELECT 1 / 0")
        return execute(connection, query, *args, **kwargs)

    def commit_outcome(connection):
        if outcome.startswith("unknown_") and calls and not faulted:
            faulted.append(1)
            if outcome == "unknown_commit":
                commit(connection)
            else:
                rollback(connection)
            raise psycopg.OperationalError("test COMMIT receipt loss")
        return commit(connection)

    def failed_rollback(connection):
        if outcome == "rollback_failure" and calls and not faulted:
            faulted.append(1)
            raise psycopg.OperationalError("test rollback failure")
        return rollback(connection)

    def callback(connection):
        calls.append(connection.info.backend_pid)
        if operation == "transact":
            connection.execute("INSERT INTO committed_values VALUES (1, 'real work')")
        else:
            connection.execute("SELECT 42")
        if outcome == "rollback_failure":
            raise ValueError("original callback failure")
        if outcome == "callback_cancellation":
            raise cancellation
        return receipt

    def run():
        published.append(getattr(database, operation)(callback))

    monkeypatch.setattr(database._pool, "connection", held_exit)
    monkeypatch.setattr(psycopg.Connection, "execute", fail_begin)
    monkeypatch.setattr(psycopg.Connection, "commit", commit_outcome)
    monkeypatch.setattr(psycopg.Connection, "rollback", failed_rollback)
    work = Call(run)
    try:
        assert at_exit.wait(2)
        closing = Call(
            lambda: database.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        )
        assert draining.wait(2)
        _assert_stopped(database)
        assert not published and not finalized and not closes
        assert work.thread.is_alive() and closing.thread.is_alive()
        persisted = operation == "transact" and outcome in ("known", "unknown_commit")
        _wait_for(
            lambda: (
                postgres_sandbox.connection.execute(
                    "SELECT count(*) FROM committed_values"
                ).fetchone()
                == (int(persisted),)
            )
        )
        if outcome == "rollback_failure":
            _wait_for(
                lambda: (
                    postgres_sandbox.connection.execute(
                        "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (calls[0],)
                    ).fetchone()
                    == (0,)
                )
            )
    finally:
        release.set()
        work.join()
        if closing is not None:
            closing.join()
    if outcome == "known" and exit_error == "none":
        assert work.result() is None
        assert published == [receipt]
    else:
        expected = {
            "known": PostgresUnavailable,
            "unknown_rollback": CommitOutcomeUnknown,
            "unknown_commit": CommitOutcomeUnknown,
            "rollback_failure": ValueError,
            "begin_error": psycopg.errors.DivisionByZero,
            "begin_cancellation": _Cancelled,
            "callback_cancellation": _Cancelled,
        }[outcome]
        expected_cancellation = cancellation
        if exit_error == "cancellation" and expected not in (CommitOutcomeUnknown, _Cancelled):
            expected = _Cancelled
            expected_cancellation = cleanup_cancellation
        with pytest.raises(expected) as failure:
            work.result()
        if expected is _Cancelled:
            assert failure.value is expected_cancellation
        assert not published
    assert len(calls) == int(not outcome.startswith("begin_"))
    assert closing.result() is None
    assert finalized == [1] and len(closes) == 1


@pytest.mark.parametrize("privileged", [False, True])
@pytest.mark.parametrize("stage", ["rollback", "pool_exit"])
def test_cleanup_cancellation_survives_an_ordinary_callback_failure(
    postgres_sandbox, monkeypatch, privileged, stage
):
    owner = postgres_sandbox.owner()
    at_exit, release = Event(), Event()
    calls, finalized, faulted = [], [], []
    work = closing = None
    cancellation = _Cancelled("cleanup cancelled")
    try:
        owner.start()
        pid = owner.read(lambda connection: connection.info.backend_pid)
        acquire, rollback = owner._pool.connection, psycopg.Connection.rollback
        draining, closes = _observe_shutdown(owner, monkeypatch)

        @contextmanager
        def held_exit(*, timeout):
            try:
                with acquire(timeout=timeout) as connection:
                    yield connection
            finally:
                at_exit.set()
                assert release.wait(3), "test did not release cancelled cleanup"
                if stage == "pool_exit":
                    raise cancellation

        def cancelled_rollback(connection):
            should_cancel = (
                stage == "rollback" and connection.info.backend_pid == pid and calls and not faulted
            )
            result = rollback(connection)
            if should_cancel:
                faulted.append(1)
                raise cancellation
            return result

        def write(connection):
            calls.append(1)
            connection.execute("INSERT INTO committed_values VALUES (1, 'tentative')")
            raise ValueError("ordinary callback failure")

        def finalize():
            finalized.append(1)
            owner.transact(write)

        monkeypatch.setattr(owner._pool, "connection", held_exit)
        monkeypatch.setattr(psycopg.Connection, "rollback", cancelled_rollback)
        work = Call(
            lambda: (
                owner.close(timeout_sec=3.0, finalizer=finalize)
                if privileged
                else owner.transact(write)
            )
        )
        assert at_exit.wait(2)
        if not privileged:
            closing = Call(lambda: owner.close(timeout_sec=3.0))
            assert draining.wait(2)
        _assert_stopped(owner)
        assert not closes and work.thread.is_alive()
        assert postgres_sandbox.connection.execute(
            "SELECT count(*) FROM committed_values"
        ).fetchone() == (0,)
    finally:
        release.set()
        for call in (work, closing):
            if call is not None:
                call.join()
        with suppress(PostgresShutdownError):
            owner.close(timeout_sec=3.0)
    with pytest.raises(_Cancelled) as failure:
        work.result()
    assert failure.value is cancellation
    assert calls == [1] and len(closes) == 1
    if privileged:
        assert finalized == [1]
        with pytest.raises(PostgresShutdownError):
            owner.close(timeout_sec=3.0, finalizer=finalize)
        assert finalized == [1]
    else:
        assert closing.result() is None and not finalized


@pytest.mark.parametrize("stage", ["open", "checkout", "pool_exit"])
@pytest.mark.parametrize("outcome", ["stopped", "failure", "cancellation"])
def test_close_waits_for_startup_reservation_without_publishing_running(
    postgres_sandbox, monkeypatch, stage, outcome
):
    owner = postgres_sandbox.owner()
    entered, release = Event(), Event()
    draining, closes = _observe_shutdown(owner, monkeypatch)
    open_pool, acquire = owner._pool.open, owner._pool.connection
    finalized, pids = [], []
    cancellation = _Cancelled("startup cancelled")
    closing = None

    def pause():
        entered.set()
        assert release.wait(3), "test did not release startup"
        if outcome == "failure":
            raise OSError("private startup failure")
        if outcome == "cancellation":
            raise cancellation

    def paused_open(*, wait):
        open_pool(wait=wait)
        if stage == "open":
            pause()

    @contextmanager
    def paused_checkout(*, timeout):
        with acquire(timeout=timeout) as connection:
            pids.append(connection.info.backend_pid)
            if stage == "checkout":
                pause()
            yield connection
        if stage == "pool_exit":
            pause()

    monkeypatch.setattr(owner._pool, "open", paused_open)
    monkeypatch.setattr(owner._pool, "connection", paused_checkout)
    starting = Call(owner.start)
    try:
        assert entered.wait(2)
        # A second start must not block behind the real startup checkout/I/O.
        with pytest.raises(PostgresStartupError, match="already in progress"):
            Call(owner.start).result()
        closing = Call(lambda: owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1)))
        assert draining.wait(2)
        assert Call(owner.stop_admission).result() is None
        assert isinstance(Call(owner.stats).result(), dict)
        _assert_stopped(owner)
        with pytest.raises(PostgresShutdownError, match="already in progress"):
            Call(lambda: owner.close(timeout_sec=3.0)).result()
        assert not finalized and not closes
        assert starting.thread.is_alive() and closing.thread.is_alive()
    finally:
        release.set()
        starting.join()
        if closing is not None:
            closing.join()
        owner.close(timeout_sec=3.0)
    with pytest.raises(
        _Cancelled if outcome == "cancellation" else PostgresStartupError
    ) as failure:
        starting.result()
    if outcome == "cancellation":
        assert failure.value is cancellation
    else:
        assert "private startup failure" not in str(failure.value)
    assert closing.result() is None
    assert len(closes) == 1 and not finalized
    _assert_stopped(owner)
    with pytest.raises(PostgresStartupError, match="closed"):
        owner.start()
    for pid in pids:
        _wait_for(
            lambda pid=pid: (
                postgres_sandbox.connection.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (pid,)
                ).fetchone()
                == (0,)
            )
        )


@pytest.mark.parametrize("phase", ["startup", "operation"])
@pytest.mark.parametrize("state", ["autocommit", "transaction"])
def test_foreground_checkout_rejects_non_idle_session_before_callback(
    postgres_sandbox, monkeypatch, phase, state
):
    owner = postgres_sandbox.owner()
    if phase == "operation":
        owner.start()
    acquire = owner._pool.connection
    borrowed, pids, callbacks = [], [], []

    @contextmanager
    def unsafe_checkout(*, timeout):
        with acquire(timeout=timeout) as connection:
            borrowed.append(connection)
            pids.append(connection.info.backend_pid)
            if state == "autocommit":
                connection.autocommit = False
            else:
                connection.execute("BEGIN")
            yield connection

    monkeypatch.setattr(owner._pool, "connection", unsafe_checkout)
    try:
        with pytest.raises(
            PostgresStartupError if phase == "startup" else PostgresTransactionStateError
        ):
            if phase == "startup":
                owner.start()
            else:
                owner.transact(lambda connection: callbacks.append(True))
        assert not callbacks and len(borrowed) == 1 and borrowed[0].closed
        _wait_for(
            lambda: (
                postgres_sandbox.connection.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE pid=%s", (pids[0],)
                ).fetchone()
                == (0,)
            )
        )
    finally:
        owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("phase", ["startup", "operation"])
@pytest.mark.parametrize("cleanup_at", ["connection_close", "pool_exit"])
@pytest.mark.parametrize(
    "body_cancelled,cleanup_cancelled", [(True, False), (False, True), (False, False)]
)
def test_foreground_configuration_failure_keeps_precedence_through_complete_exit(
    postgres_sandbox, monkeypatch, phase, cleanup_at, body_cancelled, cleanup_cancelled
):
    owner = postgres_sandbox.owner()
    if phase == "operation":
        owner.start()
    acquire = owner._pool.connection
    body = (
        _Cancelled("configuration cancelled")
        if body_cancelled
        else ValueError("configuration failed")
    )
    cleanup = _Cancelled("cleanup cancelled") if cleanup_cancelled else OSError("cleanup failed")
    expected = body if body_cancelled or not cleanup_cancelled else cleanup
    borrowed, pids, callbacks, closed = [], [], [], []

    def fail_configuration(connection):
        raise body

    @contextmanager
    def controlled_exit(*, timeout):
        try:
            with acquire(timeout=timeout) as connection:
                borrowed.append(connection)
                pids.append(connection.info.backend_pid)
                if cleanup_at == "connection_close":
                    real_close = type(connection).close

                    def fail_after_close(target):
                        real_close(target)
                        if target is connection and not closed:
                            closed.append(True)
                            raise cleanup

                    monkeypatch.setattr(type(connection), "close", fail_after_close)
                yield connection
        finally:
            if cleanup_at == "pool_exit":
                raise cleanup

    monkeypatch.setattr(owner, "_configure", fail_configuration)
    monkeypatch.setattr(owner._pool, "connection", controlled_exit)
    try:
        translated = phase == "startup" and isinstance(expected, Exception)
        with pytest.raises(PostgresStartupError if translated else type(expected)) as caught:
            if phase == "startup":
                owner.start()
            else:
                owner.transact(lambda connection: callbacks.append(True))
        if translated:
            assert str(caught.value) == "PostgreSQL bounded startup failed"
        else:
            assert caught.value is expected
        assert not callbacks and len(borrowed) == 1 and borrowed[0].closed
        _wait_for(
            lambda: (
                postgres_sandbox.connection.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE pid=%s", (pids[0],)
                ).fetchone()
                == (0,)
            )
        )
    finally:
        owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("primary_kind", ["none", "ordinary", "cancellation"])
@pytest.mark.parametrize("cleanup_cancelled", [False, True])
@pytest.mark.parametrize("pool_exit_kind", ["none", "ordinary", "cancellation"])
def test_synchronous_return_validation_preserves_owned_exception_precedence(
    database, postgres_sandbox, monkeypatch, primary_kind, cleanup_cancelled, pool_exit_kind
):
    primary = {
        "none": None,
        "ordinary": ValueError("callback failed"),
        "cancellation": _Cancelled("callback cancelled"),
    }[primary_kind]
    cleanup = (
        _Cancelled("return validation cancelled")
        if cleanup_cancelled
        else OSError("return validation failed")
    )
    expected = (
        primary
        if primary_kind == "cancellation"
        else (cleanup if primary is None or cleanup_cancelled else primary)
    )
    pool_error = {
        "none": None,
        "ordinary": OSError("later pool exit failed"),
        "cancellation": _Cancelled("later pool exit cancelled"),
    }[pool_exit_kind]
    if isinstance(expected, Exception) and pool_exit_kind == "cancellation":
        expected = pool_error
    configure = database._configure
    acquire = database._pool.connection
    calls, borrowed = [], []

    def controlled_configuration(connection):
        configure(connection)
        calls.append(True)
        if len(calls) == 2:
            raise cleanup

    def callback(connection):
        borrowed.append(connection)
        connection.execute("INSERT INTO committed_values VALUES (1,'value')")
        if primary is not None:
            raise primary
        return "must not escape failed owned return"

    @contextmanager
    def later_exit_failure(*, timeout):
        try:
            with acquire(timeout=timeout) as connection:
                yield connection
        finally:
            if pool_error is not None:
                raise pool_error

    monkeypatch.setattr(database, "_configure", controlled_configuration)
    monkeypatch.setattr(database._pool, "connection", later_exit_failure)
    with pytest.raises(type(expected)) as caught:
        database.transact(callback)
    assert caught.value is expected
    assert len(calls) == 2 and len(borrowed) == 1 and borrowed[0].closed
    assert postgres_sandbox.connection.execute(
        "SELECT count(*) FROM committed_values"
    ).fetchone() == (1 if primary is None else 0,)


@pytest.mark.parametrize("fail_checkout", [False, True])
def test_close_disposes_checked_out_connection_without_background_reset_gc(
    postgres_sandbox, monkeypatch, fail_checkout
):
    owner = postgres_sandbox.owner()
    schedule, acquire = owner._pool.run_task, owner._pool.connection
    pending_returns, connections, pids = [], [], []

    def queue_at_shutdown(task):
        if isinstance(task, ReturnConnection):
            pending_returns.append(task)
        else:
            if isinstance(task, StopWorker):
                # Deterministically exercise a return task first scheduled
                # after close set pool.closed. Keep its connection referenced;
                # neither successful close nor admission drain may rely on GC.
                for pending in pending_returns:
                    schedule(pending)
            schedule(task)

    @contextmanager
    def capture_checkout(*, timeout):
        with acquire(timeout=timeout) as connection:
            connections.append(connection)
            pids.append(connection.info.backend_pid)
            if fail_checkout:
                raise OSError("controlled startup checkout failure")
            yield connection

    monkeypatch.setattr(owner._pool, "run_task", queue_at_shutdown)
    monkeypatch.setattr(owner._pool, "connection", capture_checkout)
    try:
        if fail_checkout:
            with pytest.raises(PostgresStartupError):
                owner.start()
        else:
            owner.start()
        assert len(connections) == 1
        pid = pids[0]
        owner.close(timeout_sec=3.0)
        assert connections[0].closed, "owned close left a returned connection open"
        _wait_for(
            lambda: (
                postgres_sandbox.connection.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE pid=%s", (pid,)
                ).fetchone()
                == (0,)
            )
        )
    finally:
        for connection in connections:
            connection.close()
        owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_startup_cleans_once_outside_state_lock_and_preserves_cancellation(
    postgres_sandbox, monkeypatch, cancelled
):
    owner = postgres_sandbox.owner()
    acquire, close_pool = owner._pool.connection, owner._pool.close
    cleaning, release = Event(), Event()
    closes, pids = [], []
    cancellation = _Cancelled("startup cancelled")

    @contextmanager
    def failed_checkout(*, timeout):
        with acquire(timeout=timeout) as connection:
            pids.append(connection.info.backend_pid)
            connection.execute("SELECT 42")
            if cancelled:
                raise cancellation
            connection.execute("SELECT 1 / 0")
            yield connection

    def paused_cleanup(*, timeout):
        closes.append(timeout)
        cleaning.set()
        assert release.wait(3), "test did not release startup cleanup"
        close_pool(timeout=timeout)

    monkeypatch.setattr(owner._pool, "connection", failed_checkout)
    monkeypatch.setattr(owner._pool, "close", paused_cleanup)
    starting = Call(owner.start)
    try:
        assert cleaning.wait(2)
        assert Call(owner.stop_admission).result() is None
        _assert_stopped(owner)
        with pytest.raises(PostgresShutdownError, match="already in progress"):
            Call(lambda: owner.close(timeout_sec=3.0)).result()
        with pytest.raises(PostgresStartupError, match="closed"):
            Call(owner.start).result()
    finally:
        release.set()
        starting.join()
        owner.close(timeout_sec=3.0)
    with pytest.raises(_Cancelled if cancelled else PostgresStartupError) as failure:
        starting.result()
    if cancelled:
        assert failure.value is cancellation
    assert len(closes) == 1 and len(pids) == 1
    _wait_for(
        lambda: (
            postgres_sandbox.connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (pids[0],)
            ).fetchone()
            == (0,)
        )
    )


def test_startup_drain_timeout_never_disposes_a_live_checkout(postgres_sandbox, monkeypatch):
    owner = postgres_sandbox.owner()
    acquire = owner._pool.connection
    entered, release = Event(), Event()
    _, closes = _observe_shutdown(owner, monkeypatch)
    finalized = []

    @contextmanager
    def held_startup(*, timeout):
        with acquire(timeout=timeout) as connection:
            entered.set()
            assert release.wait(3), "test did not release startup checkout"
            yield connection

    monkeypatch.setattr(owner._pool, "connection", held_startup)
    starting = Call(owner.start)
    try:
        assert entered.wait(2)
        with pytest.raises(PostgresShutdownTimeout):
            owner.close(timeout_sec=0.02, finalizer=lambda: finalized.append(1))
        assert not closes and not finalized and not owner._pool.closed
        _assert_stopped(owner)
    finally:
        release.set()
        starting.join()
        owner.close(timeout_sec=3.0)
    with pytest.raises(PostgresStartupError):
        starting.result()
    assert len(closes) == 1 and not finalized


def test_shutdown_uses_one_remaining_budget_across_drain_finalizer_and_pool(database, monkeypatch):
    clock = [100.0]
    entered, release, draining = Event(), Event(), Event()
    wait, close_pool = database._condition.wait, database._pool.close
    budgets, finalized = [], []
    closing = None

    def observed_wait(timeout=None):
        budgets.append(("drain", timeout))
        draining.set()
        result = wait(timeout)
        clock[0] += 1.0
        return result

    def finalize():
        finalized.append(1)
        database.transact(lambda connection: connection.execute("SELECT 42").fetchone())
        clock[0] += 0.5

    def observed_close(*, timeout):
        budgets.append(("pool", timeout))
        close_pool(timeout=timeout)

    def work(connection):
        connection.execute("SELECT 42")
        entered.set()
        assert release.wait(3), "test did not release the budget probe"

    monkeypatch.setattr(postgres, "monotonic", lambda: clock[0])
    monkeypatch.setattr(database._condition, "wait", observed_wait)
    monkeypatch.setattr(database._pool, "close", observed_close)
    operation = Call(lambda: database.read(work))
    try:
        assert entered.wait(2)
        closing = Call(lambda: database.close(timeout_sec=3.0, finalizer=finalize))
        assert draining.wait(2)
    finally:
        release.set()
        operation.join()
        if closing is not None:
            closing.join()
    assert operation.result() is None
    assert closing.result() is None
    assert finalized == [1]
    assert budgets == [("drain", 3.0), ("pool", 1.5)]


def test_finalizer_overrun_is_not_interruptible_or_replayed_and_later_cleanup_still_fails(
    postgres_sandbox, monkeypatch
):
    owner = postgres_sandbox.owner()
    clock = [100.0]
    entered, release = Event(), Event()
    finalizations = []
    closing = None
    try:
        owner.start()
        _, closes = _observe_shutdown(owner, monkeypatch)
        monkeypatch.setattr(postgres, "monotonic", lambda: clock[0])

        def finalize():
            finalizations.append(1)
            owner.transact(
                lambda connection: connection.execute(
                    "INSERT INTO committed_values VALUES (1, 'finalizer committed')"
                )
            )
            entered.set()
            assert release.wait(3), "test did not release synchronous finalizer"

        closing = Call(lambda: owner.close(timeout_sec=3.0, finalizer=finalize))
        assert entered.wait(2)
        clock[0] += 4.0
        assert Call(owner.stop_admission).result() is None
        _assert_stopped(owner)
        with pytest.raises(PostgresShutdownError, match="already in progress"):
            owner.close(timeout_sec=3.0)
        assert not closes and closing.thread.is_alive()
        release.set()
        with pytest.raises(PostgresShutdownTimeout):
            closing.result()
        assert not closes and not owner._pool.closed
        with pytest.raises(PostgresShutdownTimeout, match="finalizer exceeded"):
            owner.close(timeout_sec=3.0, finalizer=finalize)
        assert len(closes) == 1 and owner._pool.closed
        with pytest.raises(PostgresShutdownTimeout, match="finalizer exceeded"):
            owner.close(timeout_sec=3.0)
        assert finalizations == [1] and len(closes) == 1
        assert postgres_sandbox.connection.execute("SELECT * FROM committed_values").fetchall() == [
            (1, "finalizer committed")
        ]
    finally:
        release.set()
        if closing is not None:
            closing.join()
        with suppress(PostgresShutdownError):
            owner.close(timeout_sec=3.0)


def test_pool_disposal_after_deadline_is_not_a_successful_shutdown(postgres_sandbox, monkeypatch):
    owner = postgres_sandbox.owner()
    clock = [100.0]
    closes, finalized = [], []
    try:
        owner.start()
        close_pool = owner._pool.close
        monkeypatch.setattr(postgres, "monotonic", lambda: clock[0])

        def late_close(*, timeout):
            closes.append(timeout)
            close_pool(timeout=timeout)
            clock[0] += 4.0

        monkeypatch.setattr(owner._pool, "close", late_close)
        with pytest.raises(PostgresShutdownTimeout):
            owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        assert owner._pool.closed
        for _ in range(2):
            with pytest.raises(PostgresShutdownTimeout, match="pool cleanup exceeded"):
                owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        assert closes == [3.0] and finalized == [1]
    finally:
        with suppress(PostgresShutdownError):
            owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("cancelled", [False, True])
def test_pool_cleanup_failure_never_becomes_success_or_repeats_finalization(
    postgres_sandbox, monkeypatch, cancelled
):
    owner = postgres_sandbox.owner()
    closes, finalized = [], []
    cancellation = _Cancelled("pool cleanup cancelled")
    try:
        owner.start()
        close_pool = owner._pool.close

        def failed_close(*, timeout):
            close_pool(timeout=timeout)
            closes.append(1)
            if len(closes) == 1:
                if cancelled:
                    raise cancellation
                raise OSError("private pool cleanup failure")

        monkeypatch.setattr(owner._pool, "close", failed_close)
        with pytest.raises(_Cancelled if cancelled else PostgresShutdownError) as failure:
            owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        if cancelled:
            assert failure.value is cancellation
        assert "private" not in str(failure.value)
        assert owner._pool.closed
        with pytest.raises(PostgresShutdownError, match="pool cleanup"):
            owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        assert closes == [1, 1] and finalized == [1]
        with pytest.raises(PostgresShutdownError, match="pool cleanup"):
            owner.close(timeout_sec=3.0)
        assert closes == [1, 1]
    finally:
        with suppress(PostgresShutdownError):
            owner.close(timeout_sec=3.0)


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_completed_nested_operation_does_not_release_outer_thread_depth(
    postgres_sandbox, operation
):
    owner = postgres_sandbox.owner(replace(_budget(), max_connections=2, acquire_timeout_sec=1.0))
    pids = []

    def inner(connection):
        pids.append(connection.info.backend_pid)
        return connection.execute("SELECT 42").fetchone()

    def outer(connection):
        pids.append(connection.info.backend_pid)
        assert getattr(owner, operation)(inner) == (42,)
        with pytest.raises(PostgresShutdownError, match="own scope"):
            owner.close(timeout_sec=3.0)
        return connection.execute("SELECT 43").fetchone()

    try:
        owner.start()
        assert getattr(owner, operation)(outer) == (43,)
        assert len(pids) == 2 and pids[0] != pids[1]
    finally:
        owner.close(timeout_sec=3.0)


def test_expired_drain_deadline_does_not_consume_uninvoked_finalizer(database, monkeypatch):
    clock = [100.0]
    entered, release, draining = Event(), Event(), Event()
    wait = database._condition.wait
    _, closes = _observe_shutdown(database, monkeypatch)
    finalized = []
    closing = None

    def expire_after_wait(timeout=None):
        draining.set()
        result = wait(timeout)
        clock[0] += 4.0
        return result

    def work(connection):
        connection.execute("SELECT 42")
        entered.set()
        assert release.wait(3), "test did not release deadline probe"

    monkeypatch.setattr(postgres, "monotonic", lambda: clock[0])
    monkeypatch.setattr(database._condition, "wait", expire_after_wait)
    operation = Call(lambda: database.read(work))
    try:
        assert entered.wait(2)
        closing = Call(
            lambda: database.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
        )
        assert draining.wait(2)
    finally:
        release.set()
        operation.join()
        if closing is not None:
            closing.join()
    assert operation.result() is None
    with pytest.raises(PostgresShutdownTimeout):
        closing.result()
    assert not finalized and not closes
    database.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1))
    assert finalized == [1] and closes == [3.0]


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_live_pool_waiter_cancellation_keeps_other_admitted_work_draining(
    postgres_sandbox, monkeypatch, operation
):
    owner = postgres_sandbox.owner(replace(_budget(), acquire_timeout_sec=3.0))
    held, release = Event(), Event()
    waiting = Queue()
    finalized, callbacks = [], []
    first = queued = closing = None
    cancellation = _Cancelled("live waiter cancelled")
    wait = WaitingClient.wait

    def observed_wait(client, timeout):
        waiting.put(client)
        return wait(client, timeout)

    def hold(connection):
        connection.execute("INSERT INTO committed_values VALUES (1, 'held')")
        held.set()
        assert release.wait(3), "test did not release held transaction"

    def forbidden(connection):
        callbacks.append(1)
        pytest.fail("cancelled pool waiter entered a transaction callback")

    try:
        owner.start()
        draining, closes = _observe_shutdown(owner, monkeypatch)
        first = Call(lambda: owner.transact(hold))
        assert held.wait(2)
        # Observe the real enqueued WaitingClient. Cancellation is delivered
        # through its condition/error path, not converted from a PoolTimeout.
        monkeypatch.setattr(WaitingClient, "wait", observed_wait)
        queued = Call(lambda: getattr(owner, operation)(forbidden))
        client = waiting.get(timeout=2)
        assert owner.stats()["requests_waiting"] == 1
        closing = Call(lambda: owner.close(timeout_sec=3.0, finalizer=lambda: finalized.append(1)))
        assert draining.wait(2)
        assert client.fail(cancellation)
        with pytest.raises(_Cancelled) as failure:
            queued.result()
        assert failure.value is cancellation
        _assert_stopped(owner)
        assert not callbacks and not finalized and not closes
        assert closing.thread.is_alive()
    finally:
        release.set()
        for call in (first, queued, closing):
            if call is not None:
                call.join()
        owner.close(timeout_sec=3.0)
    assert first.result() is None
    assert closing.result() is None
    assert finalized == [1] and len(closes) == 1
    assert postgres_sandbox.connection.execute(
        "SELECT value FROM committed_values WHERE id=1"
    ).fetchone() == ("held",)


@pytest.mark.parametrize("exit_error", [False, True])
def test_privileged_finalizer_commit_cannot_overtake_its_pool_exit(
    postgres_sandbox, monkeypatch, exit_error
):
    owner = postgres_sandbox.owner()
    entered, release = Event(), Event()
    finalized, published = [], []
    closing = None
    try:
        owner.start()
        acquire = owner._pool.connection
        _, closes = _observe_shutdown(owner, monkeypatch)

        @contextmanager
        def paused_exit(*, timeout):
            with acquire(timeout=timeout) as connection:
                yield connection
                assert connection.info.transaction_status is TransactionStatus.IDLE
                entered.set()
                assert release.wait(3), "test did not release privileged pool exit"
                if exit_error:
                    raise psycopg.OperationalError("synthetic privileged exit failure")

        def finalize():
            finalized.append(1)
            result = owner.transact(
                lambda connection: connection.execute(
                    "INSERT INTO committed_values VALUES (1, 'closed') RETURNING value"
                ).fetchone()
            )
            published.append(result)

        monkeypatch.setattr(owner._pool, "connection", paused_exit)
        closing = Call(lambda: owner.close(timeout_sec=3.0, finalizer=finalize))
        assert entered.wait(2)
        assert postgres_sandbox.connection.execute(
            "SELECT value FROM committed_values WHERE id=1"
        ).fetchone() == ("closed",)
        _assert_stopped(owner)
        assert not published and not closes
        assert closing.thread.is_alive()
    finally:
        release.set()
        if closing is not None:
            closing.join()
        with suppress(PostgresShutdownError):
            owner.close(timeout_sec=3.0)
    if exit_error:
        with pytest.raises(PostgresShutdownError):
            closing.result()
        with pytest.raises(PostgresShutdownError):
            owner.close(timeout_sec=3.0, finalizer=finalize)
        assert not published
    else:
        assert closing.result() is None
        owner.close(timeout_sec=3.0, finalizer=finalize)
        assert published == [("closed",)]
    assert finalized == [1] and len(closes) == 1
