"""Real PostgreSQL checks. Missing SEEON_TEST_POSTGRES_DSN is not field acceptance."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from queue import Queue
from threading import Event
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.pq import TransactionStatus

from backend.app.edge_db import postgres
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PoolBudget,
    PostgresDatabase,
    PostgresPoolBusy,
    PostgresStartupError,
    PostgresTransactionStateError,
    PostgresUnavailable,
)

pytest_plugins = ("tests_support.postgres_sandbox",)

_OPERATIONS = ("read", "read_snapshot", "transact")


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
        pytest.skip("requires an isolated SEEON_TEST_POSTGRES_DSN; not field acceptance")
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
        owner.close()


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
def test_sandbox_missing_opt_in_skips_but_explicit_invalid_input_fails_without_connecting(
    monkeypatch, sandbox_kind, dsn
):
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
        error_type = pytest.skip.Exception if dsn is None else pytest.fail.Exception
        with pytest.raises(error_type) as failure:
            next(fixture)
        if dsn is None:
            assert "requires an isolated SEEON_TEST_POSTGRES_DSN" in str(failure.value)
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
        owner.close()


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
        writer.close()


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
        owner.close()


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
        owner.close()
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
        owner.close()
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
        owner.close()
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
        owner.close()


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
        owner.close()


@pytest.mark.parametrize("role", ["replica", "local"])
def test_pool_reset_discards_unsafe_replication_role_instead_of_repairing_it(
    database, postgres_sandbox, role, caplog
):
    def poison(connection):
        # Deliberate callback-contract violation to exercise the real pool reset.
        connection.execute(sql.SQL("SET session_replication_role TO {}").format(sql.Literal(role)))
        assert connection.execute("SHOW session_replication_role").fetchone() == (role,)
        return connection.info.backend_pid

    poisoned_pid = database.transact(poison)
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
    assert "PostgreSQL durability or namespace configuration is unsafe" in caplog.text
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
        owner.close()


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
        owner.close()


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_close_is_terminal_and_releases_test_owned_backend(postgres_sandbox, operation):
    owner = postgres_sandbox.owner()
    owner.start()
    pid = owner.read(lambda connection: connection.info.backend_pid)
    owner.close()
    owner.close()
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
