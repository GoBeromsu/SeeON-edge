"""Native credential component tests; not HTTP/lifespan or deployment admission.

Real database cases require the isolated PostgreSQL sandbox. A missing DSN
fails; it never skips. Owner-return injections below do not simulate or
prove actual wire-level COMMIT loss.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.pq import TransactionStatus

from backend.app.edge_db.authority import AuthorityFenced, AuthorityToken, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown, PoolBudget, PostgresDatabase
from backend.app.shared import dashboard_credentials as credentials
from backend.app.shared.dashboard_credentials import (
    DashboardCredentialsStoreError,
    PersistedDashboardCredentials,
)
from backend.app.shared.postgres_dashboard_credentials import PostgresDashboardCredentialsStore

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_TIME = "2026-09-27T04:01:00.456Z"
_COLUMNS = "username,algorithm,salt,password_hash,updated_at"


class _Cancelled(BaseException):
    pass


@contextmanager
def _independent_database(sandbox: ProductSandbox) -> Iterator[PostgresDatabase]:
    database = PostgresDatabase(
        sandbox.dsn,
        sandbox.schema,
        PoolBudget(
            max_connections=2,
            max_waiting=4,
            acquire_timeout_sec=1.0,
            statement_timeout_ms=5000,
            lock_timeout_ms=3000,
            startup_timeout_sec=5.0,
        ),
    )
    try:
        database.start()
        yield database
    finally:
        database.close(timeout_sec=3.0)


def _row(sandbox: ProductSandbox):
    return sandbox.admin.execute(f"SELECT {_COLUMNS} FROM credentials WHERE id=1").fetchone()


def _record_row(record: PersistedDashboardCredentials):
    return record.username, record.algorithm, record.salt, record.password_hash, record.updated_at


def _hook_write(connection: psycopg.Connection) -> None:
    connection.execute(
        "INSERT INTO locations(location_id,kind,name,order_index,created_at,updated_at) "
        "VALUES('credentials-hook','FLOOR','Credential callback',0,%s,%s)",
        (_TIME, _TIME),
    )


def _hook_count(sandbox: ProductSandbox) -> int:
    return sandbox.admin.execute(
        "SELECT count(*) FROM locations WHERE location_id='credentials-hook'"
    ).fetchone()[0]


def test_shared_factory_preserves_hash_parameters_and_fresh_salt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = iter((b"a" * 16, b"b" * 16))
    lengths = []

    def random_bytes(size: int) -> bytes:
        lengths.append(size)
        return next(supplied)

    class Clock:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return datetime.fromisoformat(_TIME)

    monkeypatch.setattr(credentials, "os", SimpleNamespace(urandom=random_bytes))
    monkeypatch.setattr(credentials, "datetime", Clock)
    first = PersistedDashboardCredentials.from_password(username="운영자", password="새 암호")
    second = PersistedDashboardCredentials.from_password(username="운영자", password="새 암호")
    assert lengths == [16, 16]
    assert first.salt == b"a" * 16 and second.salt == b"b" * 16
    assert first.password_hash != second.password_hash
    assert first.password_hash == hashlib.scrypt(
        "새 암호".encode(), salt=b"a" * 16, n=2**14, r=8, p=1, dklen=64
    )
    assert first.algorithm == "scrypt" and first.updated_at == _TIME
    assert first.verify_password("새 암호") and not first.verify_password("다른 암호")


def test_missing_owner_is_rejected_without_constructing_ambient_storage() -> None:
    with pytest.raises(TypeError, match="require a PostgreSQL owner"):
        PostgresDashboardCredentialsStore(None, AuthorityToken(1, uuid4()))


def test_real_absence_and_rotations_survive_independent_owners_and_restart(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox

    def reject_sqlite(*args, **kwargs):
        raise AssertionError("native credentials attempted SQLite")

    monkeypatch.setattr(sqlite3, "connect", reject_sqlite)
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    assert store.database is sandbox.database
    assert store.load() is None
    first = store.save(username="운영자", password="첫 번째 암호")
    assert _row(sandbox) == _record_row(first)
    with _independent_database(sandbox) as database:
        other = PostgresDashboardCredentialsStore(database, sandbox.authority)
        assert other.load() == first
        second = other.save(username="운영자-2", password="두 번째 암호")
        assert second.salt != first.salt and second.password_hash != first.password_hash
        assert store.load() == second
        assert second.verify_password("두 번째 암호")
        assert not second.verify_password("첫 번째 암호")
    sandbox.database.close(timeout_sec=3.0)
    with _independent_database(sandbox) as database:
        restarted = PostgresDashboardCredentialsStore(database, sandbox.authority)
        assert restarted.load() == second
        third = restarted.save(username="operator", password="third password")
        assert _row(sandbox) == _record_row(third)
    assert sandbox.admin.execute("SELECT count(*) FROM credentials").fetchone() == (1,)


def test_load_uses_the_owned_readonly_transaction(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    expected = store.save(username="operator", password="secret")
    read = sandbox.database.read
    observed = []

    def instrumented_read(callback):
        def inside(connection):
            observed.append(
                (
                    connection.info.transaction_status,
                    connection.execute("SHOW transaction_read_only").fetchone()[0],
                )
            )
            return callback(connection)

        return read(inside)

    monkeypatch.setattr(sandbox.database, "read", instrumented_read)
    assert store.load() == expected
    assert observed == [(TransactionStatus.INTRANS, "on")]


def test_hook_and_credentials_remain_tentative_until_the_owned_operation_returns(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    previous = store.save(username="previous", password="old secret")
    entered, release = Event(), Event()
    observed = []

    def after_write(connection):
        assert isinstance(connection, psycopg.Connection)
        observed.append(connection.info.transaction_status)
        _hook_write(connection)
        entered.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("credential callback release was not observed")

    with _independent_database(sandbox) as database:
        observer = PostgresDashboardCredentialsStore(database, sandbox.authority)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            store.save, username="next", password="new secret", after_write=after_write
        )
        try:
            assert entered.wait(timeout=5.0)
            assert not future.done()
            assert _row(sandbox) == _record_row(previous)
            assert _hook_count(sandbox) == 0
            assert observer.load() == previous
        finally:
            release.set()
            _, pending = wait((future,), timeout=5.0)
            executor.shutdown(wait=not pending, cancel_futures=True)
            assert not pending, "credential operation did not terminate"
    committed = future.result(timeout=0)
    assert observed == [TransactionStatus.INTRANS]
    assert _row(sandbox) == _record_row(committed)
    assert _hook_count(sandbox) == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_callback_failure_rolls_back_all_effects_and_preserves_identity(
    postgres_product_sandbox: ProductSandbox, cancel: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    previous = store.save(username="previous", password="old secret")
    failure = _Cancelled("callback cancelled") if cancel else ValueError("callback refused")
    calls = []

    def after_write(connection):
        calls.append(connection.info.transaction_status)
        _hook_write(connection)
        raise failure

    with pytest.raises(type(failure)) as raised:
        store.save(username="next", password="new secret", after_write=after_write)
    assert raised.value is failure
    assert calls == [TransactionStatus.INTRANS]
    assert _row(sandbox) == _record_row(previous)
    assert _hook_count(sandbox) == 0
    assert store.load() == previous


@pytest.mark.parametrize("frozen", [False, True])
def test_wrong_or_frozen_authority_cannot_rotate_credentials_or_invoke_hook(
    postgres_product_sandbox: ProductSandbox, frozen: bool
) -> None:
    sandbox = postgres_product_sandbox
    good = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    previous = good.save(username="previous", password="old secret")
    authority = sandbox.authority
    if frozen:
        freeze_authority(sandbox.database, authority)
    else:
        authority = AuthorityToken(authority.generation, uuid4())
    calls = []
    store = PostgresDashboardCredentialsStore(sandbox.database, authority)
    with pytest.raises(AuthorityFenced):
        store.save(username="next", password="new secret", after_write=calls.append)
    assert calls == []
    assert _row(sandbox) == _record_row(previous)


@pytest.mark.parametrize("unknown", [False, True])
def test_injected_owner_return_loss_preserves_failure_and_never_retries_a_known_commit(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, unknown: bool
) -> None:
    """Store propagation only: inject AFTER real owned COMMIT, not on the wire."""
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    store.save(username="previous", password="old secret")
    transact = sandbox.database.transact
    failure = CommitOutcomeUnknown() if unknown else OSError("owned return lost")
    attempts, hooks = [], []

    def lose_result(operation):
        attempts.append(True)
        transact(operation)
        raise failure

    monkeypatch.setattr(sandbox.database, "transact", lose_result)
    with pytest.raises(type(failure)) as raised:
        store.save(username="committed", password="new secret", after_write=hooks.append)
    assert raised.value is failure
    assert len(attempts) == len(hooks) == 1
    assert _row(sandbox)[0] == "committed"
    assert store.load().verify_password("new secret")


def test_unreadable_relation_is_not_absence_and_does_not_fall_back_or_leak(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, capsys, caplog
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    record = store.save(username="private-operator", password="private-password")
    sandbox.admin.execute("DROP TABLE credentials")

    def reject_sqlite(*args, **kwargs):
        raise AssertionError("native credentials attempted SQLite")

    monkeypatch.setattr(sqlite3, "connect", reject_sqlite)
    with pytest.raises(DashboardCredentialsStoreError) as raised:
        store.load()
    assert str(raised.value) == "dashboard credentials store unreadable"
    assert raised.value.__suppress_context__
    captured = capsys.readouterr()
    exposed = captured.out + captured.err + caplog.text + str(raised.value)
    for forbidden in (
        sandbox.dsn,
        "private-operator",
        "private-password",
        record.salt.hex(),
        record.password_hash.hex(),
    ):
        assert forbidden not in exposed
    assert "does not exist" not in exposed


@pytest.mark.parametrize(
    "column,value",
    [
        ("id", 2),
        ("username", ""),
        ("algorithm", "other"),
        ("salt", b"short"),
        ("password_hash", b"short"),
        ("updated_at", "not-a-timestamp"),
    ],
)
def test_present_corrupt_rows_fail_closed_instead_of_enabling_bootstrap(
    postgres_product_sandbox: ProductSandbox, column: str, value: object
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    store.save(username="operator", password="secret")
    sandbox.admin.execute(
        sql.SQL("ALTER TABLE credentials DROP CONSTRAINT {}").format(
            sql.Identifier(f"credentials_{column}_check")
        )
    )
    sandbox.admin.execute(
        sql.SQL("UPDATE credentials SET {}=%s WHERE id=1").format(sql.Identifier(column)), (value,)
    )
    with pytest.raises(
        DashboardCredentialsStoreError, match="^dashboard credentials store unreadable$"
    ):
        store.load()


def test_multiple_credential_rows_are_not_silently_reduced_to_id_one(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    store.save(username="operator", password="secret")
    sandbox.admin.execute("ALTER TABLE credentials DROP CONSTRAINT credentials_id_check")
    sandbox.admin.execute(
        f"INSERT INTO credentials(id,{_COLUMNS}) SELECT 2,{_COLUMNS} FROM credentials WHERE id=1"
    )
    with pytest.raises(
        DashboardCredentialsStoreError, match="^dashboard credentials store unreadable$"
    ):
        store.load()


def test_load_preserves_schema_admitted_timestamp_bytes_without_datetime_normalization(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    original = store.save(username="operator", password="secret")
    raw_time = "0000-02-29T23:59:59.123456Z"
    sandbox.admin.execute("UPDATE credentials SET updated_at=%s WHERE id=1", (raw_time,))
    actual = store.load()
    assert actual.updated_at == raw_time
    assert actual.password_hash == original.password_hash and actual.verify_password("secret")


def test_load_cancellation_remains_cancellation(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = PostgresDashboardCredentialsStore(sandbox.database, sandbox.authority)
    cancelled = _Cancelled("read cancelled")

    def cancel_read(callback):
        raise cancelled

    monkeypatch.setattr(sandbox.database, "read", cancel_read)
    with pytest.raises(_Cancelled) as raised:
        store.load()
    assert raised.value is cancelled
