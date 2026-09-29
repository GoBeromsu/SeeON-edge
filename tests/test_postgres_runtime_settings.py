"""Singleton export settings against the real PostgreSQL product sandbox."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING
from uuid import uuid4

import psycopg
import pytest
from fastapi import FastAPI
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown, PoolBudget, PostgresDatabase
from backend.app.features.runtime_settings import store as runtime_store
from backend.app.features.runtime_settings.store import (
    RuntimeSetting,
    RuntimeSettingsNotInitialized,
    RuntimeSettingsStore,
    RuntimeSettingsVersionConflict,
    get_runtime_settings_store,
)

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_FIRST_TIME = "2026-09-27T04:00:00.123Z"
_SECOND_TIME = "2026-09-27T04:01:00.456Z"


class _Clock:
    @staticmethod
    def now(tz):
        assert tz is UTC
        return datetime.fromisoformat(_SECOND_TIME)


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
        # Close every independent pool before the sandbox drops its unique namespace.
        database.close(timeout_sec=3.0)


def _seed(sandbox: ProductSandbox, enabled: bool = True, version: int = 7) -> None:
    sandbox.admin.execute(
        "UPDATE edge_site SET clip_export_enabled=%s,runtime_settings_version=%s,"
        "updated_at=%s WHERE id=1",
        (int(enabled), version, _FIRST_TIME),
    )


def _site(sandbox: ProductSandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def _write_location(connection: psycopg.Connection) -> None:
    connection.execute(
        "INSERT INTO locations(location_id,kind,name,order_index,created_at,updated_at) "
        "VALUES (%s,'FLOOR',%s,0,%s,%s)",
        ("hook-floor", "Transactional hook", _FIRST_TIME, _FIRST_TIME),
    )


def test_committed_settings_survive_independent_owners_and_pool_restart_without_recording_changes(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox, enabled=False)
    sandbox.admin.execute(
        "UPDATE edge_site SET clip_store_subdir='capture/local',registry_version=23,"
        "fall_on=1,fall_mode='always',storage_state='ready',recording_suspended=0,"
        "audit_state='healthy',degraded_observed_at=%s WHERE id=1",
        (_FIRST_TIME,),
    )
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    first = store.set_clip_export_enabled(True, expected_version=7)
    assert first == RuntimeSetting(True, 8)
    assert first.as_dict() == {"clip_export_enabled": True, "version": 8}
    with _independent_database(sandbox) as database:
        other = RuntimeSettingsStore(database, sandbox.authority)
        assert other.get() == first
        second = other.set_clip_export_enabled(False)
        assert second == RuntimeSetting(False, 9)
        assert store.get() == second
    sandbox.database.close(timeout_sec=3.0)
    with _independent_database(sandbox) as database:
        restarted = RuntimeSettingsStore(database, sandbox.authority)
        assert restarted.get() == second
        assert restarted.set_clip_export_enabled(True) == RuntimeSetting(True, 10)
    assert sandbox.admin.execute(
        "SELECT clip_export_enabled,runtime_settings_version,clip_store_subdir,registry_version,"
        "fall_on,fall_mode,storage_state,recording_suspended,audit_state,degraded_observed_at "
        "FROM edge_site WHERE id=1"
    ).fetchone() == (1, 10, "capture/local", 23, 1, "always", "ready", 0, "healthy", _FIRST_TIME)


def test_get_borrows_one_read_committed_readonly_transaction(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox, version=2**63 - 1)
    before = _site(sandbox)
    read = sandbox.database.read
    reads = []

    def observed_read(callback):
        def observe(connection):
            reads.append(connection.info.backend_pid)
            assert connection.row_factory is tuple_row
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert connection.execute("SHOW transaction_read_only").fetchone() == ("on",)
            assert connection.execute("SHOW transaction_isolation").fetchone() == (
                "read committed",
            )
            return callback(connection)

        return read(observe)

    monkeypatch.setattr(sandbox.database, "read", observed_read)
    setting = RuntimeSettingsStore(sandbox.database, sandbox.authority).get()
    assert setting == RuntimeSetting(True, 2**63 - 1)
    assert type(setting.clip_export_enabled) is bool and type(setting.version) is int
    assert len(reads) == 1 and _site(sandbox) == before


@pytest.mark.parametrize("no_op", [False, True], ids=["change", "no-op"])
@pytest.mark.parametrize("fail", [False, True], ids=["commit", "rollback"])
def test_hook_shares_owned_write_transaction_and_rolls_back_another_table_atomically(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    no_op: bool,
    fail: bool,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    monkeypatch.setattr(runtime_store, "datetime", _Clock)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    before = _site(sandbox)
    original = RuntimeSetting(True, 7)
    candidate = original if no_op else RuntimeSetting(False, 8)
    timestamp = _FIRST_TIME if no_op else _SECOND_TIME
    transact = sandbox.database.transact
    connections = []
    hooks = []

    def observed_transaction(callback):
        def observe(connection):
            connections.append(id(connection))
            return callback(connection)

        return transact(observe)

    def hook(connection: psycopg.Connection) -> None:
        hooks.append(id(connection))
        assert hooks == connections
        assert connection.row_factory is tuple_row
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert connection.execute("SHOW transaction_read_only").fetchone() == ("off",)
        assert connection.execute("SHOW transaction_isolation").fetchone() == ("read committed",)
        assert connection.execute(
            "SELECT clip_export_enabled,runtime_settings_version,updated_at FROM edge_site"
        ).fetchone() == (int(candidate.clip_export_enabled), candidate.version, timestamp)
        assert _site(sandbox) == before and store.get() == original
        _write_location(connection)
        assert sandbox.admin.execute("SELECT count(*) FROM locations").fetchone() == (0,)
        if fail:
            raise RuntimeError("injected after_write failure")

    monkeypatch.setattr(sandbox.database, "transact", observed_transaction)
    if fail:
        with pytest.raises(RuntimeError, match="injected after_write failure"):
            store.set_clip_export_enabled(no_op, expected_version=7, after_write=hook)
    else:
        assert (
            store.set_clip_export_enabled(no_op, expected_version=7, after_write=hook) == candidate
        )
    assert len(connections) == len(hooks) == 1
    assert store.get() == (original if fail else candidate)
    if fail or no_op:
        assert _site(sandbox) == before
    assert sandbox.admin.execute("SELECT updated_at FROM edge_site").fetchone() == (
        _FIRST_TIME if fail else timestamp,
    )
    assert sandbox.admin.execute("SELECT count(*) FROM locations").fetchone() == (int(not fail),)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("expected_version", [6, 8])
def test_version_conflict_including_no_op_returns_current_without_hook_or_timestamp_change(
    postgres_product_sandbox: ProductSandbox, enabled: bool, expected_version: int
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    before = _site(sandbox)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    hooks = []
    with pytest.raises(
        RuntimeSettingsVersionConflict, match="runtime settings version conflict"
    ) as e:
        store.set_clip_export_enabled(
            enabled, expected_version=expected_version, after_write=hooks.append
        )
    assert e.value.current == RuntimeSetting(True, 7)
    assert not hooks and _site(sandbox) == before
    assert store.get() == e.value.current


def test_version_overflow_is_a_known_failure_but_max_version_no_op_remains_valid(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox, version=2**63 - 1)
    before = _site(sandbox)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    hooks = []
    with pytest.raises(psycopg.DataError):
        store.set_clip_export_enabled(False, after_write=hooks.append)
    assert not hooks and _site(sandbox) == before
    assert store.set_clip_export_enabled(True, expected_version=2**63 - 1) == RuntimeSetting(
        True, 2**63 - 1
    )
    assert _site(sandbox) == before


@pytest.mark.parametrize("fence", ["frozen", "stale-generation", "stale-token", "missing"])
@pytest.mark.parametrize(("enabled", "expected_version"), [(True, 7), (True, 0), (False, None)])
def test_authority_precedes_site_and_version_reads_even_for_no_op(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    fence: str,
    enabled: bool,
    expected_version: int | None,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    if fence == "frozen":
        freeze_authority(sandbox.database, sandbox.authority)
    elif fence == "stale-generation":
        sandbox.admin.execute("UPDATE deployment_authority SET generation=generation+1")
    elif fence == "stale-token":
        sandbox.admin.execute("UPDATE deployment_authority SET writer_token=%s", (uuid4(),))
    else:
        sandbox.admin.execute("DELETE FROM deployment_authority")
    before = _site(sandbox)
    execute = psycopg.Cursor.execute
    site_queries = []
    hooks = []

    def observe(cursor, query, *args, **kwargs):
        if isinstance(query, str) and "edge_site" in query:
            site_queries.append(query)
        return execute(cursor, query, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Cursor, "execute", observe)
        with pytest.raises(AuthorityFenced):
            store.set_clip_export_enabled(
                enabled, expected_version=expected_version, after_write=hooks.append
            )
    assert not site_queries and not hooks and _site(sandbox) == before
    assert store.get() == RuntimeSetting(True, 7)
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    with pytest.raises(AuthorityFenced):
        store.set_clip_export_enabled(enabled, after_write=hooks.append)
    assert not hooks


def test_missing_bootstrap_refuses_reads_no_ops_and_changes_without_inserting_defaults(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    with pytest.raises(RuntimeSettingsNotInitialized, match="bootstrap row is missing"):
        store.get()
    hooks = []
    for enabled in (False, True):
        with pytest.raises(RuntimeSettingsNotInitialized):
            store.set_clip_export_enabled(enabled, expected_version=0, after_write=hooks.append)
    assert not hooks
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


@pytest.mark.parametrize(
    "compare_version", [False, True], ids=["no-op-follower", "conflict-follower"]
)
def test_concurrent_independent_writers_lock_singleton_before_version_and_no_op_checks(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    compare_version: bool,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox, enabled=False)
    ready = Barrier(3)
    pids: Queue[int] = Queue()
    hooks: Queue[int] = Queue()

    def attempt(store):
        try:
            return store.set_clip_export_enabled(
                True,
                expected_version=7 if compare_version else None,
                after_write=lambda connection: hooks.put(connection.info.backend_pid),
            )
        except RuntimeSettingsVersionConflict as error:
            return error

    with _independent_database(sandbox) as database, monkeypatch.context() as patch:
        for owner in (sandbox.database, database):
            transact = owner.transact

            def synchronized_transaction(callback, transact=transact):
                def synchronize(connection):
                    pids.put(connection.info.backend_pid)
                    ready.wait(timeout=2)
                    return callback(connection)

                return transact(synchronize)

            patch.setattr(owner, "transact", synchronized_transaction)
        first = RuntimeSettingsStore(sandbox.database, sandbox.authority)
        second = RuntimeSettingsStore(database, sandbox.authority)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with sandbox.admin.transaction():
                sandbox.admin.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE")
                one = pool.submit(attempt, first)
                two = pool.submit(attempt, second)
                worker_pids = [pids.get(timeout=2), pids.get(timeout=2)]
                assert len(set(worker_pids)) == 2
                ready.wait(timeout=2)
                deadline = monotonic() + 2
                pause = Event()
                while monotonic() < deadline:
                    waiting = sandbox.admin.execute(
                        "SELECT count(DISTINCT pid) FROM pg_locks "
                        "WHERE pid=ANY(%s) AND NOT granted",
                        (worker_pids,),
                    ).fetchone()
                    if waiting == (2,):
                        break
                    pause.wait(0.01)
                else:
                    pytest.fail("both runtime writers must wait on the site singleton")
            results = [one.result(timeout=5), two.result(timeout=5)]
        conflicts = [
            result for result in results if isinstance(result, RuntimeSettingsVersionConflict)
        ]
        successes = [result for result in results if isinstance(result, RuntimeSetting)]
        assert len(conflicts) == int(compare_version)
        assert successes == [RuntimeSetting(True, 8)] * (2 - int(compare_version))
        assert all(error.current == RuntimeSetting(True, 8) for error in conflicts)
        assert hooks.qsize() == len(successes)
        assert first.get() == second.get() == RuntimeSetting(True, 8)


@pytest.mark.parametrize("release_fails", [False, True], ids=["released", "release-failure"])
def test_return_is_own_candidate_after_pool_release_not_a_racing_later_writer(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    release_fails: bool,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox, enabled=False)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    borrow = sandbox.database._pool.connection
    released = []
    published = []
    with _independent_database(sandbox) as database:
        later = RuntimeSettingsStore(database, sandbox.authority)

        @contextmanager
        def interleaved_release(*args, **kwargs):
            with borrow(*args, **kwargs) as connection:
                yield connection
            released.append(True)
            assert later.get() == RuntimeSetting(True, 8)
            later.set_clip_export_enabled(False, expected_version=8)
            if release_fails:
                raise RuntimeError("injected pool release failure")

        with monkeypatch.context() as patch:
            patch.setattr(sandbox.database._pool, "connection", interleaved_release)
            if release_fails:
                with pytest.raises(RuntimeError, match="injected pool release failure"):
                    published.append(store.set_clip_export_enabled(True, expected_version=7))
            else:
                published.append(store.set_clip_export_enabled(True, expected_version=7))
        assert released == [True]
        assert published == ([] if release_fails else [RuntimeSetting(True, 8)])
        assert store.get() == later.get() == RuntimeSetting(False, 9)


@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
@pytest.mark.parametrize("no_op", [False, True], ids=["change", "no-op"])
def test_actual_unknown_commit_never_replays_callback_or_publishes_a_candidate(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
    no_op: bool,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    monkeypatch.setattr(runtime_store, "datetime", _Clock)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    before = _site(sandbox)
    commit = psycopg.Connection.commit
    hook_pids = []
    commit_pids = []
    published = []

    def hook(connection: psycopg.Connection) -> None:
        hook_pids.append(connection.info.backend_pid)
        _write_location(connection)

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        if pid in hook_pids:
            if committed:
                commit(connection)
            commit_pids.append(pid)
            raise psycopg.OperationalError("injected COMMIT receipt loss")
        commit(connection)

    with monkeypatch.context() as patch:
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(
                store.set_clip_export_enabled(no_op, expected_version=7, after_write=hook)
            )
    assert not published and len(hook_pids) == len(commit_pids) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM locations").fetchone() == (int(committed),)
    expected = RuntimeSetting(False, 8) if committed and not no_op else RuntimeSetting(True, 7)
    assert store.get() == expected
    with _independent_database(sandbox) as database:
        assert RuntimeSettingsStore(database, sandbox.authority).get() == expected
    if not committed or no_op:
        assert _site(sandbox) == before
    else:
        assert sandbox.admin.execute("SELECT updated_at FROM edge_site").fetchone() == (
            _SECOND_TIME,
        )


@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
def test_unknown_read_commit_does_not_publish_or_replay_the_read(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    read = sandbox.database.read
    commit = psycopg.Connection.commit
    reads = []
    commits = []
    published = []

    def observed_read(callback):
        def observe(connection):
            reads.append(connection.info.backend_pid)
            return callback(connection)

        return read(observe)

    def lose_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        if pid in reads:
            if committed:
                commit(connection)
            commits.append(pid)
            raise psycopg.OperationalError("injected read COMMIT receipt loss")
        commit(connection)

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.database, "read", observed_read)
        patch.setattr(psycopg.Connection, "commit", lose_receipt)
        with pytest.raises(CommitOutcomeUnknown):
            published.append(store.get())
    assert not published and len(reads) == len(commits) == 1
    assert store.get() == RuntimeSetting(True, 7)


def test_getter_returns_only_the_injected_owner(postgres_product_sandbox: ProductSandbox) -> None:
    sandbox = postgres_product_sandbox
    app = FastAPI()
    store = RuntimeSettingsStore(sandbox.database, sandbox.authority)
    app.state.runtime_settings_store = store
    assert get_runtime_settings_store(app) is store
    assert get_runtime_settings_store(app) is store
    assert store.database is sandbox.database and store.authority is sandbox.authority


def test_getter_refuses_missing_owner_without_installing_a_fallback() -> None:
    app = FastAPI()
    with pytest.raises(RuntimeError, match="runtime settings store is not injected"):
        get_runtime_settings_store(app)
    assert not hasattr(app.state, "runtime_settings_store")


@pytest.mark.parametrize(
    ("wrong_owner", "error", "message"),
    [
        (None, RuntimeError, "runtime settings store is not injected"),
        (object(), TypeError, "runtime settings store has invalid type"),
        ("not-a-store", TypeError, "runtime settings store has invalid type"),
    ],
)
def test_getter_refuses_wrong_owner_without_replacing_it(
    wrong_owner: object, error: type[Exception], message: str
) -> None:
    app = FastAPI()
    app.state.runtime_settings_store = wrong_owner
    with pytest.raises(error, match=message):
        get_runtime_settings_store(app)
    assert app.state.runtime_settings_store is wrong_owner
