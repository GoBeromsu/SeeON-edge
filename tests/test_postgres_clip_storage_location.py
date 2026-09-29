"""Clip selection persistence against the real PostgreSQL product sandbox."""

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
from psycopg.pq import TransactionStatus
from psycopg.rows import tuple_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown, PoolBudget, PostgresDatabase
from backend.app.features.clips import storage_location_store as location_store
from backend.app.features.clips.storage_location_store import (
    ClipStorageLocationNotInitialized,
    ClipStorageLocationStore,
)
from backend.app.features.runtime_settings.store import RuntimeSetting, RuntimeSettingsStore

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
        # The sandbox drops only its unique namespace, after all owners close.
        database.close(timeout_sec=3.0)


def _seed(sandbox: ProductSandbox, selected_path: str = "initial/clips") -> None:
    sandbox.admin.execute(
        "UPDATE edge_site SET clip_store_subdir=%s,updated_at=%s WHERE id=1",
        (selected_path or None, _FIRST_TIME),
    )


def _site(sandbox: ProductSandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def _write_location(connection: psycopg.Connection) -> None:
    connection.execute(
        "INSERT INTO locations(location_id,kind,name,order_index,created_at,updated_at) "
        "VALUES (%s,'FLOOR',%s,0,%s,%s)",
        ("hook-floor", "Transactional hook", _FIRST_TIME, _FIRST_TIME),
    )


def test_selection_survives_independent_owners_and_pool_restart_without_sibling_changes(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    sandbox.admin.execute(
        "UPDATE edge_site SET clip_export_enabled=1,runtime_settings_version=9,registry_version=23,"
        "fall_on=1,fall_mode='always',storage_state='ready',recording_suspended=0,"
        "audit_state='healthy',degraded_observed_at=%s WHERE id=1",
        (_FIRST_TIME,),
    )
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
    assert store.put("external/클립") == "external/클립"
    with _independent_database(sandbox) as database:
        other = ClipStorageLocationStore(database, sandbox.authority)
        assert other.get() == "external/클립"
        assert other.put("other/clips") == "other/clips"
        assert store.get() == "other/clips"
    sandbox.database.close(timeout_sec=3.0)
    with _independent_database(sandbox) as database:
        restarted = ClipStorageLocationStore(database, sandbox.authority)
        assert restarted.get() == "other/clips"
        assert restarted.put("") == ""
        assert restarted.get() == ""
    assert sandbox.admin.execute(
        "SELECT clip_store_subdir,clip_export_enabled,runtime_settings_version,registry_version,"
        "fall_on,fall_mode,storage_state,recording_suspended,audit_state,degraded_observed_at "
        "FROM edge_site WHERE id=1"
    ).fetchone() == (None, 1, 9, 23, 1, "always", "ready", 0, "healthy", _FIRST_TIME)


@pytest.mark.parametrize("selected_path", ["", "rack/clips"])
def test_get_borrows_one_read_committed_readonly_transaction_and_keeps_null_representation(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    selected_path: str,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox, selected_path)
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
    assert ClipStorageLocationStore(sandbox.database, sandbox.authority).get() == selected_path
    assert len(reads) == 1 and _site(sandbox) == before
    assert sandbox.admin.execute("SELECT clip_store_subdir FROM edge_site").fetchone() == (
        selected_path or None,
    )


@pytest.mark.parametrize(
    "selected_path",
    ["usb/클립", "a" * 512, ".", " folder /clips ", "folder'; DELETE FROM edge_site; --"],
    ids=["unicode", "length-limit", "dot", "whitespace", "sql-text"],
)
def test_selection_is_parameterized_and_preserved_without_normalization_or_filesystem_policy(
    postgres_product_sandbox: ProductSandbox, selected_path: str
) -> None:
    sandbox = postgres_product_sandbox
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
    assert store.put(selected_path) == selected_path
    assert store.get() == selected_path
    assert sandbox.admin.execute("SELECT clip_store_subdir FROM edge_site").fetchone() == (
        selected_path,
    )
    assert store.put("") == "" and store.get() == ""
    assert sandbox.admin.execute("SELECT clip_store_subdir FROM edge_site").fetchone() == (None,)


@pytest.mark.parametrize("invalid", ["/absolute", "../escape", "a/../b", "a\\b", "x" * 513])
def test_native_relpath_constraints_refuse_invalid_selection_without_partial_effects(
    postgres_product_sandbox: ProductSandbox, invalid: str
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    before = _site(sandbox)
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
    hooks = []
    with pytest.raises(psycopg.IntegrityError):
        store.put(invalid, after_write=hooks.append)
    assert not hooks and _site(sandbox) == before
    assert store.get() == "initial/clips"


@pytest.mark.parametrize("selected_path", ["replacement/clips", "", "initial/clips"])
@pytest.mark.parametrize("fail", [False, True], ids=["commit", "rollback"])
def test_hook_shares_owned_write_transaction_and_rolls_back_another_table_atomically(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    selected_path: str,
    fail: bool,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    monkeypatch.setattr(location_store, "datetime", _Clock)
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
    before = _site(sandbox)
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
            "SELECT clip_store_subdir,updated_at FROM edge_site WHERE id=1"
        ).fetchone() == (selected_path or None, _SECOND_TIME)
        assert _site(sandbox) == before and store.get() == "initial/clips"
        _write_location(connection)
        assert sandbox.admin.execute("SELECT count(*) FROM locations").fetchone() == (0,)
        if fail:
            raise RuntimeError("injected after_write failure")

    monkeypatch.setattr(sandbox.database, "transact", observed_transaction)
    if fail:
        with pytest.raises(RuntimeError, match="injected after_write failure"):
            store.put(selected_path, after_write=hook)
        assert _site(sandbox) == before
    else:
        assert store.put(selected_path, after_write=hook) == selected_path
    assert len(connections) == len(hooks) == 1
    assert store.get() == ("initial/clips" if fail else selected_path)
    # Unlike runtime export no-ops, repeated clip selections still update the timestamp.
    assert sandbox.admin.execute("SELECT updated_at FROM edge_site").fetchone() == (
        _FIRST_TIME if fail else _SECOND_TIME,
    )
    assert sandbox.admin.execute("SELECT count(*) FROM locations").fetchone() == (int(not fail),)


@pytest.mark.parametrize("fence", ["frozen", "stale-generation", "stale-token", "missing"])
@pytest.mark.parametrize("selected_path", ["initial/clips", "", "../invalid"])
def test_authority_precedes_site_reads_and_validation_even_for_repeated_selection(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    fence: str,
    selected_path: str,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
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
            store.put(selected_path, after_write=hooks.append)
    assert not site_queries and not hooks and _site(sandbox) == before
    assert store.get() == "initial/clips"
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    with pytest.raises(AuthorityFenced):
        store.put(selected_path, after_write=hooks.append)
    assert not hooks


def test_missing_bootstrap_refuses_read_reset_and_selection_without_inserting_defaults(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
    with pytest.raises(ClipStorageLocationNotInitialized, match="bootstrap row is missing"):
        store.get()
    hooks = []
    for selected_path in ("", "folder/clips", "../invalid"):
        with pytest.raises(ClipStorageLocationNotInitialized):
            store.put(selected_path, after_write=hooks.append)
    assert not hooks
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


@pytest.mark.parametrize("second_kind", ["clip", "export"])
def test_concurrent_independent_writers_share_the_global_singleton_lock(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    second_kind: str,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    sandbox.admin.execute(
        "UPDATE edge_site SET clip_export_enabled=0,runtime_settings_version=7 WHERE id=1"
    )
    ready = Barrier(3)
    pids: Queue[int] = Queue()
    hooks: Queue[str] = Queue()

    def clip_hook(connection: psycopg.Connection, selected_path: str) -> None:
        assert connection.execute("SELECT clip_store_subdir FROM edge_site").fetchone() == (
            selected_path,
        )
        hooks.put(selected_path)

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
        first = ClipStorageLocationStore(sandbox.database, sandbox.authority)
        second = ClipStorageLocationStore(database, sandbox.authority)
        runtime = RuntimeSettingsStore(database, sandbox.authority)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with sandbox.admin.transaction():
                sandbox.admin.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE")
                one = pool.submit(
                    first.put,
                    "first/clips",
                    after_write=lambda connection: clip_hook(connection, "first/clips"),
                )
                if second_kind == "clip":
                    two = pool.submit(
                        second.put,
                        "second/clips",
                        after_write=lambda connection: clip_hook(connection, "second/clips"),
                    )
                else:
                    two = pool.submit(
                        runtime.set_clip_export_enabled,
                        True,
                        expected_version=7,
                        after_write=lambda connection: hooks.put("export"),
                    )
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
                    pytest.fail("both singleton writers must wait on the site row")
            assert one.result(timeout=5) == "first/clips"
            assert two.result(timeout=5) == (
                "second/clips" if second_kind == "clip" else RuntimeSetting(True, 8)
            )
        order = [hooks.get(timeout=2), hooks.get(timeout=2)]
        assert set(order) == {"first/clips", "second/clips" if second_kind == "clip" else "export"}
        assert (
            first.get() == second.get() == (order[-1] if second_kind == "clip" else "first/clips")
        )
        assert runtime.get() == (
            RuntimeSetting(False, 7) if second_kind == "clip" else RuntimeSetting(True, 8)
        )


@pytest.mark.parametrize("release_fails", [False, True], ids=["released", "release-failure"])
def test_return_is_own_candidate_after_pool_release_not_a_racing_later_writer(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    release_fails: bool,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
    borrow = sandbox.database._pool.connection
    released = []
    published = []
    with _independent_database(sandbox) as database:
        later = ClipStorageLocationStore(database, sandbox.authority)

        @contextmanager
        def interleaved_release(*args, **kwargs):
            with borrow(*args, **kwargs) as connection:
                yield connection
            released.append(True)
            assert later.get() == "own/clips"
            later.put("later/clips")
            if release_fails:
                raise RuntimeError("injected pool release failure")

        with monkeypatch.context() as patch:
            patch.setattr(sandbox.database._pool, "connection", interleaved_release)
            if release_fails:
                with pytest.raises(RuntimeError, match="injected pool release failure"):
                    published.append(store.put("own/clips"))
            else:
                published.append(store.put("own/clips"))
        assert released == [True]
        assert published == ([] if release_fails else ["own/clips"])
        assert store.get() == later.get() == "later/clips"


@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
@pytest.mark.parametrize("selected_path", ["changed/clips", "", "initial/clips"])
def test_actual_unknown_commit_never_replays_callback_or_publishes_a_candidate(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
    selected_path: str,
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    monkeypatch.setattr(location_store, "datetime", _Clock)
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
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
            published.append(store.put(selected_path, after_write=hook))
    assert not published and len(hook_pids) == len(commit_pids) == 1
    assert sandbox.admin.execute("SELECT count(*) FROM locations").fetchone() == (int(committed),)
    expected = selected_path if committed else "initial/clips"
    assert store.get() == expected
    with _independent_database(sandbox) as database:
        assert ClipStorageLocationStore(database, sandbox.authority).get() == expected
    assert sandbox.admin.execute(
        "SELECT clip_store_subdir,updated_at FROM edge_site"
    ).fetchone() == (
        expected or None,
        _SECOND_TIME if committed else _FIRST_TIME,
    )
    if not committed:
        assert _site(sandbox) == before


@pytest.mark.parametrize(
    "committed", [False, True], ids=["lost-before-commit", "lost-after-commit"]
)
def test_unknown_read_commit_does_not_publish_or_replay_the_read(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    sandbox = postgres_product_sandbox
    _seed(sandbox)
    store = ClipStorageLocationStore(sandbox.database, sandbox.authority)
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
    assert store.get() == "initial/clips"
