"""Native connection settings against an explicitly isolated real PostgreSQL server."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from queue import Queue
from threading import Barrier, Event
from time import monotonic
from typing import TYPE_CHECKING

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row, tuple_row

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_store import append_postgres_audit
from backend.app.features.audit.store import AuditEvent
from backend.app.features.connection import store as connection_store
from backend.app.features.connection.repository import (
    ConnectionData,
    ConnectionSettingsNotInitialized,
)
from backend.app.features.connection.store import (
    API_BACKEND_BASE_URL_ENV,
    ConnectionSettingsStore,
    InvalidConnectionSettingError,
)

if TYPE_CHECKING:
    from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_FIRST_TIME = "2026-09-27T04:00:00.123Z"
_SECOND_TIME = "2026-09-27T04:01:00.456Z"
_CLIENT_TOKEN = "synthetic-client-token-1234"


def _enrollment() -> ConnectionData:
    return {
        "facility_code": "NH-1234",
        "client_installation_ref": "install-1",
        "facility_id": "facility-1",
        "facility_token": _CLIENT_TOKEN,
        "edge_installation_id": "edge-1",
        "enrollment_generation": 1,
    }


def _audit_event() -> AuditEvent:
    return AuditEvent(
        occurred_at=_FIRST_TIME,
        actor_id="test-operator",
        action=AuditAction.CONNECTION_UPDATE,
        target_id="facility-1",
        detail=empty_detail(AuditAction.CONNECTION_UPDATE),
    )


def test_unconfigured_identity_is_db_only_and_load_borrows_read_transaction(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    monkeypatch.delenv(API_BACKEND_BASE_URL_ENV, raising=False)
    monkeypatch.setenv("API_FACILITY_ID", "environment-facility")
    monkeypatch.setenv("EDGE_FACILITY_TOKEN", "synthetic-environment-token")
    monkeypatch.setenv("API_BACKEND_EVENTS_URL", "https://retired.example/events")
    monkeypatch.setenv("API_BACKEND_CONFIG_URL", "https://retired.example/config")
    read = sandbox.database.read
    calls = []

    def observed_read(callback):
        def observe(connection):
            calls.append(connection.info.backend_pid)
            assert connection.info.transaction_status is TransactionStatus.INTRANS
            assert connection.execute("SHOW transaction_read_only").fetchone() == ("on",)
            return callback(connection)

        return read(observe)

    monkeypatch.setattr(sandbox.database, "read", observed_read)
    settings = ConnectionSettingsStore(sandbox.database, sandbox.authority).load()
    assert len(calls) == 1
    assert settings.facility_id is None and settings.facility_token is None
    assert settings.edge_installation_id is None and settings.enrollment_generation is None
    assert settings.events_url is None and settings.config_url is None
    assert sandbox.admin.execute("SELECT updated_at FROM edge_site").fetchone() == (
        settings.updated_at,
    )
    assert settings.updated_at is not None and settings.updated_at.endswith("Z")
    assert datetime.fromisoformat(settings.updated_at).tzinfo == UTC
    assert "dsn=" not in repr(sandbox) and "admin=" not in repr(sandbox)


def test_save_persists_complete_enrollment_and_preserves_creation_time_on_partial_update(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    times = iter((_FIRST_TIME, _SECOND_TIME))
    monkeypatch.setattr(connection_store, "utc_now_iso", lambda: next(times))
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    first = store.save(_enrollment())
    second = store.save({"facility_id": "facility-2", "enrollment_generation": 2})
    assert first.updated_at == first.enrollment_created_at == first.enrollment_updated_at
    assert first.updated_at == _FIRST_TIME
    assert second.enrollment_created_at == _FIRST_TIME
    assert second.updated_at == second.enrollment_updated_at == _SECOND_TIME
    assert second.facility_id == "facility-2" and second.enrollment_generation == 2
    assert second.facility_token == first.facility_token
    assert second.client_installation_ref == "install-1"
    assert ConnectionSettingsStore(sandbox.database, sandbox.authority).load() == second
    assert sandbox.admin.execute(
        "SELECT facility_id,enrollment_generation,enrollment_created_at,updated_at "
        "FROM edge_site WHERE id=1"
    ).fetchone() == ("facility-2", 2, _FIRST_TIME, _SECOND_TIME)


@pytest.mark.parametrize("missing_field", tuple(_enrollment()))
def test_partial_enrollment_and_partial_clear_are_rejected_without_changes(
    postgres_product_sandbox: ProductSandbox, missing_field: str
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    before = sandbox.admin.execute("SELECT * FROM edge_site").fetchone()
    partial = _enrollment()
    del partial[missing_field]
    with pytest.raises(InvalidConnectionSettingError, match="saved atomically"):
        store.save(partial)
    assert sandbox.admin.execute("SELECT * FROM edge_site").fetchone() == before
    complete = store.save(_enrollment())
    with pytest.raises(InvalidConnectionSettingError, match="saved atomically"):
        store.save({missing_field: None})
    assert store.load() == complete


def test_complete_clear_removes_enrollment_timestamps_without_environment_fallback(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    store.save(_enrollment())
    monkeypatch.setenv("EDGE_FACILITY_TOKEN", "synthetic-environment-token")
    monkeypatch.setenv("API_FACILITY_ID", "environment-facility")
    cleared = store.save(dict.fromkeys(_enrollment()))
    assert cleared.facility_token is None and cleared.facility_id is None
    assert cleared.enrollment_created_at is None and cleared.enrollment_updated_at is None
    assert (
        sandbox.admin.execute(
            "SELECT facility_code,client_installation_ref,facility_id,facility_token,"
            "edge_installation_id,enrollment_generation,"
            "enrollment_created_at,enrollment_updated_at "
            "FROM edge_site WHERE id=1"
        ).fetchone()
        == (None,) * 8
    )
    assert store.load() == cleared


def test_after_write_uses_same_active_tuple_connection_and_commits_before_return(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    transact = sandbox.database.transact
    borrowed_ids = []
    hook_calls = []

    def observed_transaction(callback):
        def observe(connection):
            borrowed_ids.append(id(connection))
            return callback(connection)

        return transact(observe)

    def after_write(connection: psycopg.Connection) -> None:
        hook_calls.append(id(connection))
        assert hook_calls == borrowed_ids
        assert connection.info.transaction_status is TransactionStatus.INTRANS
        assert connection.row_factory is tuple_row
        assert connection.execute("SELECT facility_id FROM edge_site").fetchone() == ("facility-1",)
        assert sandbox.admin.execute("SELECT facility_id FROM edge_site").fetchone() == (None,)
        append_postgres_audit(connection, _audit_event())
        assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)

    monkeypatch.setattr(sandbox.database, "transact", observed_transaction)
    saved = store.save(_enrollment(), after_write=after_write)
    assert len(borrowed_ids) == len(hook_calls) == 1
    assert sandbox.admin.execute("SELECT facility_id FROM edge_site").fetchone() == (
        saved.facility_id,
    )
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)
    assert sandbox.database.read(lambda connection: connection.row_factory is tuple_row)


def test_after_write_failure_rolls_back_settings_topology_and_audit_together(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    store.save(_enrollment())
    sandbox.admin.execute(
        "UPDATE edge_site SET topology_client_revision=7,topology_server_revision=8"
    )
    before = sandbox.admin.execute("SELECT * FROM edge_site").fetchone()

    def failing_hook(connection: psycopg.Connection) -> None:
        append_postgres_audit(connection, _audit_event())
        connection.execute("UPDATE edge_site SET clip_export_enabled=1 WHERE id=1")
        raise RuntimeError("injected after_write failure")

    with pytest.raises(RuntimeError, match="injected after_write failure"):
        store.save({"enrollment_generation": 2}, after_write=failing_hook)
    assert sandbox.admin.execute("SELECT * FROM edge_site").fetchone() == before
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert store.load().enrollment_generation == 1


def test_save_returns_its_own_committed_snapshot_not_a_postcommit_reread(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    transact = sandbox.database.transact
    calls = []

    def interleaved_transaction(callback):
        candidate = transact(callback)
        calls.append(candidate.facility_id)
        sandbox.admin.execute("UPDATE edge_site SET facility_id=%s WHERE id=1", ("later-writer",))
        return candidate

    monkeypatch.setattr(sandbox.database, "transact", interleaved_transaction)
    saved = store.save(_enrollment())
    assert calls == ["facility-1"]
    assert saved.facility_id == "facility-1"
    assert store.load().facility_id == "later-writer"


def test_concurrent_partial_updates_lock_before_merge_without_losing_fields(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    first = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    second = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    first.save(_enrollment())
    transact = sandbox.database.transact
    ready = Barrier(3)
    pids: Queue[int] = Queue()

    def synchronized_transaction(callback):
        def synchronize(connection):
            pids.put(connection.info.backend_pid)
            ready.wait(timeout=2)
            return callback(connection)

        return transact(synchronize)

    monkeypatch.setattr(sandbox.database, "transact", synchronized_transaction)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with sandbox.admin.transaction():
            sandbox.admin.execute("SELECT id FROM edge_site WHERE id=1 FOR UPDATE")
            one = pool.submit(first.save, {"facility_id": "facility-concurrent"})
            two = pool.submit(second.save, {"facility_code": "NH-CONCURRENT"})
            worker_pids = [pids.get(timeout=2), pids.get(timeout=2)]
            ready.wait(timeout=2)
            deadline = monotonic() + 2
            pause = Event()
            while monotonic() < deadline:
                waiting = sandbox.admin.execute(
                    "SELECT count(DISTINCT pid) FROM pg_locks WHERE pid=ANY(%s) AND NOT granted",
                    (worker_pids,),
                ).fetchone()
                if waiting == (2,):
                    break
                pause.wait(0.01)
            else:
                pytest.fail("both settings writers must wait on the singleton row lock")
        assert one.result(timeout=5).facility_id == "facility-concurrent"
        assert two.result(timeout=5).facility_code == "NH-CONCURRENT"
    saved = first.load()
    assert saved.facility_id == "facility-concurrent"
    assert saved.facility_code == "NH-CONCURRENT"
    assert saved.facility_token == _CLIENT_TOKEN


@pytest.mark.parametrize("invalid_update", [False, True])
def test_freeze_authority_precedes_validation_and_blocks_every_partial_effect(
    postgres_product_sandbox: ProductSandbox, invalid_update: bool
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    saved = store.save(_enrollment())
    before = sandbox.admin.execute("SELECT * FROM edge_site").fetchone()
    freeze_authority(sandbox.database, sandbox.authority)
    hooks = []
    updates = {"unknown": "value"} if invalid_update else {"facility_id": "fenced-facility"}
    with pytest.raises(AuthorityFenced):
        store.save(updates, after_write=lambda connection: hooks.append(True))
    assert not hooks
    assert sandbox.admin.execute("SELECT * FROM edge_site").fetchone() == before
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (0,)
    assert store.load() == saved


def test_missing_bootstrap_singleton_fails_closed_without_creating_it(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    with pytest.raises(ConnectionSettingsNotInitialized, match="bootstrap row is missing"):
        store.load()
    hooks = []
    with pytest.raises(ConnectionSettingsNotInitialized) as error:
        store.save(_enrollment(), after_write=lambda connection: hooks.append(True))
    assert str(error.value) == "connection settings bootstrap row is missing"
    assert not hooks
    assert sandbox.admin.execute("SELECT count(*) FROM edge_site").fetchone() == (0,)


@pytest.mark.parametrize(
    "generation",
    [0, -1, True, False, 1.5, float("nan"), float("inf"), float("-inf"), 2**63, "1"],
)
def test_generation_requires_positive_finite_signed_bigint_without_coercion(
    postgres_product_sandbox: ProductSandbox, generation: object
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    before = store.load()
    with pytest.raises(InvalidConnectionSettingError) as error:
        store.save(_enrollment() | {"enrollment_generation": generation})
    assert str(error.value) == "invalid connection setting field: enrollment_generation"
    assert store.load() == before


@pytest.mark.parametrize(
    ("field_name", "limit"),
    [
        ("facility_code", 64),
        ("client_installation_ref", 128),
        ("facility_id", 128),
        ("facility_token", 512),
        ("edge_installation_id", 128),
    ],
)
def test_postgres_character_bounds_accept_limit_and_reject_overflow_before_sql(
    postgres_product_sandbox: ProductSandbox, field_name: str, limit: int
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    # PostgreSQL length(text) counts characters, not UTF-8 bytes.
    value = "가" * limit
    saved = store.save(_enrollment() | {field_name: value, "enrollment_generation": 2**63 - 1})
    assert getattr(saved, field_name) == value
    assert saved.enrollment_generation == 2**63 - 1
    assert store.load() == saved
    with pytest.raises(InvalidConnectionSettingError) as error:
        store.save({field_name: value + "가"})
    assert str(error.value) == f"invalid connection setting field: {field_name}"
    assert store.load() == saved


@pytest.mark.parametrize("value", ["", " \t", 7, True, b"bytes", "bad\x00text", "bad\ud800text"])
def test_invalid_text_is_rejected_without_echoing_credentials(
    postgres_product_sandbox: ProductSandbox, value: object
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    for field_name in (
        "facility_code",
        "client_installation_ref",
        "facility_id",
        "facility_token",
        "edge_installation_id",
    ):
        with pytest.raises(InvalidConnectionSettingError) as error:
            store.save(_enrollment() | {field_name: value})
        assert str(error.value) == f"invalid connection setting field: {field_name}"
        assert _CLIENT_TOKEN not in repr(error.value)
    assert sandbox.admin.execute("SELECT facility_token FROM edge_site").fetchone() == (None,)


@pytest.mark.parametrize(
    ("token", "masked"),
    [("ab", "****"), ("abcd", "****"), (_CLIENT_TOKEN, "****1234")],
    ids=("short", "four-characters", "long"),
)
def test_tokens_are_masked_and_absent_from_repr_and_validation_errors(
    postgres_product_sandbox: ProductSandbox,
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    masked: str,
) -> None:
    sandbox = postgres_product_sandbox
    monkeypatch.setenv(API_BACKEND_BASE_URL_ENV, "https://hub.example/api/")
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    assert store.masked()["facility_token_set"] is False
    assert store.masked()["facility_token_masked"] is None
    saved = store.save(_enrollment() | {"facility_token": token})
    assert "facility_token" not in repr(saved)
    assert token not in repr(saved)
    public = store.masked()
    assert public["facility_token_masked"] == masked and public["facility_token_set"] is True
    assert "facility_token" not in public
    assert public["events_url"] == "https://hub.example/api/v1/events"
    assert public["config_url"] == "https://hub.example/api/v1/ml-config"
    for updates in (
        {"facility_token": _CLIENT_TOKEN * 30},
        {_CLIENT_TOKEN: _CLIENT_TOKEN},
    ):
        with pytest.raises(InvalidConnectionSettingError) as error:
            store.save(updates)
        assert _CLIENT_TOKEN not in str(error.value)
        assert _CLIENT_TOKEN not in repr(error.value)
    assert store.load() == saved


def test_principal_change_resets_topology_but_preserves_unrelated_bigint_flags(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    store.save(_enrollment())
    sandbox.admin.execute(
        "UPDATE edge_site SET registry_version=9,clip_export_enabled=1,"
        "topology_snapshot_registry_version=9,topology_client_revision=4,topology_server_revision=3,"
        "topology_pending_snapshot_id='pending-1',topology_pending_body=%s,"
        "topology_pending_registry_version=9,topology_pending_client_revision=5,"
        "topology_pending_expected_server_revision=3,topology_consecutive_failures=2,"
        "topology_next_retry_at=42.0,topology_pause_reason='conflict',topology_last_accepted_at=40.0,"
        "topology_dirty_registry_version=9,topology_dirty_created_at=%s,"
        "topology_confirmation_id='confirm-1',topology_confirmation_digest=%s,"
        "topology_confirmation_expires_at=%s,topology_confirmation_snapshot_id='snapshot-1',"
        "topology_confirmation_client_revision=4,topology_confirmation_server_revision=3,"
        "topology_confirmation_registry_version=9,topology_confirmation_cameras=1,"
        "topology_confirmation_rooms=1,topology_confirmation_floors=1,"
        "topology_confirmation_confirmed=1,topology_confirmation_result='accepted' WHERE id=1",
        (b"{}", _FIRST_TIME, "a" * 64, _SECOND_TIME),
    )
    store.save({"facility_id": "facility-renamed"})
    assert sandbox.admin.execute(
        "SELECT topology_client_revision,topology_confirmation_id FROM edge_site"
    ).fetchone() == (4, "confirm-1")
    store.save({"edge_installation_id": "edge-2"})
    with sandbox.admin.cursor(row_factory=dict_row) as cursor:
        row = cursor.execute("SELECT * FROM edge_site WHERE id=1").fetchone()
    assert row is not None
    for name, value in row.items():
        if name in (
            "topology_snapshot_registry_version",
            "topology_client_revision",
            "topology_server_revision",
            "topology_consecutive_failures",
        ):
            assert value == 0
        elif name == "topology_dirty_registry_version":
            assert value == 9
        elif name == "topology_dirty_created_at":
            assert value == _FIRST_TIME
        elif name.startswith("topology_"):
            assert value is None
    assert row["registry_version"] == 9 and row["clip_export_enabled"] == 1
    assert type(row["clip_export_enabled"]) is int


def test_unknown_commit_propagates_without_replay_or_reloading_local_state(
    postgres_product_sandbox: ProductSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = postgres_product_sandbox
    store = ConnectionSettingsStore(sandbox.database, sandbox.authority)
    commit = psycopg.Connection.commit
    hook_pids = []
    commit_calls = []

    def after_write(connection: psycopg.Connection) -> None:
        hook_pids.append(connection.info.backend_pid)
        append_postgres_audit(connection, _audit_event())

    def lose_commit_receipt(connection: psycopg.Connection) -> None:
        pid = connection.info.backend_pid
        commit(connection)
        if pid in hook_pids:
            commit_calls.append(pid)
            raise psycopg.OperationalError("injected loss of commit receipt")

    monkeypatch.setattr(psycopg.Connection, "commit", lose_commit_receipt)
    with pytest.raises(CommitOutcomeUnknown):
        store.save(_enrollment(), after_write=after_write)
    monkeypatch.setattr(psycopg.Connection, "commit", commit)
    assert len(hook_pids) == len(commit_calls) == 1
    # The real commit happened, but save must not turn an unknown outcome into success.
    assert sandbox.admin.execute("SELECT facility_id FROM edge_site").fetchone() == ("facility-1",)
    assert sandbox.admin.execute("SELECT count(*) FROM audit_events").fetchone() == (1,)
    assert store.load().facility_id == "facility-1"
