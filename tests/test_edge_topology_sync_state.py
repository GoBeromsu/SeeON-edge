"""Real PostgreSQL topology state contracts, not coordinator/network qualification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Event
from time import monotonic

import psycopg
import pytest

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PostgresError,
    PostgresTransactionStateError,
)
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    TopologyPauseReason,
    TopologySyncStateConflictError,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.store import ConnectionSettingsStore
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)
pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def sandbox(postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    ConnectionSettingsStore(sandbox.database, sandbox.authority).save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "token-1",
            "edge_installation_id": PRINCIPAL.edge_installation_id,
            "enrollment_generation": PRINCIPAL.enrollment_generation,
        }
    )
    return sandbox


@dataclass(frozen=True, slots=True)
class _Builder:
    registry_version: int
    principal: MachinePrincipal
    snapshot_id: str
    marker: bytes

    def build(self, client_revision: int, expected_server_revision: int) -> bytes:
        return self.marker + f":{client_revision}:{expected_server_revision}".encode()


def _accepted(snapshot_id, client_revision, server_revision):
    counts = MutationCounts(0, 0, 1)
    return TopologySuccessEnvelope(
        snapshot_id,
        client_revision,
        server_revision,
        TopologyMutationResult(counts, counts, counts),
        None,
    )


def _store(sandbox):
    return EdgeTopologySyncStateStore(sandbox.database, sandbox.authority)


def _row(sandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def test_pending_snapshot_and_backoff_survive_restart_byte_identically(sandbox):
    store = _store(sandbox)
    store.ensure_principal(PRINCIPAL)
    pending = store.create_pending(_Builder(7, PRINCIPAL, "snapshot-a", b"canonical"))
    store.record_retry(pending.snapshot_id, now_epoch=100.0)
    restarted = _store(sandbox).load()
    assert restarted.pending == pending and restarted.pending is not None
    assert restarted.pending.body == b"canonical:1:0"
    assert restarted.consecutive_failures == 1 and restarted.next_retry_at == 105.0


def test_accept_clears_only_the_represented_dirty_registry_version(sandbox):
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    represented_version = registry.topology_snapshot().registry_version
    state = _store(sandbox)
    state.ensure_principal(PRINCIPAL)
    pending = state.create_pending(_Builder(represented_version, PRINCIPAL, "snapshot-a", b"body"))
    registry.create_floor(edge_ref="floor-2", name="Second", order_index=2)
    accepted = state.accept(
        pending.snapshot_id,
        _accepted(pending.snapshot_id, pending.client_revision, 1),
    )
    assert accepted.pending is None
    assert accepted.last_snapshotted_registry_version == represented_version
    dirty = registry.topology_snapshot().dirty
    assert dirty is not None and dirty.registry_version == represented_version + 1


def test_generation_change_discards_old_pending_but_keeps_registry_dirty(sandbox):
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create_floor(edge_ref="floor-1", name="First", order_index=1)
    state = _store(sandbox)
    state.ensure_principal(PRINCIPAL)
    state.create_pending(_Builder(1, PRINCIPAL, "snapshot-a", b"old"))
    ConnectionSettingsStore(sandbox.database, sandbox.authority).save(
        {
            "facility_code": "NH-1234",
            "client_installation_ref": "install-1",
            "facility_id": "facility-1",
            "facility_token": "token-2",
            "edge_installation_id": PRINCIPAL.edge_installation_id,
            "enrollment_generation": 2,
        }
    )
    changed = state.ensure_principal(MachinePrincipal(PRINCIPAL.edge_installation_id, 2))
    assert changed.pending is None and changed.last_client_revision == 0
    assert registry.topology_snapshot().dirty is not None


def test_compound_operation_shares_one_connection_and_returns_after_pool_exit(sandbox, monkeypatch):
    store = _store(sandbox)
    before, trace, pids = _row(sandbox), [], []
    acquire = sandbox.database._pool.connection

    @contextmanager
    def observed_exit(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            pids.append(connection.info.backend_pid)
            yield connection
        trace.append("pool-exit")

    def body(connection):
        store.ensure_principal(PRINCIPAL, connection=connection)
        pending = store.create_pending(
            _Builder(7, PRINCIPAL, "snapshot-a", b"body"),
            connection=connection,
        )
        store.record_retry(pending.snapshot_id, now_epoch=100.0, connection=connection)
        assert _row(sandbox) == before
        trace.append("body")
        return store.load(connection=connection)

    def callback(connection):
        assert connection.info.backend_pid == pids[0]
        assert store.load(connection=connection).next_retry_at == 105.0
        assert _row(sandbox) == before
        trace.append("callback")

    monkeypatch.setattr(sandbox.database._pool, "connection", observed_exit)
    result = store.operation(body, after_write=callback)
    trace.append("returned")
    assert result.next_retry_at == 105.0 and len(pids) == 1
    assert trace == ["body", "callback", "pool-exit", "returned"]


def test_noop_operation_still_calls_explicit_callback_once(sandbox):
    store = _store(sandbox)
    before, calls = _row(sandbox), []
    assert (
        store.operation(lambda connection: "not-attempted", after_write=lambda c: calls.append(1))
        == "not-attempted"
    )
    assert calls == [1] and _row(sandbox) == before


def test_compound_snapshot_uses_the_supplied_owner_transaction(sandbox, monkeypatch):
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    before = registry.topology_snapshot().registry_version

    def refuse_nested_read(callback):
        pytest.fail("borrowed topology snapshot opened a second read owner")

    def body(connection):
        connection.execute("UPDATE edge_site SET registry_version=registry_version+1 WHERE id=1")
        snapshot = registry.topology_snapshot(connection=connection)
        assert snapshot.registry_version == before + 1
        assert sandbox.admin.execute(
            "SELECT registry_version FROM edge_site WHERE id=1"
        ).fetchone() == (before,)
        return snapshot

    monkeypatch.setattr(sandbox.database, "read", refuse_nested_read)
    result = _store(sandbox).operation(body)
    assert result.registry_version == before + 1
    assert sandbox.admin.execute(
        "SELECT registry_version FROM edge_site WHERE id=1"
    ).fetchone() == (before + 1,)
    with pytest.raises(PostgresTransactionStateError):
        registry.topology_snapshot(connection=sandbox.admin)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("failure_stage", ["body", "callback"])
def test_compound_failure_rolls_back_all_state_stages(sandbox, cancelled, failure_stage):
    store = _store(sandbox)
    before, calls = _row(sandbox), []
    error = BaseException("cancelled") if cancelled else ValueError("refused")

    def body(connection):
        store.create_pending(_Builder(1, PRINCIPAL, "snapshot-a", b"body"), connection=connection)
        store.record_retry("snapshot-a", now_epoch=50, connection=connection)
        store.pause("snapshot-a", TopologyPauseReason.FORBIDDEN, connection=connection)
        if failure_stage == "body":
            raise error

    def fail(connection):
        assert store.load(connection=connection).pause_reason is TopologyPauseReason.FORBIDDEN
        calls.append(True)
        raise error

    with pytest.raises(type(error)) as caught:
        store.operation(body, after_write=fail)
    assert caught.value is error
    assert calls == ([] if failure_stage == "body" else [True])
    assert _row(sandbox) == before


def test_borrowed_state_never_commits_or_closes_the_outer_transaction(sandbox):
    store, before = _store(sandbox), _row(sandbox)
    failure = ValueError("outer owner rejected")

    def outer(connection):
        pending = store.create_pending(
            _Builder(1, PRINCIPAL, "snapshot-a", b"body"), connection=connection
        )
        assert pending.body == b"body:1:0" and not connection.closed
        assert _row(sandbox) == before
        raise failure

    with pytest.raises(ValueError) as caught:
        sandbox.database.transact(outer)
    assert caught.value is failure and _row(sandbox) == before
    with pytest.raises(PostgresTransactionStateError):
        store.load(connection=sandbox.admin)
    with pytest.raises(PostgresTransactionStateError):
        store.create_pending(
            _Builder(1, PRINCIPAL, "snapshot-a", b"body"), connection=sandbox.admin
        )


@pytest.mark.parametrize("fence", ["frozen", "generation"])
def test_topology_state_requires_current_deployment_authority(sandbox, fence):
    authority = sandbox.authority
    if fence == "frozen":
        freeze_authority(sandbox.database, authority)
    else:
        authority = replace(authority, generation=authority.generation + 1)
    store = EdgeTopologySyncStateStore(sandbox.database, authority)
    before, calls = _row(sandbox), []
    with pytest.raises(AuthorityFenced):
        store.operation(lambda c: calls.append(True))
    assert calls == [] and _row(sandbox) == before


def test_missing_bootstrap_is_not_synthetic_empty_topology(sandbox):
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    store = _store(sandbox)
    with pytest.raises(PostgresError, match="bootstrap row is missing"):
        store.load()
    with pytest.raises(PostgresError, match="bootstrap row is missing"):
        store.ensure_principal(PRINCIPAL)
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    with pytest.raises(PostgresError, match="bootstrap row is missing"):
        registry.camera_count()


def test_registry_camera_count_tracks_actual_rows_without_status_publication(sandbox):
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    for camera_id in ("count-a", "count-b"):
        registry.create(
            camera_id=camera_id,
            label=camera_id,
            rtsp_url=f"rtsp://camera/{camera_id}",
            space_id=None,
            status="online",
        )
    assert registry.camera_count() == 2
    assert registry.delete("count-a")
    assert registry.camera_count() == 1
    assert CameraRegistryStore(sandbox.database, sandbox.authority).camera_count() == 1
    assert registry.delete("count-b")
    assert registry.camera_count() == 0


def test_pending_identity_and_backoff_cap_survive_competing_attempts(sandbox):
    store = _store(sandbox)
    original = store.create_pending(_Builder(1, PRINCIPAL, "snapshot-a", b"a"))
    assert store.create_pending(_Builder(2, PRINCIPAL, "snapshot-b", b"b")) == original
    before = _row(sandbox)
    with pytest.raises(TopologySyncStateConflictError):
        store.record_retry("snapshot-b", now_epoch=0)
    with pytest.raises(TopologySyncStateConflictError):
        store.accept("snapshot-a", _accepted("snapshot-a", 2, 1))
    assert _row(sandbox) == before
    delays = [store.record_retry("snapshot-a", now_epoch=100).next_retry_at - 100 for _ in range(8)]
    assert delays == [5, 10, 20, 40, 80, 160, 300, 300]
    sandbox.admin.execute("UPDATE edge_site SET topology_consecutive_failures=1000000 WHERE id=1")
    assert store.record_retry("snapshot-a", now_epoch=100).next_retry_at == 400
    assert store.pause("snapshot-a", TopologyPauseReason.CONFLICT).next_retry_at is None
    assert store.resume_pending("snapshot-a").pause_reason is None
    refreshed = store.refresh_conflict("snapshot-a", 9)
    assert refreshed.pending is None and refreshed.server_revision == 9


def test_deferred_commit_failure_and_post_commit_uncertainty_are_not_replayed(sandbox, monkeypatch):
    store, before, calls = _store(sandbox), _row(sandbox), []
    sandbox.admin.execute(
        "CREATE FUNCTION reject_topology() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'topology rejected' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_topology AFTER UPDATE ON edge_site "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_topology()"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        store.operation(
            lambda c: store.create_pending(
                _Builder(1, PRINCIPAL, "snapshot-a", b"body"), connection=c
            ),
            after_write=lambda c: calls.append(True),
        )
    assert calls == [True] and _row(sandbox) == before
    sandbox.admin.execute("DROP TRIGGER reject_topology ON edge_site")
    original, error, writes = sandbox.database.transact, CommitOutcomeUnknown(), []

    def uncertain(callback):
        original(callback)
        writes.append(True)
        raise error

    monkeypatch.setattr(sandbox.database, "transact", uncertain)
    with pytest.raises(CommitOutcomeUnknown) as caught:
        store.create_pending(_Builder(1, PRINCIPAL, "snapshot-a", b"body"))
    assert caught.value is error and writes == [True]
    assert sandbox.admin.execute(
        "SELECT topology_pending_snapshot_id FROM edge_site"
    ).fetchone() == ("snapshot-a",)


def test_concurrent_state_owners_serialize_enqueue_identity_on_site_row(sandbox):
    store, entered, release = _store(sandbox), Event(), Event()
    other = _store(sandbox)
    executor = ThreadPoolExecutor(max_workers=2)
    futures, pids = [], []

    def first(connection):
        pending = store.create_pending(
            _Builder(1, PRINCIPAL, "snapshot-a", b"a"), connection=connection
        )
        pids.append(connection.info.backend_pid)
        entered.set()
        assert release.wait(2), "topology writer not released"
        return pending

    try:
        futures.append(executor.submit(store.operation, first))
        assert entered.wait(1)
        futures.append(
            executor.submit(other.create_pending, _Builder(2, PRINCIPAL, "snapshot-b", b"b"))
        )
        deadline, pacing = monotonic() + 1.5, Event()
        while not sandbox.admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)))",
            (pids[0],),
        ).fetchone()[0]:
            assert monotonic() < deadline, "competing topology writer did not block"
            pacing.wait(0.01)
        assert not any(future.done() for future in futures)
        release.set()
        accepted = futures[0].result(timeout=2)
        assert futures[1].result(timeout=2) == accepted
        assert _store(sandbox).load().pending == accepted
    finally:
        release.set()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "topology writes did not drain"
