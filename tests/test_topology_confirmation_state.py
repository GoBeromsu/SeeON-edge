"""Native confirmation persistence, including the post-network CAS boundary."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
from time import monotonic

import pytest

from backend.app.edge_db.authority import AuthorityFenced, freeze_authority
from backend.app.edge_db.postgres import (
    CommitOutcomeUnknown,
    PostgresError,
    PostgresTransactionStateError,
)
from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.topology_confirmation_state import (
    TopologyConfirmationStateConflictError,
    TopologyConfirmationStore,
)
from backend.app.features.connection.store import ConnectionSettingsStore
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 3)
SNAPSHOT_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81a"
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
DIGEST = "a" * 64
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
    # A real accepted snapshot whose client/server/registry versions match the
    # preview. These fields are revalidated by the terminal CAS, not mocked.
    sandbox.admin.execute(
        "UPDATE edge_site SET registry_version=12,topology_client_revision=4,"
        "topology_server_revision=7 WHERE id=1"
    )
    return sandbox


def _store(sandbox):
    return TopologyConfirmationStore(sandbox.database, sandbox.authority)


def _row(sandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def _result(*, deactivated=0):
    unchanged = MutationCounts(0, 0, 1)
    removed = MutationCounts(0, 0, 0, deactivated=deactivated)
    return TopologyMutationResult(removed, removed, unchanged)


def _preview_response():
    return TopologySuccessEnvelope(
        SNAPSHOT_ID,
        4,
        7,
        _result(),
        OmissionPreview(
            CONFIRMATION_ID,
            DIGEST,
            "2099-01-01T00:00:00.000Z",
            ("camera-old",),
            ("room-old",),
            ("floor-old",),
        ),
    )


def _terminal_response():
    return TopologySuccessEnvelope(SNAPSHOT_ID, 4, 8, _result(deactivated=1), None)


def _saved(sandbox):
    store = _store(sandbox)
    store.save(_preview_response(), PRINCIPAL, registry_version=12)
    preview = store.load()
    assert preview is not None
    return store, preview


def test_preview_and_terminal_response_survive_store_restart(sandbox):
    store = _store(sandbox)
    state_store = EdgeTopologySyncStateStore(sandbox.database, sandbox.authority)
    state_store.ensure_principal(PRINCIPAL)
    store.save(_preview_response(), PRINCIPAL, registry_version=12)
    persisted_preview = _store(sandbox).load()
    assert persisted_preview is not None
    store.complete(persisted_preview, _terminal_response())
    terminal_preview = _store(sandbox).load()
    assert persisted_preview.confirmed is False
    assert persisted_preview.registry_version == 12 and persisted_preview.principal == PRINCIPAL
    assert terminal_preview is not None and terminal_preview.confirmed is True
    assert terminal_preview.terminal_response == _terminal_response()


def test_confirmation_store_uses_no_feature_local_table(sandbox):
    assert _store(sandbox).load() is None
    tables = {
        row[0]
        for row in sandbox.admin.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=%s",
            (sandbox.schema,),
        )
    }
    assert "edge_topology_confirmation_preview" not in tables
    assert "edge_site" in tables


def test_borrowed_preview_and_clear_share_outer_transaction(sandbox):
    store, before = _store(sandbox), _row(sandbox)
    error = ValueError("reject compound operation")

    def body(connection):
        store.save(_preview_response(), PRINCIPAL, 12, connection=connection)
        assert store.load(connection=connection) is not None
        assert _row(sandbox) == before
        store.save(_terminal_response(), PRINCIPAL, 12, connection=connection)
        assert store.load(connection=connection) is None and not connection.closed
        raise error

    with pytest.raises(ValueError) as caught:
        sandbox.database.transact(body)
    assert caught.value is error and _row(sandbox) == before
    with pytest.raises(PostgresTransactionStateError):
        store.load(connection=sandbox.admin)


def test_preview_reset_and_clear_require_matching_enrollment(sandbox):
    store, preview = _saved(sandbox)
    store.complete(preview, _terminal_response())
    store.save(_preview_response(), PRINCIPAL, 12)
    assert store.load() == preview
    before = _row(sandbox)
    for response in (_preview_response(), _terminal_response()):
        with pytest.raises(TopologyConfirmationStateConflictError):
            store.save(response, replace(PRINCIPAL, enrollment_generation=4), 12)
        assert _row(sandbox) == before
    store.save(_terminal_response(), PRINCIPAL, 12)
    assert store.load() is None


def test_completion_callback_precedes_commit_and_result_follows_full_pool_exit(
    sandbox, monkeypatch
):
    store, preview = _saved(sandbox)
    before, trace = _row(sandbox), []
    acquire = sandbox.database._pool.connection

    @contextmanager
    def exiting(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            yield connection
        trace.append("pool-exit")

    def hook(connection):
        assert store.load(connection=connection).terminal_response == _terminal_response()
        assert _row(sandbox) == before
        trace.append("hook")

    monkeypatch.setattr(sandbox.database._pool, "connection", exiting)
    assert store.complete(preview, _terminal_response(), after_write=hook) is None
    trace.append("returned")
    assert trace == ["hook", "pool-exit", "returned"]
    assert sandbox.admin.execute(
        "SELECT topology_server_revision,topology_confirmation_confirmed FROM edge_site"
    ).fetchone() == (8, 1)


@pytest.mark.parametrize("cancelled", [False, True])
def test_completion_callback_failure_rolls_back_and_preserves_original(sandbox, cancelled):
    store, preview = _saved(sandbox)
    before, calls = _row(sandbox), []
    error = BaseException("cancelled") if cancelled else ValueError("refused")

    def fail(connection):
        assert store.load(connection=connection).confirmed
        calls.append(True)
        raise error

    with pytest.raises(type(error)) as caught:
        store.complete(preview, _terminal_response(), after_write=fail)
    assert caught.value is error and calls == [True] and _row(sandbox) == before


def test_borrowed_completion_does_not_commit_before_outer_owner(sandbox):
    store, preview = _saved(sandbox)
    before, calls = _row(sandbox), []
    error = ValueError("later outer failure")

    def body(connection):
        store.complete(
            preview,
            _terminal_response(),
            connection=connection,
            after_write=lambda c: calls.append(c.info.backend_pid),
        )
        assert calls == [connection.info.backend_pid]
        assert store.load(connection=connection).confirmed and _row(sandbox) == before
        raise error

    with pytest.raises(ValueError) as caught:
        sandbox.database.transact(body)
    assert caught.value is error and _row(sandbox) == before


@pytest.mark.parametrize(
    "assignment",
    [
        "registry_version=13",
        "topology_client_revision=5",
        "topology_server_revision=8",
        "enrollment_generation=4",
        "edge_installation_id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'",
        "topology_confirmation_digest='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'",
        "topology_confirmation_snapshot_id='other-snapshot'",
        "topology_confirmation_registry_version=11",
        "topology_confirmation_expires_at='2099-01-02T00:00:00.000Z'",
        "topology_confirmation_confirmed=1",
    ],
)
def test_post_network_stale_facts_reject_before_completion_hook(sandbox, assignment):
    store, preview = _saved(sandbox)
    # Trusted literal test cases, never request-supplied SQL.
    sandbox.admin.execute("UPDATE edge_site SET " + assignment + " WHERE id=1")
    before, calls = _row(sandbox), []
    with pytest.raises(TopologyConfirmationStateConflictError):
        store.complete(preview, _terminal_response(), after_write=lambda c: calls.append(True))
    assert not calls and _row(sandbox) == before


@pytest.mark.parametrize("field", ["cameras", "rooms", "floors"])
def test_same_identity_preview_resave_invalidates_old_completion_counts(sandbox, field):
    store, old = _saved(sandbox)
    response = _preview_response()
    changed = replace(response.omissions, **{field: (*getattr(response.omissions, field), "new")})
    store.save(replace(response, omissions=changed), PRINCIPAL, 12)
    before, calls = _row(sandbox), []
    with pytest.raises(TopologyConfirmationStateConflictError):
        store.complete(old, _terminal_response(), after_write=lambda c: calls.append(True))
    assert not calls and _row(sandbox) == before
    fresh = store.load()
    assert getattr(fresh, field) == 2
    store.complete(fresh, _terminal_response(), after_write=lambda c: calls.append(True))
    assert calls == [True] and store.load().confirmed


@pytest.mark.parametrize("field,value", [("snapshot_id", "wrong"), ("client_revision", 5)])
def test_response_identity_mismatch_never_reaches_completion_hook(sandbox, field, value):
    store, preview = _saved(sandbox)
    before, calls = _row(sandbox), []
    with pytest.raises(TopologyConfirmationStateConflictError):
        store.complete(
            preview,
            replace(_terminal_response(), **{field: value}),
            after_write=lambda c: calls.append(True),
        )
    assert not calls and _row(sandbox) == before


def test_confirmation_authority_fence_and_missing_bootstrap_are_not_empty_state(sandbox):
    store, preview = _saved(sandbox)
    freeze_authority(sandbox.database, sandbox.authority)
    before, calls = _row(sandbox), []
    with pytest.raises(AuthorityFenced):
        store.complete(preview, _terminal_response(), after_write=lambda c: calls.append(True))
    assert not calls and _row(sandbox) == before
    sandbox.admin.execute("DELETE FROM edge_site WHERE id=1")
    with pytest.raises(PostgresError, match="bootstrap row is missing"):
        store.load()


def test_constraint_rejection_redacts_credential_bearing_row(sandbox):
    store = _store(sandbox)
    secret = "confirmation-private-facility-token"
    sandbox.admin.execute("UPDATE edge_site SET facility_token=%s WHERE id=1", (secret,))
    before = _row(sandbox)
    with pytest.raises(PostgresError, match="database constraint") as caught:
        store.save(_preview_response(), PRINCIPAL, -1)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__ and caught.value.__cause__ is None
    assert _row(sandbox) == before


@pytest.mark.parametrize(
    "encoded",
    [
        "0,0,0;0,0,0,0,0;0,0,0,0,0",
        "1,1,1,1,1;1,1,1,1,1",
        "x,0,0,0,0;0,0,0,0,0;0,0,0,0,0",
        "-1,0,0,0,0;0,0,0,0,0;0,0,0,0,0",
    ],
)
def test_malformed_terminal_counts_never_become_a_confirmed_response(sandbox, encoded):
    store, _ = _saved(sandbox)
    sandbox.admin.execute(
        "UPDATE edge_site SET topology_confirmation_confirmed=1,topology_confirmation_result=%s",
        (encoded,),
    )
    with pytest.raises(PostgresError, match="stored topology confirmation result is malformed"):
        store.load()


def test_deferred_commit_rejection_is_not_successful_confirmation(sandbox):
    store, preview = _saved(sandbox)
    before, calls = _row(sandbox), []
    sandbox.admin.execute(
        "CREATE FUNCTION reject_confirmation() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'confirmation rejected' USING ERRCODE='23514'; END $$"
    )
    sandbox.admin.execute(
        "CREATE CONSTRAINT TRIGGER reject_confirmation AFTER UPDATE ON edge_site "
        "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION reject_confirmation()"
    )
    with pytest.raises(PostgresError, match="database constraint"):
        store.complete(preview, _terminal_response(), after_write=lambda c: calls.append(True))
    assert calls == [True] and _row(sandbox) == before


@pytest.mark.parametrize("kind", ["ordinary", "cancellation", "unknown"])
def test_post_commit_owner_failures_preserve_identity_without_replay(sandbox, monkeypatch, kind):
    store, preview = _saved(sandbox)
    error = {
        "ordinary": OSError("owner exit"),
        "cancellation": BaseException("cancelled"),
        "unknown": CommitOutcomeUnknown(),
    }[kind]
    original, calls = sandbox.database.transact, []

    def fail(callback):
        original(callback)
        calls.append(True)
        raise error

    monkeypatch.setattr(sandbox.database, "transact", fail)
    with pytest.raises(type(error)) as caught:
        store.complete(preview, _terminal_response())
    assert caught.value is error and calls == [True]
    assert sandbox.admin.execute(
        "SELECT topology_confirmation_confirmed FROM edge_site"
    ).fetchone() == (1,)


def test_concurrent_confirmation_cas_has_one_winner_and_one_hook(sandbox):
    store, preview = _saved(sandbox)
    second, entered, release = _store(sandbox), Event(), Event()
    hooks, pids, futures = [], [], []
    executor = ThreadPoolExecutor(max_workers=2)

    def first_hook(connection):
        pids.append(connection.info.backend_pid)
        hooks.append("first")
        entered.set()
        assert release.wait(2), "confirmation CAS not released"

    try:
        futures.append(
            executor.submit(store.complete, preview, _terminal_response(), after_write=first_hook)
        )
        assert entered.wait(1)
        futures.append(
            executor.submit(
                second.complete,
                preview,
                _terminal_response(),
                after_write=lambda c: hooks.append("second"),
            )
        )
        deadline, pacing = monotonic() + 1.5, Event()
        while not sandbox.admin.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)))",
            (pids[0],),
        ).fetchone()[0]:
            assert monotonic() < deadline, "competing confirmation did not block"
            pacing.wait(0.01)
        assert not any(future.done() for future in futures) and hooks == ["first"]
        release.set()
        assert futures[0].result(timeout=2) is None
        with pytest.raises(TopologyConfirmationStateConflictError):
            futures[1].result(timeout=2)
        assert hooks == ["first"]
    finally:
        release.set()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "confirmation clients did not drain"
