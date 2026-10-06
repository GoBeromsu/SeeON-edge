from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest

from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import AuditRuntimeUnavailable
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
    TopologyPauseReason,
    TopologySyncStateConflictError,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPaused,
    TopologyRetryable,
)
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)
from tests_support.postgres_sandbox import ObservedAuditMutation

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 1)
pytest_plugins = ("tests_support.postgres_sandbox",)


@pytest.fixture
def sandbox(postgres_product_sandbox):
    return postgres_product_sandbox


def _audit(runtime, hook):
    return ObservedAuditMutation(
        runtime,
        lambda: AuditEvent(
            occurred_at="2026-09-27T00:00:00Z",
            actor_id="operator-1",
            action=AuditAction.CONNECTION_SYNC,
            target_id="camera-roster",
            detail=empty_detail(AuditAction.CONNECTION_SYNC),
        ),
        hook,
    )


def _state(sandbox):
    return EdgeTopologySyncStateStore(sandbox.database, sandbox.authority)


def _row(sandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def _ready_registry(sandbox):
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
    store = CameraRegistryStore(sandbox.database, sandbox.authority)
    store.create_floor(edge_ref="floor-1", name="First", order_index=1)
    store.create_room(edge_ref="room-101", floor_edge_ref="floor-1", name="101")
    store.create(
        camera_id="local-1",
        label="Lobby",
        rtsp_url="rtsp://private",
        space_id=None,
        status="online",
        edge_ref="camera-1",
        room_edge_ref="room-101",
    )
    return store


def _accepted(pending: PendingTopologySnapshot, server_revision=1):
    counts = MutationCounts(0, 0, 1)
    return TopologyAccepted(
        TopologySuccessEnvelope(
            pending.snapshot_id,
            pending.client_revision,
            server_revision,
            TopologyMutationResult(counts, counts, counts),
            None,
        )
    )


class _Client:
    principal = PRINCIPAL

    def __init__(self, outcomes):
        self.outcomes, self.sent, self.confirmations = outcomes, [], []
        self.refreshed_revision = None

    def put(self, pending):
        self.sent.append(pending)
        return self.outcomes.pop(0)(pending)

    def refresh_server_revision(self):
        return self.refreshed_revision

    def confirm(self, snapshot_id, confirmation):
        self.confirmations.append((snapshot_id, confirmation))
        return TopologyRetryable("unreachable")


def test_timeout_after_commit_restarts_with_byte_identical_pending_snapshot(sandbox):
    registry = _ready_registry(sandbox)
    first_client = _Client([lambda p: TopologyRetryable("timeout")])
    first = TopologyRetryCoordinator(registry, _state(sandbox), lambda: first_client)
    first_result = first.trigger(force=True, now_epoch=100.0)
    second_client = _Client([_accepted])
    restarted = TopologyRetryCoordinator(
        CameraRegistryStore(sandbox.database, sandbox.authority),
        _state(sandbox),
        lambda: second_client,
    )
    second_result = restarted.trigger(force=True, now_epoch=101.0)
    assert first_result.status == "failed" and second_result.status == "synced"
    assert first_client.sent[0] == second_client.sent[0]
    assert first_client.sent[0].body == second_client.sent[0].body


def test_boot_recovery_enqueues_exactly_one_snapshot_for_fresh_dirty_registry(sandbox):
    registry, client = _ready_registry(sandbox), _Client([_accepted])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    first = coordinator.trigger(force=True, now_epoch=100.0)
    second = coordinator.trigger(force=True, now_epoch=101.0)
    assert first.status == "synced" and second.attempted is False and len(client.sent) == 1


def test_coordinator_crash_leaves_pending_for_exact_restart_resume(sandbox):
    registry = _ready_registry(sandbox)

    class SimulatedCrash(RuntimeError):
        pass

    def crash(pending):
        raise SimulatedCrash

    client = _Client([crash])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    with pytest.raises(SimulatedCrash):
        coordinator.trigger(force=True, now_epoch=100.0)
    resumed_client = _Client([_accepted])
    restarted = TopologyRetryCoordinator(
        CameraRegistryStore(sandbox.database, sandbox.authority),
        _state(sandbox),
        lambda: resumed_client,
    )
    resumed = restarted.trigger(force=True, now_epoch=101.0)
    assert resumed.status == "synced" and client.sent[0] == resumed_client.sent[0]


def test_conflict_pauses_until_refresh_then_rebuilds_against_refreshed_revision(sandbox):
    registry = _ready_registry(sandbox)
    client = _Client(
        [
            lambda p: TopologyPaused(TopologyPauseReason.CONFLICT, 409),
            lambda pending: _accepted(pending, server_revision=5),
        ]
    )
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    paused = coordinator.trigger(force=True, now_epoch=100.0)
    blind = coordinator.trigger(force=True, now_epoch=101.0)
    client.refreshed_revision = 4
    resumed = coordinator.trigger(force=True, refresh=True, now_epoch=102.0)
    assert paused.status == "failed" and blind.attempted is False and resumed.status == "synced"
    assert len(client.sent) == 2 and client.sent[0].snapshot_id != client.sent[1].snapshot_id
    assert json_expected_revision(client.sent[1].body) == 4


@pytest.mark.parametrize(
    ("reason", "status"), [(TopologyPauseReason.AUTH, 401), (TopologyPauseReason.FORBIDDEN, 403)]
)
def test_auth_pause_resumes_exact_pending_after_connectivity(sandbox, reason, status):
    registry = _ready_registry(sandbox)
    client = _Client([lambda p: TopologyPaused(reason, status), _accepted])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    coordinator.trigger(force=True, now_epoch=100.0)
    coordinator.trigger(force=True, refresh=True, now_epoch=101.0)
    assert client.sent[0] == client.sent[1]


def test_concurrent_trigger_is_single_flight(sandbox, postgres_audit_runtime):
    registry, entered, release = _ready_registry(sandbox), Event(), Event()

    def block_then_accept(pending):
        entered.set()
        assert release.wait(2), "first topology request not released"
        return _accepted(pending)

    client = _Client([block_then_accept])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    executor, futures, hooks = ThreadPoolExecutor(max_workers=1), [], []
    try:
        futures.append(executor.submit(coordinator.trigger, force=True, now_epoch=100.0))
        assert entered.wait(1)
        concurrent = coordinator.trigger(
            force=True,
            now_epoch=100.0,
            audit=_audit(postgres_audit_runtime, lambda c: hooks.append(True)),
        )
        release.set()
        assert futures[0].result(timeout=2).status == "synced"
        assert concurrent.attempted is False and len(client.sent) == 1
        assert not hooks
    finally:
        release.set()
        _, pending = wait(futures, timeout=2)
        executor.shutdown(wait=not pending, cancel_futures=True)
        assert not pending, "topology coordinator did not drain"


def json_expected_revision(body):
    return int(json.loads(body)["expectedServerRevision"])


def test_nonforced_retry_preserves_pending_before_deadline_and_retries_at_deadline(
    sandbox, monkeypatch, postgres_audit_runtime
):
    registry = _ready_registry(sandbox)
    client = _Client([lambda p: TopologyRetryable("timeout"), _accepted])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    assert coordinator.trigger(now_epoch=100.0).status == "failed"
    pending = _state(sandbox).load().pending
    assert pending == client.sent[0]
    before, hooks, leases = _row(sandbox), [], []
    acquire = sandbox.database._pool.connection

    @contextmanager
    def count_leases(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            leases.append(connection.info.backend_pid)
            yield connection

    monkeypatch.setattr(sandbox.database._pool, "connection", count_leases)
    skipped = coordinator.trigger(
        now_epoch=104.999,
        audit=_audit(postgres_audit_runtime, lambda c: hooks.append(c.info.backend_pid)),
    )
    assert not skipped.attempted and client.sent == [pending]
    assert _row(sandbox) == before and len(leases) == 1 and hooks == leases
    retried = coordinator.trigger(
        now_epoch=105.0,
        audit=_audit(postgres_audit_runtime, lambda c: hooks.append(c.info.backend_pid)),
    )
    assert retried.attempted and retried.status == "synced"
    assert client.sent == [pending, pending]
    assert client.sent[0].body == client.sent[1].body
    assert len(leases) == 2 and hooks == leases


@pytest.mark.parametrize("kind", ["accepted", "retryable", "paused"])
def test_explicit_sync_borrows_one_connection_through_network_state_and_projection(
    sandbox, monkeypatch, kind, postgres_audit_runtime
):
    registry, trace, pids = _ready_registry(sandbox), [], []
    before = _row(sandbox)

    def upstream(pending):
        assert _row(sandbox) == before
        trace.append("network")
        return {
            "accepted": _accepted(pending),
            "retryable": TopologyRetryable("timeout"),
            "paused": TopologyPaused(TopologyPauseReason.FORBIDDEN, 403),
        }[kind]

    client = _Client([upstream])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    acquire = sandbox.database._pool.connection

    @contextmanager
    def owner_exit(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            pids.append(connection.info.backend_pid)
            yield connection
        trace.append("owner-exit")

    def reject_nested_read(callback):
        pytest.fail("compound topology operation opened a second read owner")

    def hook(connection):
        assert pids == [connection.info.backend_pid]
        assert _row(sandbox) == before
        trace.append("hook")

    monkeypatch.setattr(sandbox.database._pool, "connection", owner_exit)
    monkeypatch.setattr(sandbox.database, "read", reject_nested_read)
    result = coordinator.trigger(
        force=True, now_epoch=100.0, audit=_audit(postgres_audit_runtime, hook)
    )
    assert result.status == ("synced" if kind == "accepted" else "failed")
    assert result.camera_count == 1 and len(pids) == 1
    assert trace == ["network", "hook", "owner-exit"]


def test_explicit_noop_audits_once_but_outer_unconfigured_exit_audits_zero(
    sandbox, postgres_audit_runtime
):
    registry, client, hooks = _ready_registry(sandbox), _Client([_accepted]), []
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    coordinator.trigger(force=True, now_epoch=100.0)
    result = coordinator.trigger(
        force=True,
        now_epoch=101.0,
        audit=_audit(postgres_audit_runtime, lambda c: hooks.append(True)),
    )
    assert result.attempted is False and hooks == [True] and len(client.sent) == 1
    unconfigured = TopologyRetryCoordinator(registry, _state(sandbox), lambda: None)
    assert (
        unconfigured.trigger(
            audit=_audit(postgres_audit_runtime, lambda c: hooks.append(False))
        ).attempted
        is False
    )
    assert hooks == [True]


@pytest.mark.parametrize("stage", ["body", "hook"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_explicit_sync_failure_rolls_back_without_pretending_to_undo_network(
    sandbox, stage, cancelled, postgres_audit_runtime
):
    registry, hooks = _ready_registry(sandbox), []
    before = _row(sandbox)
    error = BaseException("cancelled") if cancelled else ValueError("refused")

    def upstream(pending):
        if stage == "body":
            raise error
        return _accepted(pending)

    def hook(connection):
        hooks.append(True)
        raise error

    client = _Client([upstream])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    with pytest.raises(type(error)) as caught:
        coordinator.trigger(force=True, now_epoch=100.0, audit=_audit(postgres_audit_runtime, hook))
    assert caught.value is error and _row(sandbox) == before
    assert len(client.sent) == 1 and hooks == ([] if stage == "body" else [True])


def test_background_preview_and_acceptance_rollback_together_after_network(sandbox):
    registry = _ready_registry(sandbox)

    def mismatch(pending):
        accepted = _accepted(pending).response
        return TopologyAccepted(
            replace(
                accepted,
                client_revision=pending.client_revision + 1,
                omissions=OmissionPreview(
                    "confirmation-1", "a" * 64, "2099-01-01T00:00:00.000Z", ("old-camera",), (), ()
                ),
            )
        )

    client = _Client([mismatch])
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    with pytest.raises(TopologySyncStateConflictError):
        coordinator.trigger(force=True, now_epoch=100.0)
    # The pre-network pending enqueue is durable; the later preview plus
    # acceptance transaction is not. No fabricated rollback of upstream I/O.
    assert _state(sandbox).load().pending == client.sent[0]
    assert coordinator.preview() is None


@pytest.mark.parametrize("network", ["put", "refresh"])
def test_trigger_rechecks_admission_after_owner_entry_before_egress(
    sandbox, postgres_audit_runtime, monkeypatch, network
):
    registry, client = _ready_registry(sandbox), _Client([_accepted])
    state = _state(sandbox)
    coordinator = TopologyRetryCoordinator(registry, state, lambda: client)
    if network == "refresh":
        client.outcomes.insert(0, lambda p: TopologyPaused(TopologyPauseReason.CONFLICT, 409))
        coordinator.trigger(force=True, now_epoch=100.0)
    before, sent, hooks, refreshes = _row(sandbox), list(client.sent), [], []
    audit = _audit(postgres_audit_runtime, lambda c: hooks.append(True))
    ensure, refresh = state.ensure_principal, client.refresh_server_revision

    def admission_lost(*args, **kwargs):
        result = ensure(*args, **kwargs)
        postgres_audit_runtime.stop()
        return result

    def refresh_observed():
        refreshes.append(True)
        return refresh()

    monkeypatch.setattr(state, "ensure_principal", admission_lost)
    monkeypatch.setattr(client, "refresh_server_revision", refresh_observed)
    with pytest.raises(AuditRuntimeUnavailable):
        coordinator.trigger(force=True, refresh=True, now_epoch=200.0, audit=audit)
    assert client.sent == sent and not refreshes and not hooks
    assert _row(sandbox) == before and not postgres_audit_runtime._pending
