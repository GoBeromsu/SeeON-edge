from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from backend.app.edge_db.postgres import CommitOutcomeUnknown
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import AuditRuntimeUnavailable
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.edge_topology_sync_state import (
    EdgeTopologySyncStateStore,
    PendingTopologySnapshot,
)
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.cameras.topology_client import (
    TopologyAccepted,
    TopologyPaused,
    TopologyPutResult,
    TopologyRetryable,
)
from backend.app.features.cameras.topology_confirmation import (
    TopologyConfirmationCommand,
    TopologyConfirmationRejected,
)
from backend.app.features.cameras.topology_confirmation_state import TopologyConfirmationStore
from backend.app.features.connection.store import ConnectionSettingsStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from contracts.edge_provisioning_v1 import (
    MachinePrincipal,
    MutationCounts,
    OmissionPreview,
    TopologyConfirmation,
    TopologyMutationResult,
    TopologySuccessEnvelope,
)
from tests_support.postgres_sandbox import ObservedAuditMutation

PRINCIPAL = MachinePrincipal("c72bd9a7-3e04-47ba-a8cd-a56e54f98152", 3)
CONFIRMATION_ID = "0197f671-3a31-7a6c-a6e4-83ed412de81b"
DIGEST = "a" * 64
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
            action=AuditAction.TOPOLOGY_CONFIRM,
            target_id=CONFIRMATION_ID,
            detail=empty_detail(AuditAction.TOPOLOGY_CONFIRM),
        ),
        hook,
    )


class _Client:
    def __init__(
        self,
        principal: MachinePrincipal,
        put_outcomes: list[Callable[[PendingTopologySnapshot], TopologyPutResult]],
        confirm_outcomes: list[TopologyPutResult],
    ) -> None:
        self.principal, self.put_outcomes, self.confirm_outcomes = (
            principal,
            put_outcomes,
            confirm_outcomes,
        )
        self.sent: list[PendingTopologySnapshot] = []
        self.confirmations: list[tuple[str, TopologyConfirmation]] = []

    def put(self, pending):
        self.sent.append(pending)
        return self.put_outcomes.pop(0)(pending)

    def confirm(self, snapshot_id, confirmation):
        self.confirmations.append((snapshot_id, confirmation))
        return self.confirm_outcomes.pop(0)

    def refresh_server_revision(self):
        return None


def _state(sandbox):
    return EdgeTopologySyncStateStore(sandbox.database, sandbox.authority)


def _row(sandbox):
    return sandbox.admin.execute("SELECT * FROM edge_site WHERE id=1").fetchone()


def _registry(sandbox):
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


def _preview_acceptance(pending, *, expires_at="2099-01-01T00:00:00.000Z"):
    unchanged = MutationCounts(0, 0, 1)
    return TopologyAccepted(
        TopologySuccessEnvelope(
            pending.snapshot_id,
            pending.client_revision,
            7,
            TopologyMutationResult(unchanged, unchanged, unchanged),
            OmissionPreview(CONFIRMATION_ID, DIGEST, expires_at, ("camera-old",), (), ()),
        )
    )


def _terminal(pending):
    unchanged, deactivated = MutationCounts(0, 0, 1), MutationCounts(0, 0, 0, deactivated=1)
    return TopologyAccepted(
        TopologySuccessEnvelope(
            pending.snapshot_id,
            pending.client_revision,
            8,
            TopologyMutationResult(unchanged, unchanged, deactivated),
            None,
        )
    )


def _primed(sandbox, *, expires_at="2099-01-01T00:00:00.000Z"):
    registry, client = _registry(sandbox), _Client(PRINCIPAL, [], [])

    def accept(pending):
        client.confirm_outcomes.append(_terminal(pending))
        return _preview_acceptance(pending, expires_at=expires_at)

    client.put_outcomes.append(accept)
    coordinator = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    result = coordinator.trigger(force=True, now_epoch=100.0)
    assert result.status == "synced" and client.confirmations == []
    terminal = client.confirm_outcomes[0]
    assert isinstance(terminal, TopologyAccepted)
    return coordinator, registry, client, terminal


def _confirm(coordinator, **kwargs):
    return coordinator.confirm(TopologyConfirmationCommand(CONFIRMATION_ID, DIGEST, 1, 7), **kwargs)


def test_manual_confirmation_persists_terminal_result_and_advances_revision(sandbox):
    coordinator, _, client, expected = _primed(sandbox)
    result = _confirm(coordinator)
    assert result == expected and len(client.confirmations) == 1
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is True
    assert _state(sandbox).load().server_revision == 8


def test_exact_confirmation_replay_returns_terminal_result_without_second_upstream_call(sandbox):
    coordinator, registry, client, expected = _primed(sandbox)
    first = _confirm(coordinator)
    restarted = TopologyRetryCoordinator(
        CameraRegistryStore(sandbox.database, sandbox.authority),
        _state(sandbox),
        lambda: client,
    )
    replay = _confirm(restarted)
    assert first == replay == expected and len(client.confirmations) == 1
    assert registry.topology_snapshot().registry_version == 3


def test_expired_confirmation_has_zero_upstream_calls_or_local_mutation(sandbox):
    coordinator, _, client, _ = _primed(sandbox, expires_at="2000-01-01T00:00:00.000Z")
    before = _state(sandbox).load()
    result = _confirm(coordinator)
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 410
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is False
    assert _state(sandbox).load() == before


def test_changed_registry_rejects_confirmation_without_mutation(sandbox):
    coordinator, registry, client, _ = _primed(sandbox)
    before = _state(sandbox).load()
    registry.update("local-1", {"label": "Changed"})
    result = _confirm(coordinator)
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 409
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is False
    assert _state(sandbox).load() == before


@pytest.mark.parametrize(("client_revision", "server_revision"), [(2, 7), (1, 8)])
def test_stale_request_revision_rejects_without_upstream_call(
    sandbox, client_revision, server_revision
):
    coordinator, _, client, _ = _primed(sandbox)
    result = coordinator.confirm(
        TopologyConfirmationCommand(CONFIRMATION_ID, DIGEST, client_revision, server_revision)
    )
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 409
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is False


def test_changed_digest_rejects_without_upstream_call_or_local_mutation(sandbox):
    coordinator, _, client, _ = _primed(sandbox)
    before = _state(sandbox).load()
    result = coordinator.confirm(TopologyConfirmationCommand(CONFIRMATION_ID, "b" * 64, 1, 7))
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 409
    assert client.confirmations == [] and _state(sandbox).load() == before
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is False


def test_changed_generation_rejects_without_upstream_call(sandbox):
    coordinator, registry, original, _ = _primed(sandbox)
    client = _Client(MachinePrincipal(PRINCIPAL.edge_installation_id, 4), [], [])
    changed = TopologyRetryCoordinator(registry, _state(sandbox), lambda: client)
    result = _confirm(changed)
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 409
    assert original.confirmations == client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is False


def test_changed_local_server_revision_fails_confirmation_cas(sandbox):
    coordinator, _, client, _ = _primed(sandbox)
    sandbox.admin.execute("UPDATE edge_site SET topology_server_revision=9 WHERE id=1")
    result = _confirm(coordinator)
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 409
    assert client.confirmations == []
    preview = coordinator.preview()
    assert preview is not None and preview.confirmed is False
    assert _state(sandbox).load().server_revision == 9


def test_confirmation_network_precedes_completion_transaction_and_one_hook(
    sandbox, monkeypatch, postgres_audit_runtime
):
    coordinator, _, client, expected = _primed(sandbox)
    confirm, transact = client.confirm, sandbox.database.transact
    trace = []

    def network(snapshot_id, command):
        assert trace == []
        trace.append("network")
        return confirm(snapshot_id, command)

    def transaction(callback):
        assert trace == ["network"]
        trace.append("transaction")
        result = transact(callback)
        trace.append("complete-owned-return")
        return result

    publish = postgres_audit_runtime.publish_committed

    def publication(token):
        assert trace == ["network", "transaction", "hook", "complete-owned-return"]
        assert sandbox.admin.execute(
            "SELECT count(*) FROM audit_events WHERE action='topology.confirm'"
        ).fetchone() == (1,)
        trace.append("publication")
        return publish(token)

    monkeypatch.setattr(client, "confirm", network)
    monkeypatch.setattr(sandbox.database, "transact", transaction)
    monkeypatch.setattr(postgres_audit_runtime, "publish_committed", publication)
    result = _confirm(
        coordinator, audit=_audit(postgres_audit_runtime, lambda c: trace.append("hook"))
    )
    assert result == expected and trace == [
        "network",
        "transaction",
        "hook",
        "complete-owned-return",
        "publication",
    ]
    assert (
        _confirm(
            coordinator, audit=_audit(postgres_audit_runtime, lambda c: trace.append("replay-hook"))
        )
        == expected
    )
    assert len(client.confirmations) == 1 and "replay-hook" not in trace
    assert sandbox.admin.execute(
        "SELECT count(*) FROM audit_events WHERE action='topology.confirm'"
    ).fetchone() == (1,)


@pytest.mark.parametrize(
    "assignment",
    [
        "registry_version=registry_version+1",
        "topology_server_revision=9",
        "topology_client_revision=2",
        "enrollment_generation=4",
    ],
)
def test_post_network_stale_state_rejects_cas_with_zero_hooks(
    sandbox, monkeypatch, assignment, postgres_audit_runtime
):
    coordinator, _, client, _ = _primed(sandbox)
    confirm, hooks, after_network = client.confirm, [], []

    def network(snapshot_id, command):
        result = confirm(snapshot_id, command)
        sandbox.admin.execute("UPDATE edge_site SET " + assignment + " WHERE id=1")
        after_network.append(_row(sandbox))
        return result

    monkeypatch.setattr(client, "confirm", network)
    result = _confirm(
        coordinator, audit=_audit(postgres_audit_runtime, lambda c: hooks.append(True))
    )
    assert isinstance(result, TopologyConfirmationRejected) and result.status_code == 409
    assert len(client.confirmations) == 1 and not hooks
    assert _row(sandbox) == after_network[0]


@pytest.mark.parametrize("cancelled", [False, True])
def test_completion_hook_failure_does_not_hide_network_or_commit_local_result(
    sandbox, cancelled, postgres_audit_runtime
):
    coordinator, _, client, _ = _primed(sandbox)
    before, calls = _row(sandbox), []
    error = BaseException("cancelled") if cancelled else ValueError("audit rejected")

    def fail(connection):
        assert connection.execute(
            "SELECT topology_confirmation_confirmed FROM edge_site"
        ).fetchone() == (1,)
        calls.append(True)
        raise error

    with pytest.raises(type(error)) as caught:
        _confirm(coordinator, audit=_audit(postgres_audit_runtime, fail))
    assert caught.value is error and calls == [True]
    assert len(client.confirmations) == 1 and _row(sandbox) == before


@pytest.mark.parametrize("kind", ["retryable", "paused", "mismatched"])
def test_nonterminal_upstream_outcomes_never_call_completion_hook(
    sandbox, kind, postgres_audit_runtime
):
    from backend.app.features.cameras.edge_topology_sync_state import TopologyPauseReason

    coordinator, _, client, accepted = _primed(sandbox)
    outcome = {
        "retryable": TopologyRetryable("timeout"),
        "paused": TopologyPaused(TopologyPauseReason.FORBIDDEN, 403),
        "mismatched": TopologyAccepted(replace(accepted.response, snapshot_id="other")),
    }[kind]
    client.confirm_outcomes[:] = [outcome]
    before, hooks = _row(sandbox), []
    result = _confirm(
        coordinator, audit=_audit(postgres_audit_runtime, lambda c: hooks.append(True))
    )
    assert isinstance(result, TopologyRetryable if kind != "paused" else TopologyPaused)
    assert len(client.confirmations) == 1 and not hooks and _row(sandbox) == before


def test_unknown_completion_does_not_replay_the_network_operation(
    sandbox, monkeypatch, postgres_audit_runtime
):
    coordinator, _, client, _ = _primed(sandbox)
    transact, error, writes = sandbox.database.transact, CommitOutcomeUnknown(), []

    def unknown(callback):
        transact(callback)
        writes.append(True)
        raise error

    monkeypatch.setattr(sandbox.database, "transact", unknown)
    with pytest.raises(CommitOutcomeUnknown) as caught:
        _confirm(coordinator, audit=_audit(postgres_audit_runtime, lambda c: None))
    assert caught.value is error and writes == [True] and len(client.confirmations) == 1
    assert coordinator.preview().confirmed
    assert postgres_audit_runtime.snapshot().indeterminate and not postgres_audit_runtime._pending


@pytest.mark.parametrize("stage", ["before_network", "after_network"])
def test_confirmation_rechecks_admission_without_claiming_network_rollback(
    sandbox, postgres_audit_runtime, monkeypatch, stage
):
    coordinator, _, client, _ = _primed(sandbox)
    before, hooks = _row(sandbox), []
    audit = _audit(postgres_audit_runtime, lambda c: hooks.append(True))
    if stage == "before_network":
        original = coordinator._confirmation._pending_is_stale

        def stale(preview):
            result = original(preview)
            postgres_audit_runtime.stop()
            return result

        monkeypatch.setattr(coordinator._confirmation, "_pending_is_stale", stale)
    else:
        original = client.confirm

        def network(*args):
            result = original(*args)
            postgres_audit_runtime.stop()
            return result

        monkeypatch.setattr(client, "confirm", network)
    with pytest.raises(AuditRuntimeUnavailable):
        _confirm(coordinator, audit=audit)
    assert len(client.confirmations) == (0 if stage == "before_network" else 1)
    assert _row(sandbox) == before and not hooks and not postgres_audit_runtime._pending


def test_confirmation_owner_mismatch_refuses_before_upstream_effect(
    sandbox, postgres_audit_runtime
):
    coordinator, _, client, _ = _primed(sandbox)
    coordinator._confirmation._store = TopologyConfirmationStore(
        sandbox.database, replace(sandbox.authority, generation=2)
    )
    before, hooks = _row(sandbox), []
    with pytest.raises(ValueError, match="share database and authority"):
        _confirm(coordinator, audit=_audit(postgres_audit_runtime, lambda c: hooks.append(True)))
    assert not client.confirmations and not hooks and _row(sandbox) == before
