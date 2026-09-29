"""Declared optional-owner publication over real PostgreSQL, not HTTP admission."""

import traceback
from contextlib import contextmanager
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.app.edge_db.postgres import CommitOutcomeUnknown, PostgresDatabase
from backend.app.features.audit.catalog import AuditAction, empty_detail
from backend.app.features.audit.postgres_runtime import (
    AuditMutation,
    AuditRuntimeUnavailable,
    InvalidAuditPublication,
    PostgresAuditRuntime,
)
from backend.app.features.audit.postgres_store import PostgresAuditStore
from backend.app.features.audit.store import AuditEvent
from backend.app.features.cameras.edge_topology_sync_state import EdgeTopologySyncStateStore
from backend.app.features.cameras.store import CameraRegistryStore
from backend.app.features.connection.topology_retry_coordinator import TopologyRetryCoordinator
from backend.app.main import create_app, no_lifespan
from backend.app.shared.dashboard_auth import (
    DASHBOARD_SESSION_COOKIE,
    DashboardSessionStore,
    PlaintextDashboardCredentials,
)

pytest_plugins = ("tests_support.postgres_sandbox",)
_SECRET = "private-credential-in-callback"


class Cancelled(BaseException):
    pass


@pytest.fixture
def setup(postgres_product_sandbox):
    sandbox = postgres_product_sandbox
    clock = [100.0]
    runtime = PostgresAuditRuntime(
        PostgresAuditStore(sandbox.database, sandbox.authority),
        maximum_snapshot_age_sec=10,
        clock=lambda: clock[0],
    )
    assert runtime.verify_once() and runtime.start_session_once()
    registry = CameraRegistryStore(sandbox.database, sandbox.authority)
    registry.create(
        camera_id="camera-1",
        label="before",
        rtsp_url="rtsp://camera-1/live",
        space_id=None,
        status="unknown",
    )
    return sandbox, runtime, registry, clock


def _event():
    return AuditEvent(
        occurred_at="2026-09-27T00:00:00Z",
        actor_id="operator-1",
        action=AuditAction.CAMERA_UPDATE,
        target_id="camera-1",
        detail=empty_detail(AuditAction.CAMERA_UPDATE),
    )


def _history(sandbox):
    return sandbox.admin.execute("SELECT * FROM audit_events ORDER BY audit_id").fetchall()


def _label(sandbox):
    return sandbox.admin.execute("SELECT label FROM cameras WHERE camera_id='camera-1'").fetchone()


def _update(audit, registry, *, camera_id="camera-1"):
    return audit.apply(
        registry,
        lambda append: registry.update(camera_id, {"label": "after"}, after_write=append),
        expects_audit=lambda result: result is not None,
    )


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_optional_absence_never_constructs_event_or_changes_audit(setup, operation):
    sandbox, runtime, registry, _ = setup
    before, factories = _history(sandbox), []

    def event():
        factories.append(True)
        return _event()

    audit = AuditMutation(runtime, event)
    if operation == "update":
        assert _update(audit, registry, camera_id="absent") is None
    else:
        assert (
            audit.apply(
                registry,
                lambda append: registry.delete("absent", after_write=append),
                expects_audit=bool,
            )
            is False
        )
    assert _history(sandbox) == before and _label(sandbox) == ("before",)
    assert not factories and not runtime._pending and runtime.snapshot().ready


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_optional_success_constructs_exactly_one_event_and_publication(setup, operation):
    sandbox, runtime, registry, _ = setup
    factories, before = [], len(_history(sandbox))

    def event():
        factories.append(True)
        assert _label(sandbox) == ("before",)
        return _event()

    audit = AuditMutation(runtime, event)
    if operation == "update":
        assert _update(audit, registry) is not None
        assert _label(sandbox) == ("after",)
    else:
        assert (
            audit.apply(
                registry,
                lambda append: registry.delete("camera-1", after_write=append),
                expects_audit=bool,
            )
            is True
        )
        assert _label(sandbox) is None
    assert factories == [True] and len(_history(sandbox)) == before + 1
    assert not runtime._pending and runtime.snapshot().ready


def test_lazy_event_and_publication_surround_complete_optional_owner(setup, monkeypatch):
    sandbox, runtime, registry, _ = setup
    before, trace = _history(sandbox), []
    acquire, update = sandbox.database._pool.connection, registry.update
    append, publish = runtime.append_borrowed, runtime.publish_committed

    def event():
        assert _label(sandbox) == ("before",) and _history(sandbox) == before
        trace.append("event")
        return _event()

    def borrowed(connection, value):
        token = append(connection, value)
        assert _history(sandbox) == before
        trace.append("hook")
        return token

    @contextmanager
    def exit_pool(*args, **kwargs):
        with acquire(*args, **kwargs) as connection:
            yield connection
        trace.append("pool-exit")

    def owned(*args, **kwargs):
        result = update(*args, **kwargs)
        assert trace == ["event", "hook", "pool-exit"]
        trace.append("owner-return")
        return result

    def published(token):
        assert trace == ["event", "hook", "pool-exit", "owner-return"]
        assert _label(sandbox) == ("after",) and len(_history(sandbox)) == len(before) + 1
        trace.append("publication")
        return publish(token)

    monkeypatch.setattr(sandbox.database._pool, "connection", exit_pool)
    monkeypatch.setattr(registry, "update", owned)
    monkeypatch.setattr(runtime, "append_borrowed", borrowed)
    monkeypatch.setattr(runtime, "publish_committed", published)
    assert _update(AuditMutation(runtime, event), registry) is not None
    assert trace == ["event", "hook", "pool-exit", "owner-return", "publication"]


@pytest.mark.parametrize("cancelled", [False, True])
def test_factory_failure_rolls_back_and_redacts_ordinary_error(setup, cancelled):
    sandbox, runtime, registry, _ = setup
    before = _history(sandbox)
    error = Cancelled(_SECRET) if cancelled else ValueError(_SECRET)

    def event():
        raise error

    with pytest.raises(Cancelled if cancelled else AuditRuntimeUnavailable) as caught:
        _update(AuditMutation(runtime, event), registry)
    if cancelled:
        assert caught.value is error
    else:
        assert _SECRET not in "".join(traceback.format_exception(caught.value))
        assert str(caught.value) == "mutation audit event is invalid"
    assert _label(sandbox) == ("before",) and _history(sandbox) == before
    assert not runtime._pending and not runtime.snapshot().ready


@pytest.mark.parametrize("cancelled", [False, True])
def test_optional_owner_exception_rolls_back_and_preserves_identity(setup, cancelled):
    sandbox, runtime, registry, _ = setup
    before, error = _history(sandbox), Cancelled("cancel") if cancelled else OSError("owner")

    def write(append):
        def fail(connection):
            append(connection)
            raise error

        return registry.update("camera-1", {"label": "after"}, after_write=fail)

    with pytest.raises(type(error)) as caught:
        AuditMutation(runtime, _event).apply(registry, write, expects_audit=bool)
    assert caught.value is error
    assert _label(sandbox) == ("before",) and _history(sandbox) == before
    assert not runtime._pending and not runtime.snapshot().ready


def test_repeated_callback_never_constructs_second_event(setup):
    sandbox, runtime, registry, _ = setup
    before, factories = _history(sandbox), []

    def event():
        factories.append(True)
        return _event()

    def write(append):
        def twice(connection):
            append(connection)
            append(connection)

        return registry.update("camera-1", {"label": "after"}, after_write=twice)

    with pytest.raises(AuditRuntimeUnavailable, match="was repeated"):
        AuditMutation(runtime, event).apply(registry, write, expects_audit=bool)
    assert factories == [True] and _history(sandbox) == before
    assert _label(sandbox) == ("before",) and not runtime._pending


@pytest.mark.parametrize("expectation", [False, None, 0, 1])
def test_committed_owner_contract_failure_never_claims_rollback(setup, expectation):
    sandbox, runtime, registry, _ = setup
    before = len(_history(sandbox))
    with pytest.raises(AuditRuntimeUnavailable, match="unexpected|expectation is invalid"):
        AuditMutation(runtime, _event).apply(
            registry,
            lambda append: registry.update("camera-1", {"label": "after"}, after_write=append),
            expects_audit=lambda result: expectation,
        )
    assert _label(sandbox) == ("after",) and len(_history(sandbox)) == before + 1
    assert not runtime._pending and not runtime.snapshot().ready


@pytest.mark.parametrize("kind", ["ordinary", "cancelled", "unknown"])
@pytest.mark.parametrize("secondary", [False, True])
def test_classifier_failure_preserves_outcome_and_unrelated_token(
    setup, monkeypatch, caplog, kind, secondary
):
    sandbox, runtime, registry, _ = setup
    unrelated = sandbox.database.transact(lambda c: runtime.append_borrowed(c, _event()))
    before = len(_history(sandbox))
    error = {
        "ordinary": ValueError(_SECRET),
        "cancelled": Cancelled(_SECRET),
        "unknown": CommitOutcomeUnknown(),
    }[kind]
    publish, failures = runtime.publish_failed, []

    def classified(result):
        assert _label(sandbox) == ("after",)
        raise error

    def failed(token, original):
        assert token is not unrelated
        failures.append(original)
        publish(token, original)
        if secondary:
            raise InvalidAuditPublication(_SECRET)

    monkeypatch.setattr(runtime, "publish_failed", failed)
    with pytest.raises(type(error)) as caught:
        AuditMutation(runtime, _event).apply(
            registry,
            lambda append: registry.update("camera-1", {"label": "after"}, after_write=append),
            expects_audit=classified,
        )
    assert caught.value is error and failures == [error]
    assert _label(sandbox) == ("after",) and len(_history(sandbox)) == before + 1
    assert runtime._pending == {unrelated} and not runtime.snapshot().ready
    assert runtime.snapshot().indeterminate is (kind == "unknown")
    assert _SECRET not in caplog.text
    if secondary:
        assert caplog.records[-1].exc_info is None
    publish(unrelated, OSError("test cleanup"))


@pytest.mark.parametrize("kind", ["invalid", "raising"])
def test_zero_callback_classifier_failure_marks_unavailable(setup, kind):
    sandbox, runtime, registry, _ = setup
    before, error = _history(sandbox), ValueError("classifier")

    def classify(result) -> None:
        if kind == "raising":
            raise error

    with pytest.raises(ValueError if kind == "raising" else AuditRuntimeUnavailable) as caught:
        AuditMutation(runtime, _event).apply(
            registry,
            lambda append: registry.update("absent", {}, after_write=append),
            expects_audit=classify,
        )
    if kind == "raising":
        assert caught.value is error
    assert _history(sandbox) == before and not runtime._pending
    assert not runtime.snapshot().ready


@pytest.mark.parametrize("state", ["expired", "stopped"])
def test_capability_rechecks_before_network_and_actual_owner(setup, state):
    sandbox, runtime, registry, clock = setup
    before, entered, factories = _history(sandbox), [], []

    def event():
        factories.append(True)
        return _event()

    capability = AuditMutation(runtime, event)
    assert not factories and _history(sandbox) == before
    if state == "expired":
        clock[0] += 10
    else:
        runtime.stop()
    with pytest.raises(AuditRuntimeUnavailable):
        AuditMutation(runtime, event)
        entered.append("network")
    with pytest.raises(AuditRuntimeUnavailable):
        capability.require_admission()
        entered.append("network")
    with pytest.raises(AuditRuntimeUnavailable):
        capability.apply(registry, lambda append: entered.append("owner"))
    assert not entered and not factories and _history(sandbox) == before
    assert _label(sandbox) == ("before",) and not runtime._pending


@pytest.mark.parametrize("mismatch", ["database", "authority"])
def test_capability_checks_owner_before_external_effects(setup, mismatch):
    sandbox, runtime, _, _ = setup
    other = PostgresDatabase(sandbox.dsn, sandbox.schema, sandbox.database._budget)
    owner = CameraRegistryStore(
        other if mismatch == "database" else sandbox.database,
        replace(sandbox.authority, generation=2) if mismatch == "authority" else sandbox.authority,
    )
    audit, effects = AuditMutation(runtime, _event), []
    try:
        with pytest.raises(ValueError, match="share database and authority"):
            audit.require_admission(owner)
            effects.append("network")
        assert not effects and runtime.snapshot().ready and not runtime._pending
        assert _label(sandbox) == ("before",)
    finally:
        other.close(timeout_sec=3.0)


@pytest.mark.parametrize("state", ["missing", "ineligible", "pool_closed", "schema_missing"])
def test_confirmation_http_refuses_before_egress_without_audit_or_database(setup, state):
    sandbox, runtime, registry, _ = setup
    before, effects = _history(sandbox), []
    app = create_app(lifespan=no_lifespan)
    sessions = DashboardSessionStore(PlaintextDashboardCredentials("operator", "test-password"))
    token = sessions.authenticate("operator", "test-password")
    assert token is not None
    app.state.dashboard_sessions = sessions

    def client_provider():
        effects.append(True)
        pytest.fail("refused confirmation reached its external client")

    app.state.topology_retry_coordinator = TopologyRetryCoordinator(
        registry, EdgeTopologySyncStateStore(sandbox.database, sandbox.authority), client_provider
    )
    if state != "missing":
        app.state.audit_runtime = runtime
    if state == "ineligible":
        runtime.record_failure(OSError("test admission loss"))
    if state == "pool_closed":
        sandbox.database.close(timeout_sec=3.0)
    if state == "schema_missing":
        sandbox.admin.execute("ALTER TABLE edge_site RENAME TO detached_site")
    with TestClient(app) as client:
        payload = {
            "confirmation_id": "confirmation",
            "digest": "a" * 64,
            "client_revision": 1,
            "server_revision": 0,
        }
        path = "/api/v1/connection/topology-preview/confirm"
        assert client.post(path, json=payload).status_code == 401
        client.cookies.set(DASHBOARD_SESSION_COOKIE, token)
        response = client.post(path, json=payload)
    assert (response.status_code, response.content) == (503, b"")
    assert not effects and _history(sandbox) == before
    assert not hasattr(app.state, "audit_store") and not hasattr(app.state, "audit_readiness")
    if state != "missing":
        assert not runtime.snapshot().ready
