"""The API-owned audit runtime on the PostgreSQL root and the readiness it publishes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.features.audit.startup import (
    close_audit_session,
    configure_audit_readiness,
    start_audit_verification,
    verify_audit_runtime,
)
from backend.app.main import create_app, no_lifespan
from backend.app.postgres_root import PostgresRoot, install_postgres_stores
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

pytest_plugins = ("tests_support.postgres_sandbox",)

_READY = {"ready": True, "status": "ready"}
_DEGRADED = {"ready": False, "status": "degraded", "reason": "audit unavailable"}
# Long enough that only an explicit verification or the close wakes the verifier.
_IDLE_INTERVAL_SEC = 3600.0


def _root_app(sandbox: ProductSandbox) -> FastAPI:
    app = create_app(lifespan=no_lifespan)
    root = PostgresRoot(sandbox.database, sandbox.authority)
    install_postgres_stores(app, root)
    app.state.postgres_root = root
    return app


def _reject_audit_inserts(sandbox: ProductSandbox) -> None:
    sandbox.admin.execute(
        "CREATE OR REPLACE FUNCTION reject_audit_test() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'injected audit failure'; END $$"
    )
    sandbox.admin.execute(
        "CREATE TRIGGER reject_audit_test BEFORE INSERT ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_audit_test()"
    )


def _session_rows(sandbox: ProductSandbox) -> list[tuple[str, str]]:
    return sandbox.admin.execute(
        "SELECT action, target_id FROM audit_events WHERE action IN (%s, %s, %s) ORDER BY audit_id",
        (
            AuditAction.AUDIT_SESSION_START.value,
            AuditAction.AUDIT_SESSION_CLOSE.value,
            AuditAction.RECOVERY_FENCE.value,
        ),
    ).fetchall()


def _readiness(client: TestClient) -> tuple[int, object]:
    response = client.get("/health/ready")
    return response.status_code, response.json()


def _wait_until(predicate: Callable[[], bool], *, timeout_sec: float, what: str) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail(f"timed out waiting for {what}")


def test_missing_postgres_root_refuses_audit_wiring() -> None:
    # Given: an app with neither a PostgreSQL root nor an injected audit runtime.
    app = FastAPI()

    # When / Then: configuration and the boot publish both refuse instead of degrading.
    with pytest.raises(RuntimeError):
        configure_audit_readiness(app, clock=lambda: 0.0)
    with pytest.raises(RuntimeError):
        start_audit_verification(app)


def test_boot_publishes_ready_and_owned_close_writes_one_session_close(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: an owned runtime verified and started on the PostgreSQL root.
    sandbox = postgres_product_sandbox
    app = _root_app(sandbox)
    assert configure_audit_readiness(app, clock=lambda: 0.0) is True
    client = TestClient(app)
    try:
        # When: a verification runs before the boot publish.
        assert verify_audit_runtime(app, app.state.audit_runtime) is True

        # Then: only the boot publish turns the booting probe ready.
        assert _readiness(client) == (503, {"ready": False, "reason": "booting"})
        start_audit_verification(app, verify_interval_sec=_IDLE_INTERVAL_SEC)
        assert _readiness(client) == (200, _READY)
    finally:
        closed = close_audit_session(app)

    # Then: the healthy owned session closes exactly once in PostgreSQL.
    assert closed is True
    rows = _session_rows(sandbox)
    assert [action for action, _ in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]
    assert rows[0][1] == rows[1][1]


def test_verifier_tick_heals_a_degraded_boot_once_audit_is_restored(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: boot runs while audit_events carries an extra, unverifiable trigger.
    sandbox = postgres_product_sandbox
    _reject_audit_inserts(sandbox)
    app = _root_app(sandbox)
    assert configure_audit_readiness(app, clock=lambda: 0.0) is False
    client = TestClient(app)
    try:
        start_audit_verification(app, verify_interval_sec=0.01)
        assert _readiness(client) == (503, _DEGRADED)

        # When: the audit contract is restored while the owned verifier keeps ticking.
        sandbox.admin.execute("DROP TRIGGER reject_audit_test ON audit_events")

        # Then: the tick re-verifies, finishes the deferred session start, and heals.
        _wait_until(
            lambda: _readiness(client) == (200, _READY),
            timeout_sec=5.0,
            what="the audit verifier to heal readiness",
        )
    finally:
        closed = close_audit_session(app)

    # Then: with no append to fence the boot degradation, the session is left unclean.
    assert closed is False
    assert [action for action, _ in _session_rows(sandbox)] == [
        AuditAction.AUDIT_SESSION_START,
    ]


def test_postgres_failure_degrades_readiness_until_verification_heals_it(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: a ready owned runtime and a governed write whose own UPDATE PostgreSQL rejects.
    sandbox = postgres_product_sandbox
    app = _root_app(sandbox)
    assert configure_audit_readiness(app, clock=lambda: 0.0) is True
    try:
        with TestClient(app) as client:
            start_audit_verification(app, verify_interval_sec=_IDLE_INTERVAL_SEC)
            login = client.post(
                "/api/v1/auth/session", json={"username": "admin", "password": "admin"}
            )
            assert login.status_code == 204
            sandbox.admin.execute(
                "CREATE OR REPLACE FUNCTION reject_site_test() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected site failure'; END $$"
            )
            sandbox.admin.execute(
                "CREATE TRIGGER reject_site_test BEFORE UPDATE ON edge_site "
                "FOR EACH ROW EXECUTE FUNCTION reject_site_test()"
            )

            # When: the route fails on PostgreSQL.
            response = client.put(
                "/api/v1/runtime-settings",
                json={"clip_export_enabled": True, "expected_version": 0},
            )

            # Then: the response is an empty 503 and the probe reports audit degradation.
            assert response.status_code == 503
            assert response.content == b""
            assert _readiness(client) == (503, _DEGRADED)

            # When: the failure is gone and one verification runs.
            sandbox.admin.execute("DROP TRIGGER reject_site_test ON edge_site")
            healed = verify_audit_runtime(app, app.state.audit_runtime)

            # Then: the same runtime is ready again.
            assert healed is True
            assert _readiness(client) == (200, _READY)
    finally:
        close_audit_session(app)


def test_degraded_close_leaves_an_unclean_marker_that_the_next_owner_fences(
    postgres_product_sandbox: ProductSandbox,
) -> None:
    # Given: an owned session that degrades before shutdown.
    sandbox = postgres_product_sandbox
    first = _root_app(sandbox)
    assert configure_audit_readiness(first, clock=lambda: 0.0) is True
    _reject_audit_inserts(sandbox)
    assert verify_audit_runtime(first, first.state.audit_runtime) is False

    # When: the degraded owner shuts down and a new owner boots on the restored audit.
    closed = close_audit_session(first)
    sandbox.admin.execute("DROP TRIGGER reject_audit_test ON audit_events")
    second = _root_app(sandbox)
    try:
        assert configure_audit_readiness(second, clock=lambda: 0.0) is True
    finally:
        assert close_audit_session(second) is True

    # Then: no close was written for the degraded session; the next start fenced it once.
    assert closed is False
    rows = _session_rows(sandbox)
    assert [action for action, _ in rows] == [
        AuditAction.AUDIT_SESSION_START,
        AuditAction.RECOVERY_FENCE,
        AuditAction.AUDIT_SESSION_START,
        AuditAction.AUDIT_SESSION_CLOSE,
    ]
    assert rows[1][1] == rows[0][1]
    assert rows[2][1] == rows[3][1] != rows[0][1]
    (fence_detail,) = sandbox.admin.execute(
        "SELECT detail_json FROM audit_events WHERE action=%s",
        (AuditAction.RECOVERY_FENCE.value,),
    ).fetchone()
    assert json.loads(fence_detail)["failure_code"] == "unclean_restart"


def test_injected_runtime_is_published_but_never_stopped_or_closed(
    postgres_product_sandbox: ProductSandbox, postgres_audit_runtime: PostgresAuditRuntime
) -> None:
    # Given: a test-owned runtime injected into an app that also has a PostgreSQL root.
    sandbox = postgres_product_sandbox
    app = postgres_api_app(sandbox, postgres_audit_runtime)
    app.state.postgres_root = PostgresRoot(sandbox.database, sandbox.authority)
    before = _session_rows(sandbox)

    # When: the startup owner configures, publishes, and closes around it.
    configured = configure_audit_readiness(app, clock=lambda: 0.0)
    start_audit_verification(app, verify_interval_sec=0.01)
    status = _readiness(TestClient(app))
    closed = close_audit_session(app)

    # Then: readiness reflects it, but it is neither replaced, stopped, nor closed.
    assert configured is True
    assert status == (200, _READY)
    assert closed is False
    assert app.state.audit_runtime is postgres_audit_runtime
    snapshot = postgres_audit_runtime.snapshot()
    assert snapshot.stopping is False
    assert snapshot.eligible_to_attempt is True
    assert _session_rows(sandbox) == before
