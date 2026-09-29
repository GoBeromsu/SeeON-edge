"""Relay route app on the real PostgreSQL product sandbox, plus row oracles.

The app is ``postgres_api_app`` (production composition root, no lifespan) with
the relay token set and one registered camera. The oracles read the committed
rows through the sandbox admin connection, never through the stores under test.
Register next to the sandbox plugin: ``pytest_plugins = ("tests_support.postgres_sandbox",)``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from tests_support.postgres_api_app import postgres_api_app
from tests_support.postgres_sandbox import ProductSandbox

RELAY_TOKEN = "relay-token"
RELAY_HEADERS = {"X-Edge-Relay-Token": RELAY_TOKEN}


def relay_postgres_app(
    sandbox: ProductSandbox,
    audit_runtime: PostgresAuditRuntime,
    *,
    client: Any = None,
    camera_id: str | None = "camera-1",
    backend_camera_id: str | None = "camera-1",
    space_id: str | None = None,
    rtsp_url: str = "rtsp://example/camera-1",
    label: str | None = None,
) -> FastAPI:
    """Build the relay app; ``camera_id=None`` leaves the registry empty."""
    app = postgres_api_app(sandbox, audit_runtime)
    app.state.edge_relay_token = RELAY_TOKEN
    if camera_id is not None:
        app.state.camera_registry.create(
            camera_id=camera_id,
            label=label or camera_id,
            rtsp_url=rtsp_url,
            space_id=space_id,
            status="online",
            backend_camera_id=backend_camera_id,
        )
    if client is not None:
        app.state.backend_ingest_client = client
    return app


def incident_rows(sandbox: ProductSandbox) -> list[tuple[Any, ...]]:
    return sandbox.admin.execute(
        "SELECT edge_event_id, camera_id, facility_id, event_type, probability, detected_at "
        "FROM incidents ORDER BY edge_event_id"
    ).fetchall()


def outbox_rows(sandbox: ProductSandbox) -> list[tuple[Any, ...]]:
    return sandbox.admin.execute(
        "SELECT edge_event_id, state, backend_camera_id, attempt_count "
        "FROM event_outbox ORDER BY edge_event_id"
    ).fetchall()


def row_counts(sandbox: ProductSandbox) -> tuple[int, int]:
    """(incidents, event_outbox) row counts."""
    row = sandbox.admin.execute(
        "SELECT (SELECT count(*) FROM incidents), (SELECT count(*) FROM event_outbox)"
    ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


def artifact_count(sandbox: ProductSandbox) -> int:
    row = sandbox.admin.execute("SELECT count(*) FROM artifacts").fetchone()
    assert row is not None
    return int(row[0])


def camera_revision(sandbox: ProductSandbox, camera_id: str) -> tuple[int, int]:
    """(revision, never_connected) of one registry row."""
    row = sandbox.admin.execute(
        "SELECT revision, never_connected FROM cameras WHERE camera_id = %s", (camera_id,)
    ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


def fail_inserts(sandbox: ProductSandbox, table: str, *, sqlstate: str = "08006") -> None:
    """Install a BEFORE INSERT trigger that raises ``sqlstate`` on ``table``.

    ``08006`` (connection_failure) is classified as unavailable by the product
    pool, so the admission transaction rolls back and nothing is acknowledged.
    """
    function = f"relay_fault_{table}"
    sandbox.admin.execute(
        f"CREATE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $$ "
        f"BEGIN RAISE EXCEPTION 'injected {table} fault' USING ERRCODE = '{sqlstate}'; END $$"
    )
    sandbox.admin.execute(
        f"CREATE TRIGGER {function} BEFORE INSERT ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {function}()"
    )


def clear_insert_fault(sandbox: ProductSandbox, table: str) -> None:
    function = f"relay_fault_{table}"
    sandbox.admin.execute(f"DROP TRIGGER {function} ON {table}")
    sandbox.admin.execute(f"DROP FUNCTION {function}()")


__all__ = [
    "RELAY_HEADERS",
    "RELAY_TOKEN",
    "artifact_count",
    "camera_revision",
    "clear_insert_fault",
    "fail_inserts",
    "incident_rows",
    "outbox_rows",
    "relay_postgres_app",
    "row_counts",
]
