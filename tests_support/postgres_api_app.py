"""No-lifespan app whose governed stores and audit runtime share one sandbox database.

Route tests that skip the real lifespan still go through the production
composition root: ``install_postgres_stores`` wires every governed store from the
sandbox pool and the verified audit runtime appends to the same database. Register
next to the sandbox plugin: ``pytest_plugins = ("tests_support.postgres_sandbox",)``.
"""

from __future__ import annotations

from fastapi import FastAPI

from backend.app.features.audit.postgres_runtime import PostgresAuditRuntime
from backend.app.main import create_app, no_lifespan
from backend.app.postgres_root import PostgresRoot, install_postgres_stores
from tests_support.postgres_sandbox import ProductSandbox


def postgres_api_app(sandbox: ProductSandbox, audit_runtime: PostgresAuditRuntime) -> FastAPI:
    """Build a no-lifespan app on the sandbox root with its audit runtime installed."""
    app = create_app(lifespan=no_lifespan)
    install_postgres_stores(app, PostgresRoot(sandbox.database, sandbox.authority))
    app.state.audit_runtime = audit_runtime
    return app
