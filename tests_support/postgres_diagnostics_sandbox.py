"""Real diagnostics PostgreSQL fixtures on the isolated test DSN; never an ambient database.

Execution records live in their own ``<product schema>_diagnostics`` schema, applied
from ``postgres_diagnostics.sql``. ``postgres_diagnostics_sandbox`` owns a fresh
schema and a started pool per test; ``create_diagnostics_schema`` prepares the
schema the real lifespan opens next to a product sandbox, and
``diagnostics_database_for`` serves stacks that build the store themselves.
Register with ``pytest_plugins = ("tests_support.postgres_diagnostics_sandbox",)``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase
from backend.app.features.diagnostics.postgres_database import DIAGNOSTICS_SCHEMA_SUFFIX
from tests_support.postgres_sandbox import ProductSandbox

_DDL = Path(__file__).resolve().parents[1] / "backend/app/edge_db/postgres_diagnostics.sql"
_BUDGET = PoolBudget(
    max_connections=4,
    max_waiting=8,
    acquire_timeout_sec=1.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=3000,
    startup_timeout_sec=5.0,
)


@dataclass(frozen=True, slots=True)
class DiagnosticsSandbox:
    admin: psycopg.Connection = field(repr=False)
    database: PostgresDatabase
    schema: str
    dsn: str = field(repr=False)


def _test_dsn() -> str:
    dsn = os.environ.get("SEEON_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.fail(
            "SEEON_TEST_POSTGRES_DSN is required; point it at an isolated test database",
            pytrace=False,
        )
    if not dsn.strip() or "\x00" in dsn:
        pytest.fail("SEEON_TEST_POSTGRES_DSN must be nonblank without NUL bytes", pytrace=False)
    return dsn


@contextmanager
def _admin_connection(dsn: str) -> Iterator[psycopg.Connection]:
    try:
        admin = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except (psycopg.Error, OSError, ValueError, TypeError):
        admin = None
    if admin is None:
        # Outside the except block: do not chain a libpq error containing the DSN.
        pytest.fail("isolated PostgreSQL test database is unreachable", pytrace=False)
    try:
        admin.execute("SET statement_timeout TO 5000")
        admin.execute("SET lock_timeout TO 3000")
        yield admin
    finally:
        admin.close()


def create_diagnostics_schema(admin: psycopg.Connection, base_schema: str) -> str:
    """Create ``<base_schema>_diagnostics`` with its DDL once; return its name.

    The DDL runs with the diagnostics schema first on a transaction-local
    search_path, so an admin session pinned to the product schema keeps its path.
    """
    schema = base_schema + DIAGNOSTICS_SCHEMA_SUFFIX
    with admin.transaction():
        admin.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        applied = admin.execute(
            "SELECT to_regclass(format('%%I.execution_batches', %s::text))", (schema,)
        ).fetchone()
        if applied is None or applied[0] is None:
            admin.execute(
                sql.SQL("SET LOCAL search_path TO {}, pg_catalog, pg_temp").format(
                    sql.Identifier(schema)
                )
            )
            admin.execute(_DDL.read_text(), prepare=False)
    return schema


def drop_diagnostics_schema(admin: psycopg.Connection, schema: str) -> None:
    admin.rollback()
    admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def postgres_diagnostics_sandbox() -> Iterator[DiagnosticsSandbox]:
    """Own one diagnostics schema and bounded pool per test, with the real DDL.

    The admin connection is independent of the pool and its search_path is the
    diagnostics schema, so oracles read the same unqualified tables the store writes.
    """
    dsn = _test_dsn()
    with _admin_connection(dsn) as admin:
        schema = create_diagnostics_schema(admin, "seeon_diag_test_" + uuid4().hex)
        database = None
        try:
            admin.execute(
                sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(sql.Identifier(schema))
            )
            database = PostgresDatabase(dsn, schema, _BUDGET)
            database.start()
            yield DiagnosticsSandbox(admin=admin, database=database, schema=schema, dsn=dsn)
        finally:
            if database is not None:
                database.close(timeout_sec=3.0)
            drop_diagnostics_schema(admin, schema)


@pytest.fixture
def postgres_lifespan_diagnostics_schema(postgres_product_sandbox: ProductSandbox) -> Iterator[str]:
    """Provision the schema the real lifespan opens beside the product sandbox.

    Migration provision owns this in a deployment; the fixture drops it afterwards
    because the product sandbox only drops its own schema.
    """
    admin = postgres_product_sandbox.admin
    schema = create_diagnostics_schema(admin, postgres_product_sandbox.schema)
    try:
        yield schema
    finally:
        drop_diagnostics_schema(admin, schema)


@contextmanager
def diagnostics_database_for(base_schema: str) -> Iterator[PostgresDatabase]:
    """Serve a started pool on ``<base_schema>_diagnostics``; close and drop it on exit."""
    dsn = _test_dsn()
    with _admin_connection(dsn) as admin:
        schema = create_diagnostics_schema(admin, base_schema)
        database = None
        try:
            database = PostgresDatabase(dsn, schema, _BUDGET)
            database.start()
            yield database
        finally:
            if database is not None:
                database.close(timeout_sec=3.0)
            drop_diagnostics_schema(admin, schema)


__all__ = [
    "DiagnosticsSandbox",
    "create_diagnostics_schema",
    "diagnostics_database_for",
    "drop_diagnostics_schema",
    "postgres_diagnostics_sandbox",
    "postgres_lifespan_diagnostics_schema",
]
