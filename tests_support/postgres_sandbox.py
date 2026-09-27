"""Opt-in real product PostgreSQL fixture; never an ambient production database."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.authority import AuthorityToken
from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase

_DDL = Path(__file__).resolve().parents[1] / "backend/app/edge_db"


@dataclass(frozen=True, slots=True)
class ProductSandbox:
    admin: psycopg.Connection = field(repr=False)
    database: PostgresDatabase
    authority: AuthorityToken
    schema: str
    dsn: str = field(repr=False)


@pytest.fixture
def postgres_product_sandbox() -> Iterator[ProductSandbox]:
    """Own one quoted namespace and bounded pool per test, with real product DDL.

    Import this fixture from tests_support.postgres_sandbox or register that
    module in pytest_plugins. Missing opt-in skips, which is not qualification.
    The admin connection is independent of the pool for visibility/fault checks.
    """
    dsn = os.environ.get("SEEON_TEST_POSTGRES_DSN")
    if dsn is None:
        pytest.skip("requires an isolated SEEON_TEST_POSTGRES_DSN; not qualification evidence")
    if not dsn.strip() or "\x00" in dsn:
        pytest.fail("SEEON_TEST_POSTGRES_DSN must be nonblank without NUL bytes", pytrace=False)
    try:
        admin = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except (psycopg.Error, OSError, ValueError, TypeError):
        admin = None
    if admin is None:
        # Outside the except block: do not chain a libpq error containing the DSN.
        pytest.fail("isolated PostgreSQL test database is unreachable", pytrace=False)

    schema = "seeon_product_test_" + uuid4().hex
    database = None
    try:
        admin.execute("SET statement_timeout TO 5000")
        admin.execute("SET lock_timeout TO 3000")
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            # Product triggers capture this path with SET search_path FROM CURRENT.
            admin.execute(
                sql.SQL("SET search_path TO {}, pg_catalog, pg_temp").format(sql.Identifier(schema))
            )
            authority = AuthorityToken(generation=1, writer_token=uuid4())
            with admin.transaction():
                admin.execute((_DDL / "postgres_product.sql").read_text(), prepare=False)
                admin.execute((_DDL / "postgres_delivery.sql").read_text(), prepare=False)
                admin.execute(
                    "INSERT INTO deployment_authority "
                    "(singleton,generation,writer_token,accepting,egress_enabled) "
                    "VALUES (1,%s,%s,true,true)",
                    (authority.generation, authority.writer_token),
                )
                admin.execute(
                    "INSERT INTO edge_site (id,updated_at) VALUES (1,%s)",
                    (datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),),
                )
            database = PostgresDatabase(
                dsn,
                schema,
                PoolBudget(
                    max_connections=4,
                    max_waiting=8,
                    acquire_timeout_sec=1.0,
                    statement_timeout_ms=5000,
                    lock_timeout_ms=3000,
                    startup_timeout_sec=5.0,
                ),
            )
            database.start()
            yield ProductSandbox(
                admin=admin,
                database=database,
                authority=authority,
                schema=schema,
                dsn=dsn,
            )
        finally:
            if database is not None:
                database.close()
            admin.rollback()
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    finally:
        admin.close()


__all__ = ["ProductSandbox", "postgres_product_sandbox"]
