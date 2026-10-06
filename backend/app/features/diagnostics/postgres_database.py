"""Diagnostics PostgreSQL owner: its own schema and pool beside the product root.

Execution records are observability, not product authority. They live in
``<product schema>_diagnostics`` on the same cluster so retention pressure and
diagnostic reads never share the product pool. A configuration error refuses
startup (ADR-0002); an unreachable server only degrades the diagnostics API to
503. The DSN is never logged, echoed in an error, or chained into a traceback.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from backend.app.edge_db.postgres import PoolBudget, PostgresDatabase, PostgresError

# Same deployment variables the product root reads; the diagnostics owner does
# not import the root so the two pools stay independently composable.
API_POSTGRES_DSN_FILE_ENV = "API_POSTGRES_DSN_FILE"
API_POSTGRES_SCHEMA_ENV = "API_POSTGRES_SCHEMA"
DEFAULT_POSTGRES_SCHEMA = "seeon_edge"
DIAGNOSTICS_SCHEMA_SUFFIX = "_diagnostics"
DIAGNOSTICS_POOL_BUDGET = PoolBudget(
    max_connections=2,
    max_waiting=8,
    acquire_timeout_sec=1.0,
    statement_timeout_ms=5000,
    lock_timeout_ms=3000,
    startup_timeout_sec=10.0,
)

logger = logging.getLogger(__name__)


class DiagnosticsDatabaseConfigError(RuntimeError):
    """Startup refusal whose text never carries connection secrets."""


def diagnostics_schema(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    base = env.get(API_POSTGRES_SCHEMA_ENV, "").strip() or DEFAULT_POSTGRES_SCHEMA
    return base + DIAGNOSTICS_SCHEMA_SUFFIX


def _read_conninfo(environ: Mapping[str, str]) -> str:
    raw_path = environ.get(API_POSTGRES_DSN_FILE_ENV, "").strip()
    if not raw_path:
        raise DiagnosticsDatabaseConfigError(
            f"PostgreSQL DSN file is not configured; set {API_POSTGRES_DSN_FILE_ENV}"
        )
    try:
        text = Path(raw_path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        text = None
    if text is None:
        raise DiagnosticsDatabaseConfigError(
            f"PostgreSQL DSN file named by {API_POSTGRES_DSN_FILE_ENV} is unreadable"
        )
    if not text:
        raise DiagnosticsDatabaseConfigError(
            f"PostgreSQL DSN file named by {API_POSTGRES_DSN_FILE_ENV} is empty"
        )
    return text


def open_diagnostics_database(
    environ: Mapping[str, str] | None = None, *, budget: PoolBudget = DIAGNOSTICS_POOL_BUDGET
) -> PostgresDatabase:
    """Return the diagnostics database, started when the server answers in budget.

    A database that fails to start is still returned: it refuses every
    transaction with ``PostgresUnavailable``, which the diagnostics API maps
    to 503 while the rest of the product keeps serving.
    """
    env = os.environ if environ is None else environ
    conninfo = _read_conninfo(env)
    schema = diagnostics_schema(env)
    try:
        database: PostgresDatabase | None = PostgresDatabase(conninfo, schema, budget)
    except (TypeError, ValueError, PostgresError):
        database = None
    del conninfo
    if database is None:
        raise DiagnosticsDatabaseConfigError("diagnostics PostgreSQL settings are invalid")
    try:
        database.start()
        started = True
    except PostgresError:
        started = False
    if not started:
        logger.warning(
            "diagnostics PostgreSQL schema %s is unavailable; execution records answer 503",
            schema,
        )
    return database


__all__ = [
    "API_POSTGRES_DSN_FILE_ENV",
    "API_POSTGRES_SCHEMA_ENV",
    "DEFAULT_POSTGRES_SCHEMA",
    "DIAGNOSTICS_POOL_BUDGET",
    "DIAGNOSTICS_SCHEMA_SUFFIX",
    "DiagnosticsDatabaseConfigError",
    "diagnostics_schema",
    "open_diagnostics_database",
]
