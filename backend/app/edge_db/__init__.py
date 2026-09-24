"""DDL-free public surface for the single local edge database."""

from backend.app.edge_db.compatibility import CURRENT_SCHEMA_RANGE, SchemaCompatibility
from backend.app.edge_db.connection import (
    BusyPolicy,
    RuntimeActor,
    best_effort_zero_wait_write,
    open_runtime_database,
    write_transaction,
)
from backend.app.edge_db.diagnostics_connection import open_diagnostics_database
from backend.app.edge_db.paths import (
    DIAGNOSTICS_DATABASE_FILENAME,
    EDGE_DATABASE_PATH,
    EDGE_STATE_DIRECTORY,
)

__all__ = [
    "CURRENT_SCHEMA_RANGE",
    "DIAGNOSTICS_DATABASE_FILENAME",
    "EDGE_DATABASE_PATH",
    "EDGE_STATE_DIRECTORY",
    "BusyPolicy",
    "RuntimeActor",
    "SchemaCompatibility",
    "best_effort_zero_wait_write",
    "open_diagnostics_database",
    "open_runtime_database",
    "write_transaction",
]
