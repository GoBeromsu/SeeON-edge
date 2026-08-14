"""DDL-free public surface for the single local edge database."""

from shared.edge_db.compatibility import CURRENT_SCHEMA_RANGE, SchemaCompatibility
from shared.edge_db.connection import (
    BusyPolicy,
    RuntimeActor,
    best_effort_zero_wait_write,
    open_runtime_database,
    write_transaction,
)
from shared.edge_db.paths import EDGE_DATABASE_PATH, EDGE_STATE_DIRECTORY

__all__ = [
    "CURRENT_SCHEMA_RANGE",
    "EDGE_DATABASE_PATH",
    "EDGE_STATE_DIRECTORY",
    "BusyPolicy",
    "RuntimeActor",
    "SchemaCompatibility",
    "best_effort_zero_wait_write",
    "open_runtime_database",
    "write_transaction",
]
