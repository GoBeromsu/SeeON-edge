"""Table writer ownership for the one local edge database."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from backend.app.edge_db.compact_schema import COMPACT_API_TABLES, COMPACT_APPLICATION_TABLES


class Writer(StrEnum):
    API = "api"
    MIGRATOR = "migrator"


SCHEMA_LEDGER_TABLE: Final = "schema_migrations"


def writer_for_table(table: str) -> Writer | None:
    """Return the declared writer for *table*; ``None`` for anything outside schema 18."""
    if table in COMPACT_API_TABLES:
        return Writer.API
    if table == SCHEMA_LEDGER_TABLE:
        return Writer.MIGRATOR
    return None


__all__ = [
    "COMPACT_API_TABLES",
    "COMPACT_APPLICATION_TABLES",
    "SCHEMA_LEDGER_TABLE",
    "Writer",
    "writer_for_table",
]
