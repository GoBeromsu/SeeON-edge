"""Schema 18 compact ten-table DDL plus the schema-19 create statements."""

from __future__ import annotations

from typing import Final

from backend.app.edge_db.compact_schema_ddl import COMPACT_SCHEMA_CREATE_STATEMENTS
from backend.app.edge_db.execution_records_ddl import (
    EXECUTION_RECORD_CREATE_STATEMENTS,
    EXECUTION_RECORD_TABLES,
)

# The persistent `schema_migrations` ledger table. Its CREATE text and the three
# provenance columns below are byte-for-byte what every deployed schema-18
# database carries (the columns were added with ALTER TABLE, so the stored
# `sqlite_schema.sql` text differs from an inline CREATE). Keeping the same
# statements means a freshly created schema-19 SQLite source and a deployed one
# compile to the identical structural manifest.
SCHEMA_MIGRATIONS_LEDGER_TABLE_SQL: Final = """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY CHECK (version > 0),
            name TEXT NOT NULL UNIQUE,
            checksum TEXT NOT NULL CHECK (length(checksum) = 64),
            applied_at TEXT NOT NULL
        ) STRICT
        """

SCHEMA_MIGRATIONS_PROVENANCE_STATEMENTS: Final = (
    """
    ALTER TABLE schema_migrations ADD COLUMN source_schema_version INTEGER
        CHECK (source_schema_version IS NULL OR source_schema_version > 0)
    """,
    """
    ALTER TABLE schema_migrations ADD COLUMN source_db_sha256 TEXT
        CHECK (
            source_db_sha256 IS NULL
            OR (
                length(source_db_sha256) = 64
                AND source_db_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        )
    """,
    """
    ALTER TABLE schema_migrations ADD COLUMN reconciliation_sha256 TEXT
        CHECK (
            reconciliation_sha256 IS NULL
            OR (
                length(reconciliation_sha256) = 64
                AND reconciliation_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        )
    """,
)

COMPACT_APPLICATION_TABLES: Final = frozenset(
    {
        "artifacts",
        "audit_events",
        "cameras",
        "clips",
        "credentials",
        "edge_site",
        "incidents",
        "locations",
        "policies",
        "schema_migrations",
    }
)
COMPACT_API_TABLES: Final = frozenset(
    table for table in COMPACT_APPLICATION_TABLES if table != "schema_migrations"
)

APPLICATION_TABLES: Final = COMPACT_APPLICATION_TABLES | EXECUTION_RECORD_TABLES
APPLICATION_API_TABLES: Final = frozenset(
    table for table in APPLICATION_TABLES if table != "schema_migrations"
)

# Every DDL statement that turns an empty database into schema 18, in order.
SCHEMA_18_STATEMENTS: Final = (
    SCHEMA_MIGRATIONS_LEDGER_TABLE_SQL,
    *SCHEMA_MIGRATIONS_PROVENANCE_STATEMENTS,
    *COMPACT_SCHEMA_CREATE_STATEMENTS,
)

# Schema 19 is schema 18 plus the six execution-record tables and their indexes.
SCHEMA_19_STATEMENTS: Final = (
    *SCHEMA_18_STATEMENTS,
    *EXECUTION_RECORD_CREATE_STATEMENTS,
)

__all__ = [
    "APPLICATION_API_TABLES",
    "APPLICATION_TABLES",
    "COMPACT_API_TABLES",
    "COMPACT_APPLICATION_TABLES",
    "SCHEMA_18_STATEMENTS",
    "SCHEMA_19_STATEMENTS",
    "SCHEMA_MIGRATIONS_LEDGER_TABLE_SQL",
    "SCHEMA_MIGRATIONS_PROVENANCE_STATEMENTS",
]
