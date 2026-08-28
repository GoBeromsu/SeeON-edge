"""Schema-18 compatibility guard shared by every DDL-free runtime connection."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from backend.app.edge_db.ownership import COMPACT_APPLICATION_TABLES
from backend.app.edge_db.schema18_manifest import (
    compile_schema18_manifest,
    read_schema18_manifest,
)
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION


class EdgeDatabaseError(RuntimeError):
    """Base failure for the edge database foundation."""


@dataclass(slots=True)
class MigrationRequiredError(EdgeDatabaseError):
    """The database is absent or below schema 18; only a fresh bootstrap can create it."""

    found: int
    minimum: int

    def __str__(self) -> str:
        if self.found == 0:
            return f"edge database is not bootstrapped; schema {self.minimum} is required"
        return (
            f"edge database schema {self.found} is below required {self.minimum} "
            "and no migration path exists"
        )


@dataclass(slots=True)
class NewerSchemaError(EdgeDatabaseError):
    found: int
    maximum: int

    def __str__(self) -> str:
        return f"edge database schema {self.found} is newer than supported {self.maximum}"


class SchemaLedgerError(EdgeDatabaseError):
    """The version marker and the on-disk schema disagree with the compiled contract."""


@dataclass(frozen=True, slots=True)
class SchemaCompatibility:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("schema compatibility must be a non-negative inclusive range")


class CompatibilityDisposition(StrEnum):
    MIGRATION_REQUIRED = "migration_required"
    COMPATIBLE = "compatible"
    NEWER_SCHEMA = "newer_schema"


MigrationIdentity = tuple[int, str, str]

# The frozen ledger identity of schema 18. The checksum is the historical
# release value every deployed schema-18 database already records; it is a
# constant, not a hash of the current DDL, so it stays stable across the
# retirement of the v1-v18 migration ledger.
SCHEMA_18_IDENTITY: Final[MigrationIdentity] = (
    EDGE_DATABASE_SCHEMA_VERSION,
    "strict_ten_table_application_schema",
    "d43dbc02e395e3df5117f7dc96814a87299f949cac7195cc72fb950d60964c9c",
)

# Explicit rolling-version matrix: exactly one schema is supported.
CURRENT_SCHEMA_RANGE: Final = SchemaCompatibility(
    minimum=EDGE_DATABASE_SCHEMA_VERSION,
    maximum=EDGE_DATABASE_SCHEMA_VERSION,
)
COMPATIBILITY_MATRIX: Final = (
    ("database_version < minimum", CompatibilityDisposition.MIGRATION_REQUIRED),
    ("minimum <= database_version <= maximum", CompatibilityDisposition.COMPATIBLE),
    ("database_version > maximum", CompatibilityDisposition.NEWER_SCHEMA),
)


def classify_schema(
    version: int,
    compatibility: SchemaCompatibility = CURRENT_SCHEMA_RANGE,
) -> CompatibilityDisposition:
    if version < compatibility.minimum:
        return CompatibilityDisposition.MIGRATION_REQUIRED
    if version > compatibility.maximum:
        return CompatibilityDisposition.NEWER_SCHEMA
    return CompatibilityDisposition.COMPATIBLE


def verify_runtime_schema(
    connection: sqlite3.Connection,
    compatibility: SchemaCompatibility = CURRENT_SCHEMA_RANGE,
) -> int:
    """Verify the version marker, the schema-18 ledger row, and the structural contract."""
    row = connection.execute("PRAGMA user_version").fetchone()
    version = 0 if row is None else int(row[0])
    disposition = classify_schema(version, compatibility)
    if disposition is CompatibilityDisposition.MIGRATION_REQUIRED:
        raise MigrationRequiredError(found=version, minimum=compatibility.minimum)
    if disposition is CompatibilityDisposition.NEWER_SCHEMA:
        raise NewerSchemaError(found=version, maximum=compatibility.maximum)

    try:
        ledger = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
    except sqlite3.Error as error:
        raise SchemaLedgerError("edge database schema ledger is missing or unreadable") from error
    # A freshly bootstrapped database records only the schema-18 row; a database
    # that reached schema 18 through the retired migration ledger also carries
    # the historical rows 1-17. Both are schema 18: the newest row must be the
    # frozen identity and nothing may sit beyond it.
    if not ledger or tuple(ledger[-1]) != SCHEMA_18_IDENTITY:
        raise SchemaLedgerError("applied schema ledger does not end at schema 18")
    if [int(entry[0]) for entry in ledger] != sorted({int(entry[0]) for entry in ledger}):
        raise SchemaLedgerError("applied schema ledger is not a strict version sequence")

    _verify_compact_application_tables(connection)
    return version


def _verify_compact_application_tables(connection: sqlite3.Connection) -> None:
    """Require the exact schema-18 table allowlist and structural contract."""
    try:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    except sqlite3.Error as error:
        raise SchemaLedgerError("edge database application table set is unreadable") from error
    tables = {str(name) for name, _sql in rows}
    if tables != COMPACT_APPLICATION_TABLES:
        raise SchemaLedgerError("edge database application table set is invalid")
    if any(sql is None or " STRICT" not in sql.upper() for _name, sql in rows):
        raise SchemaLedgerError("edge database application tables must be STRICT")
    try:
        actual = read_schema18_manifest(connection)
    except sqlite3.Error as error:
        raise SchemaLedgerError("schema 18 contract is unreadable") from error
    delta = actual.diff(compile_schema18_manifest())
    if delta:
        raise SchemaLedgerError("schema 18 contract is invalid")


__all__ = [
    "COMPATIBILITY_MATRIX",
    "CURRENT_SCHEMA_RANGE",
    "SCHEMA_18_IDENTITY",
    "CompatibilityDisposition",
    "EdgeDatabaseError",
    "MigrationIdentity",
    "MigrationRequiredError",
    "NewerSchemaError",
    "SchemaCompatibility",
    "SchemaLedgerError",
    "classify_schema",
    "verify_runtime_schema",
]
