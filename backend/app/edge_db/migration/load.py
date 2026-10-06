"""Load a verified SQLite snapshot into an empty product schema in one transaction."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import psycopg
from psycopg import sql

from backend.app.edge_db.migration.compatibility import verify_runtime_schema
from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.mapping import (
    COPY_TYPES,
    EXPECTED_TARGET_TABLES,
    TableMapping,
    build_mappings,
    require_identifier,
)
from backend.app.edge_db.migration.provision import SCHEMA_NAME, SCHEMA_VERSION, schema_checksum
from backend.app.edge_db.migration.reconcile import (
    compare_tables,
    fingerprint,
    source_identity_floors,
    sqlite_rows,
)
from backend.app.edge_db.migration.snapshot import open_snapshot, snapshot_sha256
from backend.app.edge_db.postgres import PostgresDatabase

# Tables the import fills; provisioning owns the ledger and the fenced authority row.
_PROVISIONED: Final = frozenset({"schema_migrations", "deployment_authority"})
_PYTHON_TYPES: Final = {"bigint": int, "text": str, "bytea": bytes, "double precision": float}


@dataclass(frozen=True, slots=True)
class ImportResult:
    rows: dict[str, int]
    source_db_sha256: str
    reconciliation_sha256: str


def import_snapshot(
    database: PostgresDatabase, *, schema: str, snapshot_path: Path
) -> ImportResult:
    """Copy every migrated table, verify it and stamp the ledger, or change nothing."""
    require_identifier(schema, "schema")
    source_sha256 = snapshot_sha256(snapshot_path)
    with closing(open_snapshot(snapshot_path)) as snapshot:
        schema_version = verify_runtime_schema(snapshot)

        def load(connection: psycopg.Connection) -> ImportResult:
            lock_all(connection, schema)
            require_unimported(connection, schema)
            mappings = build_mappings(snapshot, connection, schema)
            require_empty(connection, schema)
            rows = {
                mapping.name: _copy(connection, schema, snapshot, mapping) for mapping in mappings
            }
            _advance_identities(
                connection, schema, mappings, source_identity_floors(snapshot, mappings)
            )
            results = compare_tables(snapshot, connection, mappings, schema)
            failed = [result.name for result in results if not result.passed]
            if failed:
                raise MigrationError(f"post-load reconciliation failed: {', '.join(failed)}")
            reconciliation = fingerprint(results)
            if snapshot_sha256(snapshot_path) != source_sha256:
                raise MigrationError("snapshot changed during import")
            stamped = connection.execute(
                sql.SQL(
                    "UPDATE {} SET source_schema_version = %s, source_db_sha256 = %s, "
                    "reconciliation_sha256 = %s WHERE version = %s AND source_db_sha256 IS NULL"
                ).format(sql.Identifier(schema, "schema_migrations")),
                (schema_version, source_sha256, reconciliation, SCHEMA_VERSION),
            ).rowcount
            if stamped != 1:
                raise MigrationError("schema ledger could not be stamped")
            return ImportResult(
                rows=rows, source_db_sha256=source_sha256, reconciliation_sha256=reconciliation
            )

        return database.transact(load)


def lock_all(connection: psycopg.Connection, schema: str) -> None:
    # Every product table, so no runtime can read a half-loaded schema.
    connection.execute(
        sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(
            sql.SQL(", ").join(
                sql.Identifier(schema, table) for table in sorted(EXPECTED_TARGET_TABLES)
            )
        )
    )


def require_unimported(connection: psycopg.Connection, schema: str) -> None:
    ledger = [
        tuple(row)
        for row in connection.execute(
            sql.SQL(
                "SELECT version, name, checksum, source_schema_version, source_db_sha256, "
                "reconciliation_sha256 FROM {} ORDER BY version"
            ).format(sql.Identifier(schema, "schema_migrations"))
        ).fetchall()
    ]
    if len(ledger) == 1 and ledger[0][4] is not None:
        raise MigrationError("target already holds an imported snapshot")
    if ledger != [(SCHEMA_VERSION, SCHEMA_NAME, schema_checksum(), None, None, None)]:
        raise MigrationError("target schema ledger does not match this tool's schema")
    authority = connection.execute(
        sql.SQL("SELECT generation, accepting, egress_enabled FROM {}").format(
            sql.Identifier(schema, "deployment_authority")
        )
    ).fetchall()
    if [tuple(row) for row in authority] != [(1, False, False)]:
        raise MigrationError("target authority must be the fenced provisioning generation")


def require_empty(connection: psycopg.Connection, schema: str) -> None:
    occupied = [
        table
        for table in sorted(EXPECTED_TARGET_TABLES - _PROVISIONED)
        if connection.execute(
            sql.SQL("SELECT EXISTS (SELECT 1 FROM {})").format(sql.Identifier(schema, table))
        ).fetchone()[0]
    ]
    if occupied:
        raise MigrationError(f"target tables are not empty: {', '.join(occupied)}")


def _copy(
    connection: psycopg.Connection,
    schema: str,
    snapshot: sqlite3.Connection,
    mapping: TableMapping,
) -> int:
    target_types = [mapping.target_type(name) for name in mapping.column_names]
    kinds = [_PYTHON_TYPES[target] for target in target_types]
    statement = sql.SQL("COPY {} ({}) FROM STDIN (FORMAT BINARY)").format(
        sql.Identifier(schema, mapping.name),
        sql.SQL(", ").join(sql.Identifier(name) for name in mapping.column_names),
    )
    count = 0
    with (
        connection.cursor() as cursor,
        cursor.copy(statement) as copy,
        closing(sqlite_rows(snapshot, mapping, load_order=True)) as rows,
    ):
        copy.set_types([COPY_TYPES[target] for target in target_types])
        for row in rows:
            for name, kind, value in zip(mapping.column_names, kinds, row, strict=True):
                if value is not None and type(value) is not kind:
                    # Name the column only; the value may be resident data.
                    raise MigrationError(f"{mapping.name}.{name} holds a value of another type")
            copy.write_row(row)
            count += 1
    return count


def _advance_identities(
    connection: psycopg.Connection,
    schema: str,
    mappings: Iterable[TableMapping],
    floors: dict[str, int],
) -> None:
    for mapping in mappings:
        floor = floors.get(mapping.name, 0)
        if mapping.identity_column is None or floor <= 0:
            continue
        connection.execute(
            "SELECT pg_catalog.setval(pg_catalog.pg_get_serial_sequence(%s, %s), %s)",
            (f"{schema}.{mapping.name}", mapping.identity_column, floor),
        )


__all__ = [
    "ImportResult",
    "import_snapshot",
    "lock_all",
    "require_empty",
    "require_unimported",
]
