"""The schema-19 SQLite source the migration tool reads, and how it is classified."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from backend.app.edge_db.migration.compatibility import (
    COMPATIBILITY_MATRIX,
    CURRENT_SCHEMA_RANGE,
    SCHEMA_18_IDENTITY,
    SCHEMA_19_IDENTITY,
    CompatibilityDisposition,
    MigrationRequiredError,
    NewerSchemaError,
    SchemaCompatibility,
    SchemaLedgerError,
    classify_schema,
    schema19_identity_checksum,
    verify_runtime_schema,
)
from tests_support.sqlite_source import create_schema19_source


def test_schema19_identity_checksum_matches_execution_record_ddl() -> None:
    assert (
        19,
        "strict_sixteen_table_application_schema",
        schema19_identity_checksum(),
    ) == SCHEMA_19_IDENTITY
    assert SCHEMA_18_IDENTITY == (
        18,
        "strict_ten_table_application_schema",
        "d43dbc02e395e3df5117f7dc96814a87299f949cac7195cc72fb950d60964c9c",
    )


def test_forward_backward_compatibility_matrix_is_explicit() -> None:
    assert COMPATIBILITY_MATRIX == (
        ("database_version < minimum", CompatibilityDisposition.MIGRATION_REQUIRED),
        ("minimum <= database_version <= maximum", CompatibilityDisposition.COMPATIBLE),
        ("database_version > maximum", CompatibilityDisposition.NEWER_SCHEMA),
    )
    assert SchemaCompatibility(minimum=19, maximum=19) == CURRENT_SCHEMA_RANGE
    supported = SchemaCompatibility(minimum=3, maximum=4)
    assert classify_schema(2, supported) is CompatibilityDisposition.MIGRATION_REQUIRED
    assert classify_schema(3, supported) is CompatibilityDisposition.COMPATIBLE
    assert classify_schema(4, supported) is CompatibilityDisposition.COMPATIBLE
    assert classify_schema(5, supported) is CompatibilityDisposition.NEWER_SCHEMA


def test_runtime_refuses_absent_and_out_of_range_schemas(tmp_path: Path) -> None:
    # The migration fence runs verify_runtime_schema on every source it opens.
    with (
        closing(sqlite3.connect(tmp_path / "empty.sqlite3")) as empty,
        pytest.raises(MigrationRequiredError) as refused,
    ):
        verify_runtime_schema(empty)
    assert (refused.value.found, refused.value.minimum) == (0, 19)

    source = create_schema19_source(tmp_path / "edge-state" / "edge.sqlite3")
    with closing(sqlite3.connect(source)) as connection:
        assert verify_runtime_schema(connection) == 19
        with pytest.raises(MigrationRequiredError):
            verify_runtime_schema(connection, SchemaCompatibility(minimum=20, maximum=20))
        with pytest.raises(NewerSchemaError):
            verify_runtime_schema(connection, SchemaCompatibility(minimum=0, maximum=0))


def test_a_forged_schema19_ledger_row_is_refused(tmp_path: Path) -> None:
    source = create_schema19_source(tmp_path / "edge-state" / "edge.sqlite3")
    with closing(sqlite3.connect(source)) as connection:
        with connection:
            connection.execute(
                "UPDATE schema_migrations SET name = 'forged', checksum = ? WHERE version = 19",
                ("f" * 64,),
            )
        with pytest.raises(
            SchemaLedgerError, match="^applied schema ledger does not end at schema 19$"
        ):
            verify_runtime_schema(connection)
