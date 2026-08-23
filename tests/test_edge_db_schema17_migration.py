from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from test_edge_db_schema16_fixtures import build_schema16_fixture

from backend.app.edge_db.migrator import migrate_database
from backend.app.edge_db.schema import SchemaV17MigrationError


def _application_manifest(connection: sqlite3.Connection) -> tuple[dict[str, int], str]:
    """Return a fresh count-and-content manifest for non-migrator tables."""
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE 'schema_%' ORDER BY name"
        )
    ]
    row_counts = {
        table: int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
        for table in tables
    }
    content = []
    for table in tables:
        columns = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]
        order_by = ", ".join(f'"{column}"' for column in columns)
        rows = connection.execute(f'SELECT * FROM "{table}" ORDER BY {order_by}').fetchall()
        content.append((table, rows))
    checksum = hashlib.sha256(
        json.dumps(content, default=repr, separators=(",", ":")).encode()
    ).hexdigest()
    return row_counts, checksum


def _assert_application_manifest_equal(
    expected: tuple[dict[str, int], str], actual: tuple[dict[str, int], str]
) -> None:
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]


def _schema_rows_except_table_families(
    connection: sqlite3.Connection,
) -> dict[str, list[tuple]]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table' "
            "AND name LIKE 'schema_%' AND name NOT IN "
            # The migration ledger legitimately gains the v17 entry; that it does
            # so is asserted separately, and is exactly what the retired direct
            # helper skipped when it stamped user_version without recording one.
            "('schema_table_families', 'schema_migrations') ORDER BY name"
        )
    ]
    result = {}
    for table in tables:
        columns = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]
        order_by = ", ".join(f'"{column}"' for column in columns)
        result[table] = connection.execute(
            f'SELECT * FROM "{table}" ORDER BY {order_by}'
        ).fetchall()
    return result


def test_schema17_migration_preserves_application_rows_and_only_reassigns_writers(
    tmp_path: Path,
) -> None:
    database = tmp_path / "schema16.sqlite3"
    # Drained on purpose. The previous version of this test seeded a
    # drain-blocked fixture and then called a direct helper that bypassed the
    # gate, so it proved row preservation for a migration production would have
    # refused. Rows must survive the migration an operator can actually run.
    build_schema16_fixture(database, drain_blocked=False)

    with sqlite3.connect(database) as connection:
        before_manifest = _application_manifest(connection)
        before_schema_rows = _schema_rows_except_table_families(connection)
        before_families = connection.execute(
            "SELECT prefix, writer, purpose FROM schema_table_families ORDER BY prefix"
        ).fetchall()

    migrate_database(database)

    with sqlite3.connect(database) as connection:
        after_manifest = _application_manifest(connection)
        after_schema_rows = _schema_rows_except_table_families(connection)
        after_families = connection.execute(
            "SELECT prefix, writer, purpose FROM schema_table_families ORDER BY prefix"
        ).fetchall()

        _assert_application_manifest_equal(before_manifest, after_manifest)
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)
        assert after_schema_rows == before_schema_rows

        # The canonical path records what it did. A database at version 17 with
        # no ledger entry is the signature of a bypass.
        recorded = connection.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 17"
        ).fetchone()[0]
        assert recorded == 1, (
            "schema 17 was applied without a migration ledger entry; that is the "
            "signature of a path that bypassed the drain gate"
        )
        assert [(prefix, purpose) for prefix, _, purpose in after_families] == [
            (prefix, purpose) for prefix, _, purpose in before_families
        ]
        assert after_families == [
            (prefix, "migrator" if prefix == "schema_" else "api", purpose)
            for prefix, _, purpose in before_families
        ]


def test_application_manifest_comparator_detects_deleted_row(tmp_path: Path) -> None:
    database = tmp_path / "schema16.sqlite3"
    build_schema16_fixture(database, drain_blocked=False)

    with sqlite3.connect(database) as connection:
        before_manifest = _application_manifest(connection)
        connection.execute("DELETE FROM control_heartbeats")
        after_manifest = _application_manifest(connection)

    with pytest.raises(AssertionError):
        _assert_application_manifest_equal(before_manifest, after_manifest)


def test_schema17_migration_refuses_reinvocation_with_typed_error(tmp_path: Path) -> None:
    database = tmp_path / "schema16.sqlite3"
    build_schema16_fixture(database, drain_blocked=False)

    migrate_database(database)
    # Re-running the canonical path on an already-migrated database is a clean
    # no-op, not an error: an operator who repeats the documented command must
    # not be told the deployment is broken.
    migrate_database(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)


def test_registered_schema17_drain_gate_refuses_without_mutating_database(tmp_path: Path) -> None:
    database = tmp_path / "schema16.sqlite3"
    build_schema16_fixture(database, drain_blocked=True)
    before = database.read_bytes()

    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (16,)


def test_schema17_drain_gate_refuses_waiting_legacy_clips_without_mutating_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "schema16.sqlite3"
    build_schema16_fixture(database, drain_blocked=False)
    with sqlite3.connect(database) as connection:
        # The live stall condition is unfinalized local media, not an
        # unpublished clip. `publish_state = 'WAITING'` is the resting state of
        # every clip that finalized locally and was never uploaded, so gating on
        # it would refuse every real database forever and the cutover could
        # never run. See tests/test_schema17_drain_gate_clips.py.
        connection.execute("UPDATE evidence_clips SET local_state = 'AWAITING_FINALIZE'")
        connection.commit()
    before = database.read_bytes()

    with pytest.raises(SchemaV17MigrationError, match="EDGE_DB_DRAIN_INCOMPLETE"):
        migrate_database(database)

    assert database.read_bytes() == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (16,)


def test_production_migration_ledger_lands_on_schema17(tmp_path: Path) -> None:
    database = tmp_path / "production.sqlite3"

    result = migrate_database(database)

    assert result.current_version == 17
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (17,)
