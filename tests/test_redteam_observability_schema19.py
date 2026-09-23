"""Adversarial schema-19 bootstrap: create / extend-with-backup / verify / refuse."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import UnsupportedSchemaError, bootstrap_database
from backend.app.edge_db.compact_schema import SCHEMA_18_STATEMENTS
from backend.app.edge_db.compatibility import (
    SCHEMA_18_IDENTITY,
    NewerSchemaError,
    SchemaLedgerError,
)
from backend.app.edge_db.ownership import APPLICATION_TABLES
from backend.app.edge_db.paths import schema18_backup_path

_TS = "2026-08-24T00:00:00Z"


def _raw_execute(path: Path, sql: str, parameters: tuple[object, ...] = ()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(sql, parameters)
        connection.commit()
    finally:
        connection.close()


def _build_schema18(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in SCHEMA_18_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO schema_migrations (version, name, checksum, applied_at)
            VALUES (?, ?, ?, '2026-01-01T00:00:00.000Z')
            """,
            SCHEMA_18_IDENTITY,
        )
        connection.execute("PRAGMA user_version = 18")
        connection.commit()
    finally:
        connection.close()


def _seed_location(path: Path) -> None:
    _raw_execute(
        path,
        "INSERT INTO locations "
        "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
        "created_at, updated_at) VALUES "
        "('floor-1', 'FLOOR', NULL, NULL, 'Floor 1', 0, ?, ?)",
        (_TS, _TS),
    )


def _compact_rows(path: Path) -> dict[str, list[tuple[object, ...]]]:
    connection = sqlite3.connect(path)
    try:
        rows: dict[str, list[tuple[object, ...]]] = {}
        for table in sorted(APPLICATION_TABLES):
            if table.startswith("execution_") or table == "schema_migrations":
                continue
            rows[table] = connection.execute(f"SELECT * FROM {table}").fetchall()
        return rows
    finally:
        connection.close()


def test_e1_schema18_extension_preserves_existing_rows_byte_identical(tmp_path: Path) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    _build_schema18(database)
    _seed_location(database)
    before = _compact_rows(database)
    result = bootstrap_database(database)
    assert result.created is False
    assert result.extended is True
    assert result.schema_version == 19
    assert _compact_rows(database) == before
    backup = schema18_backup_path(database)
    assert backup.is_file()
    backup_connection = sqlite3.connect(backup)
    live_connection = sqlite3.connect(database)
    try:
        assert backup_connection.execute("SELECT * FROM locations").fetchall() == (
            live_connection.execute("SELECT * FROM locations").fetchall()
        )
    finally:
        backup_connection.close()
        live_connection.close()


def test_e2_refuses_when_schema18_backup_already_exists(tmp_path: Path) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    _build_schema18(database)
    backup = schema18_backup_path(database)
    backup.write_bytes(b"occupied")
    with pytest.raises(SchemaLedgerError, match="schema-18 backup already exists"):
        bootstrap_database(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (18,)
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert "execution_records" not in tables
    finally:
        connection.close()


def test_e3_second_extension_on_schema19_is_noop_verify(tmp_path: Path) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    _build_schema18(database)
    first = bootstrap_database(database)
    assert first.extended is True
    second = bootstrap_database(database)
    assert (second.created, second.extended) == (False, False)
    assert second.schema_version == 19


def test_e4_user_version_20_is_refused(tmp_path: Path) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    bootstrap_database(database)
    _raw_execute(database, "PRAGMA user_version = 20")
    with pytest.raises(NewerSchemaError):
        bootstrap_database(database)


def test_e5_schema_17_is_refused(tmp_path: Path) -> None:
    database = tmp_path / "edge-state" / "edge.sqlite3"
    database.parent.mkdir(parents=True)
    _raw_execute(database, "CREATE TABLE evidence_events (id INTEGER PRIMARY KEY)")
    _raw_execute(database, "PRAGMA user_version = 17")
    with pytest.raises(UnsupportedSchemaError):
        bootstrap_database(database)
