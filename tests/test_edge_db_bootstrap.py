"""The create-or-extend schema-19 bootstrap is the sole DDL owner of the edge database."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.edge_db.bootstrap import (
    DeploymentLockError,
    UnsupportedSchemaError,
    bootstrap_database,
    deployment_lock,
)
from backend.app.edge_db.compact_schema import SCHEMA_18_STATEMENTS
from backend.app.edge_db.compatibility import (
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
    verify_schema18_contract,
)
from backend.app.edge_db.connection import (
    BusyPolicy,
    RuntimeActor,
    open_runtime_database,
    write_transaction,
)
from backend.app.edge_db.ownership import APPLICATION_TABLES, writer_for_table
from backend.app.edge_db.paths import schema18_backup_path
from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION

_PROVENANCE_ID = "ab" * 32
_TS = "2026-08-24T00:00:00Z"


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "edge-state" / "edge.sqlite3"


def _raw_execute(path: Path, sql: str, parameters: tuple[object, ...] = ()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(sql, parameters)
        connection.commit()
    finally:
        connection.close()


def _build_schema18(path: Path, *, historical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in SCHEMA_18_STATEMENTS:
            connection.execute(statement)
        if historical:
            for version in range(1, 18):
                connection.execute(
                    "INSERT INTO schema_migrations (version, name, checksum, applied_at) "
                    "VALUES (?, ?, ?, '2026-01-01T00:00:00.000Z')",
                    (version, f"historical_{version}", "0" * 64),
                )
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


def _compact_application_rows(path: Path) -> dict[str, list[tuple[object, ...]]]:
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


def test_fresh_bootstrap_creates_schema_19_with_secure_local_files(database_path: Path) -> None:
    result = bootstrap_database(database_path)

    assert result.created is True
    assert result.extended is False
    assert result.schema_version == EDGE_DATABASE_SCHEMA_VERSION == CURRENT_SCHEMA_RANGE.maximum
    assert database_path.parent.stat().st_mode & 0o777 == 0o700
    assert database_path.stat().st_mode & 0o777 == 0o600

    connection = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (19,)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        tables = {str(name) for name, _sql in rows}
        assert tables == APPLICATION_TABLES
        assert len(tables) == 16
        assert all(" STRICT" in str(sql).upper() for _name, sql in rows)
        assert {table: writer_for_table(table) for table in tables} == {
            table: ("migrator" if table == "schema_migrations" else "api") for table in tables
        }
        assert writer_for_table("system_test_runs") is None
        assert connection.execute(
            "SELECT version, name, checksum FROM schema_migrations"
        ).fetchall() == [SCHEMA_19_IDENTITY]
        names = {path.name for path in database_path.parent.iterdir()}
        assert {"edge.sqlite3", "edge.sqlite3-wal", "edge.sqlite3-shm"} <= names
    finally:
        connection.close()


def test_bootstrap_is_idempotent_and_runtime_open_does_not_mutate_schema(
    database_path: Path,
) -> None:
    first = bootstrap_database(database_path)
    second = bootstrap_database(database_path)
    assert (first.created, first.extended) == (True, False)
    assert (second.created, second.extended) == (False, False)
    assert first.schema_version == second.schema_version == 19

    def schema() -> list[tuple[object, ...]]:
        connection = sqlite3.connect(database_path)
        try:
            return connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
        finally:
            connection.close()

    before = schema()
    runtime = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        assert runtime.total_changes == 0
    finally:
        runtime.close()
    assert schema() == before


def test_bootstrap_extends_exact_schema_18_preserving_application_rows(
    database_path: Path,
) -> None:
    _build_schema18(database_path)
    _seed_location(database_path)
    before = _compact_application_rows(database_path)

    result = bootstrap_database(database_path)

    assert result.created is False
    assert result.extended is True
    assert result.schema_version == 19
    assert _compact_application_rows(database_path) == before

    backup = schema18_backup_path(database_path)
    assert backup.is_file()
    assert backup.stat().st_mode & 0o777 == 0o600
    backup_sha256 = hashlib.sha256(backup.read_bytes()).hexdigest()

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (19,)
        ledger = connection.execute(
            "SELECT version, name, checksum, source_schema_version, source_db_sha256 "
            "FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [(row[0], row[1], row[2]) for row in ledger] == [
            SCHEMA_18_IDENTITY,
            SCHEMA_19_IDENTITY,
        ]
        assert ledger[-1][3] == 18
        assert ledger[-1][4] == backup_sha256
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == APPLICATION_TABLES
        assert len(tables) == 16
    finally:
        connection.close()

    backup_connection = sqlite3.connect(backup)
    try:
        verify_schema18_contract(backup_connection)
        assert backup_connection.execute("PRAGMA user_version").fetchone() == (18,)
        assert backup_connection.execute("SELECT location_id, name FROM locations").fetchall() == [
            ("floor-1", "Floor 1")
        ]
    finally:
        backup_connection.close()

    open_runtime_database(database_path, actor=RuntimeActor.API).close()
    second = bootstrap_database(database_path)
    assert (second.created, second.extended) == (False, False)


def test_bootstrap_extends_schema_18_that_still_carries_retired_ledger_rows(
    database_path: Path,
) -> None:
    _build_schema18(database_path, historical=True)
    result = bootstrap_database(database_path)
    assert result.extended is True
    connection = sqlite3.connect(database_path)
    try:
        versions = [
            int(row[0])
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]
        assert versions == list(range(1, 20))
        assert (
            connection.execute(
                "SELECT version, name, checksum FROM schema_migrations WHERE version = 19"
            ).fetchone()
            == SCHEMA_19_IDENTITY
        )
    finally:
        connection.close()


def test_bootstrap_refuses_extension_when_schema18_backup_already_exists(
    database_path: Path,
) -> None:
    _build_schema18(database_path)
    backup = schema18_backup_path(database_path)
    backup.write_bytes(b"occupied")

    with pytest.raises(SchemaLedgerError, match="schema-18 backup already exists"):
        bootstrap_database(database_path)

    connection = sqlite3.connect(database_path)
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


@pytest.mark.parametrize("version", [1, 16, 17])
def test_bootstrap_refuses_any_older_schema_instead_of_migrating(
    database_path: Path, version: int
) -> None:
    database_path.parent.mkdir(parents=True)
    _raw_execute(database_path, "CREATE TABLE evidence_events (id INTEGER PRIMARY KEY)")
    _raw_execute(database_path, f"PRAGMA user_version = {version}")

    with pytest.raises(UnsupportedSchemaError) as raised:
        bootstrap_database(database_path)
    assert raised.value.found == version
    assert "never migrates" in str(raised.value)
    with pytest.raises(MigrationRequiredError):
        open_runtime_database(database_path, actor=RuntimeActor.API)
    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (version,)
        assert connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        ).fetchall() == [("evidence_events",)]
    finally:
        connection.close()


def test_bootstrap_refuses_a_newer_schema(database_path: Path) -> None:
    bootstrap_database(database_path)
    _raw_execute(database_path, "PRAGMA user_version = 20")

    with pytest.raises(NewerSchemaError):
        bootstrap_database(database_path)
    with pytest.raises(NewerSchemaError):
        open_runtime_database(database_path, actor=RuntimeActor.API)


def test_bootstrap_refuses_a_versionless_database_that_already_holds_tables(
    database_path: Path,
) -> None:
    database_path.parent.mkdir(parents=True)
    _raw_execute(database_path, "CREATE TABLE stray (id INTEGER PRIMARY KEY)")

    with pytest.raises(SchemaLedgerError, match="no schema version"):
        bootstrap_database(database_path)


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


def test_runtime_refuses_absent_and_out_of_range_schemas(database_path: Path) -> None:
    with pytest.raises(MigrationRequiredError, match="not bootstrapped"):
        open_runtime_database(database_path, actor=RuntimeActor.API)

    bootstrap_database(database_path)
    with pytest.raises(MigrationRequiredError):
        open_runtime_database(
            database_path,
            actor=RuntimeActor.API,
            compatibility=SchemaCompatibility(minimum=20, maximum=20),
        )
    with pytest.raises(NewerSchemaError):
        open_runtime_database(
            database_path,
            actor=RuntimeActor.API,
            compatibility=SchemaCompatibility(minimum=0, maximum=0),
        )


def test_runtime_denies_ddl_and_schema_ledger_writes_and_allows_execution_api_writes(
    database_path: Path,
) -> None:
    bootstrap_database(database_path)

    api = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        with write_transaction(api):
            api.execute(
                "INSERT INTO locations "
                "(location_id, kind, parent_location_id, parent_kind, name, order_index, "
                "created_at, updated_at) VALUES "
                "('floor-1', 'FLOOR', NULL, NULL, 'Floor 1', 0, "
                "'2026-08-24T00:00:00Z', '2026-08-24T00:00:00Z')"
            )
        with write_transaction(api):
            api.execute(
                "INSERT INTO execution_provenance ("
                "provenance_id, worker_build_revision, worker_image_digest, model_digest, "
                "calibration_digest, preprocessing_identity, config_digest, policy_identity, "
                "backend_build_revision, first_seen_ns"
                ") VALUES (?, 'rev', 'img', 'model', 'cal', 'pre', 'cfg', 'pol', 'be', 0)",
                (_PROVENANCE_ID,),
            )
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api.execute("UPDATE schema_migrations SET name = 'forged' WHERE version = 19")
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            api.execute("CREATE TABLE runtime_illegal (id INTEGER)")
    finally:
        api.close()


def test_forged_ledger_fails_runtime_and_bootstrap_consistently(database_path: Path) -> None:
    bootstrap_database(database_path)
    _raw_execute(
        database_path,
        "UPDATE schema_migrations SET name = 'forged', checksum = ? WHERE version = 19",
        ("f" * 64,),
    )

    with pytest.raises(SchemaLedgerError) as runtime_error:
        open_runtime_database(database_path, actor=RuntimeActor.API)
    with pytest.raises(SchemaLedgerError) as bootstrap_error:
        bootstrap_database(database_path)
    assert str(runtime_error.value) == str(bootstrap_error.value)
    assert str(runtime_error.value) == "applied schema ledger does not end at schema 19"


def test_bootstrap_refuses_while_a_runtime_holds_the_deployment_lock(
    database_path: Path,
) -> None:
    bootstrap_database(database_path)
    runtime = open_runtime_database(database_path, actor=RuntimeActor.API)
    try:
        with pytest.raises(DeploymentLockError, match="held by a running runtime"):
            bootstrap_database(database_path)
    finally:
        runtime.close()
    with deployment_lock(database_path.parent) as lock:
        with pytest.raises(DeploymentLockError, match="does not cover"):
            lock.require_for(database_path.parent.parent / "elsewhere.sqlite3")
        assert bootstrap_database(database_path, lock=lock).created is False


def test_bootstrap_cli_is_the_only_schema_entrypoint(database_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "backend.app.edge_db", "--database", os.fspath(database_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        f"EDGE_DB_BOOTSTRAP_OK path={database_path} schema=19 created=true extended=false\n"
    )


@pytest.mark.parametrize(
    ("corrupt_sql", "expected_error"),
    [
        (
            "UPDATE schema_migrations SET checksum = '" + ("0" * 64) + "' WHERE version = 19",
            "ledger",
        ),
        (
            "CREATE TABLE extra_table (id INTEGER PRIMARY KEY) STRICT",
            "application table",
        ),
        ("PRAGMA user_version = 17", "never migrates"),
    ],
)
def test_bootstrap_cli_refuses_corrupt_or_foreign_schema_without_success_marker(
    database_path: Path,
    corrupt_sql: str,
    expected_error: str,
) -> None:
    bootstrap_database(database_path)
    _raw_execute(database_path, corrupt_sql)

    completed = subprocess.run(
        [sys.executable, "-m", "backend.app.edge_db", "--database", os.fspath(database_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "EDGE_DB_BOOTSTRAP_OK" not in completed.stdout
    assert "EDGE_DB_BOOTSTRAP_FAILED" in completed.stderr
    assert expected_error in completed.stderr.lower()


def test_zero_wait_policy_is_explicit(database_path: Path) -> None:
    bootstrap_database(database_path)
    connection = open_runtime_database(
        database_path,
        actor=RuntimeActor.API,
        busy_policy=BusyPolicy.ZERO_WAIT,
    )
    try:
        assert connection.execute("PRAGMA busy_timeout").fetchone() == (0,)
    finally:
        connection.close()
