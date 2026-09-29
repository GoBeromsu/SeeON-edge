"""The isolated live diagnostics schema provisioned beside the product schema."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.mapping import (
    DIAGNOSTICS_TABLES,
    DIAGNOSTICS_TARGET_TABLES,
    diagnostics_schema_name,
    postgres_table_names,
)
from backend.app.edge_db.migration.provision import (
    DIAGNOSTICS_SCHEMA_NAME,
    DIAGNOSTICS_SCHEMA_VERSION,
    diagnostics_checksum,
)
from backend.app.edge_db.migration.reconcile import reconcile
from backend.app.edge_db.migration.snapshot import export_snapshot
from backend.app.features.diagnostics.postgres_database import (
    DIAGNOSTICS_POOL_BUDGET,
    diagnostics_schema,
)
from backend.app.features.diagnostics.records import StorageState
from tests_support.postgres_migration import (
    MigrationNames,
    MigrationTarget,
    ingest_live_record,
    provision_target,
    runtime_role_database,
    source_and_destination,
    table_counts,
)

pytest_plugins = ("tests_support.postgres_migration",)


def _names(target: MigrationTarget) -> MigrationNames:
    return MigrationNames(
        admin=target.admin, dsn=target.dsn, schema=target.schema, runtime_role=target.runtime_role
    )


def _diagnostics_ledger(admin: psycopg.Connection, schema: str) -> list[tuple[object, ...]]:
    rows = admin.execute(
        sql.SQL("SELECT version, name, checksum, applied_at FROM {} ORDER BY version").format(
            sql.Identifier(diagnostics_schema_name(schema), "schema_migrations")
        )
    ).fetchall()
    return [tuple(row) for row in rows]


def _schema_exists(admin: psycopg.Connection, schema: str) -> bool:
    (exists,) = admin.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = %s)", (schema,)
    ).fetchone()
    return exists


@pytest.mark.parametrize("schema", ["seeon_edge", "site_a"])
def test_the_provisioned_name_is_the_one_the_runtime_derives(schema: str) -> None:
    derived = diagnostics_schema({"API_POSTGRES_SCHEMA": schema})

    assert derived == diagnostics_schema_name(schema) == f"{schema}_diagnostics"
    assert diagnostics_schema({}) == diagnostics_schema_name("seeon_edge")


def test_a_schema_without_room_for_the_suffix_is_refused_before_any_change(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    names = MigrationNames(
        admin=migration_names.admin,
        dsn=migration_names.dsn,
        schema="s" * 60,
        runtime_role=migration_names.runtime_role,
    )
    authority_path = tmp_path / "authority.json"

    with pytest.raises(
        MigrationError,
        match="^diagnostics schema must be a lowercase SQL identifier of at most 63 bytes$",
    ):
        provision_target(names, authority_path)

    assert not _schema_exists(names.admin, names.schema)
    assert not authority_path.exists()


def test_provision_creates_the_diagnostics_schema_with_its_own_ledger(
    migration_target: MigrationTarget,
) -> None:
    target = migration_target
    diagnostics = diagnostics_schema_name(target.schema)
    ledger_before = _diagnostics_ledger(target.admin, target.schema)
    (owner,) = target.admin.execute(
        "SELECT pg_catalog.pg_get_userbyid(nspowner) = current_user "
        "FROM pg_catalog.pg_namespace WHERE nspname = %s",
        (diagnostics,),
    ).fetchone()

    provision_target(_names(target), target.authority_path)

    assert [entry[:3] for entry in ledger_before] == [
        (DIAGNOSTICS_SCHEMA_VERSION, DIAGNOSTICS_SCHEMA_NAME, diagnostics_checksum())
    ]
    assert _diagnostics_ledger(target.admin, target.schema) == ledger_before
    assert owner is True
    assert postgres_table_names(target.admin, diagnostics) == DIAGNOSTICS_TARGET_TABLES
    assert postgres_table_names(target.admin, target.schema) >= DIAGNOSTICS_TABLES


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("UPDATE {ledger} SET version = 2", "is newer than this tool supports"),
        (
            "UPDATE {ledger} SET checksum = repeat('b', 64)",
            "ledger does not match this tool's schema",
        ),
        ("CREATE TABLE {stray} (id bigint)", "tables drifted from the provisioned set"),
    ],
    ids=["newer", "foreign-ledger", "drifted"],
)
def test_provision_refuses_a_diagnostics_schema_it_did_not_leave(
    migration_target: MigrationTarget, statement: str, message: str
) -> None:
    target = migration_target
    diagnostics = diagnostics_schema_name(target.schema)
    target.admin.execute(
        sql.SQL(statement).format(
            ledger=sql.Identifier(diagnostics, "schema_migrations"),
            stray=sql.Identifier(diagnostics, "stray"),
        )
    )
    ledger_before = _diagnostics_ledger(target.admin, target.schema)
    counts_before = table_counts(target.admin, diagnostics)
    authority_bytes = target.authority_path.read_bytes()

    with pytest.raises(MigrationError, match=f"^diagnostics schema {message}$"):
        provision_target(_names(target), target.authority_path)

    assert _diagnostics_ledger(target.admin, target.schema) == ledger_before
    assert table_counts(target.admin, diagnostics) == counts_before
    assert target.authority_path.read_bytes() == authority_bytes


def test_provision_refuses_a_foreign_diagnostics_schema_and_creates_nothing(
    migration_names: MigrationNames, tmp_path: Path
) -> None:
    names = migration_names
    diagnostics = diagnostics_schema_name(names.schema)
    names.admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(diagnostics)))
    names.admin.execute(
        sql.SQL("CREATE TABLE {} (id bigint)").format(sql.Identifier(diagnostics, "stray"))
    )
    authority_path = tmp_path / "authority.json"

    with pytest.raises(
        MigrationError, match="^diagnostics schema has objects but no migration ledger$"
    ):
        provision_target(names, authority_path)

    assert postgres_table_names(names.admin, diagnostics) == frozenset({"stray"})
    assert not _schema_exists(names.admin, names.schema)
    assert not authority_path.exists()


@pytest.mark.parametrize(
    ("statement", "table"),
    [
        ("CREATE TABLE {} (id bigint)", "stray"),
        ("DROP TABLE {}", "execution_records"),
        ("TRUNCATE {}", "execution_records"),
        ("ALTER TABLE {} ADD COLUMN stray bigint", "execution_records"),
        ("UPDATE {} SET version = version", "schema_migrations"),
    ],
    ids=["create", "drop", "truncate", "alter", "rewrite-ledger"],
)
def test_runtime_role_holds_dml_but_no_ddl_on_diagnostics(
    migration_target: MigrationTarget, statement: str, table: str
) -> None:
    target = migration_target
    admin = target.admin
    diagnostics = diagnostics_schema_name(target.schema)

    with admin.transaction():
        admin.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(target.runtime_role)))
        visible = {
            name: admin.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(diagnostics, name))
            ).fetchone()[0]
            for name in sorted(DIAGNOSTICS_TARGET_TABLES)
        }
        admin.execute(
            sql.SQL("DELETE FROM {}").format(sql.Identifier(diagnostics, "execution_records"))
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege), admin.transaction():
            admin.execute(sql.SQL(statement).format(sql.Identifier(diagnostics, table)))

    assert visible == {name: int(name == "schema_migrations") for name in visible}
    assert postgres_table_names(admin, diagnostics) == DIAGNOSTICS_TARGET_TABLES


def test_lane_e_store_writes_live_records_as_the_runtime_role(
    migration_target: MigrationTarget,
) -> None:
    target = migration_target
    diagnostics = diagnostics_schema({"API_POSTGRES_SCHEMA": target.schema})
    product_before = table_counts(target.admin, target.schema)

    with runtime_role_database(
        target.dsn, diagnostics, target.runtime_role, DIAGNOSTICS_POOL_BUDGET
    ) as database:
        session_user = database.transact(
            lambda connection: connection.execute("SELECT current_user").fetchone()[0]
        )
        receipt = ingest_live_record(database, "live-1")
        replay = ingest_live_record(database, "live-1")

    assert session_user == target.runtime_role
    assert (receipt.accepted, receipt.storage_state) == (1, StorageState.COMMITTED)
    assert replay == receipt
    assert table_counts(target.admin, diagnostics)["execution_records"] == 1
    assert table_counts(target.admin, target.schema) == product_before


def test_reconcile_reports_diagnostics_as_live_not_reconciled(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    diagnostics = diagnostics_schema_name(target.schema)
    source, destination = source_and_destination(tmp_path)
    snapshot = export_snapshot(source, destination).path
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)
    with runtime_role_database(
        target.dsn, diagnostics, target.runtime_role, DIAGNOSTICS_POOL_BUDGET
    ) as database:
        ingest_live_record(database, "live-1")

    live = reconcile(target.database, schema=target.schema, snapshot_path=snapshot)
    written = table_counts(target.admin, diagnostics)
    del written["schema_migrations"]
    target.admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(diagnostics)))
    missing = reconcile(target.database, schema=target.schema, snapshot_path=snapshot)

    assert (live["result"], live["failures"]) == ("PASS", [])
    assert live["diagnostics"] == {
        "schema": diagnostics,
        "mode": "live",
        "reconciled": False,
        "ledger": "PASS",
        "tables": "PASS",
        "rows": written,
        "result": "PASS",
    }
    assert (sorted(written), written["execution_records"]) == (sorted(DIAGNOSTICS_TABLES), 1)
    assert (missing["result"], missing["failures"]) == ("FAIL", ["diagnostics:schema"])
    assert missing["diagnostics"] == {
        "schema": diagnostics,
        "mode": "live",
        "reconciled": False,
        "ledger": "FAIL",
        "tables": "FAIL",
        "rows": None,
        "result": "FAIL",
    }
    assert missing["tables"] == live["tables"]
