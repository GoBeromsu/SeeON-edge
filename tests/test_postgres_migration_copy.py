"""Export, import and reconcile of the old SQLite database into a real PG18 target."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from backend.app.edge_db.migration.errors import MigrationError
from backend.app.edge_db.migration.load import import_snapshot
from backend.app.edge_db.migration.reconcile import reconcile
from backend.app.edge_db.migration.rollback import rollback_check
from backend.app.edge_db.migration.snapshot import export_snapshot
from backend.app.edge_db.migration.sqlite_fence import fence_sqlite
from backend.app.edge_db.postgres import PostgresUnavailable
from tests_support.postgres_migration import (
    NOW,
    MigrationTarget,
    add_incident,
    append_audit,
    authority_file_token,
    authority_row,
    ledger,
    open_source_writer,
    source_and_destination,
    table_counts,
)
from tests_support.sqlite_source import hold_runtime_lock

pytest_plugins = ("tests_support.postgres_migration",)

_PROVISIONED_ONLY = {"schema_migrations": 1, "deployment_authority": 1}


def _fail_copy_after(patch: pytest.MonkeyPatch, rows: int) -> None:
    """The COPY stream to the server breaks after `rows` rows were sent."""
    original = psycopg.Copy.write_row
    written = [0]

    def write_row(self: psycopg.Copy, row: object) -> None:
        written[0] += 1
        if written[0] > rows:
            raise psycopg.OperationalError("injected COPY stream loss")
        original(self, row)

    patch.setattr(psycopg.Copy, "write_row", write_row)


def _import(target: MigrationTarget, snapshot: Path) -> None:
    import_snapshot(target.database, schema=target.schema, snapshot_path=snapshot)


def _reconcile(target: MigrationTarget, snapshot: Path, **options: object) -> dict[str, object]:
    return reconcile(target.database, schema=target.schema, snapshot_path=snapshot, **options)


def _fence(target: MigrationTarget, source: Path, snapshot: Path, root: Path) -> Path:
    """Fence the old database at the provisioned generation and return its receipt."""
    receipts = root / "receipts"
    receipts.mkdir(mode=0o700)
    receipt = receipts / "fence.json"
    generation, _ = authority_file_token(target.authority_path)
    fence_sqlite(source, snapshot=snapshot, generation=generation, receipt=receipt)
    return receipt


def test_export_refuses_a_source_held_by_a_runtime(tmp_path: Path) -> None:
    source, destination = source_and_destination(tmp_path)

    with (
        hold_runtime_lock(source),
        pytest.raises(MigrationError, match="^source database is in use by a running runtime$"),
    ):
        export_snapshot(source, destination)

    assert list(destination.parent.iterdir()) == []


def test_snapshot_is_readable_only_by_its_owner(tmp_path: Path) -> None:
    source, destination = source_and_destination(tmp_path)

    snapshot = export_snapshot(source, destination).path

    assert snapshot.stat().st_mode & 0o777 == 0o600
    assert [path.name for path in destination.parent.iterdir()] == [destination.name]


def test_write_committed_before_the_boundary_reaches_postgres(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source, destination = source_and_destination(tmp_path)
    with closing(open_source_writer(source)) as writer:
        add_incident(writer, 2)
        append_audit(writer)
        # The writer is still open, so both commits live only in its WAL.
        snapshot = export_snapshot(source, destination).path

    _import(target, snapshot)
    report = _reconcile(target, snapshot, source_path=source)

    with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as reader:
        (source_incidents,) = reader.execute("SELECT count(*) FROM incidents").fetchone()
    counts = table_counts(target.admin, target.schema)
    assert source_incidents == 2
    assert (report["result"], report["failures"]) == ("PASS", [])
    assert report["boundary"]["live_source"]["result"] == "PASS"
    assert (counts["incidents"], counts["audit_events"]) == (2, 3)
    assert report["boundary"]["snapshot_audit_tail"] == 3
    assert report["boundary"]["target_audit_tail"] == 3


def test_write_accepted_after_the_boundary_is_reported(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source, destination = source_and_destination(tmp_path)
    snapshot = export_snapshot(source, destination).path
    with closing(open_source_writer(source)) as writer:
        add_incident(writer, 2)
        append_audit(writer)

    _import(target, snapshot)
    report = _reconcile(target, snapshot, source_path=source)

    live = {entry["table"]: entry for entry in report["boundary"]["live_source"]["tables"]}
    assert report["result"] == "FAIL"
    assert report["failures"] == ["live_source:incidents", "live_source:audit_events"]
    assert (live["incidents"]["only_in_snapshot"], live["incidents"]["only_in_live"]) == (0, 1)
    assert (live["audit_events"]["only_in_snapshot"], live["audit_events"]["only_in_live"]) == (
        0,
        1,
    )
    assert table_counts(target.admin, target.schema)["incidents"] == 1


@pytest.mark.parametrize(
    ("statement", "parameters", "table", "drift", "statuses"),
    [
        (
            "UPDATE {} SET label = 'relabelled' WHERE camera_id = 'camera-1'",
            (),
            "cameras",
            (0, 0, 1),
            ({"mapping_state": {"UNMAPPED": 1}}, {"mapping_state": {"UNMAPPED": 1}}),
        ),
        ("DELETE FROM {}", (), "credentials", (1, 0, 0), ({}, {})),
        (
            "UPDATE {} SET status = 'applied', applied_at = %s",
            (NOW,),
            "policies",
            (0, 0, 1),
            ({"status": {"pending": 1}}, {"status": {"applied": 1}}),
        ),
    ],
    ids=["mutated-row", "missing-key", "status-flip"],
)
def test_reconcile_detects_target_drift(
    migration_target: MigrationTarget,
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
    table: str,
    drift: tuple[int, int, int],
    statuses: tuple[dict[str, object], dict[str, object]],
) -> None:
    target = migration_target
    source, destination = source_and_destination(tmp_path)
    snapshot = export_snapshot(source, destination).path
    _import(target, snapshot)
    assert _reconcile(target, snapshot)["result"] == "PASS"

    target.admin.execute(
        sql.SQL(statement).format(sql.Identifier(target.schema, table)), parameters
    )
    report = _reconcile(target, snapshot)

    entry = next(entry for entry in report["tables"] if entry["table"] == table)
    assert report["result"] == "FAIL"
    assert f"table:{table}" in report["failures"]
    assert (entry["only_in_source"], entry["only_in_target"], entry["changed"]) == drift
    assert (entry["source"]["status"], entry["target"]["status"]) == statuses


def _delete_audit_row(target: MigrationTarget, audit_id: int) -> None:
    """Delete one audit row as the schema owner, past the trigger the runtime meets."""
    table = sql.Identifier(target.schema, "audit_events")
    trigger = sql.Identifier("audit_events_immutable_delete")
    with target.admin.transaction():
        target.admin.execute(sql.SQL("ALTER TABLE {} DISABLE TRIGGER {}").format(table, trigger))
        target.admin.execute(
            sql.SQL("DELETE FROM {} WHERE audit_id = %s").format(table), (audit_id,)
        )
        target.admin.execute(sql.SQL("ALTER TABLE {} ENABLE TRIGGER {}").format(table, trigger))


def test_reconcile_fails_a_target_that_lost_the_snapshot_audit_tail(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source, destination = source_and_destination(tmp_path)
    snapshot = export_snapshot(source, destination).path
    with closing(sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)) as old:
        audit_ids = [audit_id for (audit_id,) in old.execute("SELECT audit_id FROM audit_events")]
    _import(target, snapshot)
    imported = _reconcile(target, snapshot)

    _delete_audit_row(target, max(audit_ids))
    report = _reconcile(target, snapshot)

    assert (imported["result"], imported["failures"]) == ("PASS", [])
    assert report["result"] == "FAIL"
    assert "boundary:audit_tail" in report["failures"]
    assert (report["boundary"]["snapshot_audit_tail"], report["boundary"]["target_audit_tail"]) == (
        max(audit_ids),
        sorted(audit_ids)[-2],
    )


def test_copy_failure_leaves_the_target_empty_and_retryable(
    migration_target: MigrationTarget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = migration_target
    source, destination = source_and_destination(tmp_path)
    snapshot = export_snapshot(source, destination).path
    receipt = _fence(target, source, snapshot, tmp_path)

    with monkeypatch.context() as patch:
        _fail_copy_after(patch, 5)
        with pytest.raises(PostgresUnavailable):
            _import(target, snapshot)

    counts = table_counts(target.admin, target.schema)
    generation, token = authority_file_token(target.authority_path)
    verdict = rollback_check(
        target.database,
        schema=target.schema,
        snapshot_path=snapshot,
        source=source,
        fence_receipt=receipt,
    )
    assert {name: count for name, count in counts.items() if count} == _PROVISIONED_ONLY
    assert [entry[4] for entry in ledger(target.admin, target.schema)] == [None]
    assert authority_row(target.admin, target.schema) == (generation, token, False, False)
    assert (verdict["result"], verdict["reasons"]) == ("ALLOW", [])

    _import(target, snapshot)
    assert _reconcile(target, snapshot)["result"] == "PASS"


def test_second_import_is_refused_and_changes_nothing(
    migration_target: MigrationTarget, tmp_path: Path
) -> None:
    target = migration_target
    source, destination = source_and_destination(tmp_path)
    snapshot = export_snapshot(source, destination).path
    _import(target, snapshot)
    counts = table_counts(target.admin, target.schema)
    stamped = ledger(target.admin, target.schema)

    with pytest.raises(MigrationError, match="^target already holds an imported snapshot$"):
        _import(target, snapshot)

    assert table_counts(target.admin, target.schema) == counts
    assert ledger(target.admin, target.schema) == stamped
